"""Local backend bootstrap for desktop app (cross-platform).

Starts weather-engine automatically if available next to the desktop app repo
and not already running on localhost:8321.
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger("weather_trader.bootstrap")

BACKEND_URL = os.getenv("WEATHER_API_BASE", "http://localhost:8321").rstrip("/")
HEALTH_URL = f"{BACKEND_URL}/health"

_BOOTSTRAP_STATE: dict[str, object] = {
    "attempted": False,
    "started": False,
    "reason": None,
    "engine_dir": None,
    "command": None,
    "log_path": None,
}


def _health_ok(timeout: float = 1.2) -> bool:
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.get(HEALTH_URL)
            return r.status_code == 200
    except Exception:
        return False


def _candidate_engine_dirs() -> list[Path]:
    candidates: list[Path] = []

    env_dir = os.getenv("WEATHER_ENGINE_DIR")
    if env_dir:
        candidates.append(Path(env_dir))

    # Source checkout: Bot/desktop-app/weather_trader/backend_bootstrap.py -> sibling weather-engine
    try:
        candidates.append(Path(__file__).resolve().parents[2] / "weather-engine")
    except Exception:
        pass

    # Frozen app next to a copied "weather-engine" directory
    try:
        exe_dir = Path(sys.executable).resolve().parent
        candidates.append(exe_dir / "weather-engine")
    except Exception:
        pass

    # macOS .app bundle common layouts (Contents/MacOS/WeatherTrader -> sibling Resources/weather-engine)
    try:
        exe_dir = Path(sys.executable).resolve().parent
        candidates.append(exe_dir.parent / "Resources" / "weather-engine")
    except Exception:
        pass

    uniq: list[Path] = []
    seen = set()
    for c in candidates:
        key = str(c)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    return uniq


def _find_engine_dir() -> Optional[Path]:
    for p in _candidate_engine_dirs():
        if (p / "pyproject.toml").exists() and (p / "src" / "weather_engine").exists():
            return p
    return None


def _build_backend_command(engine_dir: Path) -> tuple[list[str], bool]:
    """Return (command, use_shell)."""
    is_win = platform.system().lower().startswith("win")
    venv_dir = engine_dir / ".venv"
    uvicorn_bin = venv_dir / ("Scripts/uvicorn.exe" if is_win else "bin/uvicorn")
    py_bin = venv_dir / ("Scripts/python.exe" if is_win else "bin/python")

    if uvicorn_bin.exists():
        return [str(uvicorn_bin), "weather_engine.main:app", "--host", "0.0.0.0", "--port", "8321"], False

    if py_bin.exists():
        return [str(py_bin), "-m", "uvicorn", "weather_engine.main:app", "--host", "0.0.0.0", "--port", "8321"], False

    if shutil.which("uv"):
        return ["uv", "run", "uvicorn", "weather_engine.main:app", "--host", "0.0.0.0", "--port", "8321"], False

    python_exe = shutil.which("python3") or shutil.which("python")
    if python_exe:
        return [python_exe, "-m", "uvicorn", "weather_engine.main:app", "--host", "0.0.0.0", "--port", "8321"], False

    raise RuntimeError("Nessun runtime trovato (uv / python / .venv) per avviare weather-engine")


def get_bootstrap_state() -> dict[str, object]:
    return dict(_BOOTSTRAP_STATE)


def _start_embedded(wait_seconds: float) -> bool:
    """Try to start the backend in-process via a daemon thread (no external process needed)."""
    try:
        import uvicorn
        from weather_engine.main import app as backend_app  # noqa: F811
    except ImportError as e:
        logger.info("Embedded backend not available: %s", e)
        return False

    import threading

    def _serve():
        uvicorn.run(backend_app, host="127.0.0.1", port=8321, log_level="warning")

    thread = threading.Thread(target=_serve, daemon=True, name="weather-engine")
    thread.start()
    logger.info("Embedded backend thread started")

    _BOOTSTRAP_STATE.update({"command": "embedded (in-process uvicorn)", "engine_dir": "bundled"})

    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if _health_ok(timeout=1.0):
            return True
        time.sleep(0.5)
    return False


def ensure_backend_started(wait_seconds: float = 12.0) -> dict[str, object]:
    """Ensure backend is running; start it in background if possible.

    Tries in order:
    1. Check if already running externally
    2. Start embedded (in-process, bundled weather_engine package)
    3. Start external process (sibling weather-engine directory)

    Safe to call multiple times.
    """
    _BOOTSTRAP_STATE["attempted"] = True

    if _health_ok():
        _BOOTSTRAP_STATE.update({"started": True, "reason": "already_running"})
        return get_bootstrap_state()

    # --- Try embedded backend first (works in distributed binaries) ---
    if _start_embedded(wait_seconds=min(wait_seconds, 15.0)):
        _BOOTSTRAP_STATE.update({"started": True, "reason": "embedded"})
        return get_bootstrap_state()

    # --- Fallback: external process (dev mode, sibling directory) ---
    engine_dir = _find_engine_dir()
    if not engine_dir:
        _BOOTSTRAP_STATE.update({"started": False, "reason": "engine_not_found"})
        return get_bootstrap_state()

    try:
        cmd, use_shell = _build_backend_command(engine_dir)
    except Exception as e:
        _BOOTSTRAP_STATE.update({"started": False, "reason": f"command_error: {e}", "engine_dir": str(engine_dir)})
        return get_bootstrap_state()

    log_path = Path(tempfile.gettempdir()) / "weather-engine.log"
    _BOOTSTRAP_STATE.update(
        {
            "engine_dir": str(engine_dir),
            "command": " ".join(cmd),
            "log_path": str(log_path),
        }
    )

    try:
        log_file = open(log_path, "a", encoding="utf-8")
        popen_kwargs = dict(
            cwd=str(engine_dir),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            shell=use_shell,
            env=os.environ.copy(),
        )
        if platform.system().lower().startswith("win"):
            popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(
                subprocess, "DETACHED_PROCESS", 0
            )
        else:
            popen_kwargs["start_new_session"] = True

        subprocess.Popen(cmd, **popen_kwargs)
        logger.info("Backend bootstrap started: %s (cwd=%s)", cmd, engine_dir)
    except Exception as e:
        _BOOTSTRAP_STATE.update({"started": False, "reason": f"spawn_failed: {e}"})
        logger.error("Backend bootstrap failed: %s", e, exc_info=True)
        return get_bootstrap_state()

    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if _health_ok(timeout=1.0):
            _BOOTSTRAP_STATE.update({"started": True, "reason": "started_by_desktop"})
            return get_bootstrap_state()
        time.sleep(0.5)

    _BOOTSTRAP_STATE.update({"started": False, "reason": "startup_timeout"})
    return get_bootstrap_state()
