import os
import json
import logging
import platform
import shutil
import subprocess
import threading
import time
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException

from weather_engine.api.routes_analysis import get_accuracy
from weather_engine.api.routes_forecast import get_forecast
from weather_engine.api.routes_health import health, metrics
from weather_engine.api.routes_market import scan_markets
from weather_engine.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ai", tags=["ai"])

_INSTALL_LOCK = threading.Lock()
_INSTALL_STATE: dict[str, Any] = {
    "running": False,
    "status": "idle",  # idle | running | done | error
    "step": None,
    "error": None,
    "model": None,
    "logs": [],
    "updated_at": None,
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _install_log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    with _INSTALL_LOCK:
        _INSTALL_STATE["logs"].append(line)
        _INSTALL_STATE["logs"] = _INSTALL_STATE["logs"][-120:]
        _INSTALL_STATE["updated_at"] = time.time()
    logger.info("AI install: %s", msg)


def _set_install_state(**kwargs) -> None:
    with _INSTALL_LOCK:
        _INSTALL_STATE.update(kwargs)
        _INSTALL_STATE["updated_at"] = time.time()


def _run_cmd(command: list[str] | str, shell: bool = False, timeout: int = 1800) -> tuple[int, str]:
    proc = subprocess.run(
        command,
        shell=shell,
        text=True,
        capture_output=True,
        timeout=timeout,
        env=os.environ.copy(),
    )
    out = (proc.stdout or "") + ("\n" if proc.stdout and proc.stderr else "") + (proc.stderr or "")
    return proc.returncode, out.strip()


def _ollama_reachable(base_url: str | None = None) -> bool:
    base = (base_url or str(settings.ai_ollama_url or "http://127.0.0.1:11434")).rstrip("/")
    try:
        with httpx.Client(timeout=2.0) as client:
            r = client.get(f"{base}/api/tags")
            r.raise_for_status()
        return True
    except Exception:
        return False


def _run_ollama_install(model: str) -> None:
    try:
        _set_install_state(running=True, status="running", step="checking", error=None, model=model)
        _install_log(f"Avvio installazione Ollama + modello '{model}'")
        _install_log(f"Piattaforma: {platform.system()} {platform.release()}")

        if platform.system().lower() != "linux":
            raise RuntimeError("Installazione automatica supportata solo su Linux in questa versione")

        ollama_bin = shutil.which("ollama")
        if not ollama_bin:
            _set_install_state(step="installing_ollama")
            _install_log("Ollama non trovato. Eseguo installer ufficiale (curl | sh)")
            code, out = _run_cmd("curl -fsSL https://ollama.com/install.sh | sh", shell=True, timeout=1800)
            if out:
                for line in out.splitlines()[-20:]:
                    _install_log(line[:500])
            if code != 0:
                raise RuntimeError(f"Installer Ollama fallito (exit {code})")
            ollama_bin = shutil.which("ollama")
            if not ollama_bin:
                ollama_bin = "ollama"  # try anyway
        else:
            _install_log(f"Ollama già presente: {ollama_bin}")

        if not _ollama_reachable():
            _set_install_state(step="starting_service")
            _install_log("Ollama non raggiungibile su 11434. Provo ad avviare 'ollama serve' in background")
            try:
                subprocess.Popen(
                    [ollama_bin or "ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                time.sleep(3)
            except Exception as e:
                _install_log(f"Avvio 'ollama serve' fallito: {e}")
            if _ollama_reachable():
                _install_log("Ollama API raggiungibile")
            else:
                _install_log("Ollama API ancora non raggiungibile (potrebbe partire come servizio separato)")

        _set_install_state(step="pull_model")
        _install_log(f"Eseguo pull del modello: {model}")
        code, out = _run_cmd([ollama_bin or "ollama", "pull", model], shell=False, timeout=5400)
        if out:
            for line in out.splitlines()[-30:]:
                _install_log(line[:500])
        if code != 0:
            raise RuntimeError(f"'ollama pull {model}' fallito (exit {code})")

        _set_install_state(running=False, status="done", step="completed", error=None)
        _install_log("Installazione completata")
    except Exception as e:
        _set_install_state(running=False, status="error", step="failed", error=str(e))
        _install_log(f"ERRORE: {e}")


def _summarize_forecast(city_slug: str) -> dict[str, Any]:
    data = get_forecast(city_slug, days=7)
    rows = (data or {}).get("forecast", []) or []
    city = (data or {}).get("city", {}) or {}

    if not rows:
        return {"city": city, "days": [], "summary": "Nessun forecast disponibile"}

    compact_days = []
    hottest = None
    windiest = None
    rainiest = None
    for row in rows[:7]:
        ens = row.get("ensemble", {}) or {}
        spread = None
        if ens.get("ensemble_max") is not None and ens.get("ensemble_min") is not None:
            spread = round(_safe_float(ens.get("ensemble_max")) - _safe_float(ens.get("ensemble_min")), 1)
        day = {
            "date": row.get("date"),
            "temp_max": row.get("temp_max"),
            "temp_min": row.get("temp_min"),
            "precip_mm": row.get("precipitation_sum"),
            "precip_probability": row.get("precip_probability"),
            "snow_mm": row.get("snowfall_sum"),
            "wind_max": row.get("wind_max"),
            "wind_gusts_max": row.get("wind_gusts_max"),
            "uv_max": row.get("uv_max"),
            "ensemble_spread": spread,
        }
        compact_days.append(day)

        if hottest is None or _safe_float(day.get("temp_max"), -999) > _safe_float(hottest.get("temp_max"), -999):
            hottest = day
        if windiest is None or _safe_float(day.get("wind_gusts_max"), -999) > _safe_float(windiest.get("wind_gusts_max"), -999):
            windiest = day
        if rainiest is None or _safe_float(day.get("precip_mm"), -1) > _safe_float(rainiest.get("precip_mm"), -1):
            rainiest = day

    return {
        "city": city,
        "days": compact_days,
        "summary": {
            "hottest_day": hottest,
            "windiest_day": windiest,
            "rainiest_day": rainiest,
        },
    }


def _summarize_live_markets(city_slug: str) -> dict[str, Any]:
    try:
        scan = scan_markets(min_edge=0.0)
    except Exception as e:
        logger.warning("AI market scan failed: %s", e)
        return {"ok": False, "error": str(e), "total": 0, "matched": []}

    markets = (scan or {}).get("markets", []) or []
    matched = []
    for m in markets:
        meta = m.get("metadata", {}) or {}
        rec = m.get("recommendation", {}) or {}
        city_meta = (meta.get("city") or {}).get("slug")
        rec_city = str(rec.get("city") or "").strip().lower()
        if city_meta != city_slug and rec_city != city_slug:
            continue
        best_bet = rec.get("best_bet")
        if not best_bet:
            continue
        outcomes = rec.get("outcomes", []) or []
        best_row = next((o for o in outcomes if o.get("outcome") == best_bet), {})
        matched.append({
            "question": rec.get("market_question") or m.get("question"),
            "date": rec.get("date") or meta.get("target_date"),
            "best_bet": best_bet,
            "edge_pct": round(_safe_float(best_row.get("edge")) * 100, 2),
            "prob_pct": round(_safe_float(best_row.get("our_probability")) * 100, 1),
            "price_pct": round(_safe_float(best_row.get("market_price")) * 100, 1),
            "confidence_pct": round(_safe_float(best_row.get("confidence")) * 100, 1),
            "ev_pct": round(_safe_float(rec.get("expected_value")) * 100, 2),
            "volume": (m.get("market") or {}).get("volume") or m.get("volume"),
        })

    matched.sort(key=lambda x: (x.get("edge_pct") or 0, x.get("ev_pct") or 0), reverse=True)
    return {
        "ok": True,
        "total": len(markets),
        "matched": matched[:5],
        "message": scan.get("message"),
        "n_scanned": scan.get("n_scanned"),
        "n_parseable": scan.get("n_parseable"),
    }


def _summarize_system(city_slug: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        out["health"] = health()
    except Exception as e:
        out["health"] = {"error": str(e)}
    try:
        out["metrics"] = metrics()
    except Exception as e:
        out["metrics"] = {"error": str(e)}
    try:
        out["accuracy"] = get_accuracy(city_slug=city_slug, variable="temperature_2m_max")
    except Exception as e:
        out["accuracy"] = {"error": str(e)}
    return out


def _build_context(city_slug: str, section: str | None = None) -> dict[str, Any]:
    return {
        "city_slug": city_slug,
        "section": section or "",
        "forecast": _summarize_forecast(city_slug),
        "live_markets": _summarize_live_markets(city_slug),
        "system": _summarize_system(city_slug),
    }


def _build_fallback_reply(question: str, ctx: dict[str, Any]) -> str:
    forecast = (ctx.get("forecast") or {})
    live = (ctx.get("live_markets") or {})
    system = (ctx.get("system") or {})
    summary = (forecast.get("summary") or {})
    hottest = summary.get("hottest_day") or {}
    rainiest = summary.get("rainiest_day") or {}
    windiest = summary.get("windiest_day") or {}
    accuracy = (system.get("accuracy") or {}).get("accuracy") or {}
    mae = accuracy.get("mae")
    rmse = accuracy.get("rmse")
    bias = accuracy.get("bias")

    lines = [
        "Assistente locale (fallback): non ho ottenuto risposta da Ollama, ma ti lascio analisi su dati reali dell'app.",
        "",
        f"Domanda: {question}",
        "",
        f"Citta': {ctx.get('city_slug', '?')}",
    ]
    if hottest:
        lines.append(f"- Giorno piu' caldo: {hottest.get('date')} · Tmax {hottest.get('temp_max')}°C")
    if rainiest:
        lines.append(
            f"- Giorno piu' piovoso: {rainiest.get('date')} · {rainiest.get('precip_mm')} mm "
            f"(P pioggia {rainiest.get('precip_probability')}%)"
        )
    if windiest:
        lines.append(
            f"- Giorno piu' ventoso: {windiest.get('date')} · raffiche {windiest.get('wind_gusts_max')} km/h"
        )
    if mae is not None or rmse is not None or bias is not None:
        lines.append(
            f"- Accuracy Tmax: MAE={mae if mae is not None else 'n/a'} | "
            f"RMSE={rmse if rmse is not None else 'n/a'} | Bias={bias if bias is not None else 'n/a'}"
        )

    matched = live.get("matched") or []
    if matched:
        lines.append("")
        lines.append("Top mercati LIVE collegati:")
        for i, m in enumerate(matched[:3], 1):
            lines.append(
                f"{i}. {m.get('date')} · {m.get('best_bet')} | edge {m.get('edge_pct')}% | "
                f"conf {m.get('confidence_pct')}% | EV {m.get('ev_pct')}%"
            )
    else:
        lines.append("")
        lines.append("Mercati LIVE collegati: nessun match trovato o scan non disponibile.")

    lines.extend([
        "",
        "Per risposte AI complete installa/avvia Ollama locale e assicurati che il modello configurato sia disponibile.",
    ])
    return "\n".join(lines)


def _call_ollama_chat(question: str, ctx: dict[str, Any], history: list[dict[str, str]]) -> tuple[str, str]:
    base_url = str(settings.ai_ollama_url or "http://127.0.0.1:11434").rstrip("/")
    model = str(settings.ai_ollama_model or "").strip()
    if not model:
        raise RuntimeError("Modello Ollama non configurato")

    system_prompt = (
        "Sei l'assistente di Weather Trader. Rispondi in italiano, in modo operativo per scommesse meteo.\n"
        "Usa solo i dati forniti nel contesto. Se mancano dati, dillo chiaramente.\n"
        "Struttura preferita: Verdetto, Motivi, Rischi, Dati usati."
    )

    safe_history = []
    for msg in (history or [])[-8:]:
        role = str(msg.get("role") or "").lower()
        if role not in {"user", "assistant"}:
            continue
        content = str(msg.get("content") or "").strip()
        if not content:
            continue
        safe_history.append({"role": role, "content": content[:4000]})

    user_content = (
        f"Domanda utente:\n{question.strip()}\n\n"
        "Contesto reale app (JSON):\n"
        f"{json.dumps(ctx, ensure_ascii=False)}"
    )

    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            *safe_history,
            {"role": "user", "content": user_content},
        ],
        "options": {"temperature": 0.2},
    }

    with httpx.Client(timeout=float(settings.ai_ollama_timeout or 45)) as client:
        r = client.post(f"{base_url}/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()

    content = (((data or {}).get("message") or {}).get("content") or "").strip()
    if not content:
        raise RuntimeError("Risposta Ollama vuota")
    return content, model


@router.get("/status")
def ai_status():
    base_url = str(settings.ai_ollama_url or "http://127.0.0.1:11434").rstrip("/")
    model = str(settings.ai_ollama_model or "").strip()
    info: dict[str, Any] = {
        "provider": "ollama",
        "configured": bool(model),
        "model": model or None,
        "base_url": base_url,
        "reachable": False,
    }
    if not model:
        return info

    try:
        with httpx.Client(timeout=2.5) as client:
            r = client.get(f"{base_url}/api/tags")
            r.raise_for_status()
            tags = r.json()
        info["reachable"] = True
        names = [m.get("name") for m in (tags.get("models") or []) if isinstance(m, dict)]
        info["installed_models"] = names[:20]
        info["model_installed"] = model in names
    except Exception as e:
        info["error"] = str(e)
    return info


@router.get("/install/status")
def ai_install_status():
    with _INSTALL_LOCK:
        state = dict(_INSTALL_STATE)
        state["logs"] = list(_INSTALL_STATE.get("logs", []))
    state["ollama_in_path"] = bool(shutil.which("ollama"))
    state["ollama_reachable"] = _ollama_reachable()
    state["configured_model"] = str(settings.ai_ollama_model or "").strip() or None
    return state


@router.post("/install/ollama")
def ai_install_ollama(payload: dict[str, Any]):
    confirm = bool((payload or {}).get("confirm"))
    if not confirm:
        raise HTTPException(400, "Confirmation required: pass {'confirm': true}")

    model = str((payload or {}).get("model") or settings.ai_ollama_model or "").strip()
    if not model:
        raise HTTPException(400, "No model configured")
    # Apply selected model immediately for current process (used by /ai/chat and /ai/status).
    try:
        settings.ai_ollama_model = model
    except Exception:
        pass

    with _INSTALL_LOCK:
        if _INSTALL_STATE.get("running"):
            return {"started": False, "message": "Installazione già in corso", "state": dict(_INSTALL_STATE)}
        _INSTALL_STATE["logs"] = []
        _INSTALL_STATE["status"] = "running"
        _INSTALL_STATE["step"] = "queued"
        _INSTALL_STATE["error"] = None
        _INSTALL_STATE["running"] = True
        _INSTALL_STATE["model"] = model
        _INSTALL_STATE["updated_at"] = time.time()

    thread = threading.Thread(target=_run_ollama_install, args=(model,), daemon=True)
    thread.start()
    return {"started": True, "message": "Installazione avviata in background", "model": model}


@router.post("/chat")
def ai_chat(payload: dict[str, Any]):
    question = str((payload or {}).get("message") or "").strip()
    if not question:
        raise HTTPException(400, "Missing 'message'")

    city_slug = str((payload or {}).get("city_slug") or "nyc").strip().lower()
    section = str((payload or {}).get("section") or "").strip()
    history = (payload or {}).get("history") or []
    if not isinstance(history, list):
        history = []

    ctx = _build_context(city_slug=city_slug, section=section)

    provider = "fallback"
    model = None
    error = None
    try:
        reply, model = _call_ollama_chat(question, ctx, history)
        provider = "ollama"
    except Exception as e:
        logger.warning("AI chat fallback: %s", e)
        error = str(e)
        reply = _build_fallback_reply(question, ctx)

    return {
        "reply": reply,
        "provider": provider,
        "model": model,
        "error": error,
        "context": {
            "city_slug": city_slug,
            "section": section,
            "forecast_days": len(((ctx.get("forecast") or {}).get("days") or [])),
            "live_matches": len(((ctx.get("live_markets") or {}).get("matched") or [])),
        },
    }
