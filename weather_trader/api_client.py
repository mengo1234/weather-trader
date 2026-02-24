"""API client with logging and response time tracking."""

import logging
import time

import httpx

from weather_trader.constants import API_BASE

logger = logging.getLogger("weather_trader.api")


def api_get(path: str, timeout: float = 15) -> dict | None:
    """GET request to weather engine API."""
    t0 = time.monotonic()
    try:
        r = httpx.get(f"{API_BASE}{path}", timeout=timeout)
        elapsed = (time.monotonic() - t0) * 1000
        r.raise_for_status()
        logger.debug("GET %s → %d (%.0fms)", path, r.status_code, elapsed)
        return r.json()
    except Exception as e:
        elapsed = (time.monotonic() - t0) * 1000
        logger.warning("GET %s failed (%.0fms): %s", path, elapsed, e)
        return None


def api_post(path: str, data: dict | None = None, timeout: float = 30) -> dict | None:
    """POST request to weather engine API."""
    t0 = time.monotonic()
    try:
        r = httpx.post(f"{API_BASE}{path}", json=data or {}, timeout=timeout)
        elapsed = (time.monotonic() - t0) * 1000
        r.raise_for_status()
        logger.debug("POST %s → %d (%.0fms)", path, r.status_code, elapsed)
        return r.json()
    except Exception as e:
        elapsed = (time.monotonic() - t0) * 1000
        logger.warning("POST %s failed (%.0fms): %s", path, elapsed, e)
        return None
