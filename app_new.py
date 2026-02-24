"""Weather Trader — Professional Desktop App (Flet 0.80+).

Slim entry point. All logic is in the weather_trader package.
"""

import logging

import flet as ft

from weather_trader.backend_bootstrap import ensure_backend_started, get_bootstrap_state
from weather_trader.main_layout import main

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("weather_trader.entry")


def _bootstrap_backend():
    """Try to auto-start local backend for portable/source usage."""
    state = ensure_backend_started(wait_seconds=10.0)
    logger.info("Backend bootstrap state: %s", state)


_bootstrap_backend()
ft.app(target=main)
