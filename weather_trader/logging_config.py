"""Logging configuration with file rotation."""

import logging
import logging.handlers
from pathlib import Path

LOG_DIR = Path.home() / ".weather-trader" / "logs"


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure logging with console + rotating file handler."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / "app.log"

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Rotating file handler: 5 MB max, 3 backups
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    file_handler.setLevel(level)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    console_handler.setLevel(level)

    root = logging.getLogger("weather_trader")
    root.setLevel(level)
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    return root
