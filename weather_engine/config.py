from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_prefix": "WEATHER_", "env_file": ".env", "extra": "ignore"}

    engine_host: str = "0.0.0.0"
    engine_port: int = 8321
    db_path: str = str(Path.home() / ".weather-trader" / "weather.duckdb")

    # Open-Meteo base URLs
    forecast_url: str = "https://api.open-meteo.com/v1/forecast"
    ensemble_url: str = "https://ensemble-api.open-meteo.com/v1/ensemble"
    historical_url: str = "https://archive-api.open-meteo.com/v1/archive"
    climate_url: str = "https://climate-api.open-meteo.com/v1/climate"
    air_quality_url: str = "https://air-quality-api.open-meteo.com/v1/air-quality"
    marine_url: str = "https://marine-api.open-meteo.com/v1/marine"
    flood_url: str = "https://flood-api.open-meteo.com/v1/flood"

    # NOAA CPC teleconnection URLs
    oni_url: str = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
    nao_url: str = "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii.table"
    ao_url: str = "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/ao.shtml"
    pna_url: str = "https://www.cpc.ncep.noaa.gov/data/teledoc/pna.shtml"
    teleconnection_interval: int = 43200  # 12h

    # Scheduling intervals (seconds)
    forecast_interval: int = 3600  # 1h
    ensemble_interval: int = 21600  # 6h
    air_quality_interval: int = 10800  # 3h
    marine_interval: int = 21600  # 6h
    deterministic_interval: int = 10800  # 3h
    flood_interval: int = 21600  # 6h

    # Trading safety
    max_bet_pct: float = 0.10  # 10% wallet per bet
    max_exposure_pct: float = 0.50  # 50% total exposure
    min_edge: float = 0.05  # 5% minimum edge
    min_confidence: float = 0.60
    min_hours_to_expiry: int = 2

    # Polymarket
    polymarket_private_key: str = ""
    polymarket_api_key: str = ""
    polymarket_api_secret: str = ""
    polymarket_api_passphrase: str = ""
    polygon_rpc_url: str = "https://polygon-rpc.com"

    # Telegram
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    telegram_allowed_chat_ids: str = ""
    telegram_allowed_user_ids: str = ""

    # Local AI assistant (Ollama)
    ai_ollama_url: str = "http://127.0.0.1:11434"
    ai_ollama_model: str = "qwen3:8b"
    ai_ollama_timeout: int = 45


settings = Settings()
