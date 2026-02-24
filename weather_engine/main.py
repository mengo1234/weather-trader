import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI

from weather_engine.api.routes_analysis import router as analysis_router
from weather_engine.api.routes_ai import router as ai_router
from weather_engine.api.routes_forecast import router as forecast_router
from weather_engine.api.routes_health import router as health_router
from weather_engine.api.routes_market import router as market_router
from weather_engine.config import settings
from weather_engine.db import get_db
from weather_engine.scheduler import start_scheduler, stop_scheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _initial_collection():
    """Raccolta iniziale in background per non bloccare lo startup."""
    from weather_engine.collectors.ensemble import EnsembleCollector
    from weather_engine.collectors.forecast import ForecastCollector
    from weather_engine.db import get_cities

    try:
        db = get_db()
        cities = get_cities(db)

        logger.info("Background: raccolta forecast per %d citta'...", len(cities))
        ForecastCollector(db).collect_all(cities)

        logger.info("Background: raccolta ensemble per %d citta'...", len(cities))
        EnsembleCollector(db).collect_all(cities)

        logger.info("Background: raccolta iniziale completata")
    except Exception as e:
        logger.error("Raccolta iniziale fallita: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Avvio Weather Engine sulla porta %d", settings.engine_port)
    get_db()  # Inizializza DB
    start_scheduler()

    # Raccolta dati in background (non blocca il server)
    thread = threading.Thread(target=_initial_collection, daemon=True)
    thread.start()

    yield

    # Shutdown
    stop_scheduler()
    logger.info("Weather Engine fermato")


app = FastAPI(
    title="Weather Engine",
    description="Weather analysis engine for Polymarket trading",
    version="0.1.0",
    lifespan=lifespan,
)

from weather_engine.api.error_handlers import register_error_handlers

register_error_handlers(app)

app.include_router(health_router)
app.include_router(forecast_router)
app.include_router(analysis_router)
app.include_router(market_router)
app.include_router(ai_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.engine_host, port=settings.engine_port)
