import logging
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import duckdb
import openmeteo_requests
import requests_cache
from retry_requests import retry

from weather_engine.db import get_db

logger = logging.getLogger(__name__)

_session = requests_cache.CachedSession(".cache/open_meteo", expire_after=1800)
_retry_session = retry(_session, retries=3, backoff_factor=0.5)
om_client = openmeteo_requests.Client(session=_retry_session)


class BaseCollector(ABC):
    name: str = "base"
    max_workers: int = 6

    def __init__(self, db: duckdb.DuckDBPyConnection | None = None):
        self.db = db or get_db()

    def collect_all(self, cities: list[dict]) -> int:
        """Collect data for all cities in parallel using ThreadPoolExecutor."""
        total = 0
        t0 = time.monotonic()

        def _collect_city(city):
            # Each thread gets its own DB cursor for thread safety
            db = get_db()

            # Circuit breaker check
            source_key = f"{self.name}_{city['slug']}"
            try:
                from weather_engine.resilience.circuit_breaker import is_available, record_success, record_failure
                if not is_available(db, source_key):
                    logger.info("Circuit open for %s, skipping", source_key)
                    return city["slug"], 0, "circuit_open"
            except Exception:
                pass  # If circuit breaker module not available, proceed normally

            try:
                n = self.collect(city, db=db)
                self._log(city["slug"], "success", n, db=db)
                try:
                    from weather_engine.resilience.circuit_breaker import record_success
                    record_success(db, source_key)
                except Exception:
                    pass
                return city["slug"], n, None
            except Exception as e:
                logger.error("Collector %s failed for %s: %s", self.name, city["slug"], e)
                self._log(city["slug"], "error", 0, str(e), db=db)
                try:
                    from weather_engine.resilience.circuit_breaker import record_failure
                    record_failure(db, source_key)
                except Exception:
                    pass
                return city["slug"], 0, str(e)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(_collect_city, city): city for city in cities}
            for future in as_completed(futures):
                slug, n, err = future.result()
                total += n

        elapsed = time.monotonic() - t0
        logger.info("Collector %s: %d cities, %d rows total in %.1fs",
                     self.name, len(cities), total, elapsed)
        return total

    @abstractmethod
    def collect(self, city: dict, db=None) -> int:
        ...

    def _log(self, city_slug: str, status: str, rows: int, error: str | None = None, db=None) -> None:
        db = db or self.db
        now = datetime.now(timezone.utc)
        db.execute(
            """INSERT INTO collection_log (collector, city_slug, started_at, finished_at, status, rows_inserted, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [self.name, city_slug, now, now, status, rows, error],
        )
