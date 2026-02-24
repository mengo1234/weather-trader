import logging
from datetime import datetime, timezone

import httpx

from weather_engine.collectors.base import BaseCollector

logger = logging.getLogger(__name__)

SEASONAL_URL = "https://seasonal-api.open-meteo.com/v1/seasonal"

VARIABLES = [
    "temperature_2m",
    "precipitation",
    "wind_speed_10m",
]


class SeasonalCollector(BaseCollector):
    name = "seasonal"

    def collect(self, city: dict, db=None) -> int:
        try:
            resp = httpx.get(
                SEASONAL_URL,
                params={
                    "latitude": city["latitude"],
                    "longitude": city["longitude"],
                    "six_hourly": ",".join(VARIABLES),
                    "temperature_unit": "fahrenheit",
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            n = 0
            if "six_hourly" in data and "time" in data["six_hourly"]:
                times = data["six_hourly"]["time"]
                temps = data["six_hourly"].get("temperature_2m", [])
                precips = data["six_hourly"].get("precipitation", [])
                n = len(times)

                self._store(db, city["slug"], times, temps, precips)

            logger.info("Seasonal %s: %d six-hourly readings stored", city["slug"], n)
            return n
        except Exception as e:
            logger.warning("Seasonal %s failed: %s", city["slug"], e)
            return 0

    def _store(self, db, city_slug, times, temps, precips):
        """Aggregate six-hourly data into monthly anomalies and store."""
        db = db or self.db

        # Group by month/year and compute monthly means
        monthly = {}
        for i, t_str in enumerate(times):
            try:
                dt = datetime.fromisoformat(t_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                continue
            key = (dt.month, dt.year)
            if key not in monthly:
                monthly[key] = {"temps": [], "precips": []}
            if i < len(temps) and temps[i] is not None:
                monthly[key]["temps"].append(temps[i])
            if i < len(precips) and precips[i] is not None:
                monthly[key]["precips"].append(precips[i])

        for (month, year), vals in monthly.items():
            temp_anomaly = None
            precip_anomaly = None
            if vals["temps"]:
                # Anomaly relative to midpoint (rough approximation)
                mean_temp = sum(vals["temps"]) / len(vals["temps"])
                temp_anomaly = mean_temp - 50.0  # rough baseline in F
            if vals["precips"]:
                mean_precip = sum(vals["precips"]) / len(vals["precips"])
                precip_anomaly = mean_precip

            try:
                db.execute(
                    """INSERT OR REPLACE INTO seasonal_forecast
                    (city_slug, month, year, temp_anomaly, precip_anomaly, collected_at)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                    [city_slug, month, year, temp_anomaly, precip_anomaly,
                     datetime.now(timezone.utc)],
                )
            except Exception as e:
                logger.debug("Seasonal insert failed for %s %d/%d: %s",
                             city_slug, month, year, e)
