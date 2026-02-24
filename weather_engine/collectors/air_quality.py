import logging
from datetime import date

import numpy as np
import pandas as pd

from weather_engine.collectors.base import BaseCollector, om_client
from weather_engine.config import settings

logger = logging.getLogger(__name__)

HOURLY_VARIABLES = [
    "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
    "sulphur_dioxide", "ozone", "european_aqi", "us_aqi",
    "dust", "uv_index", "ammonia",
]

# Indices into HOURLY_VARIABLES for extraction
_IDX = {v: i for i, v in enumerate(HOURLY_VARIABLES)}


class AirQualityCollector(BaseCollector):
    name = "air_quality"

    def collect(self, city: dict, db=None) -> int:
        response = om_client.weather_api(
            settings.air_quality_url,
            params={
                "latitude": city["latitude"],
                "longitude": city["longitude"],
                "hourly": ",".join(HOURLY_VARIABLES),
                "forecast_days": 5,
            },
        )[0]

        hourly = response.Hourly()
        if hourly is None:
            return 0

        times = pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        )

        n = len(times)
        if n == 0:
            return 0

        # Extract all variable arrays
        arrays = {}
        for var_name, idx in _IDX.items():
            if idx < hourly.VariablesLength():
                arrays[var_name] = hourly.Variables(idx).ValuesAsNumpy()
            else:
                arrays[var_name] = np.full(n, np.nan)

        self._store(db, city["slug"], times, arrays)
        logger.info("Air quality %s: %d hourly readings stored", city["slug"], n)
        return n

    def _store(self, db, city_slug, times, arrays):
        """Store air quality data into DuckDB."""
        db = db or self.db
        rows = []
        for i, t in enumerate(times):
            forecast_date = t.date()
            hour = t.hour

            def _f(arr, idx):
                v = arr[idx]
                return float(v) if not np.isnan(v) else None

            def _i(arr, idx):
                v = arr[idx]
                return int(v) if not np.isnan(v) else None

            rows.append([
                city_slug, forecast_date, hour,
                _f(arrays["pm10"], i),
                _f(arrays["pm2_5"], i),
                _f(arrays["ozone"], i),
                _f(arrays["nitrogen_dioxide"], i),
                _i(arrays["european_aqi"], i),
                _f(arrays["carbon_monoxide"], i),
                _f(arrays["sulphur_dioxide"], i),
                _i(arrays["us_aqi"], i),
                _f(arrays["dust"], i),
                _f(arrays["uv_index"], i),
                _f(arrays["ammonia"], i),
            ])

        if rows:
            try:
                db.executemany(
                    """INSERT OR REPLACE INTO air_quality
                    (city_slug, forecast_date, hour, pm10, pm25, ozone, no2, aqi_european,
                     co, so2, us_aqi, dust, uv_index, ammonia)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    rows,
                )
            except Exception as e:
                logger.debug("Air quality batch insert failed for %s: %s", city_slug, e)
