import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from weather_engine.collectors.base import BaseCollector, om_client
from weather_engine.config import settings

logger = logging.getLogger(__name__)

DAILY_VARIABLES = [
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "precipitation_sum", "wind_speed_10m_max",
]

CLIMATE_MODELS = ["EC_Earth3P_HR", "FGOALS_f3_H", "HiRAM_SIT_HR"]


class ClimateCollector(BaseCollector):
    name = "climate"

    def collect(self, city: dict, db=None, start_date: str = "2020-01-01", end_date: str = "2050-12-31") -> int:
        total = 0
        for model in CLIMATE_MODELS:
            try:
                response = om_client.weather_api(
                    settings.climate_url,
                    params={
                        "latitude": city["latitude"],
                        "longitude": city["longitude"],
                        "daily": ",".join(DAILY_VARIABLES),
                        "models": model,
                        "start_date": start_date,
                        "end_date": end_date,
                        "temperature_unit": "fahrenheit",
                    },
                )[0]

                daily = response.Daily()
                if daily is None:
                    continue

                times = pd.date_range(
                    start=pd.to_datetime(daily.Time(), unit="s", utc=True),
                    end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
                    freq=pd.Timedelta(seconds=daily.Interval()),
                    inclusive="left",
                )

                n = len(times)
                if n > 0:
                    # Extract variable arrays
                    temp_max = daily.Variables(0).ValuesAsNumpy() if daily.VariablesLength() > 0 else np.full(n, np.nan)
                    temp_min = daily.Variables(1).ValuesAsNumpy() if daily.VariablesLength() > 1 else np.full(n, np.nan)
                    temp_mean = daily.Variables(2).ValuesAsNumpy() if daily.VariablesLength() > 2 else np.full(n, np.nan)
                    precip = daily.Variables(3).ValuesAsNumpy() if daily.VariablesLength() > 3 else np.full(n, np.nan)
                    wind = daily.Variables(4).ValuesAsNumpy() if daily.VariablesLength() > 4 else np.full(n, np.nan)

                    self._store(db, city["slug"], model, times,
                                temp_max, temp_min, temp_mean, precip, wind)

                total += n
                logger.info("Climate %s/%s: %d days stored", city["slug"], model, n)
            except Exception as e:
                logger.warning("Climate %s/%s failed: %s", city["slug"], model, e)

        return total

    def _store(self, db, city_slug, model, times, temp_max, temp_min, temp_mean, precip, wind):
        """Store climate indicator data into DuckDB."""
        db = db or self.db
        now = datetime.now(timezone.utc)

        for i, t in enumerate(times):
            d = t.date()
            t_max = float(temp_max[i]) if not np.isnan(temp_max[i]) else None
            t_min = float(temp_min[i]) if not np.isnan(temp_min[i]) else None
            t_mean = float(temp_mean[i]) if not np.isnan(temp_mean[i]) else None
            p = float(precip[i]) if not np.isnan(precip[i]) else None
            w = float(wind[i]) if not np.isnan(wind[i]) else None

            try:
                db.execute(
                    """INSERT OR REPLACE INTO climate_indicators
                    (city_slug, date, model, temp_max, temp_min, temp_mean,
                     precip_sum, wind_max, collected_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    [city_slug, d, model, t_max, t_min, t_mean, p, w, now],
                )
            except Exception as e:
                logger.debug("Climate insert failed for %s %s %s: %s",
                             city_slug, d, model, e)
