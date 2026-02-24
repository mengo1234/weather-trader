import logging
from datetime import datetime, timezone

import pandas as pd

from weather_engine.collectors.base import BaseCollector, om_client
from weather_engine.config import settings

logger = logging.getLogger(__name__)

HOURLY_VARIABLES = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "apparent_temperature", "pressure_msl", "surface_pressure",
    "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
    "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
    "precipitation", "rain", "snowfall", "snow_depth",
    "visibility", "cape",
    "shortwave_radiation", "direct_radiation", "diffuse_radiation",
    "uv_index",
]

DAILY_VARIABLES = [
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "apparent_temperature_max", "apparent_temperature_min",
    "precipitation_sum", "rain_sum", "snowfall_sum",
    "precipitation_hours", "precipitation_probability_max",
    "wind_speed_10m_max", "wind_gusts_10m_max", "wind_direction_10m_dominant",
    "shortwave_radiation_sum", "uv_index_max",
]


class ForecastCollector(BaseCollector):
    name = "forecast"

    def collect(self, city: dict, db=None) -> int:
        db = db or self.db
        model_run = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

        response = om_client.weather_api(
            settings.forecast_url,
            params={
                "latitude": city["latitude"],
                "longitude": city["longitude"],
                "hourly": ",".join(HOURLY_VARIABLES),
                "daily": ",".join(DAILY_VARIABLES),
                "timezone": city["timezone"],
                "forecast_days": 16,
                "temperature_unit": "fahrenheit",
            },
        )[0]

        n_hourly = self._store_hourly(response, city["slug"], model_run, db=db)
        n_daily = self._store_daily(response, city["slug"], model_run, db=db)

        logger.info("Forecast %s: %d hourly, %d daily rows", city["slug"], n_hourly, n_daily)
        return n_hourly + n_daily

    def _store_hourly(self, response, city_slug: str, model_run: datetime, db=None) -> int:
        db = db or self.db
        hourly = response.Hourly()
        if hourly is None:
            return 0

        times = pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        )

        rows = []
        for i, t in enumerate(times):
            row = [city_slug, t.to_pydatetime(), model_run]
            for vi, var in enumerate(HOURLY_VARIABLES):
                val = hourly.Variables(vi).ValuesAsNumpy()[i]
                row.append(float(val) if pd.notna(val) else None)
            rows.append(row)

        if rows:
            cols = ["city_slug", "time", "model_run"] + HOURLY_VARIABLES
            placeholders = ", ".join(["?"] * len(cols))
            db.executemany(
                f"INSERT OR REPLACE INTO forecasts_hourly ({', '.join(cols)}) VALUES ({placeholders})",
                rows,
            )
        return len(rows)

    def _store_daily(self, response, city_slug: str, model_run: datetime, db=None) -> int:
        db = db or self.db
        daily = response.Daily()
        if daily is None:
            return 0

        times = pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s", utc=True),
            end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left",
        )

        rows = []
        for i, t in enumerate(times):
            row = [city_slug, t.date(), model_run]
            for vi, var in enumerate(DAILY_VARIABLES):
                val = daily.Variables(vi).ValuesAsNumpy()[i]
                row.append(float(val) if pd.notna(val) else None)
            rows.append(row)

        if rows:
            # Map to DB columns (skip sunrise/sunset which aren't in API response)
            db_cols = ["city_slug", "date", "model_run"] + DAILY_VARIABLES
            placeholders = ", ".join(["?"] * len(db_cols))
            db.executemany(
                f"INSERT OR REPLACE INTO forecasts_daily ({', '.join(db_cols)}) VALUES ({placeholders})",
                rows,
            )
        return len(rows)
