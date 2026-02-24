"""Deterministic multi-model forecast collector.

Fetches daily forecasts from 8 deterministic models via the Open-Meteo
/v1/forecast endpoint with models= parameter for cross-model comparison.
"""
import logging

import pandas as pd

from weather_engine.collectors.base import BaseCollector, om_client
from weather_engine.config import settings

logger = logging.getLogger(__name__)

DETERMINISTIC_MODELS = [
    "best_match",
    "ecmwf_ifs025",
    "gfs_seamless",
    "icon_seamless",
    "gem_seamless",
    "meteofrance_seamless",
    "jma_seamless",
    "ukmo_seamless",
]

DAILY_VARIABLES = [
    "temperature_2m_max", "temperature_2m_min", "precipitation_sum",
    "wind_speed_10m_max", "wind_gusts_10m_max",
]

HOURLY_VARIABLES = [
    "pressure_msl", "cape", "snow_depth",
    "soil_temperature_0_to_10cm", "soil_moisture_0_to_10cm", "visibility",
]

# Map from API hourly aggregation to DB column
_HOURLY_AGG = {
    "pressure_msl": ("AVG", "pressure_msl"),
    "cape": ("MAX", "cape"),
    "snow_depth": ("MAX", "snow_depth"),
    "soil_temperature_0_to_10cm": ("AVG", "soil_temp"),
    "soil_moisture_0_to_10cm": ("AVG", "soil_moisture"),
    "visibility": ("MIN", "visibility"),
}


class DeterministicMultiCollector(BaseCollector):
    name = "deterministic_multi"

    def collect(self, city: dict, db=None) -> int:
        db = db or self.db
        total = 0
        for model in DETERMINISTIC_MODELS:
            try:
                n = self._collect_model(city, model, db)
                total += n
                logger.info("Deterministic %s/%s: %d rows", city["slug"], model, n)
            except Exception as e:
                logger.warning("Deterministic %s/%s failed: %s", city["slug"], model, e)
        return total

    def _collect_model(self, city: dict, model: str, db) -> int:
        response = om_client.weather_api(
            settings.forecast_url,
            params={
                "latitude": city["latitude"],
                "longitude": city["longitude"],
                "daily": ",".join(DAILY_VARIABLES),
                "hourly": ",".join(HOURLY_VARIABLES),
                "models": model,
                "temperature_unit": "fahrenheit",
                "forecast_days": 10,
            },
        )[0]

        model_run = pd.to_datetime(response.Hourly().Time(), unit="s", utc=True) if response.Hourly() else None

        # --- Daily data ---
        daily = response.Daily()
        if daily is None:
            return 0

        times = pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s", utc=True),
            end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left",
        )

        n_daily_vars = len(DAILY_VARIABLES)
        daily_data = {}
        for i, t in enumerate(times):
            d = t.date()
            row = {}
            for vi in range(min(n_daily_vars, daily.VariablesLength())):
                val = daily.Variables(vi).ValuesAsNumpy()[i]
                row[DAILY_VARIABLES[vi]] = float(val) if pd.notna(val) else None
            daily_data[d] = row

        # --- Hourly data → aggregate to daily ---
        hourly = response.Hourly()
        hourly_agg = {}
        if hourly is not None:
            h_times = pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left",
            )

            for vi, var_name in enumerate(HOURLY_VARIABLES):
                if vi >= hourly.VariablesLength():
                    break
                vals = hourly.Variables(vi).ValuesAsNumpy()
                agg_type, db_col = _HOURLY_AGG[var_name]

                day_vals: dict[object, list] = {}
                for i, t in enumerate(h_times):
                    d = t.date()
                    v = float(vals[i]) if pd.notna(vals[i]) else None
                    if v is not None:
                        day_vals.setdefault(d, []).append(v)

                for d, vlist in day_vals.items():
                    if d not in hourly_agg:
                        hourly_agg[d] = {}
                    if agg_type == "AVG":
                        hourly_agg[d][db_col] = sum(vlist) / len(vlist)
                    elif agg_type == "MAX":
                        hourly_agg[d][db_col] = max(vlist)
                    elif agg_type == "MIN":
                        hourly_agg[d][db_col] = min(vlist)

        # --- Merge and store ---
        rows = []
        for d, daily_row in daily_data.items():
            hagg = hourly_agg.get(d, {})
            rows.append([
                city["slug"], d, model, model_run,
                daily_row.get("temperature_2m_max"),
                daily_row.get("temperature_2m_min"),
                daily_row.get("precipitation_sum"),
                daily_row.get("wind_speed_10m_max"),
                daily_row.get("wind_gusts_10m_max"),
                hagg.get("pressure_msl"),
                hagg.get("cape"),
                hagg.get("snow_depth"),
                hagg.get("soil_temp"),
                hagg.get("soil_moisture"),
                hagg.get("visibility"),
            ])

        if rows:
            db.executemany(
                """INSERT OR REPLACE INTO deterministic_forecasts
                (city_slug, date, model, model_run,
                 temp_max, temp_min, precip_sum, wind_max, wind_gusts_max,
                 pressure_msl, cape, snow_depth, soil_temp, soil_moisture, visibility)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
        return len(rows)
