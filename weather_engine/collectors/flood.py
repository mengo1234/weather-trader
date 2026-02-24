"""Flood / river discharge collector from Open-Meteo Flood API."""
import logging

import pandas as pd

from weather_engine.collectors.base import BaseCollector, om_client

logger = logging.getLogger(__name__)

FLOOD_URL = "https://flood-api.open-meteo.com/v1/flood"

DAILY_VARIABLES = [
    "river_discharge",
    "river_discharge_mean",
    "river_discharge_max",
]


class FloodCollector(BaseCollector):
    name = "flood"

    def collect(self, city: dict, db=None) -> int:
        db = db or self.db

        response = om_client.weather_api(
            FLOOD_URL,
            params={
                "latitude": city["latitude"],
                "longitude": city["longitude"],
                "daily": ",".join(DAILY_VARIABLES),
                "forecast_days": 10,
            },
        )[0]

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
            d = t.date()
            vals = []
            for vi in range(min(len(DAILY_VARIABLES), daily.VariablesLength())):
                v = daily.Variables(vi).ValuesAsNumpy()[i]
                vals.append(float(v) if pd.notna(v) else None)
            while len(vals) < len(DAILY_VARIABLES):
                vals.append(None)

            rows.append([city["slug"], d] + vals)

        if rows:
            db.executemany(
                """INSERT OR REPLACE INTO flood_data
                (city_slug, date, river_discharge, river_discharge_mean, river_discharge_max)
                VALUES (?, ?, ?, ?, ?)""",
                rows,
            )

        logger.info("Flood %s: %d daily rows stored", city["slug"], len(rows))
        return len(rows)
