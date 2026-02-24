import logging
from datetime import date

import pandas as pd

from weather_engine.collectors.base import BaseCollector, om_client
from weather_engine.config import settings

logger = logging.getLogger(__name__)

DAILY_VARIABLES = [
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "precipitation_sum", "rain_sum", "snowfall_sum",
    "wind_speed_10m_max", "wind_gusts_10m_max", "wind_direction_10m_dominant",
    "shortwave_radiation_sum",
]


class HistoricalCollector(BaseCollector):
    name = "historical"

    def collect(self, city: dict, db=None, start_date: str = "2000-01-01", end_date: str | None = None) -> int:
        db = db or self.db
        if end_date is None:
            end_date = str(date.today())

        response = om_client.weather_api(
            settings.historical_url,
            params={
                "latitude": city["latitude"],
                "longitude": city["longitude"],
                "daily": ",".join(DAILY_VARIABLES),
                "timezone": city["timezone"],
                "start_date": start_date,
                "end_date": end_date,
                "temperature_unit": "fahrenheit",
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
            row = [city["slug"], t.date()]
            for vi, var in enumerate(DAILY_VARIABLES):
                val = daily.Variables(vi).ValuesAsNumpy()[i]
                row.append(float(val) if pd.notna(val) else None)
            rows.append(row)

        if rows:
            cols = ["city_slug", "date"] + DAILY_VARIABLES
            placeholders = ", ".join(["?"] * len(cols))
            db.executemany(
                f"INSERT OR REPLACE INTO observations ({', '.join(cols)}) VALUES ({placeholders})",
                rows,
            )

        logger.info("Historical %s: %d observations (%s to %s)", city["slug"], len(rows), start_date, end_date)
        return len(rows)

    def compute_climate_normals(self, city_slug: str) -> int:
        """Compute 30-year climate normals from observations."""
        result = self.db.execute("""
            INSERT OR REPLACE INTO climate_normals
            SELECT
                city_slug,
                DAYOFYEAR(date) as day_of_year,
                AVG(temperature_2m_max) as temperature_2m_max_mean,
                STDDEV(temperature_2m_max) as temperature_2m_max_std,
                AVG(temperature_2m_min) as temperature_2m_min_mean,
                STDDEV(temperature_2m_min) as temperature_2m_min_std,
                AVG(temperature_2m_mean) as temperature_2m_mean_mean,
                AVG(precipitation_sum) as precipitation_sum_mean,
                STDDEV(precipitation_sum) as precipitation_sum_std,
                AVG(wind_speed_10m_max) as wind_speed_10m_max_mean
            FROM observations
            WHERE city_slug = ?
            GROUP BY city_slug, DAYOFYEAR(date)
            HAVING COUNT(*) >= 3
        """, [city_slug])
        count = self.db.execute(
            "SELECT COUNT(*) FROM climate_normals WHERE city_slug = ?", [city_slug]
        ).fetchone()[0]
        logger.info("Climate normals %s: %d days", city_slug, count)
        return count
