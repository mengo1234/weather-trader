import logging

import numpy as np
import pandas as pd

from weather_engine.collectors.base import BaseCollector, om_client
from weather_engine.config import settings

logger = logging.getLogger(__name__)

HOURLY_VARIABLES = [
    "wave_height", "wave_direction", "wave_period",
    "wind_wave_height", "wind_wave_direction", "wind_wave_period",
    "swell_wave_height", "swell_wave_direction", "swell_wave_period",
    "ocean_current_velocity", "ocean_current_direction",
]

DAILY_VARIABLES = [
    "wave_height_max", "wave_direction_dominant", "wave_period_max",
    "wind_wave_height_max", "swell_wave_height_max",
]

# Coastal cities (distance to coast < 100km)
COASTAL_CITIES = {
    "nyc", "miami", "los_angeles", "seattle", "london", "napoli",
    "sao_paulo", "buenos_aires", "wellington", "amsterdam", "toronto",
}


class MarineCollector(BaseCollector):
    name = "marine"

    def collect(self, city: dict, db=None) -> int:
        db = db or self.db

        # Skip non-coastal cities
        if city["slug"] not in COASTAL_CITIES:
            logger.debug("Marine: skipping inland city %s", city["slug"])
            return 0

        response = om_client.weather_api(
            settings.marine_url,
            params={
                "latitude": city["latitude"],
                "longitude": city["longitude"],
                "hourly": ",".join(HOURLY_VARIABLES),
                "daily": ",".join(DAILY_VARIABLES),
                "forecast_days": 8,
            },
        )[0]

        # Store daily data
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
            wave_h = self._get_val(daily, 0, i)
            wave_dir = self._get_val(daily, 1, i)
            wave_p = self._get_val(daily, 2, i)
            swell_h = self._get_val(daily, 4, i)

            # Aggregate ocean_current_velocity from hourly if available
            ocv = None
            hourly = response.Hourly()
            if hourly is not None and hourly.VariablesLength() > 9:
                h_times = pd.date_range(
                    start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                    end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                    freq=pd.Timedelta(seconds=hourly.Interval()),
                    inclusive="left",
                )
                ocv_vals = hourly.Variables(9).ValuesAsNumpy()  # ocean_current_velocity index
                day_ocvs = [
                    float(ocv_vals[j]) for j, ht in enumerate(h_times)
                    if ht.date() == d and pd.notna(ocv_vals[j])
                ]
                if day_ocvs:
                    ocv = max(day_ocvs)

            rows.append([
                city["slug"], d,
                wave_h, wave_p,
                int(wave_dir) if wave_dir is not None else None,
                swell_h, ocv,
            ])

        if rows:
            db.executemany(
                """INSERT OR REPLACE INTO marine_data
                (city_slug, date, wave_height_max, wave_period_max,
                 wave_direction_dominant, swell_wave_height_max, ocean_current_velocity)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )

        logger.info("Marine %s: %d daily rows stored", city["slug"], len(rows))
        return len(rows)

    @staticmethod
    def _get_val(daily, var_idx: int, time_idx: int) -> float | None:
        if var_idx >= daily.VariablesLength():
            return None
        val = daily.Variables(var_idx).ValuesAsNumpy()[time_idx]
        return float(val) if pd.notna(val) else None
