import logging

import pandas as pd

from weather_engine.collectors.base import BaseCollector, om_client
from weather_engine.config import settings

logger = logging.getLogger(__name__)

# Batch 1 (ogni 4h): core models
ENSEMBLE_BATCH_1 = [
    "icon_seamless",
    "gfs_seamless",
    "ecmwf_ifs025",
    "gem_global",
    "ecmwf_ifs04",
]

# Batch 2 (ogni 6h): secondary models
ENSEMBLE_BATCH_2 = [
    "bom_access_global_ensemble",
    "cma_grapes_global_ensemble",
    "meteo_france_arpege_world_ensemble",
    "ncep_gfs025",
    "ncep_gefs05",
]

# Batch 3 (ogni 8h): regional/supplementary models
ENSEMBLE_BATCH_3 = [
    "dwd_icon_eu",
    "dwd_icon_d2",
    "knmi_seamless",
    "ukmo_global_deterministic_10km",
    "jma_gsm",
]

ENSEMBLE_MODELS = ENSEMBLE_BATCH_1 + ENSEMBLE_BATCH_2 + ENSEMBLE_BATCH_3

ENSEMBLE_VARIABLES = [
    "temperature_2m", "precipitation", "wind_speed_10m", "wind_gusts_10m",
    "cloud_cover", "pressure_msl", "relative_humidity_2m", "dew_point_2m",
    "shortwave_radiation", "cape",
    "surface_pressure", "soil_temperature_0_to_10cm", "soil_moisture_0_to_10cm",
    "visibility", "snow_depth",
]


class EnsembleCollector(BaseCollector):
    name = "ensemble"

    def __init__(self, db=None, batch: int | None = None):
        super().__init__(db)
        self.batch = batch

    def _get_models(self) -> list[str]:
        if self.batch == 1:
            return ENSEMBLE_BATCH_1
        elif self.batch == 2:
            return ENSEMBLE_BATCH_2
        elif self.batch == 3:
            return ENSEMBLE_BATCH_3
        return ENSEMBLE_MODELS

    def collect(self, city: dict, db=None) -> int:
        db = db or self.db
        total = 0
        for model in self._get_models():
            try:
                n = self._collect_model(city, model, db=db)
                total += n
                logger.info("Ensemble %s/%s: %d rows", city["slug"], model, n)
            except Exception as e:
                logger.warning("Ensemble %s/%s failed: %s", city["slug"], model, e)
        return total

    def _collect_model(self, city: dict, model: str, db=None) -> int:
        db = db or self.db
        response = om_client.weather_api(
            settings.ensemble_url,
            params={
                "latitude": city["latitude"],
                "longitude": city["longitude"],
                "hourly": ",".join(ENSEMBLE_VARIABLES),
                "models": model,
                "temperature_unit": "fahrenheit",
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

        # Each variable has multiple ensemble members
        # The SDK returns variables sequentially: var0_member0, var0_member1, ..., var1_member0, ...
        n_vars = len(ENSEMBLE_VARIABLES)
        total_vars = hourly.VariablesLength()
        if total_vars == 0:
            return 0

        n_members = total_vars // n_vars

        rows = []
        for member_id in range(n_members):
            for i, t in enumerate(times):
                row = [city["slug"], t.to_pydatetime(), model, member_id]
                for vi in range(n_vars):
                    idx = vi * n_members + member_id
                    if idx < total_vars:
                        val = hourly.Variables(idx).ValuesAsNumpy()[i]
                        row.append(float(val) if pd.notna(val) else None)
                    else:
                        row.append(None)
                rows.append(row)

        if rows:
            # Map API variable names to DB column names
            db_cols = [_API_TO_DB_COL.get(v, v) for v in ENSEMBLE_VARIABLES]
            cols = ["city_slug", "time", "model", "member_id"] + db_cols
            placeholders = ", ".join(["?"] * len(cols))
            db.executemany(
                f"INSERT OR REPLACE INTO ensemble_members ({', '.join(cols)}) VALUES ({placeholders})",
                rows,
            )
        return len(rows)


# API variable name → DB column name mapping
_API_TO_DB_COL = {
    "soil_temperature_0_to_10cm": "soil_temperature",
    "soil_moisture_0_to_10cm": "soil_moisture",
}
