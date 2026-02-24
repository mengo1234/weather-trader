import logging
from datetime import date

import numpy as np

logger = logging.getLogger(__name__)


def get_climate_normal(db, city_slug: str, target_date: date, variable: str = "temperature_2m_max") -> dict:
    """Get climate normal for a specific day of year."""
    doy = target_date.timetuple().tm_yday

    col_map = {
        "temperature_2m_max": "temperature_2m_max_mean",
        "temperature_2m_min": "temperature_2m_min_mean",
        "temperature_2m_mean": "temperature_2m_mean_mean",
        "precipitation_sum": "precipitation_sum_mean",
    }

    mean_col = col_map.get(variable)
    std_col = mean_col.replace("_mean", "_std") if mean_col else None

    if not mean_col:
        return {"error": f"No climate normal for {variable}"}

    # Get normal for the exact day and surrounding window (+/- 7 days)
    row = db.execute(
        f"""SELECT {mean_col}, {std_col} FROM climate_normals
        WHERE city_slug = ? AND day_of_year = ?""",
        [city_slug, doy],
    ).fetchone()

    if row is None:
        return {"error": f"No climate normal for {city_slug} day {doy}"}

    normal_mean = row[0]
    normal_std = row[1] if row[1] is not None else 5.0

    return {
        "city_slug": city_slug,
        "date": str(target_date),
        "day_of_year": doy,
        "variable": variable,
        "normal_mean": float(normal_mean) if normal_mean is not None else None,
        "normal_std": float(normal_std),
    }


def get_historical_values_for_doy(
    db,
    city_slug: str,
    target_date: date,
    variable: str = "temperature_2m_max",
    window_days: int = 10,
) -> np.ndarray:
    """Get historical values for the same day of year (+/- window) across all years.

    Con 35 anni di dati e window=10, otteniamo ~700 campioni per stima robusta.
    """
    doy = target_date.timetuple().tm_yday
    doy_low = doy - window_days
    doy_high = doy + window_days

    col_map = {
        "temperature_2m_max": "temperature_2m_max",
        "temperature_2m_min": "temperature_2m_min",
        "temperature_2m_mean": "temperature_2m_mean",
        "precipitation_sum": "precipitation_sum",
    }

    obs_col = col_map.get(variable, variable)

    if doy_low < 1:
        # Handle year boundary
        rows = db.execute(
            f"""SELECT {obs_col} FROM observations
            WHERE city_slug = ? AND {obs_col} IS NOT NULL
            AND (DAYOFYEAR(date) >= ? OR DAYOFYEAR(date) <= ?)""",
            [city_slug, 366 + doy_low, doy_high],
        ).fetchall()
    elif doy_high > 365:
        rows = db.execute(
            f"""SELECT {obs_col} FROM observations
            WHERE city_slug = ? AND {obs_col} IS NOT NULL
            AND (DAYOFYEAR(date) >= ? OR DAYOFYEAR(date) <= ?)""",
            [city_slug, doy_low, doy_high - 365],
        ).fetchall()
    else:
        rows = db.execute(
            f"""SELECT {obs_col} FROM observations
            WHERE city_slug = ? AND {obs_col} IS NOT NULL
            AND DAYOFYEAR(date) BETWEEN ? AND ?""",
            [city_slug, doy_low, doy_high],
        ).fetchall()

    return np.array([r[0] for r in rows]) if rows else np.array([])


def compute_deviation(value: float, normal_mean: float, normal_std: float) -> dict:
    """Compute deviation from climate normal."""
    deviation = value - normal_mean
    z_score = deviation / normal_std if normal_std > 0 else 0

    # Percentile rank using normal distribution
    from scipy.stats import norm
    percentile = norm.cdf(value, loc=normal_mean, scale=normal_std) * 100

    label = "normal"
    if abs(z_score) > 2:
        label = "extreme"
    elif abs(z_score) > 1:
        label = "unusual"

    return {
        "deviation": float(deviation),
        "z_score": float(z_score),
        "percentile": float(percentile),
        "label": label,
    }
