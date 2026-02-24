"""Analog Ensemble — find historically similar weather days and use their outcomes.

Uses weighted Euclidean distance across multiple variables to find the K most
similar historical days, then computes probability from their observed outcomes.
"""
import logging
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

logger = logging.getLogger(__name__)

# Variable weights for distance calculation
_DEFAULT_WEIGHTS = {
    "temperature": 0.30,
    "precipitation": 0.20,
    "pressure": 0.20,
    "wind": 0.15,
    "humidity": 0.15,
}


@dataclass
class AnalogDay:
    date: date
    distance: float
    weight: float  # inverse-distance weight
    observed_value: float | None = None
    conditions: dict | None = None


@dataclass
class AnalogResult:
    analogs: list[AnalogDay]
    n_candidates: int
    mean_outcome: float | None
    std_outcome: float | None
    variable: str


def find_analogs(
    db,
    city_slug: str,
    target_date: date,
    variable: str = "temperature_2m_max",
    k: int = 20,
    window_days: int = 30,
) -> AnalogResult:
    """Find K most similar historical days based on multi-variable similarity.

    1. Get current conditions (ensemble mean for each variable)
    2. Get historical days within DOY ± window
    3. Weighted Euclidean distance
    4. Top-K analogs with observed outcomes
    """
    # 1. Current conditions from ensemble
    current = _get_current_conditions(db, city_slug, target_date)
    if not current:
        return AnalogResult(analogs=[], n_candidates=0, mean_outcome=None, std_outcome=None, variable=variable)

    # 2. Historical days within DOY window
    doy = target_date.timetuple().tm_yday
    doy_low = (doy - window_days) % 366
    doy_high = (doy + window_days) % 366

    if doy_low < doy_high:
        doy_clause = "EXTRACT(DOY FROM o.date) BETWEEN ? AND ?"
        doy_params = [doy_low, doy_high]
    else:
        # Wraps around year boundary
        doy_clause = "(EXTRACT(DOY FROM o.date) >= ? OR EXTRACT(DOY FROM o.date) <= ?)"
        doy_params = [doy_low, doy_high]

    # Get observations with multi-variable data
    obs_rows = db.execute(
        f"""SELECT o.date,
                   o.temperature_2m_max, o.temperature_2m_min,
                   o.precipitation_sum, o.wind_speed_10m_max,
                   o.pressure_msl_mean
        FROM observations o
        WHERE o.city_slug = ? AND {doy_clause}
        AND o.date < ?
        AND o.temperature_2m_max IS NOT NULL
        ORDER BY o.date DESC
        LIMIT 2000""",
        [city_slug] + doy_params + [target_date],
    ).fetchall()

    if not obs_rows:
        return AnalogResult(analogs=[], n_candidates=0, mean_outcome=None, std_outcome=None, variable=variable)

    # 3. Compute normalization stats from historical
    temps = [r[1] for r in obs_rows if r[1] is not None]
    precips = [r[3] for r in obs_rows if r[3] is not None]
    winds = [r[4] for r in obs_rows if r[4] is not None]
    pressures = [r[5] for r in obs_rows if r[5] is not None]

    norm = {
        "temperature": np.std(temps) if temps else 1.0,
        "precipitation": np.std(precips) if precips else 1.0,
        "wind": np.std(winds) if winds else 1.0,
        "pressure": np.std(pressures) if pressures else 1.0,
    }
    # Avoid zero division
    for key in norm:
        if norm[key] < 0.01:
            norm[key] = 1.0

    # 4. Compute distances
    candidates = []
    obs_var_map = {
        "temperature_2m_max": 1,
        "temperature_2m_min": 2,
        "precipitation_sum": 3,
        "wind_speed_10m_max": 4,
    }
    outcome_idx = obs_var_map.get(variable, 1)

    for row in obs_rows:
        hist_date = row[0]
        hist_conditions = {
            "temperature": row[1],
            "precipitation": row[3],
            "wind": row[4],
            "pressure": row[5],
        }

        # Skip if too many missing values
        n_valid = sum(1 for v in hist_conditions.values() if v is not None)
        if n_valid < 2:
            continue

        dist = _weighted_euclidean(current, hist_conditions, _DEFAULT_WEIGHTS, norm)
        observed = row[outcome_idx]

        candidates.append((hist_date, dist, observed, hist_conditions))

    if not candidates:
        return AnalogResult(analogs=[], n_candidates=0, mean_outcome=None, std_outcome=None, variable=variable)

    # 5. Select top-K
    candidates.sort(key=lambda x: x[1])
    top_k = candidates[:k]

    # Compute inverse-distance weights
    distances = [c[1] for c in top_k]
    max_dist = max(distances) if distances else 1.0
    inv_distances = [1.0 / (d + 0.01) for d in distances]
    total_inv = sum(inv_distances)

    analogs = []
    for (hist_date, dist, observed, conditions), inv_d in zip(top_k, inv_distances):
        analogs.append(AnalogDay(
            date=hist_date,
            distance=round(dist, 4),
            weight=round(inv_d / total_inv, 4),
            observed_value=float(observed) if observed is not None else None,
            conditions=conditions,
        ))

    # Compute weighted mean/std of outcomes
    obs_values = [(a.observed_value, a.weight) for a in analogs if a.observed_value is not None]
    if obs_values:
        values, weights = zip(*obs_values)
        mean_outcome = float(np.average(values, weights=weights))
        std_outcome = float(np.sqrt(np.average((np.array(values) - mean_outcome) ** 2, weights=weights)))
    else:
        mean_outcome = None
        std_outcome = None

    return AnalogResult(
        analogs=analogs,
        n_candidates=len(candidates),
        mean_outcome=round(mean_outcome, 3) if mean_outcome is not None else None,
        std_outcome=round(std_outcome, 3) if std_outcome is not None else None,
        variable=variable,
    )


def analog_probability(
    analog_result: AnalogResult,
    threshold_low: float,
    threshold_high: float,
) -> float | None:
    """Compute probability of outcome falling in [threshold_low, threshold_high].

    Uses weighted fraction of analog outcomes within the range.
    Returns None if insufficient analogs.
    """
    valid = [(a.observed_value, a.weight) for a in analog_result.analogs if a.observed_value is not None]
    if len(valid) < 3:
        return None

    weighted_in_range = 0.0
    total_weight = 0.0

    for value, weight in valid:
        total_weight += weight
        if threshold_low <= value <= threshold_high:
            weighted_in_range += weight

    if total_weight <= 0:
        return None

    prob = weighted_in_range / total_weight
    return float(np.clip(prob, 0.001, 0.999))


def _get_current_conditions(db, city_slug: str, target_date: date) -> dict | None:
    """Get current forecast conditions from ensemble means."""
    row = db.execute(
        """SELECT AVG(temperature_2m) as temp,
                  SUM(precipitation) as precip,
                  AVG(wind_speed_10m) as wind,
                  AVG(pressure_msl) as pressure,
                  AVG(relative_humidity_2m) as humidity
        FROM ensemble_members
        WHERE city_slug = ? AND time::DATE = ?
        AND temperature_2m IS NOT NULL""",
        [city_slug, target_date],
    ).fetchone()

    if row is None or row[0] is None:
        return None

    return {
        "temperature": float(row[0]) if row[0] is not None else None,
        "precipitation": float(row[1]) if row[1] is not None else None,
        "wind": float(row[2]) if row[2] is not None else None,
        "pressure": float(row[3]) if row[3] is not None else None,
        "humidity": float(row[4]) if row[4] is not None else None,
    }


def _weighted_euclidean(
    current: dict,
    historical: dict,
    weights: dict,
    normalization: dict,
) -> float:
    """Compute weighted Euclidean distance between current and historical conditions."""
    total_sq = 0.0
    total_weight = 0.0

    for var, weight in weights.items():
        cur_val = current.get(var)
        hist_val = historical.get(var)
        norm_val = normalization.get(var, 1.0)

        if cur_val is not None and hist_val is not None:
            diff = (cur_val - hist_val) / norm_val
            total_sq += weight * diff ** 2
            total_weight += weight

    if total_weight <= 0:
        return float("inf")

    # Normalize by total weight used
    return float(np.sqrt(total_sq / total_weight))
