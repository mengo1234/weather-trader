"""CRPS (Continuous Ranked Probability Score) for ensemble evaluation.

CRPS measures the quality of probabilistic forecasts. Lower = better.
It generalizes MAE to probabilistic forecasts.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def crps_ensemble(ensemble_values: np.ndarray, observation: float) -> float:
    """Compute CRPS for a single ensemble forecast against one observation.

    Uses the fair CRPS formulation:
        CRPS = E|X - y| - 0.5 * E|X - X'|
    where X, X' are independent draws from the ensemble.
    """
    clean = ensemble_values[~np.isnan(ensemble_values)]
    if len(clean) == 0:
        return float("nan")

    # E|X - y|
    mae_term = np.mean(np.abs(clean - observation))

    # E|X - X'| using the efficient formula: 2/(n^2) * sum_i sum_j |x_i - x_j|
    sorted_ens = np.sort(clean)
    n = len(sorted_ens)
    # Efficient calculation: sum of |x_i - x_j| = 2 * sum_i (2*i - n) * x_sorted_i
    indices = np.arange(n)
    spread_term = 2.0 * np.sum((2 * indices - n + 1) * sorted_ens) / (n * (n - 1)) if n > 1 else 0

    return float(mae_term - 0.5 * spread_term)


def crps_batch(db, city_slug: str, variable: str = "temperature_2m_max", days: int = 30) -> dict:
    """Compute CRPS over historical ensemble forecasts vs observations.

    Returns summary statistics of CRPS values.
    """
    # Get dates with both ensemble and observation data
    rows = db.execute(
        """SELECT e.d, e.vals, o.obs_val
        FROM (
            SELECT time::DATE as d,
                   LIST(MAX(temperature_2m)) as vals
            FROM ensemble_members
            WHERE city_slug = ? AND temperature_2m IS NOT NULL
            GROUP BY time::DATE, model, member_id
        ) e
        JOIN (
            SELECT date as d, temperature_2m_max as obs_val
            FROM observations
            WHERE city_slug = ? AND temperature_2m_max IS NOT NULL
        ) o ON e.d = o.d
        ORDER BY e.d DESC
        LIMIT ?""",
        [city_slug, city_slug, days],
    ).fetchall()

    if not rows:
        return {"n_days": 0, "crps_values": [], "mean_crps": None}

    crps_values = []
    for row in rows:
        try:
            ens = np.array(row[1], dtype=float)
            obs = float(row[2])
            c = crps_ensemble(ens, obs)
            if not np.isnan(c):
                crps_values.append({"date": str(row[0]), "crps": round(c, 3)})
        except Exception:
            continue

    if not crps_values:
        return {"n_days": 0, "crps_values": [], "mean_crps": None}

    vals = [v["crps"] for v in crps_values]
    return {
        "n_days": len(crps_values),
        "crps_values": crps_values,
        "mean_crps": round(np.mean(vals), 3),
        "std_crps": round(np.std(vals), 3),
        "min_crps": round(min(vals), 3),
        "max_crps": round(max(vals), 3),
    }


def skill_score_vs_climatology(db, city_slug: str, variable: str = "temperature_2m_max", days: int = 30) -> dict | None:
    """Compute CRPS Skill Score vs climatology.

    CRPSS = 1 - CRPS_model / CRPS_climatology
    CRPSS > 0 means model is better than climatology.
    """
    from weather_engine.analysis.climate_normals import get_historical_values_for_doy
    from datetime import date as dt_date, timedelta

    today = dt_date.today()
    model_crps_list = []
    clim_crps_list = []

    for i in range(days):
        target = today - timedelta(days=i + 1)

        # Get observation
        obs = db.execute(
            f"SELECT {variable} FROM observations WHERE city_slug = ? AND date = ?",
            [city_slug, target],
        ).fetchone()
        if not obs or obs[0] is None:
            continue
        actual = obs[0]

        # Get ensemble for that date
        ens_rows = db.execute(
            """SELECT MAX(temperature_2m) FROM ensemble_members
            WHERE city_slug = ? AND time::DATE = ? AND temperature_2m IS NOT NULL
            GROUP BY model, member_id""",
            [city_slug, target],
        ).fetchall()

        if not ens_rows or len(ens_rows) < 4:
            continue

        ens_values = np.array([r[0] for r in ens_rows])
        model_c = crps_ensemble(ens_values, actual)

        # Climatology as ensemble (historical values for same DOY)
        hist = get_historical_values_for_doy(db, city_slug, target, variable)
        if len(hist) < 10:
            continue
        clim_c = crps_ensemble(hist, actual)

        if not np.isnan(model_c) and not np.isnan(clim_c):
            model_crps_list.append(model_c)
            clim_crps_list.append(clim_c)

    if not model_crps_list:
        return None

    mean_model = np.mean(model_crps_list)
    mean_clim = np.mean(clim_crps_list)
    crpss = 1 - mean_model / mean_clim if mean_clim > 0 else 0

    return {
        "crpss": round(float(crpss), 4),
        "mean_model_crps": round(float(mean_model), 3),
        "mean_climatology_crps": round(float(mean_clim), 3),
        "n_days": len(model_crps_list),
        "interpretation": "Model better than climatology" if crpss > 0 else "Climatology better than model",
    }
