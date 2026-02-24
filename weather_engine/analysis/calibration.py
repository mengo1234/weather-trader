"""Reliability diagram and calibration analysis for probabilistic forecasts."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def reliability_diagram(db, city_slug: str | None = None, variable: str = "temperature_2m_max", n_bins: int = 10) -> dict:
    """Compute reliability diagram data from calibration bins.

    Uses scored prediction data (calibration_bins table) for accurate
    observed frequency computation.

    For each probability bin, compare predicted probability vs observed frequency.
    A well-calibrated model should have predicted ~= observed.
    """
    from weather_engine.analysis.recalibration import get_reliability_data
    return get_reliability_data(db, variable=variable)


def spread_skill(db, city_slug: str, days: int = 30) -> dict:
    """Compute spread-skill relationship.

    Compares ensemble spread (standard deviation) with forecast error (RMSE).
    Ideally, spread should be proportional to actual forecast error.
    """
    from datetime import date, timedelta

    today = date.today()

    spreads = []
    errors = []

    for i in range(days):
        target = today - timedelta(days=i + 1)

        # Get ensemble stats
        ens = db.execute(
            """SELECT AVG(temperature_2m), STDDEV(temperature_2m)
            FROM (
                SELECT MAX(temperature_2m) as temperature_2m
                FROM ensemble_members
                WHERE city_slug = ? AND time::DATE = ? AND temperature_2m IS NOT NULL
                GROUP BY model, member_id
            )""",
            [city_slug, target],
        ).fetchone()

        if not ens or ens[0] is None or ens[1] is None:
            continue

        # Get observation
        obs = db.execute(
            "SELECT temperature_2m_max FROM observations WHERE city_slug = ? AND date = ?",
            [city_slug, target],
        ).fetchone()

        if not obs or obs[0] is None:
            continue

        spread = float(ens[1])
        error = abs(float(ens[0]) - float(obs[0]))
        spreads.append(spread)
        errors.append(error)

    if len(spreads) < 5:
        return {"n_days": len(spreads), "correlation": None}

    corr = float(np.corrcoef(spreads, errors)[0, 1]) if len(spreads) > 1 else 0

    return {
        "n_days": len(spreads),
        "mean_spread": round(float(np.mean(spreads)), 2),
        "mean_error": round(float(np.mean(errors)), 2),
        "spread_skill_ratio": round(float(np.mean(spreads)) / max(0.01, float(np.mean(errors))), 3),
        "correlation": round(corr, 3),
        "interpretation": (
            "Good calibration" if 0.8 < float(np.mean(spreads)) / max(0.01, float(np.mean(errors))) < 1.2
            else "Under-dispersive (spread too small)" if float(np.mean(spreads)) < float(np.mean(errors))
            else "Over-dispersive (spread too large)"
        ),
        "points": [
            {"spread": round(s, 2), "error": round(e, 2)}
            for s, e in zip(spreads, errors)
        ],
    }
