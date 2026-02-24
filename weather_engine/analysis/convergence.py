"""Forecast convergence analysis.

Checks whether successive model runs are converging (spread decreasing)
or diverging (spread increasing) for a given city/variable/date.
Convergence = more reliable forecast, divergence = less reliable.
"""

import logging
from datetime import date

import numpy as np

logger = logging.getLogger(__name__)


def check_convergence(
    db,
    city_slug: str,
    variable: str = "temperature_2m",
    target_date: date | None = None,
) -> dict:
    """Compare successive forecast runs to check convergence.

    Queries the last 3-4 model_run timestamps for the same city/target_date,
    computes ensemble spread for each run, and checks trend.

    Returns:
        dict with trend ("converging", "diverging", "stable"),
        spread_change_pct, and runs_compared.
    """
    if target_date is None:
        target_date = date.today()

    try:
        # Get distinct model runs that have data for this city and target date
        runs = db.execute(
            """SELECT DISTINCT time::DATE as run_date,
                      STDDEV(temperature_2m) as spread,
                      COUNT(*) as n_members
            FROM ensemble_members
            WHERE city_slug = ? AND time::DATE = ?
            AND temperature_2m IS NOT NULL
            GROUP BY time::DATE
            HAVING COUNT(*) >= 3
            ORDER BY run_date DESC
            LIMIT 4""",
            [city_slug, target_date],
        ).fetchall()

        if not runs or len(runs) < 1:
            return {
                "trend": "stable",
                "spread_change_pct": 0.0,
                "runs_compared": 0,
                "latest_spread": None,
                "spreads": [],
            }

        spreads = [float(r[1]) if r[1] else 0 for r in runs]
        n_compared = len(spreads)

        if n_compared < 2:
            return {
                "trend": "stable",
                "spread_change_pct": 0.0,
                "runs_compared": n_compared,
                "latest_spread": spreads[0] if spreads else None,
                "spreads": spreads,
            }

        # Also check across different forecast horizons
        # Get spreads from forecasts made at different times for same target
        horizon_spreads = _get_horizon_spreads(db, city_slug, variable, target_date)
        if len(horizon_spreads) >= 2:
            spreads = horizon_spreads

        # Spreads are ordered most recent first
        # If spread is decreasing (newer runs have lower spread) → converging
        latest = spreads[0]
        earliest = spreads[-1]

        if earliest > 0:
            change_pct = (latest - earliest) / earliest
        else:
            change_pct = 0.0

        if change_pct < -0.10:
            trend = "converging"
        elif change_pct > 0.10:
            trend = "diverging"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "spread_change_pct": round(change_pct, 4),
            "runs_compared": n_compared,
            "latest_spread": round(latest, 3) if latest else None,
            "spreads": [round(s, 3) for s in spreads],
        }

    except Exception as e:
        logger.warning("Convergence check failed for %s/%s: %s", city_slug, variable, e)
        return {
            "trend": "stable",
            "spread_change_pct": 0.0,
            "runs_compared": 0,
            "latest_spread": None,
            "spreads": [],
        }


def _get_horizon_spreads(
    db, city_slug: str, variable: str, target_date: date,
) -> list[float]:
    """Get ensemble spreads from forecasts at different horizons for the same target.

    This compares how the spread changed as the forecast horizon shortened
    (e.g., 7-day forecast spread vs 3-day vs 1-day).
    """
    try:
        # Get daily forecasts for this target date from different model_run times
        rows = db.execute(
            f"""SELECT model_run, {variable}
            FROM forecasts_daily
            WHERE city_slug = ? AND date = ? AND {variable} IS NOT NULL
            ORDER BY model_run ASC""",
            [city_slug, target_date],
        ).fetchall()

        if len(rows) < 2:
            return []

        # Group by model_run and compute spread between successive runs
        runs = {}
        for model_run, value in rows:
            run_key = str(model_run)[:13]  # group by hour
            if run_key not in runs:
                runs[run_key] = []
            runs[run_key].append(value)

        # For each run, compute the "spread" as deviation from mean across runs
        if len(runs) < 2:
            return []

        values_by_run = list(runs.values())
        spreads = []
        for vals in values_by_run:
            if len(vals) >= 1:
                spreads.append(float(np.std(vals)) if len(vals) > 1 else 0.0)

        return spreads[-4:]  # Last 4 runs

    except Exception:
        return []


def get_convergence_for_cities(db, city_slugs: list[str], target_date: date | None = None) -> dict[str, dict]:
    """Get convergence data for multiple cities at once."""
    result = {}
    for slug in city_slugs:
        result[slug] = check_convergence(db, slug, target_date=target_date)
    return result
