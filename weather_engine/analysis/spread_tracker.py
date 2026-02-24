"""Spread evolution tracking: snapshot ensemble spread over time.

Tracks how ensemble spread converges/diverges as forecast date approaches,
providing a trajectory signal for confidence adjustment.
"""

import logging
from datetime import date, datetime, timedelta, timezone

import numpy as np

logger = logging.getLogger(__name__)


def snapshot_spread(db, city_slug: str, target_date: date, variable: str = "temperature_2m_max") -> dict | None:
    """Take a snapshot of the current ensemble spread for a city/date/variable.

    Queries ensemble_members, computes statistics, and stores in spread_snapshots.

    Returns:
        Dict with spread stats, or None if no ensemble data.
    """
    ensemble_var_map = {
        "temperature_2m_max": "temperature_2m",
        "temperature_2m_min": "temperature_2m",
        "precipitation_sum": "precipitation",
        "wind_speed_10m_max": "wind_speed_10m",
    }
    ens_var = ensemble_var_map.get(variable, variable)

    if variable == "temperature_2m_max":
        agg = "MAX"
    elif variable == "temperature_2m_min":
        agg = "MIN"
    elif variable == "precipitation_sum":
        agg = "SUM"
    else:
        agg = "MAX"

    rows = db.execute(
        f"""SELECT {agg}({ens_var}) as val
        FROM ensemble_members
        WHERE city_slug = ? AND time::DATE = ? AND {ens_var} IS NOT NULL
        GROUP BY model, member_id""",
        [city_slug, target_date],
    ).fetchall()

    if not rows:
        return None

    values = np.array([r[0] for r in rows])
    if len(values) < 2:
        return None

    now = datetime.now(timezone.utc)
    stats = {
        "ensemble_mean": float(np.mean(values)),
        "ensemble_std": float(np.std(values)),
        "ensemble_min": float(np.min(values)),
        "ensemble_max": float(np.max(values)),
        "n_members": len(values),
        "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
    }

    try:
        db.execute(
            """INSERT INTO spread_snapshots
            (city_slug, target_date, variable, collected_at,
             ensemble_mean, ensemble_std, ensemble_min, ensemble_max, n_members, iqr)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                city_slug, target_date, variable, now,
                stats["ensemble_mean"], stats["ensemble_std"],
                stats["ensemble_min"], stats["ensemble_max"],
                stats["n_members"], stats["iqr"],
            ],
        )
    except Exception as e:
        logger.warning("Failed to insert spread snapshot for %s/%s: %s", city_slug, target_date, e)
        return None

    return stats


def snapshot_all_cities(db, variable: str = "temperature_2m_max") -> dict:
    """Take spread snapshots for all cities, for dates today+1 through today+7.

    Returns:
        Dict with snapshot count.
    """
    from weather_engine.db import get_cities

    cities = get_cities(db)
    today = date.today()
    snapshots = 0

    for city in cities:
        slug = city["slug"]
        for day_offset in range(1, 8):
            target = today + timedelta(days=day_offset)
            result = snapshot_spread(db, slug, target, variable)
            if result is not None:
                snapshots += 1

    logger.info("Spread snapshots: %d collected for %d cities", snapshots, len(cities))
    return {"snapshots": snapshots}


def get_spread_trajectory(db, city_slug: str, target_date: date, variable: str = "temperature_2m_max") -> dict:
    """Get the spread trajectory for a city/date/variable.

    Returns list of (hours_before_target, std) points plus convergence_rate.

    Returns:
        Dict with trajectory points and convergence rate.
    """
    rows = db.execute(
        """SELECT collected_at, ensemble_std
        FROM spread_snapshots
        WHERE city_slug = ? AND target_date = ? AND variable = ?
        ORDER BY collected_at ASC""",
        [city_slug, target_date, variable],
    ).fetchall()

    if not rows:
        return {"points": [], "convergence_rate": None}

    target_dt = datetime.combine(target_date, datetime.min.time(), tzinfo=timezone.utc)
    points = []
    for collected_at, std in rows:
        if collected_at.tzinfo is None:
            collected_at = collected_at.replace(tzinfo=timezone.utc)
        hours_before = (target_dt - collected_at).total_seconds() / 3600
        points.append({"hours_before": round(hours_before, 1), "std": round(std, 3)})

    # Convergence rate: linear slope of std over time (negative = converging)
    convergence_rate = None
    if len(points) >= 2:
        hours = np.array([p["hours_before"] for p in points])
        stds = np.array([p["std"] for p in points])
        try:
            slope, _ = np.polyfit(hours, stds, 1)
            # Positive slope means std increases as hours_before increases (= convergence as we approach)
            convergence_rate = round(float(slope), 4)
        except Exception:
            pass

    return {
        "city_slug": city_slug,
        "target_date": str(target_date),
        "variable": variable,
        "points": points,
        "convergence_rate": convergence_rate,
    }


def spread_trajectory_signal(trajectory: dict) -> dict:
    """Convert a spread trajectory into a confidence signal.

    Analyzes the last 24h of spread evolution:
    - Rapid convergence → signal_boost +5
    - Divergence → signal_boost -10
    - Stable → signal_boost 0

    Returns:
        Dict with trend, signal_boost.
    """
    points = trajectory.get("points", [])
    if len(points) < 2:
        return {"trend": "insufficient_data", "signal_boost": 0}

    # Focus on recent points (last 24h worth)
    recent = [p for p in points if p["hours_before"] <= 48]
    if len(recent) < 2:
        recent = points[-2:]

    stds = [p["std"] for p in recent]
    first_std = stds[0]
    last_std = stds[-1]

    if first_std == 0:
        return {"trend": "stable", "signal_boost": 0}

    change_pct = (last_std - first_std) / first_std

    if change_pct < -0.15:
        return {"trend": "converging", "signal_boost": 5}
    elif change_pct > 0.15:
        return {"trend": "diverging", "signal_boost": -10}
    else:
        return {"trend": "stable", "signal_boost": 0}
