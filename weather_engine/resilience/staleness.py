"""Data staleness scoring: measures how fresh each data source is.

Score 0-100 where 100 = all data fresh (< 4h), 0 = all data stale (> 24h).
Staleness translates to a confidence penalty (0-50 points).
"""

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def check_data_staleness(db, city_slug: str, variable: str = "temperature_2m_max") -> dict:
    """Check freshness of all data sources for a city/variable.

    Returns dict with age in hours for each source and an overall staleness score.
    """
    now = datetime.now(timezone.utc)
    result = {}

    # Ensemble members
    ens_row = db.execute(
        """SELECT MAX(time) FROM ensemble_members
        WHERE city_slug = ? AND temperature_2m IS NOT NULL""",
        [city_slug],
    ).fetchone()
    if ens_row and ens_row[0] is not None:
        ens_time = ens_row[0]
        if not isinstance(ens_time, datetime):
            ens_time = datetime.fromisoformat(str(ens_time))
        if ens_time.tzinfo is None:
            ens_time = ens_time.replace(tzinfo=timezone.utc)
        result["ensemble_hours"] = (now - ens_time).total_seconds() / 3600
    else:
        result["ensemble_hours"] = 999

    # Deterministic forecasts
    det_row = db.execute(
        """SELECT MAX(model_run) FROM deterministic_forecasts
        WHERE city_slug = ? AND temp_max IS NOT NULL""",
        [city_slug],
    ).fetchone()
    if det_row and det_row[0] is not None:
        det_time = det_row[0]
        if not isinstance(det_time, datetime):
            det_time = datetime.fromisoformat(str(det_time))
        if det_time.tzinfo is None:
            det_time = det_time.replace(tzinfo=timezone.utc)
        result["deterministic_hours"] = (now - det_time).total_seconds() / 3600
    else:
        result["deterministic_hours"] = 999

    # Observations
    obs_row = db.execute(
        """SELECT MAX(date) FROM observations
        WHERE city_slug = ? AND temperature_2m_max IS NOT NULL""",
        [city_slug],
    ).fetchone()
    if obs_row and obs_row[0] is not None:
        obs_date = obs_row[0]
        if not isinstance(obs_date, datetime):
            from datetime import date as date_cls
            if isinstance(obs_date, date_cls):
                obs_time = datetime(obs_date.year, obs_date.month, obs_date.day, tzinfo=timezone.utc)
            else:
                obs_time = datetime.fromisoformat(str(obs_date)).replace(tzinfo=timezone.utc)
        else:
            obs_time = obs_date if obs_date.tzinfo else obs_date.replace(tzinfo=timezone.utc)
        result["observations_hours"] = (now - obs_time).total_seconds() / 3600
    else:
        result["observations_hours"] = 999

    # Compute staleness score (100 = fresh, 0 = stale)
    scores = []
    for key in ["ensemble_hours", "deterministic_hours", "observations_hours"]:
        hours = result[key]
        if hours < 4:
            scores.append(100)
        elif hours < 12:
            scores.append(75)
        elif hours < 24:
            scores.append(40)
        elif hours < 48:
            scores.append(15)
        else:
            scores.append(0)

    result["staleness_score"] = round(sum(scores) / len(scores)) if scores else 0
    return result


def staleness_confidence_penalty(staleness: dict) -> float:
    """Compute a confidence penalty (0-50) based on data staleness.

    - Ensemble > 12h: -15, > 24h: -30
    - Observations > 48h: -10
    - Deterministic > 12h: -5
    """
    penalty = 0.0

    ens_hours = staleness.get("ensemble_hours", 0)
    if ens_hours > 24:
        penalty += 30
    elif ens_hours > 12:
        penalty += 15

    obs_hours = staleness.get("observations_hours", 0)
    if obs_hours > 48:
        penalty += 10

    det_hours = staleness.get("deterministic_hours", 0)
    if det_hours > 12:
        penalty += 5

    return min(50.0, penalty)
