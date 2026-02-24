"""Model drift detection — monitors forecast accuracy degradation over time.

Compares recent MAE (7-day) vs longer-term MAE (30-day) to detect
performance degradation or improvement per model/variable/city.
"""
import logging
from dataclasses import dataclass
from datetime import date, timedelta

logger = logging.getLogger(__name__)


@dataclass
class DriftResult:
    model: str | None
    variable: str
    city_slug: str
    mae_7d: float
    mae_30d: float
    drift_ratio: float  # mae_7d / mae_30d
    status: str  # "stable", "improving", "degrading", "alert"


def detect_drift(
    db,
    city_slug: str,
    variable: str,
    model: str | None = None,
) -> list[DriftResult]:
    """Detect drift for a specific city/variable, optionally per model.

    drift_ratio = mae_7d / mae_30d
    - >1.5 → "alert" (significant degradation)
    - >1.3 → "degrading"
    - <0.8 → "improving"
    - else → "stable"
    """
    today = date.today()
    cutoff_7d = today - timedelta(days=7)
    cutoff_30d = today - timedelta(days=30)

    model_clause = "AND model = ?" if model else ""
    base_params = [city_slug, variable]
    if model:
        base_params.append(model)

    # 7-day MAE per model
    rows_7d = db.execute(
        f"""SELECT model, AVG(abs_error) as mae, COUNT(*) as n
        FROM model_accuracy
        WHERE city_slug = ? AND variable = ? {model_clause}
        AND date >= ?
        GROUP BY model
        HAVING COUNT(*) >= 2""",
        base_params + [cutoff_7d],
    ).fetchall()

    # 30-day MAE per model
    rows_30d = db.execute(
        f"""SELECT model, AVG(abs_error) as mae, COUNT(*) as n
        FROM model_accuracy
        WHERE city_slug = ? AND variable = ? {model_clause}
        AND date >= ?
        GROUP BY model
        HAVING COUNT(*) >= 5""",
        base_params + [cutoff_30d],
    ).fetchall()

    mae_7d = {r[0]: float(r[1]) for r in rows_7d}
    mae_30d = {r[0]: float(r[1]) for r in rows_30d}

    results = []
    all_models = set(mae_7d.keys()) | set(mae_30d.keys())

    for m in all_models:
        m7 = mae_7d.get(m)
        m30 = mae_30d.get(m)

        if m7 is None or m30 is None or m30 < 0.01:
            continue

        ratio = m7 / m30

        if ratio > 1.5:
            status = "alert"
        elif ratio > 1.3:
            status = "degrading"
        elif ratio < 0.8:
            status = "improving"
        else:
            status = "stable"

        results.append(DriftResult(
            model=m,
            variable=variable,
            city_slug=city_slug,
            mae_7d=round(m7, 3),
            mae_30d=round(m30, 3),
            drift_ratio=round(ratio, 3),
            status=status,
        ))

    return results


def detect_drift_all_cities(db) -> dict[str, list[DriftResult]]:
    """Detect drift across all cities and tracked variables."""
    from weather_engine.analysis.model_tracker import TRACKED_VARIABLES

    cities = db.execute("SELECT slug FROM cities").fetchall()
    all_results = {}

    for (city_slug,) in cities:
        city_drifts = []
        for variable in TRACKED_VARIABLES:
            try:
                drifts = detect_drift(db, city_slug, variable)
                city_drifts.extend(drifts)
            except Exception as e:
                logger.debug("Drift detection failed for %s/%s: %s", city_slug, variable, e)

        if city_drifts:
            all_results[city_slug] = city_drifts

    return all_results


def get_worst_drift(db) -> list[DriftResult]:
    """Get drift results with alert or degrading status across all cities."""
    all_drifts = detect_drift_all_cities(db)
    alerts = []
    for drifts in all_drifts.values():
        for d in drifts:
            if d.status in ("alert", "degrading"):
                alerts.append(d)

    alerts.sort(key=lambda d: -d.drift_ratio)
    return alerts
