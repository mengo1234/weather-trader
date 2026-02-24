"""Per-model accuracy tracking.

Compares deterministic forecast from each model against observations,
computes reliability scores, and provides model weights for blending.
"""
import logging
from datetime import date, timedelta

logger = logging.getLogger(__name__)

TRACKED_VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
]

# Map from tracked variable to deterministic_forecasts column and observations column
_VAR_MAP = {
    "temperature_2m_max": ("temp_max", "temperature_2m_max"),
    "temperature_2m_min": ("temp_min", "temperature_2m_min"),
    "precipitation_sum": ("precip_sum", "precipitation_sum"),
    "wind_speed_10m_max": ("wind_max", "wind_speed_10m_max"),
}


def update_model_accuracy(db, city_slug: str, target_date: date) -> int:
    """Compare deterministic forecasts vs observed for a given city/date.

    For each model, compute error and store in model_accuracy.
    Returns number of records inserted.
    """
    n_inserted = 0

    for variable in TRACKED_VARIABLES:
        det_col, obs_col = _VAR_MAP[variable]

        # Get observed value
        obs_row = db.execute(
            f"SELECT {obs_col} FROM observations WHERE city_slug = ? AND date = ?",
            [city_slug, target_date],
        ).fetchone()
        if obs_row is None or obs_row[0] is None:
            continue
        observed = float(obs_row[0])

        # Get forecast from each model (latest model_run before target date)
        fc_rows = db.execute(
            f"""SELECT model, {det_col}, model_run FROM deterministic_forecasts
            WHERE city_slug = ? AND date = ? AND {det_col} IS NOT NULL
            AND model_run < ?::TIMESTAMP
            ORDER BY model, model_run DESC""",
            [city_slug, target_date, target_date],
        ).fetchall()

        # Keep only latest model_run per model
        seen_models = set()
        for model, fc_value, model_run in fc_rows:
            if model in seen_models:
                continue
            seen_models.add(model)

            fc_val = float(fc_value)
            horizon = max(1, (target_date - model_run.date()).days) if hasattr(model_run, 'date') else 1
            error = fc_val - observed

            db.execute(
                """INSERT OR REPLACE INTO model_accuracy
                (model, city_slug, variable, horizon_days, date,
                 forecast_value, observed_value, error, abs_error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [model, city_slug, variable, horizon, target_date,
                 fc_val, observed, error, abs(error)],
            )
            n_inserted += 1

    return n_inserted


def get_model_reliability(
    db, city_slug: str, variable: str, horizon_days: int = 1, lookback: int = 30,
) -> dict[str, float]:
    """Get reliability score (0-1) per model based on recent MAE.

    Lower MAE → higher reliability. Score = 1 - normalized_mae.
    Returns {model: reliability_score}.
    """
    cutoff = date.today() - timedelta(days=lookback)
    rows = db.execute(
        """SELECT model, AVG(abs_error) as mae, COUNT(*) as n
        FROM model_accuracy
        WHERE city_slug = ? AND variable = ? AND horizon_days <= ? AND date >= ?
        GROUP BY model HAVING COUNT(*) >= 3""",
        [city_slug, variable, max(horizon_days, 2), cutoff],
    ).fetchall()

    if not rows:
        return {}

    # Normalize: best model gets ~1.0, worst gets min 0.2
    maes = {r[0]: r[1] for r in rows}
    if not maes:
        return {}

    max_mae = max(maes.values())
    if max_mae == 0:
        return {m: 1.0 for m in maes}

    return {
        model: round(max(0.2, 1.0 - mae / (max_mae * 1.2)), 3)
        for model, mae in maes.items()
    }


def get_model_weights(
    db, city_slug: str, variable: str, horizon_days: int = 1,
) -> dict[str, float]:
    """Get normalized weights for blending (models with lower MAE weigh more).

    Returns {model: weight} where sum(weights) = 1.0.
    """
    reliability = get_model_reliability(db, city_slug, variable, horizon_days)
    if not reliability:
        return {}

    total = sum(reliability.values())
    if total == 0:
        n = len(reliability)
        return {m: 1.0 / n for m in reliability}

    return {model: round(score / total, 4) for model, score in reliability.items()}


def update_model_crps(db, city_slug: str, target_date: date) -> int:
    """Compute CRPS per model for a city/date and store in model_crps table.

    Uses ensemble members vs observed value.
    """
    import numpy as np
    from weather_engine.analysis.crps import crps_ensemble

    n_inserted = 0

    for variable in TRACKED_VARIABLES:
        _, obs_col = _VAR_MAP[variable]

        # Get observed value
        obs_row = db.execute(
            f"SELECT {obs_col} FROM observations WHERE city_slug = ? AND date = ?",
            [city_slug, target_date],
        ).fetchone()
        if obs_row is None or obs_row[0] is None:
            continue
        observed = float(obs_row[0])

        # Ensemble variable mapping
        ens_var_map = {
            "temperature_2m_max": ("temperature_2m", "MAX"),
            "temperature_2m_min": ("temperature_2m", "MIN"),
            "precipitation_sum": ("precipitation", "SUM"),
            "wind_speed_10m_max": ("wind_speed_10m", "MAX"),
        }
        ens_col, agg = ens_var_map.get(variable, ("temperature_2m", "MAX"))

        # Get ensemble values per model
        rows = db.execute(
            f"""SELECT model, {agg}({ens_col}) as val
            FROM ensemble_members
            WHERE city_slug = ? AND time::DATE = ? AND {ens_col} IS NOT NULL
            GROUP BY model, member_id""",
            [city_slug, target_date],
        ).fetchall()

        if not rows:
            continue

        # Group by model
        model_members: dict[str, list[float]] = {}
        for model, val in rows:
            model_members.setdefault(model, []).append(float(val))

        for model, values in model_members.items():
            if len(values) < 3:
                continue
            ens_arr = np.array(values)
            crps_val = crps_ensemble(ens_arr, observed)
            if np.isnan(crps_val):
                continue

            horizon = max(1, (target_date - date.today()).days) if target_date >= date.today() else 1
            db.execute(
                """INSERT OR REPLACE INTO model_crps
                (model, city_slug, variable, horizon_days, date, crps_value)
                VALUES (?, ?, ?, ?, ?, ?)""",
                [model, city_slug, variable, horizon, target_date, crps_val],
            )
            n_inserted += 1

    return n_inserted


def get_model_weights_combined(
    db, city_slug: str, variable: str, horizon_days: int = 1,
    mae_weight: float = 0.4, crps_weight: float = 0.6,
) -> dict[str, float]:
    """Get model weights combining MAE-based and CRPS-based scores.

    Returns {model: weight} where sum(weights) = 1.0.
    """
    mae_reliability = get_model_reliability(db, city_slug, variable, horizon_days)

    # CRPS-based reliability
    cutoff = date.today() - timedelta(days=30)
    crps_rows = db.execute(
        """SELECT model, AVG(crps_value) as avg_crps, COUNT(*) as n
        FROM model_crps
        WHERE city_slug = ? AND variable = ? AND horizon_days <= ? AND date >= ?
        GROUP BY model HAVING COUNT(*) >= 3""",
        [city_slug, variable, max(horizon_days, 2), cutoff],
    ).fetchall()

    crps_scores = {}
    if crps_rows:
        crps_vals = {r[0]: r[1] for r in crps_rows}
        max_crps = max(crps_vals.values()) if crps_vals else 1.0
        if max_crps > 0:
            crps_scores = {m: max(0.2, 1.0 - c / (max_crps * 1.2)) for m, c in crps_vals.items()}

    # Combine
    all_models = set(mae_reliability.keys()) | set(crps_scores.keys())
    if not all_models:
        return {}

    combined = {}
    for model in all_models:
        mae_s = mae_reliability.get(model, 0.5)
        crps_s = crps_scores.get(model, 0.5)
        combined[model] = mae_weight * mae_s + crps_weight * crps_s

    total = sum(combined.values())
    if total == 0:
        n = len(combined)
        return {m: 1.0 / n for m in combined}

    return {model: round(score / total, 4) for model, score in combined.items()}


def run_model_accuracy_update(db) -> dict:
    """Run model accuracy update for all cities, for recent dates with observations.

    Called daily after verification (08:15).
    """
    cities = db.execute("SELECT slug FROM cities").fetchall()
    total = 0

    # Check last 7 days for observations we haven't tracked yet
    today = date.today()
    crps_total = 0
    for (city_slug,) in cities:
        for days_back in range(1, 8):
            target = today - timedelta(days=days_back)
            try:
                n = update_model_accuracy(db, city_slug, target)
                total += n
            except Exception as e:
                logger.debug("Model accuracy update failed for %s/%s: %s", city_slug, target, e)
            try:
                n_crps = update_model_crps(db, city_slug, target)
                crps_total += n_crps
            except Exception as e:
                logger.debug("Model CRPS update failed for %s/%s: %s", city_slug, target, e)

    logger.info("Model accuracy update: %d records, %d CRPS", total, crps_total)
    return {"records": total, "crps_records": crps_total}
