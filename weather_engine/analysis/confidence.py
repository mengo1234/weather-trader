"""Confidence scoring for betting decisions.

Multi-factor scoring that measures HOW MUCH we can trust a prediction.
Edge does NOT enter the calculation (avoids circular logic).
"""

import logging

logger = logging.getLogger(__name__)


def calculate_confidence(
    ensemble_data: dict | None,
    accuracy_data: dict | None = None,
    n_outcomes: int = 2,
    days_ahead: int = 1,
    seasonal_alignment: float | None = None,
    convergence_trend: str | None = None,
    cross_ref: dict | None = None,
    model_reliability: dict[str, float] | None = None,
    regime_info: dict | None = None,
    extreme_tail_score: float | None = None,
    drift_status: str | None = None,
    db=None,
    city_slug: str | None = None,
    variable: str | None = None,
) -> dict:
    """Calculate multi-factor confidence score for a bet.

    Factors (15):
    1. Ensemble agreement (13%) — models agree = solid prediction
    2. Forecast accuracy (10%) — low historical error = reliable model
    3. Horizon decay (7%) — day 1 much better than day 7+
    4. Market complexity (4%) — fewer outcomes = more predictable
    5. Sample size ensemble (7%) — more models = more robust
    6. Seasonal alignment (5%) — forecast aligns with seasonal trend
    7. Forecast convergence (7%) — successive runs converging = more reliable
    8. Deterministic agreement (9%) — cross_reference
    9. Atmospheric stability (7%) — cross_reference - CAPE
    10. Cross-variable consistency (6%) — cross_reference
    11. Model reliability (7%) — model_tracker
    12. Climate trend alignment (4%) — cross_reference
    13. Ensemble regime (5%) — unimodal vs bimodal
    14. Extreme tail (5%) — forecast extremeness
    15. Model drift (4%) — recent accuracy trend

    Returns dict with score 0-100 and breakdown.
    """
    scores = {}
    data_quality = True
    cross_ref = cross_ref or {}

    # 1. Ensemble spread (0-100, tight = better)
    ens_std = ensemble_data.get("ensemble_std") if ensemble_data else None
    n_members = ensemble_data.get("n_members", 0) if ensemble_data else 0

    if ens_std is None or n_members == 0:
        spread_score = 10
        data_quality = False
    else:
        spread_score = max(0, min(100, 110 - ens_std * 12))
    scores["ensemble"] = spread_score

    # 2. Model accuracy (0-100)
    if accuracy_data and isinstance(accuracy_data, dict):
        mae = accuracy_data.get("mae")
        if mae is not None:
            acc_score = max(0, min(100, 105 - mae * 18))
        else:
            acc_score = 25
            data_quality = False
    else:
        acc_score = 20
        data_quality = False
    scores["accuracy"] = acc_score

    # 3. Horizon decay
    if days_ahead <= 1:
        horizon_score = 100
    elif days_ahead <= 3:
        horizon_score = 85
    elif days_ahead <= 5:
        horizon_score = 65
    elif days_ahead <= 7:
        horizon_score = 45
    elif days_ahead <= 10:
        horizon_score = 25
    else:
        horizon_score = 10
    scores["horizon"] = horizon_score

    # 4. Market complexity (fewer outcomes = easier)
    complexity_score = max(15, min(100, 115 - n_outcomes * 15))
    scores["complexity"] = complexity_score

    # 5. Sample size ensemble
    if n_members >= 40:
        sample_score = 100
    elif n_members >= 20:
        sample_score = 75
    elif n_members >= 10:
        sample_score = 50
    elif n_members > 0:
        sample_score = 30
    else:
        sample_score = 5
        data_quality = False
    scores["sample_size"] = sample_score

    # 6. Seasonal alignment
    if seasonal_alignment is not None:
        seasonal_score = max(0, min(100, 50 + seasonal_alignment * 50))
    else:
        seasonal_score = 50
    scores["seasonal"] = seasonal_score

    # 7. Forecast convergence (optionally boosted by spread trajectory)
    if convergence_trend == "converging":
        convergence_score = 85
    elif convergence_trend == "diverging":
        convergence_score = 20
    elif convergence_trend == "stable":
        convergence_score = 55
    else:
        convergence_score = 50

    # Apply spread trajectory signal boost if available
    spread_signal = (cross_ref or {}).get("spread_signal_boost")
    if spread_signal is not None:
        convergence_score = max(0, min(100, convergence_score + spread_signal))

    scores["convergence"] = convergence_score

    # 8. Deterministic agreement (NEW) — from cross_reference
    det_agreement = cross_ref.get("deterministic_agreement")
    scores["deterministic_agreement"] = det_agreement if det_agreement is not None else 50

    # 9. Atmospheric stability (NEW) — from cross_reference CAPE
    atm_stability = cross_ref.get("atmospheric_stability")
    scores["atmospheric_stability"] = atm_stability if atm_stability is not None else 50

    # 10. Cross-variable consistency (NEW) — from cross_reference
    xvar_consistency = cross_ref.get("cross_variable_consistency")
    scores["cross_variable_consistency"] = xvar_consistency if xvar_consistency is not None else 50

    # 11. Model reliability (NEW) — from model_tracker
    if model_reliability and len(model_reliability) > 0:
        avg_reliability = sum(model_reliability.values()) / len(model_reliability)
        reliability_score = max(0, min(100, avg_reliability * 100))
    else:
        reliability_score = 50
    scores["model_reliability"] = reliability_score

    # 12. Climate trend alignment — from cross_reference
    climate_align = cross_ref.get("climate_trend_alignment")
    scores["climate_trend_alignment"] = climate_align if climate_align is not None else 50

    # 13. Ensemble regime (NEW) — unimodal=80, bimodal dominant=60, bimodal=30
    if regime_info is not None:
        is_bimodal = regime_info.get("is_bimodal", False)
        dominant_weight = regime_info.get("dominant_regime_weight", 1.0)
        if not is_bimodal:
            regime_score = 80
        elif dominant_weight > 0.75:
            regime_score = 60
        else:
            regime_score = 30
    else:
        regime_score = 50
    scores["ensemble_regime"] = regime_score

    # 14. Extreme tail (NEW) — pass-through from tail_probability_score (0-100)
    scores["extreme_tail"] = extreme_tail_score if extreme_tail_score is not None else 50

    # 15. Model drift (NEW) — stable=75, improving=90, degrading=25, alert=10
    if drift_status == "stable":
        drift_score = 75
    elif drift_status == "improving":
        drift_score = 90
    elif drift_status == "degrading":
        drift_score = 25
    elif drift_status == "alert":
        drift_score = 10
    else:
        drift_score = 50
    scores["model_drift"] = drift_score

    weights = {
        "ensemble": 0.13,
        "accuracy": 0.10,
        "horizon": 0.07,
        "complexity": 0.04,
        "sample_size": 0.07,
        "seasonal": 0.05,
        "convergence": 0.07,
        "deterministic_agreement": 0.09,
        "atmospheric_stability": 0.07,
        "cross_variable_consistency": 0.06,
        "model_reliability": 0.07,
        "climate_trend_alignment": 0.04,
        "ensemble_regime": 0.05,
        "extreme_tail": 0.05,
        "model_drift": 0.04,
    }

    # Try learned weights
    if db is not None:
        try:
            from weather_engine.analysis.weight_learner import get_effective_confidence_weights
            learned = get_effective_confidence_weights(db, variable=variable or "all")
            if learned and set(learned.keys()) == set(weights.keys()):
                weights = learned
        except Exception:
            pass

    total = sum(scores[k] * weights[k] for k in scores)

    # Data staleness penalty
    if db is not None and city_slug:
        try:
            from weather_engine.resilience.staleness import check_data_staleness, staleness_confidence_penalty
            staleness = check_data_staleness(db, city_slug, variable or "temperature_2m_max")
            penalty = staleness_confidence_penalty(staleness)
            if penalty > 0:
                total = max(total - penalty, 5)
        except Exception:
            pass

    # Penalize insufficient data
    if not data_quality:
        total = min(total, 35)

    if total >= 70:
        rating = "FORTE"
    elif total >= 50:
        rating = "BUONO"
    elif total >= 30:
        rating = "DEBOLE"
    else:
        rating = "EVITA"

    return {
        "total": round(total, 1),
        "scores": scores,
        "rating": rating,
        "data_quality": data_quality,
    }


def get_seasonal_alignment(db, city_slug: str, forecast_temp: float | None = None) -> float | None:
    """Check if current forecast aligns with seasonal trend.

    Returns a value between -1 (counter-trend) and +1 (aligned with seasonal anomaly).
    Returns None if no seasonal data available.
    """
    if forecast_temp is None:
        return None

    try:
        from datetime import date
        today = date.today()
        row = db.execute(
            """SELECT temp_anomaly FROM seasonal_forecast
            WHERE city_slug = ? AND month = ? AND year = ?
            ORDER BY collected_at DESC LIMIT 1""",
            [city_slug, today.month, today.year],
        ).fetchone()

        if row is None or row[0] is None:
            return None

        seasonal_anomaly = row[0]

        normal_row = db.execute(
            """SELECT temperature_2m_max_mean FROM climate_normals
            WHERE city_slug = ? AND day_of_year = ?""",
            [city_slug, today.timetuple().tm_yday],
        ).fetchone()

        if normal_row is None or normal_row[0] is None:
            return None

        normal_temp = normal_row[0]
        forecast_deviation = forecast_temp - normal_temp

        if seasonal_anomaly == 0:
            return 0.0

        alignment = (forecast_deviation * seasonal_anomaly) / (abs(seasonal_anomaly) * max(1, abs(forecast_deviation)))
        return max(-1.0, min(1.0, alignment))

    except Exception as e:
        logger.debug("Seasonal alignment check failed for %s: %s", city_slug, e)
        return None
