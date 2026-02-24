"""Generate probability predictions for market outcomes using ensemble + historical data."""
import logging
from datetime import date

import numpy as np

from weather_engine.analysis.climate_normals import get_historical_values_for_doy
from weather_engine.analysis.probability import (
    estimate_multiple_outcomes,
    get_bias_correction,
    ProbabilityEstimate,
)

logger = logging.getLogger(__name__)


def predict_outcomes(
    db,
    city_slug: str,
    variable: str,
    target_date: date,
    thresholds: list[tuple[float, float, str]],
    fast: bool = False,
) -> list[ProbabilityEstimate] | None:
    """Generate probability estimates for each market outcome.

    Uses ensemble members, historical data, deterministic forecast,
    bias correction, dynamic horizon-based weighting, multi-model
    deterministic data, and cross-reference scores.
    """
    # 1. Get ensemble values for target date
    ensemble_var_map = {
        "temperature_2m_max": "temperature_2m",
        "temperature_2m_min": "temperature_2m",
        "temperature_2m_mean": "temperature_2m",
        "precipitation_sum": "precipitation",
        "wind_speed_10m_max": "wind_speed_10m",
    }

    ens_var = ensemble_var_map.get(variable, variable)

    if variable == "temperature_2m_max":
        rows = db.execute(
            f"""SELECT model, member_id, MAX({ens_var}) as val
            FROM ensemble_members
            WHERE city_slug = ? AND time::DATE = ? AND {ens_var} IS NOT NULL
            GROUP BY model, member_id""",
            [city_slug, target_date],
        ).fetchall()
    elif variable == "temperature_2m_min":
        rows = db.execute(
            f"""SELECT model, member_id, MIN({ens_var}) as val
            FROM ensemble_members
            WHERE city_slug = ? AND time::DATE = ? AND {ens_var} IS NOT NULL
            GROUP BY model, member_id""",
            [city_slug, target_date],
        ).fetchall()
    elif variable == "precipitation_sum":
        rows = db.execute(
            f"""SELECT model, member_id, SUM({ens_var}) as val
            FROM ensemble_members
            WHERE city_slug = ? AND time::DATE = ? AND {ens_var} IS NOT NULL
            GROUP BY model, member_id""",
            [city_slug, target_date],
        ).fetchall()
    else:
        rows = db.execute(
            f"""SELECT model, member_id, MAX({ens_var}) as val
            FROM ensemble_members
            WHERE city_slug = ? AND time::DATE = ? AND {ens_var} IS NOT NULL
            GROUP BY model, member_id""",
            [city_slug, target_date],
        ).fetchall()

    if not rows:
        logger.warning("No ensemble data for %s/%s on %s", city_slug, variable, target_date)
        return None

    ensemble_values = np.array([r[2] for r in rows])

    # Check multi-model consensus (boost confidence if models agree)
    model_means = {}
    for model, _mid, val in rows:
        model_means.setdefault(model, []).append(val)
    model_avgs = {m: np.mean(v) for m, v in model_means.items()}
    n_models = len(model_avgs)
    if n_models >= 2:
        model_std = np.std(list(model_avgs.values()))
        model_consensus = model_std < np.std(ensemble_values) * 0.5
    else:
        model_consensus = False

    # 2. Get historical values for same day of year (finestra ±7 giorni)
    historical_values = get_historical_values_for_doy(db, city_slug, target_date, variable)

    # 3. Get deterministic forecast (single best_match for backward compat)
    det_row = db.execute(
        f"""SELECT {variable} FROM forecasts_daily
        WHERE city_slug = ? AND date = ? AND {variable} IS NOT NULL
        ORDER BY model_run DESC LIMIT 1""",
        [city_slug, target_date],
    ).fetchone()
    deterministic = det_row[0] if det_row else None

    # 3b. Get multi-model deterministic forecasts
    det_var_map = {
        "temperature_2m_max": "temp_max",
        "temperature_2m_min": "temp_min",
        "precipitation_sum": "precip_sum",
        "wind_speed_10m_max": "wind_max",
    }
    det_col = det_var_map.get(variable)
    deterministic_models = None
    if det_col:
        det_rows = db.execute(
            f"""SELECT model, {det_col} FROM deterministic_forecasts
            WHERE city_slug = ? AND date = ? AND {det_col} IS NOT NULL""",
            [city_slug, target_date],
        ).fetchall()
        if det_rows:
            deterministic_models = {r[0]: r[1] for r in det_rows}

    # 4. Calculate horizon (days from today)
    horizon_days = max(0, (target_date - date.today()).days)

    # 5. Get bias correction from verification history
    bias = get_bias_correction(db, city_slug, variable)

    # 6. Get model weights from tracker
    model_weights = None
    try:
        from weather_engine.analysis.model_tracker import get_model_weights
        model_weights = get_model_weights(db, city_slug, variable, horizon_days)
    except Exception:
        pass

    # 7. Get cross-reference scores
    cross_ref_composite = None
    atmospheric_stability = None
    try:
        xref_row = db.execute(
            """SELECT composite_score, atmospheric_stability FROM cross_reference_scores
            WHERE city_slug = ? AND target_date = ?""",
            [city_slug, target_date],
        ).fetchone()
        if xref_row:
            cross_ref_composite = float(xref_row[0]) if xref_row[0] is not None else None
            atmospheric_stability = float(xref_row[1]) if xref_row[1] is not None else None
    except Exception:
        pass

    logger.info(
        "Predict %s/%s %s: %d ensemble (mean=%.1f, std=%.1f), %d historical, "
        "det=%.1f, horizon=%dd, bias=%.2f, consensus=%s, models=%d, det_multi=%d, xref=%s",
        city_slug, variable, target_date,
        len(ensemble_values), np.mean(ensemble_values), np.std(ensemble_values),
        len(historical_values),
        deterministic if deterministic else 0,
        horizon_days, bias, model_consensus, n_models,
        len(deterministic_models) if deterministic_models else 0,
        f"{cross_ref_composite:.0f}" if cross_ref_composite else "N/A",
    )

    # 8. If model consensus, tighten the ensemble spread (less uncertainty)
    if model_consensus and n_models >= 3:
        overall_mean = np.mean(ensemble_values)
        ensemble_values = overall_mean + (ensemble_values - overall_mean) * 0.85

    # 9b. Analog ensemble for each outcome
    analog_probs = None
    try:
        from weather_engine.analysis.analog import find_analogs, analog_probability
        analog_result = find_analogs(db, city_slug, target_date, variable)
        if analog_result.analogs:
            analog_probs = []
            for low, high, _label in thresholds:
                ap = analog_probability(analog_result, low, high)
                analog_probs.append(ap)
    except Exception as e:
        logger.debug("Analog ensemble failed: %s", e)

    # 9c. BMA
    bma_probs = None
    try:
        from weather_engine.analysis.bma import get_training_data, fit_bma, bma_probability as bma_prob_fn
        training = get_training_data(db, city_slug, variable)
        if training is not None:
            model_fc, obs_arr = training
            bma_result = fit_bma(model_fc, obs_arr)
            if bma_result is not None and bma_result.converged:
                # Get current deterministic forecasts
                current_fc = deterministic_models or {}
                if current_fc:
                    bma_probs = []
                    for low, high, _label in thresholds:
                        bp = bma_prob_fn(bma_result, current_fc, low, high)
                        bma_probs.append(bp)
    except Exception as e:
        logger.debug("BMA failed: %s", e)

    # 9d. Regime info
    regime_info = None
    try:
        from weather_engine.analysis.ensemble_analysis import analyze_ensemble_from_values
        ea = analyze_ensemble_from_values(ensemble_values, variable)
        regime_info = {
            "is_bimodal": ea.regime.is_bimodal,
            "n_regimes": ea.regime.n_regimes,
            "dominant_regime_weight": ea.regime.dominant_regime_weight,
            "bimodality_coefficient": ea.regime.bimodality_coefficient,
        }
    except Exception as e:
        logger.debug("Regime analysis failed: %s", e)

    # 10. Estimate probabilities with all sources
    estimates = estimate_multiple_outcomes(
        ensemble_values,
        thresholds,
        historical_values if len(historical_values) > 0 else None,
        deterministic,
        fast=fast,
        horizon_days=horizon_days,
        bias_correction=bias,
        model_weights=model_weights,
        deterministic_models=deterministic_models,
        cross_ref_composite=cross_ref_composite,
        atmospheric_stability=atmospheric_stability,
        analog_probabilities=analog_probs,
        bma_probabilities=bma_probs,
        regime_info=regime_info,
    )

    # Apply recalibration (last step)
    try:
        from weather_engine.analysis.recalibration import calibrate_estimates
        estimates = calibrate_estimates(estimates, db=db, variable=variable)
    except Exception as e:
        logger.debug("Recalibration skipped: %s", e)

    return estimates
