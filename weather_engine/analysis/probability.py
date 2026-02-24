"""Stima probabilità avanzata con KDE, bias correction, pesi dinamici e calibrazione."""
import logging
from datetime import date

import numpy as np
from scipy import stats as sp_stats

from weather_engine.models.analysis import ProbabilityEstimate

logger = logging.getLogger(__name__)


def estimate_probability_kde(
    ensemble_values: np.ndarray,
    threshold_low: float,
    threshold_high: float,
    historical_values: np.ndarray | None = None,
    deterministic_value: float | None = None,
    outcome_label: str = "",
    fast: bool = False,
    horizon_days: int = 1,
    bias_correction: float = 0.0,
    air_quality_aqi: int | None = None,
    model_weights: dict[str, float] | None = None,
    ensemble_model_labels: list[str] | None = None,
    deterministic_models: dict[str, float] | None = None,
    cross_ref_composite: float | None = None,
    atmospheric_stability: float | None = None,
    analog_probability: float | None = None,
    bma_probability: float | None = None,
    regime_info: dict | None = None,
) -> ProbabilityEstimate:
    """Stima probabilità con KDE multi-source, blending dinamico e bias correction.

    4-source blending:
    1. Ensemble KDE (weighted by model reliability)
    2. Historical KDE
    3. Deterministic (weighted mean of multi-model)
    4. Cross-reference adjustment
    """
    clean = ensemble_values[~np.isnan(ensemble_values)]
    if len(clean) < 3:
        raise ValueError(f"Need at least 3 ensemble values, got {len(clean)}")

    # --- 1. KDE ensemble con bandwidth ottimizzato ---
    try:
        kde = sp_stats.gaussian_kde(clean, bw_method="scott")
        n_unique = len(np.unique(np.round(clean, 1)))
        if n_unique < len(clean) * 0.5:
            kde.set_bandwidth(kde.factor * 1.2)

        # Bandwidth modulation: if CAPE high (atmospheric instability), widen
        if atmospheric_stability is not None and atmospheric_stability < 35:
            # Low stability score = high CAPE = widen bandwidth
            stability_factor = 1.0 + (35 - atmospheric_stability) / 100
            kde.set_bandwidth(kde.factor * stability_factor)
    except Exception:
        kde = sp_stats.gaussian_kde(clean)

    grid_range = max(30, (max(clean) - min(clean)) * 2)
    grid_center = np.mean(clean)
    x_grid = np.linspace(grid_center - grid_range, grid_center + grid_range, 2000)
    pdf = kde(x_grid)
    dx = x_grid[1] - x_grid[0]

    mask = (x_grid >= threshold_low) & (x_grid <= threshold_high)
    ensemble_prob = float(np.sum(pdf[mask]) * dx)
    ensemble_prob = np.clip(ensemble_prob, 0.001, 0.999)

    ensemble_spread = float(np.std(clean))
    ensemble_mean = float(np.mean(clean))

    # --- 2. Historical probability con KDE ---
    historical_prob = None
    n_historical = 0
    if historical_values is not None:
        hist_clean = historical_values[~np.isnan(historical_values)]
        n_historical = len(hist_clean)
        if n_historical >= 20:
            try:
                hist_kde = sp_stats.gaussian_kde(hist_clean, bw_method="scott")
                hist_pdf = hist_kde(x_grid)
                hist_prob = float(np.sum(hist_pdf[mask]) * dx)
                historical_prob = np.clip(hist_prob, 0.001, 0.999)
            except Exception:
                pass

        if historical_prob is None and n_historical > 0:
            in_range = np.sum((hist_clean >= threshold_low) & (hist_clean <= threshold_high))
            historical_prob = float((in_range + 0.5) / (n_historical + 1))
            historical_prob = np.clip(historical_prob, 0.001, 0.999)

    if historical_prob is None:
        historical_prob = ensemble_prob

    # --- 3. Deterministic probability ---
    # Multi-model deterministic: weighted mean if available
    det_prob = ensemble_prob
    if deterministic_models and len(deterministic_models) > 0:
        # Weighted gaussian mixture from multiple deterministic models
        det_probs = []
        det_weights = []
        for model, value in deterministic_models.items():
            if value is not None and not np.isnan(value):
                corrected = value - bias_correction
                det_scale = max(ensemble_spread * 0.4, 0.8)
                det_dist = sp_stats.norm(loc=corrected, scale=det_scale)
                p = float(det_dist.cdf(threshold_high) - det_dist.cdf(threshold_low))
                det_probs.append(np.clip(p, 0.001, 0.999))
                det_weights.append(model_weights.get(model, 1.0) if model_weights else 1.0)

        if det_probs:
            total_w = sum(det_weights)
            det_prob = sum(p * w for p, w in zip(det_probs, det_weights)) / total_w
            det_prob = float(np.clip(det_prob, 0.001, 0.999))
    elif deterministic_value is not None and not np.isnan(deterministic_value):
        corrected_det = deterministic_value - bias_correction
        det_scale = max(ensemble_spread * 0.4, 0.8)
        det_dist = sp_stats.norm(loc=corrected_det, scale=det_scale)
        det_prob = float(det_dist.cdf(threshold_high) - det_dist.cdf(threshold_low))
        det_prob = np.clip(det_prob, 0.001, 0.999)

    # --- 4. Cross-reference adjustment ---
    # Cross-ref composite score (0-100) modulates toward/away from 50%
    cross_ref_adj = 0.0
    if cross_ref_composite is not None:
        # High composite → confident → no adjustment
        # Low composite → uncertain → nudge toward 50%
        uncertainty_factor = max(0, (50 - cross_ref_composite) / 500)  # max ~10% shift
        # Blend of ensemble/historical/deterministic before cross-ref
        preliminary = (ensemble_prob + det_prob) / 2
        cross_ref_adj = uncertainty_factor * (0.5 - preliminary)

    # --- 5. Blending dinamico basato sull'orizzonte ---
    has_deterministic_multi = deterministic_models is not None and len(deterministic_models) > 1
    has_cross_ref = cross_ref_composite is not None
    has_analog = analog_probability is not None
    has_bma = bma_probability is not None

    # Use BMA probability in the deterministic slot if available and convergent
    effective_det_prob = det_prob
    if has_bma:
        effective_det_prob = bma_probability

    if has_analog:
        w_ens, w_hist, w_det, w_xref, w_analog = _dynamic_weights_5(
            horizon_days, n_historical, ensemble_spread,
            has_deterministic_multi or has_bma, has_cross_ref, has_analog,
        )
        blended = (w_ens * ensemble_prob + w_hist * historical_prob +
                   w_det * effective_det_prob + w_xref * cross_ref_adj +
                   w_analog * analog_probability)
    else:
        w_ens, w_hist, w_det, w_xref = _dynamic_weights_4(
            horizon_days, n_historical, ensemble_spread,
            has_deterministic_multi or has_bma, has_cross_ref,
        )
        blended = w_ens * ensemble_prob + w_hist * historical_prob + w_det * effective_det_prob + w_xref * cross_ref_adj

    # The cross_ref_adj can be negative, so re-normalize
    blended = float(np.clip(blended, 0.001, 0.999))

    # --- 5b. Air quality uncertainty adjustment ---
    if air_quality_aqi is not None and air_quality_aqi > 100:
        aqi_factor = min(0.04, (air_quality_aqi - 100) / 2500)
        blended = blended + aqi_factor * (0.5 - blended)
        blended = float(np.clip(blended, 0.001, 0.999))

    # --- 6. Confidence interval ---
    if fast:
        uncertainty = 0.10 + 0.05 * (ensemble_spread / 5.0) + 0.10 / max(1, n_historical / 10)
        ci_lower = max(0.0, blended - uncertainty)
        ci_upper = min(1.0, blended + uncertainty)
    else:
        ci_lower, ci_upper = _bootstrap_ci(clean, threshold_low, threshold_high)

    return ProbabilityEstimate(
        outcome=outcome_label,
        probability=blended,
        confidence_lower=ci_lower,
        confidence_upper=ci_upper,
        ensemble_prob=ensemble_prob,
        historical_prob=historical_prob,
        deterministic_prob=effective_det_prob,
        blended_prob=blended,
        ensemble_spread=ensemble_spread,
        analog_prob=analog_probability,
        bma_prob=bma_probability,
        regime_info=regime_info,
    )


def _dynamic_weights_4(
    horizon_days: int,
    n_historical: int,
    ensemble_spread: float,
    has_deterministic_multi: bool = False,
    has_cross_ref: bool = False,
    db=None,
    variable: str | None = None,
) -> tuple[float, float, float, float]:
    """4-source dynamic weights: ensemble, historical, deterministic, cross-ref.

    When deterministic multi-model and cross-ref data are unavailable,
    their weight is redistributed to ensemble and historical.
    """
    # Try learned weights first
    if db is not None:
        try:
            from weather_engine.analysis.weight_learner import get_effective_blending_weights
            learned = get_effective_blending_weights(db, horizon_days, variable or "temperature_2m_max", n_sources=4)
            if learned:
                return (learned.get("ensemble", 0.4), learned.get("historical", 0.2),
                        learned.get("deterministic", 0.2), learned.get("cross_ref", 0.1))
        except Exception:
            pass

    # Base weights for 4 sources by horizon
    if horizon_days <= 2:
        w_ens, w_hist, w_det, w_xref = 0.45, 0.10, 0.35, 0.10
    elif horizon_days <= 5:
        w_ens, w_hist, w_det, w_xref = 0.40, 0.20, 0.25, 0.15
    elif horizon_days <= 8:
        w_ens, w_hist, w_det, w_xref = 0.30, 0.35, 0.15, 0.20
    else:
        w_ens, w_hist, w_det, w_xref = 0.20, 0.45, 0.10, 0.25

    # If no multi-model deterministic, redistribute that weight
    if not has_deterministic_multi:
        # Fall back to old 3-source behavior
        redistribute = w_det * 0.3 + w_xref * 0.5
        w_ens += redistribute * 0.6
        w_hist += redistribute * 0.4
        # Reduce det/xref to minimal
        w_det = w_det * 0.7
        w_xref = w_xref * 0.5

    if not has_cross_ref:
        # No cross-ref: push xref weight to ensemble and historical
        w_ens += w_xref * 0.5
        w_hist += w_xref * 0.5
        w_xref = 0.0

    # Adjust for historical data quantity
    if n_historical < 10:
        shift = w_hist * 0.6
        w_hist -= shift
        w_ens += shift * 0.7
        w_det += shift * 0.3
    elif n_historical >= 100:
        shift = 0.05
        w_hist += shift
        w_ens -= shift

    # Adjust for ensemble spread
    if ensemble_spread > 5.0:
        shift = min(0.10, (ensemble_spread - 5.0) * 0.02)
        w_ens -= shift
        w_hist += shift
    elif ensemble_spread < 2.0:
        shift = 0.05
        w_ens += shift
        w_hist -= shift

    # Normalize
    total = w_ens + w_hist + w_det + w_xref
    if total <= 0:
        return 0.4, 0.3, 0.2, 0.1
    return w_ens / total, w_hist / total, w_det / total, w_xref / total


def _dynamic_weights_5(
    horizon_days: int,
    n_historical: int,
    ensemble_spread: float,
    has_deterministic_multi: bool = False,
    has_cross_ref: bool = False,
    has_analog: bool = True,
    db=None,
    variable: str | None = None,
) -> tuple[float, float, float, float, float]:
    """5-source dynamic weights: ensemble, historical, det/BMA, cross-ref, analog.

    If analog not available, redistributes to ensemble+historical.
    """
    # Try learned weights first
    if db is not None:
        try:
            from weather_engine.analysis.weight_learner import get_effective_blending_weights
            learned = get_effective_blending_weights(db, horizon_days, variable or "temperature_2m_max", n_sources=5)
            if learned:
                return (learned.get("ensemble", 0.35), learned.get("historical", 0.15),
                        learned.get("deterministic", 0.22), learned.get("cross_ref", 0.13),
                        learned.get("analog", 0.15))
        except Exception:
            pass

    # Base weights for 5 sources by horizon
    if horizon_days <= 2:
        w_ens, w_hist, w_det, w_xref, w_analog = 0.40, 0.08, 0.30, 0.10, 0.12
    elif horizon_days <= 5:
        w_ens, w_hist, w_det, w_xref, w_analog = 0.35, 0.15, 0.22, 0.13, 0.15
    elif horizon_days <= 8:
        w_ens, w_hist, w_det, w_xref, w_analog = 0.25, 0.25, 0.12, 0.18, 0.20
    else:
        w_ens, w_hist, w_det, w_xref, w_analog = 0.15, 0.30, 0.08, 0.22, 0.25

    if not has_analog:
        # Redistribute analog weight
        w_ens += w_analog * 0.5
        w_hist += w_analog * 0.5
        w_analog = 0.0

    if not has_deterministic_multi:
        redistribute = w_det * 0.3
        w_ens += redistribute * 0.6
        w_hist += redistribute * 0.4
        w_det *= 0.7

    if not has_cross_ref:
        w_ens += w_xref * 0.5
        w_hist += w_xref * 0.5
        w_xref = 0.0

    # Normalize
    total = w_ens + w_hist + w_det + w_xref + w_analog
    if total <= 0:
        return 0.35, 0.25, 0.20, 0.10, 0.10
    return w_ens / total, w_hist / total, w_det / total, w_xref / total, w_analog / total


def _dynamic_weights(
    horizon_days: int,
    n_historical: int,
    ensemble_spread: float,
) -> tuple[float, float, float]:
    """Backward-compatible 3-source weight function.

    Returns (ensemble, historical, deterministic) weights summing to 1.
    Internally delegates to the 4-source version and collapses cross-ref into det.
    """
    w_ens, w_hist, w_det, w_xref = _dynamic_weights_4(
        horizon_days, n_historical, ensemble_spread,
        has_deterministic_multi=False, has_cross_ref=False,
    )
    # Merge cross-ref weight back into deterministic for 3-tuple compat
    total = w_ens + w_hist + w_det + w_xref
    return w_ens / total, w_hist / total, (w_det + w_xref) / total


def get_recent_aqi(db, city_slug: str) -> int | None:
    """Get the most recent European AQI for a city. Returns None if no data."""
    try:
        row = db.execute(
            """SELECT AVG(aqi_european) FROM air_quality
            WHERE city_slug = ? AND forecast_date >= CURRENT_DATE - INTERVAL '1 day'
            AND aqi_european IS NOT NULL""",
            [city_slug],
        ).fetchone()
        if row and row[0] is not None:
            return int(row[0])
    except Exception:
        pass
    return None


def get_bias_correction(db, city_slug: str, variable: str) -> float:
    """Calcola la bias correction media dalle verifiche storiche.

    Ritorna il bias medio (forecast - observed). Sottrai dal forecast per correggere.
    """
    try:
        row = db.execute(
            """SELECT AVG(error), COUNT(*) FROM forecast_verification
            WHERE city_slug = ? AND variable = ? AND horizon_hours <= 48""",
            [city_slug, variable],
        ).fetchone()
        if row and row[1] >= 5:
            return float(row[0])
    except Exception:
        pass
    return 0.0


def _bootstrap_ci(
    values: np.ndarray,
    low: float,
    high: float,
    n_bootstrap: int = 500,
    ci_level: float = 0.90,
) -> tuple[float, float]:
    """Bootstrap confidence interval for probability estimate."""
    rng = np.random.default_rng(42)
    probs = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        if np.std(sample) < 0.01:
            sample = sample + rng.normal(0, 0.1, len(sample))
        try:
            kde = sp_stats.gaussian_kde(sample)
            x = np.linspace(low - 10, high + 10, 500)
            pdf = kde(x)
            dx = x[1] - x[0]
            mask = (x >= low) & (x <= high)
            p = float(np.sum(pdf[mask]) * dx)
            probs.append(np.clip(p, 0.001, 0.999))
        except Exception:
            continue

    if len(probs) < 10:
        return 0.0, 1.0

    alpha = (1 - ci_level) / 2
    return float(np.percentile(probs, alpha * 100)), float(np.percentile(probs, (1 - alpha) * 100))


def estimate_multiple_outcomes(
    ensemble_values: np.ndarray,
    thresholds: list[tuple[float, float, str]],
    historical_values: np.ndarray | None = None,
    deterministic_value: float | None = None,
    fast: bool = False,
    horizon_days: int = 1,
    bias_correction: float = 0.0,
    air_quality_aqi: int | None = None,
    model_weights: dict[str, float] | None = None,
    ensemble_model_labels: list[str] | None = None,
    deterministic_models: dict[str, float] | None = None,
    cross_ref_composite: float | None = None,
    atmospheric_stability: float | None = None,
    analog_probabilities: list[float | None] | None = None,
    bma_probabilities: list[float | None] | None = None,
    regime_info: dict | None = None,
) -> list[ProbabilityEstimate]:
    """Stima probabilità per outcome multipli, normalizzati a somma=1."""
    estimates = []
    for i, (low, high, label) in enumerate(thresholds):
        ap = analog_probabilities[i] if analog_probabilities and i < len(analog_probabilities) else None
        bp = bma_probabilities[i] if bma_probabilities and i < len(bma_probabilities) else None
        est = estimate_probability_kde(
            ensemble_values, low, high, historical_values, deterministic_value, label,
            fast=fast, horizon_days=horizon_days, bias_correction=bias_correction,
            air_quality_aqi=air_quality_aqi,
            model_weights=model_weights,
            ensemble_model_labels=ensemble_model_labels,
            deterministic_models=deterministic_models,
            cross_ref_composite=cross_ref_composite,
            atmospheric_stability=atmospheric_stability,
            analog_probability=ap,
            bma_probability=bp,
            regime_info=regime_info,
        )
        estimates.append(est)

    # Normalizza a somma = 1
    total = sum(e.blended_prob for e in estimates)
    if total > 0:
        for e in estimates:
            ratio = e.blended_prob / total
            e.probability = ratio
            e.blended_prob = ratio

    return estimates
