"""Learned weights: optimize confidence, cross-ref, and blending weights from resolved bets.

Uses logistic regression for confidence/cross-ref weights and inverse-Brier for blending.
Smooth transition: new_weights = (1 - LEARNING_RATE) * default + LEARNING_RATE * learned.
Safety: if Brier score doesn't improve, keep defaults.
"""

import json
import logging
from datetime import datetime, timezone

import numpy as np

logger = logging.getLogger(__name__)

MIN_SAMPLES_LEARN = 50
LEARNING_RATE = 0.3
REGULARIZATION_ALPHA = 0.01

# Cache for loaded weights (TTL 1 hour)
_weights_cache: dict[str, tuple[float, dict]] = {}
_CACHE_TTL = 3600.0

# Default weights copied from confidence.py
DEFAULT_CONFIDENCE_WEIGHTS = {
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

# Default weights copied from cross_reference.py
DEFAULT_CROSS_REF_WEIGHTS = {
    "model_agreement": 0.16,
    "atmospheric_stability": 0.08,
    "pressure_patterns": 0.08,
    "soil_moisture_bias": 0.04,
    "cross_variable_consistency": 0.10,
    "marine_influence": 0.04,
    "flood_precip_consistency": 0.04,
    "climate_trend_alignment": 0.06,
    "aqi_weather_correlation": 0.04,
    "deterministic_agreement": 0.16,
    "teleconnection_alignment": 0.08,
    "ensemble_regime_score": 0.06,
    "extreme_value_score": 0.06,
}

# Default blending weights from probability.py
DEFAULT_BLENDING_4 = {
    "short": {"ensemble": 0.45, "historical": 0.10, "deterministic": 0.35, "cross_ref": 0.10},
    "medium": {"ensemble": 0.40, "historical": 0.20, "deterministic": 0.25, "cross_ref": 0.15},
    "long": {"ensemble": 0.30, "historical": 0.35, "deterministic": 0.15, "cross_ref": 0.20},
    "extended": {"ensemble": 0.20, "historical": 0.45, "deterministic": 0.10, "cross_ref": 0.25},
}

DEFAULT_BLENDING_5 = {
    "short": {"ensemble": 0.40, "historical": 0.08, "deterministic": 0.30, "cross_ref": 0.10, "analog": 0.12},
    "medium": {"ensemble": 0.35, "historical": 0.15, "deterministic": 0.22, "cross_ref": 0.13, "analog": 0.15},
    "long": {"ensemble": 0.25, "historical": 0.25, "deterministic": 0.12, "cross_ref": 0.18, "analog": 0.20},
    "extended": {"ensemble": 0.15, "historical": 0.30, "deterministic": 0.08, "cross_ref": 0.22, "analog": 0.25},
}


def _horizon_band(horizon_days: int) -> str:
    if horizon_days <= 2:
        return "short"
    elif horizon_days <= 5:
        return "medium"
    elif horizon_days <= 8:
        return "long"
    return "extended"


def _normalize_weights(weights: dict[str, float], min_weight: float = 0.01) -> dict[str, float]:
    """Normalize weights to sum=1, clamp minimum to min_weight."""
    clamped = {k: max(min_weight, v) for k, v in weights.items()}
    total = sum(clamped.values())
    return {k: v / total for k, v in clamped.items()}


def _blend_weights(default: dict[str, float], learned: dict[str, float], rate: float = LEARNING_RATE) -> dict[str, float]:
    """Smooth blend: (1 - rate) * default + rate * learned."""
    blended = {}
    for k in default:
        d = default[k]
        l = learned.get(k, d)
        blended[k] = (1 - rate) * d + rate * l
    return _normalize_weights(blended)


def _compute_brier(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_pred - y_true) ** 2))


def learn_confidence_weights(db, variable: str = "all") -> dict:
    """Learn confidence weights from resolved bets with saved confidence scores.

    Uses logistic regression on the 15 confidence factors vs binary outcome.
    """
    rows = db.execute(
        """SELECT b.confidence_scores_json, ps.outcome_binary
        FROM bets b
        JOIN prediction_scores ps ON ps.bet_id = b.id
        WHERE b.confidence_scores_json IS NOT NULL
        AND b.status IN ('won', 'lost')
        ORDER BY b.timestamp DESC
        LIMIT 500"""
    ).fetchall()

    if len(rows) < MIN_SAMPLES_LEARN:
        logger.info("Confidence weight learning: only %d samples (need %d), keeping defaults", len(rows), MIN_SAMPLES_LEARN)
        return DEFAULT_CONFIDENCE_WEIGHTS

    keys = sorted(DEFAULT_CONFIDENCE_WEIGHTS.keys())
    X = []
    y = []
    for scores_json, outcome in rows:
        try:
            scores = json.loads(scores_json)
            features = [scores.get(k, 50.0) for k in keys]
            X.append(features)
            y.append(outcome)
        except (json.JSONDecodeError, TypeError):
            continue

    if len(X) < MIN_SAMPLES_LEARN:
        return DEFAULT_CONFIDENCE_WEIGHTS

    X = np.array(X)
    y = np.array(y)

    try:
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(C=1.0 / REGULARIZATION_ALPHA, max_iter=1000, solver="lbfgs")
        model.fit(X, y)

        # Extract coefficients as weights (absolute value — direction doesn't matter for weight)
        coefs = np.abs(model.coef_[0])
        learned_raw = {k: float(c) for k, c in zip(keys, coefs)}
        learned = _normalize_weights(learned_raw)
    except Exception as e:
        logger.warning("Logistic regression failed for confidence weights: %s", e)
        return DEFAULT_CONFIDENCE_WEIGHTS

    # Safety: compare Brier with default vs learned
    # Compute weighted confidence scores with both weight sets
    default_brier = _eval_confidence_brier(X, y, keys, DEFAULT_CONFIDENCE_WEIGHTS)
    learned_brier = _eval_confidence_brier(X, y, keys, learned)

    if learned_brier >= default_brier:
        logger.info("Confidence learned weights not better (brier %.4f >= %.4f), keeping defaults",
                     learned_brier, default_brier)
        return DEFAULT_CONFIDENCE_WEIGHTS

    blended = _blend_weights(DEFAULT_CONFIDENCE_WEIGHTS, learned)

    # Save to DB
    _save_weights(db, "confidence", variable, blended, len(X), default_brier, learned_brier)
    logger.info("Confidence weights learned from %d samples: brier %.4f -> %.4f", len(X), default_brier, learned_brier)

    return blended


def _eval_confidence_brier(X: np.ndarray, y: np.ndarray, keys: list[str], weights: dict[str, float]) -> float:
    """Compute Brier score using confidence-weighted predictions."""
    w = np.array([weights.get(k, 0.05) for k in keys])
    # Weighted average of factor scores → confidence → scale to [0,1]
    confidence_scores = X @ w
    # Normalize to probability-like values
    preds = confidence_scores / 100.0
    preds = np.clip(preds, 0.01, 0.99)
    return _compute_brier(y, preds)


def learn_blending_weights(db, variable: str = "temperature_2m_max", horizon_band: str = "short") -> dict:
    """Learn blending weights from per-source Brier scores.

    Weights are inversely proportional to the mean Brier score of each source.
    """
    horizon_ranges = {
        "short": (0, 2),
        "medium": (3, 5),
        "long": (6, 8),
        "extended": (9, 30),
    }
    h_min, h_max = horizon_ranges.get(horizon_band, (0, 30))

    rows = db.execute(
        """SELECT ensemble_brier, historical_brier, deterministic_brier, analog_brier, bma_brier
        FROM prediction_scores
        WHERE variable = ? AND horizon_days BETWEEN ? AND ?
        AND ensemble_brier IS NOT NULL
        ORDER BY scored_at DESC
        LIMIT 500""",
        [variable, h_min, h_max],
    ).fetchall()

    if len(rows) < MIN_SAMPLES_LEARN:
        logger.info("Blending weight learning: only %d samples for %s/%s (need %d)",
                     len(rows), variable, horizon_band, MIN_SAMPLES_LEARN)
        return DEFAULT_BLENDING_4.get(horizon_band, DEFAULT_BLENDING_4["short"])

    # Compute mean Brier per source
    ens_brier = []
    hist_brier = []
    det_brier = []
    analog_brier = []
    bma_brier = []

    for eb, hb, db_, ab, bb in rows:
        if eb is not None:
            ens_brier.append(eb)
        if hb is not None:
            hist_brier.append(hb)
        if db_ is not None:
            det_brier.append(db_)
        if ab is not None:
            analog_brier.append(ab)
        if bb is not None:
            bma_brier.append(bb)

    has_analog = len(analog_brier) >= MIN_SAMPLES_LEARN // 2
    has_bma = len(bma_brier) >= MIN_SAMPLES_LEARN // 2

    # Use BMA score in deterministic slot if available
    if has_bma:
        det_brier = bma_brier

    # Inverse-Brier weighting
    sources = {
        "ensemble": np.mean(ens_brier) if ens_brier else 0.5,
        "historical": np.mean(hist_brier) if hist_brier else 0.5,
        "deterministic": np.mean(det_brier) if det_brier else 0.5,
        "cross_ref": 0.5,  # Cross-ref doesn't have per-source Brier
    }

    if has_analog:
        sources["analog"] = np.mean(analog_brier)
        default = DEFAULT_BLENDING_5.get(horizon_band, DEFAULT_BLENDING_5["short"])
    else:
        default = DEFAULT_BLENDING_4.get(horizon_band, DEFAULT_BLENDING_4["short"])

    # Inverse Brier: w_i = 1/brier_i
    inverse = {k: 1.0 / max(0.01, v) for k, v in sources.items()}
    learned = _normalize_weights(inverse)

    # Compute combined Brier for safety check
    default_combined = _eval_blending_brier(rows, default, has_analog)
    learned_combined = _eval_blending_brier(rows, learned, has_analog)

    if learned_combined >= default_combined:
        logger.info("Blending learned weights not better for %s/%s (brier %.4f >= %.4f)",
                     variable, horizon_band, learned_combined, default_combined)
        return default

    blended = _blend_weights(default, learned)

    n_sources = 5 if has_analog else 4
    weight_group = f"blending_{n_sources}"
    _save_weights(db, weight_group, f"{variable}_{horizon_band}", blended, len(rows), default_combined, learned_combined)
    logger.info("Blending weights learned for %s/%s from %d samples: brier %.4f -> %.4f",
                variable, horizon_band, len(rows), default_combined, learned_combined)

    return blended


def _eval_blending_brier(rows, weights: dict[str, float], has_analog: bool) -> float:
    """Evaluate combined Brier using given blending weights."""
    brier_values = []
    w_ens = weights.get("ensemble", 0.3)
    w_hist = weights.get("historical", 0.2)
    w_det = weights.get("deterministic", 0.2)
    w_analog = weights.get("analog", 0) if has_analog else 0

    for eb, hb, db_, ab, bb in rows:
        components = []
        if eb is not None:
            components.append(w_ens * eb)
        if hb is not None:
            components.append(w_hist * hb)
        if db_ is not None:
            components.append(w_det * db_)
        if has_analog and ab is not None:
            components.append(w_analog * ab)
        if components:
            brier_values.append(sum(components))

    return float(np.mean(brier_values)) if brier_values else 0.5


def learn_cross_ref_weights(db) -> dict:
    """Learn cross-reference weights from resolved bets with saved cross-ref scores.

    Uses logistic regression on the 13 cross-ref factors vs binary outcome.
    """
    rows = db.execute(
        """SELECT b.cross_ref_json, ps.outcome_binary
        FROM bets b
        JOIN prediction_scores ps ON ps.bet_id = b.id
        WHERE b.cross_ref_json IS NOT NULL
        AND b.status IN ('won', 'lost')
        ORDER BY b.timestamp DESC
        LIMIT 500"""
    ).fetchall()

    if len(rows) < MIN_SAMPLES_LEARN:
        logger.info("Cross-ref weight learning: only %d samples (need %d), keeping defaults", len(rows), MIN_SAMPLES_LEARN)
        return DEFAULT_CROSS_REF_WEIGHTS

    keys = sorted(DEFAULT_CROSS_REF_WEIGHTS.keys())
    X = []
    y = []
    for scores_json, outcome in rows:
        try:
            scores = json.loads(scores_json)
            features = [scores.get(k, 50.0) for k in keys]
            X.append(features)
            y.append(outcome)
        except (json.JSONDecodeError, TypeError):
            continue

    if len(X) < MIN_SAMPLES_LEARN:
        return DEFAULT_CROSS_REF_WEIGHTS

    X = np.array(X)
    y = np.array(y)

    try:
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(C=1.0 / REGULARIZATION_ALPHA, max_iter=1000, solver="lbfgs")
        model.fit(X, y)

        coefs = np.abs(model.coef_[0])
        learned_raw = {k: float(c) for k, c in zip(keys, coefs)}
        learned = _normalize_weights(learned_raw)
    except Exception as e:
        logger.warning("Logistic regression failed for cross-ref weights: %s", e)
        return DEFAULT_CROSS_REF_WEIGHTS

    # Safety check
    default_brier = _eval_confidence_brier(X, y, keys, DEFAULT_CROSS_REF_WEIGHTS)
    learned_brier = _eval_confidence_brier(X, y, keys, learned)

    if learned_brier >= default_brier:
        logger.info("Cross-ref learned weights not better (brier %.4f >= %.4f), keeping defaults",
                     learned_brier, default_brier)
        return DEFAULT_CROSS_REF_WEIGHTS

    blended = _blend_weights(DEFAULT_CROSS_REF_WEIGHTS, learned)

    _save_weights(db, "cross_ref", "all", blended, len(X), default_brier, learned_brier)
    logger.info("Cross-ref weights learned from %d samples: brier %.4f -> %.4f", len(X), default_brier, learned_brier)

    return blended


def _save_weights(db, weight_group: str, variable: str, weights: dict, n_samples: int,
                  brier_before: float, brier_after: float) -> None:
    """Save learned weights to the database."""
    now = datetime.now(timezone.utc)
    db.execute(
        """INSERT OR REPLACE INTO learned_weights
        (weight_group, variable, weights_json, n_samples, brier_before, brier_after, fitted_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)""",
        [weight_group, variable, json.dumps(weights), n_samples, brier_before, brier_after, now],
    )


def load_weights(db, weight_group: str, variable: str = "all") -> dict | None:
    """Load learned weights from DB with 1-hour TTL cache."""
    import time
    cache_key = f"{weight_group}:{variable}"
    now = time.monotonic()

    if cache_key in _weights_cache:
        cached_time, cached_weights = _weights_cache[cache_key]
        if now - cached_time < _CACHE_TTL:
            return cached_weights

    try:
        row = db.execute(
            "SELECT weights_json FROM learned_weights WHERE weight_group = ? AND variable = ?",
            [weight_group, variable],
        ).fetchone()

        if row and row[0]:
            weights = json.loads(row[0])
            _weights_cache[cache_key] = (now, weights)
            return weights
    except Exception as e:
        logger.debug("Failed to load weights for %s/%s: %s", weight_group, variable, e)

    return None


def get_effective_confidence_weights(db, variable: str = "all") -> dict:
    """Get confidence weights: learned if available, else defaults."""
    learned = load_weights(db, "confidence", variable)
    if learned:
        return learned
    return DEFAULT_CONFIDENCE_WEIGHTS.copy()


def get_effective_blending_weights(db, horizon_days: int, variable: str = "temperature_2m_max",
                                    n_sources: int = 4) -> dict | None:
    """Get blending weights: learned if available for the horizon band, else None (use defaults)."""
    band = _horizon_band(horizon_days)
    weight_group = f"blending_{n_sources}"
    return load_weights(db, weight_group, f"{variable}_{band}")
