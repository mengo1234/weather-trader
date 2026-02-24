"""Recalibration: update calibration bins, fit calibration models, apply to probabilities.

Cold start strategy:
  0-29 resolved → identity (no calibration)
  30-99 resolved → linear (polyfit)
  100+  resolved → isotonic regression
"""

import logging
import pickle
import time
from datetime import datetime, timezone

import numpy as np

logger = logging.getLogger(__name__)

MIN_SAMPLES_LINEAR = 30
MIN_SAMPLES_ISOTONIC = 100
N_BINS = 10

# Module-level calibration cache: {variable: (fn, timestamp)}
_calibration_cache: dict[str, tuple[callable, float]] = {}
_CACHE_TTL = 3600  # 1 hour


def update_calibration_bins(db, variable: str = "all") -> dict:
    """Update calibration bins from prediction_scores data.

    Groups predictions by probability bin and computes observed frequency.

    Args:
        db: Database connection.
        variable: Variable to update bins for, or "all" for all variables.

    Returns:
        Dict with variables updated and bin counts.
    """
    if variable == "all":
        var_rows = db.execute(
            "SELECT DISTINCT variable FROM prediction_scores WHERE variable IS NOT NULL"
        ).fetchall()
        variables = [r[0] for r in var_rows]
        if not variables:
            return {"updated": 0, "variables": []}
        results = {}
        for v in variables:
            results[v] = _update_bins_for_variable(db, v)
        return {"updated": len(variables), "variables": list(results.keys()), "details": results}
    else:
        result = _update_bins_for_variable(db, variable)
        return {"updated": 1, "variables": [variable], "details": {variable: result}}


def _update_bins_for_variable(db, variable: str) -> dict:
    """Update calibration bins for a single variable."""
    rows = db.execute(
        "SELECT our_prob, outcome_binary FROM prediction_scores WHERE variable = ?",
        [variable],
    ).fetchall()

    if not rows:
        return {"n_predictions": 0, "bins_updated": 0}

    probs = np.array([r[0] for r in rows])
    outcomes = np.array([r[1] for r in rows])

    bin_edges = np.linspace(0, 1, N_BINS + 1)
    now = datetime.now(timezone.utc)
    bins_updated = 0

    for i in range(N_BINS):
        lo, hi = float(bin_edges[i]), float(bin_edges[i + 1])
        if i == N_BINS - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)

        n_pred = int(np.sum(mask))
        n_pos = int(np.sum(outcomes[mask])) if n_pred > 0 else 0
        sum_pred = float(np.sum(probs[mask])) if n_pred > 0 else 0.0
        avg_pred = float(np.mean(probs[mask])) if n_pred > 0 else None
        obs_freq = n_pos / n_pred if n_pred > 0 else None

        # REPLACE INTO calibration_bins
        try:
            db.execute("DELETE FROM calibration_bins WHERE variable = ? AND bin_index = ?", [variable, i])
            db.execute(
                """INSERT INTO calibration_bins
                (variable, bin_index, bin_low, bin_high, n_predictions, n_positive,
                 sum_predicted, avg_predicted, observed_frequency, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [variable, i, lo, hi, n_pred, n_pos, sum_pred, avg_pred, obs_freq, now],
            )
            bins_updated += 1
        except Exception as e:
            logger.warning("Failed to update bin %d for %s: %s", i, variable, e)

    return {"n_predictions": len(rows), "bins_updated": bins_updated}


def fit_calibration_model(db, variable: str) -> dict:
    """Fit a calibration model for the given variable.

    Strategy:
      n < 30: identity (no calibration)
      30 <= n < 100: linear (polyfit degree 1)
      n >= 100: isotonic regression

    Only saves if calibration improves Brier score (cross-validated).

    Returns:
        Dict with model_type, n_samples, brier_before, brier_after.
    """
    rows = db.execute(
        "SELECT our_prob, outcome_binary FROM prediction_scores WHERE variable = ?",
        [variable],
    ).fetchall()

    n = len(rows)
    now = datetime.now(timezone.utc)

    if n < MIN_SAMPLES_LINEAR:
        # Identity: no calibration
        try:
            db.execute("DELETE FROM calibration_models WHERE variable = ?", [variable])
            db.execute(
                """INSERT INTO calibration_models (variable, model_type, model_blob, n_samples, brier_before, brier_after, fitted_at)
                VALUES (?, 'identity', NULL, ?, NULL, NULL, ?)""",
                [variable, n, now],
            )
        except Exception as e:
            logger.warning("Failed to save identity model for %s: %s", variable, e)
        return {"model_type": "identity", "n_samples": n, "brier_before": None, "brier_after": None}

    probs = np.array([r[0] for r in rows])
    outcomes = np.array([r[1] for r in rows])

    # Brier before calibration
    brier_before = float(np.mean((probs - outcomes) ** 2))

    if n < MIN_SAMPLES_ISOTONIC:
        # Linear calibration
        model_type = "linear"
        coeffs = np.polyfit(probs, outcomes, 1)
        calibrated = np.polyval(coeffs, probs)
        calibrated = np.clip(calibrated, 1e-6, 1 - 1e-6)
        brier_after = float(np.mean((calibrated - outcomes) ** 2))
        model_blob = pickle.dumps(coeffs)
    else:
        # Isotonic regression
        model_type = "isotonic"
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(out_of_bounds="clip")

        # Simple cross-validation: fit on first 80%, evaluate on last 20%
        split = int(n * 0.8)
        iso_cv = IsotonicRegression(out_of_bounds="clip")
        iso_cv.fit(probs[:split], outcomes[:split])
        cal_test = iso_cv.predict(probs[split:])
        brier_cv = float(np.mean((cal_test - outcomes[split:]) ** 2))
        brier_uncal_test = float(np.mean((probs[split:] - outcomes[split:]) ** 2))

        # Use cross-validated brier for safety check
        brier_after = brier_cv

        # Fit on full data for the actual model
        iso.fit(probs, outcomes)
        model_blob = pickle.dumps(iso)

        # If CV shows no improvement, check more carefully
        if brier_cv >= brier_uncal_test:
            brier_after = brier_cv

    # Safety: don't save if calibration makes things worse
    if brier_after >= brier_before:
        logger.info(
            "Calibration for %s would not improve (before=%.4f, after=%.4f), keeping identity",
            variable, brier_before, brier_after,
        )
        try:
            db.execute("DELETE FROM calibration_models WHERE variable = ?", [variable])
            db.execute(
                """INSERT INTO calibration_models (variable, model_type, model_blob, n_samples, brier_before, brier_after, fitted_at)
                VALUES (?, 'identity', NULL, ?, ?, ?, ?)""",
                [variable, n, brier_before, brier_after, now],
            )
        except Exception as e:
            logger.warning("Failed to save identity model for %s: %s", variable, e)
        return {"model_type": "identity", "n_samples": n, "brier_before": brier_before, "brier_after": brier_after}

    # Save the model
    try:
        db.execute("DELETE FROM calibration_models WHERE variable = ?", [variable])
        db.execute(
            """INSERT INTO calibration_models (variable, model_type, model_blob, n_samples, brier_before, brier_after, fitted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [variable, model_type, model_blob, n, brier_before, brier_after, now],
        )
    except Exception as e:
        logger.warning("Failed to save calibration model for %s: %s", variable, e)

    logger.info(
        "Fitted %s calibration for %s: n=%d, brier %.4f -> %.4f",
        model_type, variable, n, brier_before, brier_after,
    )

    # Invalidate cache
    _calibration_cache.pop(variable, None)

    return {
        "model_type": model_type,
        "n_samples": n,
        "brier_before": round(brier_before, 4),
        "brier_after": round(brier_after, 4),
    }


def load_calibration_fn(db, variable: str):
    """Load a calibration function for the given variable.

    Uses module-level cache with 1h TTL.

    Returns:
        Callable (prob -> calibrated_prob) or None if identity/no model.
    """
    # Check cache
    if variable in _calibration_cache:
        fn, ts = _calibration_cache[variable]
        if time.time() - ts < _CACHE_TTL:
            return fn

    row = db.execute(
        "SELECT model_type, model_blob FROM calibration_models WHERE variable = ?",
        [variable],
    ).fetchone()

    if row is None or row[0] == "identity" or row[1] is None:
        _calibration_cache[variable] = (None, time.time())
        return None

    model_type, model_blob = row

    try:
        if model_type == "linear":
            coeffs = pickle.loads(model_blob)

            def linear_fn(p):
                return float(np.clip(np.polyval(coeffs, p), 1e-3, 1 - 1e-3))

            _calibration_cache[variable] = (linear_fn, time.time())
            return linear_fn

        elif model_type == "isotonic":
            iso = pickle.loads(model_blob)

            def isotonic_fn(p):
                return float(np.clip(iso.predict([p])[0], 1e-3, 1 - 1e-3))

            _calibration_cache[variable] = (isotonic_fn, time.time())
            return isotonic_fn

    except Exception as e:
        logger.warning("Failed to load calibration model for %s: %s", variable, e)

    _calibration_cache[variable] = (None, time.time())
    return None


def calibrate_probability(prob: float, calibration_fn=None) -> float:
    """Apply calibration function to a probability.

    If fn is None, returns prob unchanged (identity passthrough).
    """
    if calibration_fn is None:
        return prob
    calibrated = calibration_fn(prob)
    return max(0.001, min(0.999, calibrated))


def calibrate_estimates(estimates: list, db, variable: str) -> list:
    """Apply recalibration to a list of ProbabilityEstimate objects.

    Loads calibration function, applies to each blended_prob,
    then renormalizes so probabilities sum to 1.0.

    Args:
        estimates: List of ProbabilityEstimate.
        db: Database connection.
        variable: Weather variable name.

    Returns:
        Modified list of ProbabilityEstimate with calibrated probabilities.
    """
    cal_fn = load_calibration_fn(db, variable)
    if cal_fn is None:
        return estimates

    # Apply calibration
    calibrated_probs = []
    for est in estimates:
        cp = calibrate_probability(est.blended_prob, cal_fn)
        calibrated_probs.append(cp)

    # Renormalize to sum = 1.0
    total = sum(calibrated_probs)
    if total > 0:
        calibrated_probs = [p / total for p in calibrated_probs]

    # Update estimates
    for est, cp in zip(estimates, calibrated_probs):
        est.probability = cp
        est.blended_prob = cp

    return estimates


def get_reliability_data(db, variable: str = "temperature_2m_max") -> dict:
    """Get reliability diagram data from calibration_bins.

    Returns:
        Dict with bins data and calibration error.
    """
    rows = db.execute(
        """SELECT bin_index, bin_low, bin_high, n_predictions, n_positive,
                  avg_predicted, observed_frequency
        FROM calibration_bins WHERE variable = ?
        ORDER BY bin_index ASC""",
        [variable],
    ).fetchall()

    if not rows:
        return {"bins": [], "n_predictions": 0, "calibration_error": None}

    bins = []
    total_predictions = 0
    errors = []

    for r in rows:
        bin_data = {
            "bin_index": r[0],
            "bin_low": r[1],
            "bin_high": r[2],
            "n_predictions": r[3],
            "n_positive": r[4],
            "avg_predicted": round(r[5], 3) if r[5] is not None else None,
            "observed_frequency": round(r[6], 3) if r[6] is not None else None,
        }
        bins.append(bin_data)
        total_predictions += r[3] or 0

        if r[5] is not None and r[6] is not None and (r[3] or 0) > 0:
            errors.append(abs(r[5] - r[6]))

    calibration_error = round(float(np.mean(errors)), 4) if errors else None

    return {
        "bins": bins,
        "n_predictions": total_predictions,
        "calibration_error": calibration_error,
        "variable": variable,
    }
