"""Bayesian Model Averaging (BMA) for optimal model blending.

Replaces simple weighted Gaussian blending with EM-optimized mixture.
Each model k contributes a component N(a_k + forecast_k, sigma_k^2) 
with mixing weight pi_k.
"""
import logging
from dataclasses import dataclass

import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


@dataclass
class BMAComponent:
    model: str
    weight: float     # pi_k (mixing weight)
    bias: float       # a_k (bias correction)
    variance: float   # sigma_k^2


@dataclass
class BMAResult:
    components: list[BMAComponent]
    converged: bool
    n_iterations: int
    log_likelihood: float
    n_training: int


def fit_bma(
    model_forecasts: dict[str, np.ndarray],
    observations: np.ndarray,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> BMAResult | None:
    """Fit BMA using EM algorithm.

    model_forecasts: {model_name: array of forecasts} - all same length as observations
    observations: array of actual observed values
    
    E-step: compute posterior weights z_ik for each obs i, model k
    M-step: update pi_k, a_k, sigma_k^2
    
    Requires minimum 20 forecast-observation pairs.
    Returns None if fitting fails or too few samples.
    """
    obs = observations[~np.isnan(observations)]
    if len(obs) < 20:
        return None

    models = list(model_forecasts.keys())
    K = len(models)
    if K < 2:
        return None

    N = len(obs)

    # Align forecasts with observations (only non-NaN obs)
    fc_matrix = np.zeros((N, K))
    for j, model in enumerate(models):
        fc = model_forecasts[model]
        if len(fc) != len(observations):
            return None
        # Apply same NaN mask
        fc_clean = fc[~np.isnan(observations)]
        if len(fc_clean) != N:
            return None
        fc_matrix[:, j] = fc_clean

    # Initialize parameters
    pi_k = np.ones(K) / K  # uniform mixing weights
    a_k = np.zeros(K)       # zero initial bias
    sigma2_k = np.ones(K) * np.var(obs) * 0.5  # initial variance

    prev_ll = -float("inf")
    converged = False

    for iteration in range(max_iter):
        # E-step: compute z_ik = P(model k | obs_i, params)
        z = np.zeros((N, K))
        for k in range(K):
            mean_k = a_k[k] + fc_matrix[:, k]
            std_k = np.sqrt(max(sigma2_k[k], 1e-10))
            z[:, k] = pi_k[k] * sp_stats.norm.pdf(obs, loc=mean_k, scale=std_k)

        # Normalize rows
        row_sums = z.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-300)
        z = z / row_sums

        # M-step
        for k in range(K):
            z_k = z[:, k]
            sum_z = z_k.sum()

            if sum_z < 1e-10:
                continue

            # Update mixing weight
            pi_k[k] = sum_z / N

            # Update bias (weighted regression)
            a_k[k] = np.sum(z_k * (obs - fc_matrix[:, k])) / sum_z

            # Update variance
            residuals = obs - (a_k[k] + fc_matrix[:, k])
            sigma2_k[k] = max(np.sum(z_k * residuals ** 2) / sum_z, 1e-6)

        # Log-likelihood for convergence check
        ll = 0.0
        for i in range(N):
            p = 0.0
            for k in range(K):
                mean_k = a_k[k] + fc_matrix[i, k]
                std_k = np.sqrt(max(sigma2_k[k], 1e-10))
                p += pi_k[k] * sp_stats.norm.pdf(obs[i], loc=mean_k, scale=std_k)
            ll += np.log(max(p, 1e-300))

        if abs(ll - prev_ll) < tol:
            converged = True
            break
        prev_ll = ll

    # Normalize weights (safety)
    pi_k = pi_k / pi_k.sum()

    components = []
    for k, model in enumerate(models):
        components.append(BMAComponent(
            model=model,
            weight=round(float(pi_k[k]), 6),
            bias=round(float(a_k[k]), 4),
            variance=round(float(sigma2_k[k]), 4),
        ))

    # Sort by weight descending
    components.sort(key=lambda c: -c.weight)

    return BMAResult(
        components=components,
        converged=converged,
        n_iterations=iteration + 1,
        log_likelihood=round(float(ll), 2),
        n_training=N,
    )


def bma_predict(bma: BMAResult, current_forecasts: dict[str, float]) -> tuple[float, float]:
    """BMA predictive mean and standard deviation.

    Returns (mean, std) of the BMA mixture distribution.
    """
    mean = 0.0
    var = 0.0

    for comp in bma.components:
        fc = current_forecasts.get(comp.model)
        if fc is None:
            continue
        comp_mean = comp.bias + fc
        mean += comp.weight * comp_mean

    # Variance: mixture variance = sum(w_k * (sigma_k^2 + mu_k^2)) - mu_total^2
    for comp in bma.components:
        fc = current_forecasts.get(comp.model)
        if fc is None:
            continue
        comp_mean = comp.bias + fc
        var += comp.weight * (comp.variance + comp_mean ** 2)
    var -= mean ** 2
    var = max(var, 1e-6)

    return round(mean, 4), round(float(np.sqrt(var)), 4)


def bma_cdf(bma: BMAResult, current_forecasts: dict[str, float], x: float) -> float:
    """BMA cumulative distribution function: P(Y <= x)."""
    cdf_val = 0.0
    for comp in bma.components:
        fc = current_forecasts.get(comp.model)
        if fc is None:
            continue
        comp_mean = comp.bias + fc
        comp_std = np.sqrt(max(comp.variance, 1e-10))
        cdf_val += comp.weight * sp_stats.norm.cdf(x, loc=comp_mean, scale=comp_std)
    return float(np.clip(cdf_val, 0.0, 1.0))


def bma_probability(
    bma: BMAResult,
    current_forecasts: dict[str, float],
    threshold_low: float,
    threshold_high: float,
) -> float:
    """Compute P(threshold_low <= Y <= threshold_high) using BMA."""
    p_high = bma_cdf(bma, current_forecasts, threshold_high)
    p_low = bma_cdf(bma, current_forecasts, threshold_low)
    return float(np.clip(p_high - p_low, 0.001, 0.999))


def get_training_data(
    db, city_slug: str, variable: str, lookback: int = 60,
) -> tuple[dict[str, np.ndarray], np.ndarray] | None:
    """Get training data for BMA: paired model forecasts and observations.

    Returns (model_forecasts, observations) or None if insufficient data.
    """
    from datetime import date as dt_date, timedelta

    today = dt_date.today()
    cutoff = today - timedelta(days=lookback)

    # Map variable to deterministic column
    var_map = {
        "temperature_2m_max": ("temp_max", "temperature_2m_max"),
        "temperature_2m_min": ("temp_min", "temperature_2m_min"),
        "precipitation_sum": ("precip_sum", "precipitation_sum"),
        "wind_speed_10m_max": ("wind_max", "wind_speed_10m_max"),
    }

    det_col, obs_col = var_map.get(variable, (None, None))
    if det_col is None:
        return None

    # Get dates with observations
    obs_rows = db.execute(
        f"""SELECT date, {obs_col} FROM observations
        WHERE city_slug = ? AND date >= ? AND {obs_col} IS NOT NULL
        ORDER BY date""",
        [city_slug, cutoff],
    ).fetchall()

    if len(obs_rows) < 20:
        return None

    dates = [r[0] for r in obs_rows]
    obs_values = np.array([float(r[1]) for r in obs_rows])

    # Get deterministic forecasts for each model on those dates
    det_rows = db.execute(
        f"""SELECT date, model, {det_col} FROM deterministic_forecasts
        WHERE city_slug = ? AND date >= ? AND {det_col} IS NOT NULL""",
        [city_slug, cutoff],
    ).fetchall()

    if not det_rows:
        return None

    # Organize by model
    model_data: dict[str, dict] = {}
    for d, model, val in det_rows:
        model_data.setdefault(model, {})[d] = float(val)

    # Build aligned arrays (only models that have forecasts for >= 50% of dates)
    model_forecasts = {}
    for model, date_vals in model_data.items():
        fc_arr = np.full(len(dates), np.nan)
        for i, d in enumerate(dates):
            if d in date_vals:
                fc_arr[i] = date_vals[d]
        # Require at least 50% coverage
        if np.sum(~np.isnan(fc_arr)) >= len(dates) * 0.5:
            # Fill NaNs with model mean
            fc_mean = np.nanmean(fc_arr)
            fc_arr = np.where(np.isnan(fc_arr), fc_mean, fc_arr)
            model_forecasts[model] = fc_arr

    if len(model_forecasts) < 2:
        return None

    return model_forecasts, obs_values
