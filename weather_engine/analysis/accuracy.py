import logging

import numpy as np

from weather_engine.models.analysis import AccuracyMetrics

logger = logging.getLogger(__name__)


def compute_accuracy(
    forecast_values: np.ndarray,
    observed_values: np.ndarray,
    variable: str,
    horizon_hours: int = 24,
) -> AccuracyMetrics:
    """Compute forecast accuracy metrics."""
    mask = ~np.isnan(forecast_values) & ~np.isnan(observed_values)
    fc = forecast_values[mask]
    obs = observed_values[mask]

    if len(fc) < 2:
        raise ValueError(f"Need at least 2 pairs, got {len(fc)}")

    errors = fc - obs
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    bias = float(np.mean(errors))

    # Calibration via linear regression
    try:
        slope, intercept = np.polyfit(fc, obs, 1)
    except Exception:
        slope, intercept = 1.0, 0.0

    return AccuracyMetrics(
        variable=variable,
        horizon_hours=horizon_hours,
        n_samples=len(fc),
        mae=mae,
        rmse=rmse,
        bias=bias,
        calibration_slope=float(slope),
        calibration_intercept=float(intercept),
    )


def compute_brier_score(
    predicted_probs: np.ndarray,
    outcomes: np.ndarray,
) -> float:
    """Compute Brier score for probabilistic forecasts.
    predicted_probs: probability of event happening
    outcomes: 1 if event happened, 0 if not
    """
    mask = ~np.isnan(predicted_probs) & ~np.isnan(outcomes)
    p = predicted_probs[mask]
    o = outcomes[mask]
    if len(p) == 0:
        return float("nan")
    return float(np.mean((p - o) ** 2))


def compute_accuracy_from_db(db, city_slug: str, variable: str = "temperature_2m_max") -> AccuracyMetrics | None:
    """Compute accuracy by comparing forecasts with observations."""
    rows = db.execute(
        """SELECT f.temperature_2m_max, o.temperature_2m_max
        FROM forecasts_daily f
        JOIN observations o ON f.city_slug = o.city_slug AND f.date = o.date
        WHERE f.city_slug = ? AND f.temperature_2m_max IS NOT NULL AND o.temperature_2m_max IS NOT NULL
        ORDER BY f.date DESC LIMIT 90""",
        [city_slug],
    ).fetchall()

    if len(rows) < 2:
        return None

    fc = np.array([r[0] for r in rows])
    obs = np.array([r[1] for r in rows])

    return compute_accuracy(fc, obs, variable)
