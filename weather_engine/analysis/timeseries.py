import logging

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

logger = logging.getLogger(__name__)


def decompose_stl(series: pd.Series, period: int = 365) -> dict:
    """STL decomposition into trend, seasonal, residual."""
    if len(series) < period * 2:
        period = max(7, len(series) // 3)
        if period % 2 == 0:
            period += 1

    stl = STL(series.dropna(), period=period, robust=True)
    result = stl.fit()

    return {
        "trend": result.trend.tolist(),
        "seasonal": result.seasonal.tolist(),
        "residual": result.resid.tolist(),
        "period": period,
    }


def compute_autocorrelation(values: np.ndarray, max_lag: int = 30) -> list[float]:
    """Compute autocorrelation for given lags."""
    clean = values[~np.isnan(values)]
    if len(clean) < max_lag + 1:
        max_lag = len(clean) - 1

    mean = np.mean(clean)
    var = np.var(clean)
    if var == 0:
        return [1.0] + [0.0] * max_lag

    acf = []
    for lag in range(max_lag + 1):
        if lag == 0:
            acf.append(1.0)
        else:
            c = np.mean((clean[lag:] - mean) * (clean[:-lag] - mean))
            acf.append(float(c / var))
    return acf


def detect_trend(values: np.ndarray) -> dict:
    """Simple linear trend detection."""
    clean = values[~np.isnan(values)]
    if len(clean) < 3:
        return {"slope": 0.0, "direction": "none", "strength": 0.0}

    x = np.arange(len(clean))
    slope, intercept = np.polyfit(x, clean, 1)

    # R-squared as strength
    predicted = slope * x + intercept
    ss_res = np.sum((clean - predicted) ** 2)
    ss_tot = np.sum((clean - np.mean(clean)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    direction = "increasing" if slope > 0.01 else ("decreasing" if slope < -0.01 else "stable")

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_squared),
        "direction": direction,
        "strength": float(abs(r_squared)),
    }


def forecast_arima(
    values: np.ndarray,
    steps: int = 7,
    order: tuple[int, int, int] = (2, 1, 2),
) -> dict:
    """Simple ARIMA forecast. Falls back to naive if ARIMA fails."""
    clean = values[~np.isnan(values)]
    if len(clean) < 30:
        # Not enough data for ARIMA; use naive forecast
        last = float(clean[-1]) if len(clean) > 0 else 0
        return {
            "forecast": [last] * steps,
            "method": "naive",
        }

    try:
        from statsmodels.tsa.arima.model import ARIMA

        model = ARIMA(clean, order=order)
        fitted = model.fit()
        fc = fitted.forecast(steps=steps)
        return {
            "forecast": fc.tolist(),
            "method": "arima",
            "aic": float(fitted.aic),
        }
    except Exception as e:
        logger.warning("ARIMA failed, using EWM: %s", e)
        series = pd.Series(clean)
        ewm = series.ewm(span=7).mean()
        last = float(ewm.iloc[-1])
        return {
            "forecast": [last] * steps,
            "method": "ewm_fallback",
        }
