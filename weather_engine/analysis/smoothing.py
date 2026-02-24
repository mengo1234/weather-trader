import numpy as np
import pandas as pd


def simple_moving_average(values: np.ndarray, window: int = 7) -> np.ndarray:
    """Simple moving average."""
    series = pd.Series(values)
    return series.rolling(window=window, min_periods=1).mean().to_numpy()


def exponential_smoothing(values: np.ndarray, span: int = 7) -> np.ndarray:
    """Exponential weighted moving average."""
    series = pd.Series(values)
    return series.ewm(span=span).mean().to_numpy()


def double_exponential_smoothing(
    values: np.ndarray,
    alpha: float = 0.3,
    beta: float = 0.1,
) -> np.ndarray:
    """Holt's double exponential smoothing (level + trend)."""
    n = len(values)
    if n < 2:
        return values.copy()

    level = np.zeros(n)
    trend = np.zeros(n)
    smoothed = np.zeros(n)

    level[0] = values[0]
    trend[0] = values[1] - values[0] if n > 1 else 0

    for t in range(1, n):
        if np.isnan(values[t]):
            level[t] = level[t - 1] + trend[t - 1]
            trend[t] = trend[t - 1]
        else:
            level[t] = alpha * values[t] + (1 - alpha) * (level[t - 1] + trend[t - 1])
            trend[t] = beta * (level[t] - level[t - 1]) + (1 - beta) * trend[t - 1]
        smoothed[t] = level[t] + trend[t]

    smoothed[0] = level[0] + trend[0]
    return smoothed
