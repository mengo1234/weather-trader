"""Extreme Value Theory: GEV and Peaks Over Threshold (POT) analysis.

Fits GEV distribution to block maxima and Generalized Pareto to exceedances
above a high threshold. Computes return periods and tail probability scores.
"""
import logging
from dataclasses import dataclass

import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


@dataclass
class GEVFit:
    shape: float  # xi: <0=Weibull (bounded), 0=Gumbel, >0=Frechet (heavy tail)
    loc: float    # mu
    scale: float  # sigma
    n_blocks: int


@dataclass
class POTResult:
    shape: float
    loc: float
    scale: float
    threshold: float
    n_exceedances: int
    exceedance_rate: float  # fraction of values above threshold


@dataclass
class ReturnPeriod:
    years: int
    return_level: float
    ci_lower: float | None = None
    ci_upper: float | None = None


@dataclass
class ExtremeAnalysis:
    gev: GEVFit | None
    pot: POTResult | None
    return_periods: list[ReturnPeriod]
    tail_score: float  # 0-100 (100=normal, 0=extreme)
    forecast_percentile: float  # where forecast falls in historical distribution


def fit_gev(block_maxima: np.ndarray, block_size: str = "annual") -> GEVFit | None:
    """Fit Generalized Extreme Value distribution to block maxima.

    block_maxima: array of maximum values per block (e.g., annual maxima).
    Returns GEVFit or None if fitting fails.
    """
    clean = block_maxima[~np.isnan(block_maxima)]
    if len(clean) < 5:
        return None

    try:
        # scipy genextreme uses sign convention c = -shape
        c, loc, scale = sp_stats.genextreme.fit(clean)
        shape = -c  # Convert to standard GEV convention
        return GEVFit(
            shape=round(float(shape), 4),
            loc=round(float(loc), 4),
            scale=round(float(scale), 4),
            n_blocks=len(clean),
        )
    except Exception as e:
        logger.debug("GEV fit failed: %s", e)
        return None


def fit_pot(values: np.ndarray, threshold_percentile: float = 95) -> POTResult | None:
    """Fit Generalized Pareto Distribution to exceedances over threshold.

    Uses Peaks Over Threshold method with the given percentile as threshold.
    """
    clean = values[~np.isnan(values)]
    if len(clean) < 20:
        return None

    threshold = float(np.percentile(clean, threshold_percentile))
    exceedances = clean[clean > threshold] - threshold

    if len(exceedances) < 5:
        return None

    try:
        shape, loc, scale = sp_stats.genpareto.fit(exceedances, floc=0)
        return POTResult(
            shape=round(float(shape), 4),
            loc=round(float(loc), 4),
            scale=round(float(scale), 4),
            threshold=round(threshold, 4),
            n_exceedances=len(exceedances),
            exceedance_rate=round(len(exceedances) / len(clean), 4),
        )
    except Exception as e:
        logger.debug("POT fit failed: %s", e)
        return None


def compute_return_periods(
    gev_fit: GEVFit,
    years: list[int] | None = None,
) -> list[ReturnPeriod]:
    """Compute return levels for specified return periods using fitted GEV.

    Return level for T-year event: x_T where P(X > x_T) = 1/T.
    """
    if years is None:
        years = [2, 5, 10, 25, 50, 100]

    results = []
    c = -gev_fit.shape  # Back to scipy convention
    loc = gev_fit.loc
    scale = gev_fit.scale

    for T in years:
        try:
            # Return level: quantile at probability 1 - 1/T
            p = 1.0 - 1.0 / T
            return_level = float(sp_stats.genextreme.ppf(p, c, loc=loc, scale=scale))
            results.append(ReturnPeriod(
                years=T,
                return_level=round(return_level, 2),
            ))
        except Exception:
            continue

    return results


def tail_probability_score(
    forecast_value: float,
    gev_fit: GEVFit | None = None,
    pot: POTResult | None = None,
    historical: np.ndarray | None = None,
) -> float:
    """Score 0-100 indicating how extreme a forecast value is.

    100 = perfectly normal (near median)
    0 = extremely rare (deep in the tail)
    """
    scores = []

    # Method 1: GEV-based scoring
    if gev_fit is not None:
        try:
            c = -gev_fit.shape
            cdf_val = sp_stats.genextreme.cdf(forecast_value, c, loc=gev_fit.loc, scale=gev_fit.scale)
            # Distance from median (0.5): closer to 0.5 = more normal
            distance_from_center = abs(cdf_val - 0.5) * 2  # 0 to 1
            gev_score = (1 - distance_from_center) * 100
            scores.append(max(0, min(100, gev_score)))
        except Exception:
            pass

    # Method 2: POT-based scoring (for upper tail)
    if pot is not None and forecast_value > pot.threshold:
        try:
            excess = forecast_value - pot.threshold
            survival = sp_stats.genpareto.sf(excess, pot.shape, loc=0, scale=pot.scale)
            # Low survival probability = more extreme
            pot_score = survival * 100 * pot.exceedance_rate
            scores.append(max(0, min(100, pot_score)))
        except Exception:
            pass

    # Method 3: Empirical percentile
    if historical is not None:
        clean = historical[~np.isnan(historical)]
        if len(clean) > 10:
            percentile = sp_stats.percentileofscore(clean, forecast_value) / 100
            distance = abs(percentile - 0.5) * 2
            emp_score = (1 - distance) * 100
            scores.append(max(0, min(100, emp_score)))

    if not scores:
        return 50.0  # neutral

    return round(float(np.mean(scores)), 1)


def analyze_extremes(
    db, city_slug: str, variable: str, forecast_value: float,
) -> ExtremeAnalysis:
    """Full extreme value analysis for a forecast value.

    Fetches historical data, fits GEV+POT, computes return periods and tail score.
    """
    # Get historical observations
    obs_col_map = {
        "temperature_2m_max": "temperature_2m_max",
        "temperature_2m_min": "temperature_2m_min",
        "precipitation_sum": "precipitation_sum",
        "wind_speed_10m_max": "wind_speed_10m_max",
    }
    obs_col = obs_col_map.get(variable, variable)

    rows = db.execute(
        f"""SELECT {obs_col} FROM observations
        WHERE city_slug = ? AND {obs_col} IS NOT NULL
        ORDER BY date""",
        [city_slug],
    ).fetchall()

    historical = np.array([r[0] for r in rows], dtype=float) if rows else np.array([])

    # Fit GEV to annual maxima
    gev = None
    if len(historical) >= 365:
        # Create approximate annual blocks (365-day blocks)
        n_years = len(historical) // 365
        if n_years >= 3:
            block_maxima = np.array([
                np.max(historical[i * 365:(i + 1) * 365])
                for i in range(n_years)
            ])
            gev = fit_gev(block_maxima)

    # Fit POT
    pot = fit_pot(historical) if len(historical) >= 20 else None

    # Return periods
    return_periods = compute_return_periods(gev) if gev else []

    # Tail score
    score = tail_probability_score(forecast_value, gev, pot, historical)

    # Forecast percentile in historical
    if len(historical) > 0:
        fc_pct = float(sp_stats.percentileofscore(historical, forecast_value))
    else:
        fc_pct = 50.0

    return ExtremeAnalysis(
        gev=gev,
        pot=pot,
        return_periods=return_periods,
        tail_score=score,
        forecast_percentile=round(fc_pct, 1),
    )
