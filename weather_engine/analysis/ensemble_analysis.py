"""Ensemble percentile analysis + regime detection via Gaussian Mixture Models.

Provides distributional analysis of ensemble forecasts: percentiles, spread metrics,
and detection of multi-modal (bimodal) regimes using GMM with BIC selection.
"""
import logging
from dataclasses import dataclass, field

import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


@dataclass
class EnsemblePercentiles:
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float
    iqr: float
    spread: float  # p90 - p10


@dataclass
class RegimeInfo:
    n_regimes: int
    regime_means: list[float]
    regime_weights: list[float]
    regime_stds: list[float]
    is_bimodal: bool
    bimodality_coefficient: float
    dominant_regime_weight: float


@dataclass
class EnsembleAnalysis:
    percentiles: EnsemblePercentiles
    regime: RegimeInfo
    n_members: int
    mean: float
    std: float
    skewness: float
    kurtosis: float


def compute_percentiles(values: np.ndarray) -> EnsemblePercentiles:
    """Compute ensemble percentiles and spread metrics."""
    clean = values[~np.isnan(values)]
    if len(clean) < 3:
        raise ValueError(f"Need at least 3 values, got {len(clean)}")

    p10, p25, p50, p75, p90 = np.percentile(clean, [10, 25, 50, 75, 90])
    return EnsemblePercentiles(
        p10=float(p10),
        p25=float(p25),
        p50=float(p50),
        p75=float(p75),
        p90=float(p90),
        iqr=float(p75 - p25),
        spread=float(p90 - p10),
    )


def detect_regime(values: np.ndarray, max_components: int = 3) -> RegimeInfo:
    """Detect forecast regimes using Gaussian Mixture Model with BIC selection.

    Falls back to bimodality coefficient if GMM fitting fails.
    BC > 0.555 indicates bimodality.
    """
    clean = values[~np.isnan(values)]
    if len(clean) < 5:
        return _fallback_regime(clean)

    # Compute bimodality coefficient as fallback/supplement
    bc = _bimodality_coefficient(clean)

    # Try GMM with BIC selection
    try:
        from sklearn.mixture import GaussianMixture

        best_bic = float("inf")
        best_n = 1
        best_gmm = None

        for n in range(1, min(max_components + 1, len(clean) // 3 + 1)):
            if n < 1:
                continue
            gmm = GaussianMixture(
                n_components=n, covariance_type="full",
                max_iter=200, n_init=3, random_state=42,
            )
            gmm.fit(clean.reshape(-1, 1))
            bic = gmm.bic(clean.reshape(-1, 1))
            if bic < best_bic:
                best_bic = bic
                best_n = n
                best_gmm = gmm

        if best_gmm is not None:
            means = best_gmm.means_.flatten().tolist()
            weights = best_gmm.weights_.tolist()
            stds = np.sqrt(best_gmm.covariances_.flatten()).tolist()

            # Sort by weight descending
            order = sorted(range(len(weights)), key=lambda i: -weights[i])
            means = [means[i] for i in order]
            weights = [weights[i] for i in order]
            stds = [stds[i] for i in order]

            is_bimodal = best_n >= 2 and bc > 0.555
            dominant_weight = weights[0]

            return RegimeInfo(
                n_regimes=best_n,
                regime_means=means,
                regime_weights=weights,
                regime_stds=stds,
                is_bimodal=is_bimodal,
                bimodality_coefficient=round(bc, 4),
                dominant_regime_weight=round(dominant_weight, 4),
            )

    except ImportError:
        logger.debug("sklearn not available, using bimodality coefficient only")
    except Exception as e:
        logger.debug("GMM fitting failed: %s", e)

    return _fallback_regime(clean, bc)


def _bimodality_coefficient(values: np.ndarray) -> float:
    """Compute Sarle's bimodality coefficient.

    BC = (skewness^2 + 1) / kurtosis_adjusted
    BC > 0.555 suggests bimodality.
    """
    n = len(values)
    if n < 3:
        return 0.0

    skew = float(sp_stats.skew(values))
    kurt = float(sp_stats.kurtosis(values, fisher=False))  # excess=False → Pearson kurtosis

    if kurt <= 0:
        return 0.0

    # Adjusted for sample size
    bc = (skew ** 2 + 1) / kurt
    return float(np.clip(bc, 0, 1))


def _fallback_regime(values: np.ndarray, bc: float = 0.0) -> RegimeInfo:
    """Fallback regime detection using bimodality coefficient only."""
    if len(values) == 0:
        return RegimeInfo(
            n_regimes=1, regime_means=[0.0], regime_weights=[1.0],
            regime_stds=[0.0], is_bimodal=False, bimodality_coefficient=0.0,
            dominant_regime_weight=1.0,
        )

    if bc == 0.0:
        bc = _bimodality_coefficient(values)

    return RegimeInfo(
        n_regimes=2 if bc > 0.555 else 1,
        regime_means=[float(np.mean(values))],
        regime_weights=[1.0],
        regime_stds=[float(np.std(values))],
        is_bimodal=bc > 0.555,
        bimodality_coefficient=round(bc, 4),
        dominant_regime_weight=1.0,
    )


def analyze_ensemble_from_values(values: np.ndarray, variable: str = "") -> EnsembleAnalysis:
    """Analyze ensemble from raw values (for pipeline use)."""
    clean = values[~np.isnan(values)]
    if len(clean) < 3:
        raise ValueError(f"Need at least 3 values, got {len(clean)}")

    percentiles = compute_percentiles(clean)
    regime = detect_regime(clean)

    return EnsembleAnalysis(
        percentiles=percentiles,
        regime=regime,
        n_members=len(clean),
        mean=float(np.mean(clean)),
        std=float(np.std(clean)),
        skewness=float(sp_stats.skew(clean)),
        kurtosis=float(sp_stats.kurtosis(clean)),
    )


def analyze_ensemble(db, city_slug: str, target_date, variable: str = "temperature_2m") -> EnsembleAnalysis | None:
    """Analyze ensemble for a city/date from database."""
    rows = db.execute(
        f"""SELECT {variable} FROM ensemble_members
        WHERE city_slug = ? AND time::DATE = ? AND {variable} IS NOT NULL""",
        [city_slug, target_date],
    ).fetchall()

    if not rows or len(rows) < 3:
        return None

    values = np.array([r[0] for r in rows], dtype=float)
    return analyze_ensemble_from_values(values, variable)
