from pydantic import BaseModel


class DescriptiveStats(BaseModel):
    variable: str
    count: int
    mean: float
    median: float
    std: float
    min: float
    max: float
    p10: float
    p25: float
    p75: float
    p90: float
    skewness: float
    kurtosis: float
    iqr: float


class ProbabilityEstimate(BaseModel):
    outcome: str
    probability: float
    confidence_lower: float
    confidence_upper: float
    ensemble_prob: float
    historical_prob: float
    deterministic_prob: float
    blended_prob: float
    ensemble_spread: float
    analog_prob: float | None = None
    bma_prob: float | None = None
    regime_info: dict | None = None


class ConfidenceInterval(BaseModel):
    variable: str
    level: float
    lower: float
    upper: float
    point_estimate: float


class AnomalyResult(BaseModel):
    date: str
    variable: str
    value: float
    z_score: float
    iqr_score: float
    is_anomaly: bool
    climate_normal: float | None = None
    deviation: float | None = None


class AccuracyMetrics(BaseModel):
    variable: str
    horizon_hours: int
    n_samples: int
    mae: float
    rmse: float
    bias: float
    brier_score: float | None = None
    calibration_slope: float | None = None
    calibration_intercept: float | None = None
