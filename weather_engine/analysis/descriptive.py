import numpy as np
from scipy import stats as sp_stats

from weather_engine.models.analysis import DescriptiveStats


def compute_descriptive(values: np.ndarray, variable: str) -> DescriptiveStats:
    """Compute comprehensive descriptive statistics for a variable."""
    clean = values[~np.isnan(values)]
    if len(clean) < 2:
        raise ValueError(f"Need at least 2 values for descriptive stats, got {len(clean)}")

    percentiles = np.percentile(clean, [10, 25, 50, 75, 90])

    return DescriptiveStats(
        variable=variable,
        count=len(clean),
        mean=float(np.mean(clean)),
        median=float(percentiles[2]),
        std=float(np.std(clean, ddof=1)),
        min=float(np.min(clean)),
        max=float(np.max(clean)),
        p10=float(percentiles[0]),
        p25=float(percentiles[1]),
        p75=float(percentiles[3]),
        p90=float(percentiles[4]),
        skewness=float(sp_stats.skew(clean)),
        kurtosis=float(sp_stats.kurtosis(clean)),
        iqr=float(percentiles[3] - percentiles[1]),
    )


def compute_descriptive_from_db(db, city_slug: str, variable: str, days: int = 30) -> DescriptiveStats:
    """Compute descriptive stats from recent forecast data."""
    rows = db.execute(
        f"""SELECT {variable} FROM forecasts_daily
        WHERE city_slug = ? AND {variable} IS NOT NULL
        ORDER BY date DESC LIMIT ?""",
        [city_slug, days],
    ).fetchall()

    if not rows:
        raise ValueError(f"No data found for {city_slug}/{variable}")

    values = np.array([r[0] for r in rows])
    return compute_descriptive(values, variable)
