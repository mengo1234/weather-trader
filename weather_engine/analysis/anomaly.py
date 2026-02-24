import numpy as np

from weather_engine.models.analysis import AnomalyResult


def detect_anomalies_zscore(
    values: np.ndarray,
    dates: list[str],
    variable: str,
    threshold: float = 2.5,
    climate_normals: dict | None = None,
) -> list[AnomalyResult]:
    """Detect anomalies using Z-score method."""
    clean = values.copy()
    mean = np.nanmean(clean)
    std = np.nanstd(clean)
    if std == 0:
        return []

    results = []
    for i, (val, date) in enumerate(zip(clean, dates)):
        if np.isnan(val):
            continue

        z_score = (val - mean) / std
        iqr_score = _iqr_score(val, clean)

        normal = climate_normals.get(date) if climate_normals else None
        deviation = (val - normal) if normal is not None else None

        is_anomaly = abs(z_score) > threshold or abs(iqr_score) > 1.5

        if is_anomaly:
            results.append(AnomalyResult(
                date=date,
                variable=variable,
                value=float(val),
                z_score=float(z_score),
                iqr_score=float(iqr_score),
                is_anomaly=True,
                climate_normal=normal,
                deviation=float(deviation) if deviation is not None else None,
            ))

    return results


def _iqr_score(value: float, values: np.ndarray) -> float:
    """Compute IQR-based score for a value."""
    clean = values[~np.isnan(values)]
    q1, q3 = np.percentile(clean, [25, 75])
    iqr = q3 - q1
    if iqr == 0:
        return 0.0
    if value > q3:
        return float((value - q3) / iqr)
    elif value < q1:
        return float((q1 - value) / iqr)
    return 0.0


def detect_anomalies_from_db(db, city_slug: str, variable: str = "temperature_2m_max", days: int = 90) -> list[AnomalyResult]:
    """Detect anomalies from stored forecast data."""
    rows = db.execute(
        f"""SELECT date, {variable} FROM forecasts_daily
        WHERE city_slug = ? AND {variable} IS NOT NULL
        ORDER BY date DESC LIMIT ?""",
        [city_slug, days],
    ).fetchall()

    if not rows:
        return []

    dates = [str(r[0]) for r in rows]
    values = np.array([r[1] for r in rows])

    # Get climate normals if available
    normals = {}
    try:
        normal_rows = db.execute(
            """SELECT day_of_year, temperature_2m_max_mean FROM climate_normals
            WHERE city_slug = ?""",
            [city_slug],
        ).fetchall()
        for r in normal_rows:
            normals[r[0]] = r[1]
    except Exception:
        pass

    return detect_anomalies_zscore(values, dates, variable, climate_normals=normals)
