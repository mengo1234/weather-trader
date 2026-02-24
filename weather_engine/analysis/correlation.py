import numpy as np
from scipy import stats as sp_stats


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> dict:
    """Compute Pearson correlation with p-value."""
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_clean = x[mask]
    y_clean = y[mask]

    if len(x_clean) < 3:
        return {"r": 0.0, "p_value": 1.0, "n": len(x_clean)}

    r, p = sp_stats.pearsonr(x_clean, y_clean)
    return {"r": float(r), "p_value": float(p), "n": len(x_clean)}


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> dict:
    """Compute Spearman rank correlation."""
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_clean = x[mask]
    y_clean = y[mask]

    if len(x_clean) < 3:
        return {"rho": 0.0, "p_value": 1.0, "n": len(x_clean)}

    rho, p = sp_stats.spearmanr(x_clean, y_clean)
    return {"rho": float(rho), "p_value": float(p), "n": len(x_clean)}


def cross_variable_correlation(db, city_slug: str, var1: str, var2: str, days: int = 90) -> dict:
    """Compute correlation between two weather variables from stored data."""
    rows = db.execute(
        f"""SELECT {var1}, {var2} FROM forecasts_daily
        WHERE city_slug = ? AND {var1} IS NOT NULL AND {var2} IS NOT NULL
        ORDER BY date DESC LIMIT ?""",
        [city_slug, days],
    ).fetchall()

    if len(rows) < 3:
        return {"error": "insufficient data", "n": len(rows)}

    x = np.array([r[0] for r in rows])
    y = np.array([r[1] for r in rows])

    pearson = pearson_correlation(x, y)
    spearman = spearman_correlation(x, y)

    return {
        "var1": var1,
        "var2": var2,
        "pearson": pearson,
        "spearman": spearman,
    }
