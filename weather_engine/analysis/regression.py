import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def linear_regression(x: np.ndarray, y: np.ndarray) -> dict:
    """Simple linear regression."""
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_clean = x[mask].reshape(-1, 1)
    y_clean = y[mask]

    if len(x_clean) < 2:
        return {"error": "insufficient data"}

    model = LinearRegression()
    model.fit(x_clean, y_clean)

    y_pred = model.predict(x_clean)
    ss_res = np.sum((y_clean - y_pred) ** 2)
    ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return {
        "slope": float(model.coef_[0]),
        "intercept": float(model.intercept_),
        "r_squared": float(r_squared),
    }


def polynomial_regression(x: np.ndarray, y: np.ndarray, degree: int = 2) -> dict:
    """Polynomial regression."""
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_clean = x[mask].reshape(-1, 1)
    y_clean = y[mask]

    if len(x_clean) < degree + 1:
        return {"error": "insufficient data"}

    poly = PolynomialFeatures(degree=degree)
    x_poly = poly.fit_transform(x_clean)

    model = LinearRegression()
    model.fit(x_poly, y_clean)

    y_pred = model.predict(x_poly)
    ss_res = np.sum((y_clean - y_pred) ** 2)
    ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return {
        "coefficients": model.coef_.tolist(),
        "intercept": float(model.intercept_),
        "degree": degree,
        "r_squared": float(r_squared),
    }
