from datetime import date, datetime, timezone

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from weather_engine.analysis.anomaly import detect_anomalies_from_db
from weather_engine.analysis.accuracy import compute_accuracy_from_db
from weather_engine.analysis.climate_normals import get_climate_normal, get_historical_values_for_doy
from weather_engine.analysis.descriptive import compute_descriptive_from_db
from weather_engine.analysis.probability import estimate_probability_kde, estimate_multiple_outcomes
from weather_engine.analysis.verification import run_verification, get_verification_summary
from weather_engine.db import get_db, get_city

router = APIRouter(prefix="/analysis", tags=["analysis"])


@router.get("/{city_slug}/stats")
def get_stats(city_slug: str, variable: str = "temperature_2m_max", days: int = 30):
    db = get_db()
    city = get_city(city_slug, db)
    if city is None:
        raise HTTPException(404, f"City '{city_slug}' not found")

    stats = compute_descriptive_from_db(db, city_slug, variable, days)
    return {"city": city, "stats": stats.model_dump()}


@router.get("/{city_slug}/probability")
def get_probability(
    city_slug: str,
    target_date: str = Query(..., description="Target date YYYY-MM-DD"),
    variable: str = "temperature_2m_max",
    threshold_low: float = Query(...),
    threshold_high: float = Query(...),
):
    db = get_db()
    city = get_city(city_slug, db)
    if city is None:
        raise HTTPException(404, f"City '{city_slug}' not found")

    parsed_date = date.fromisoformat(target_date)

    # Mappa variabili daily → ensemble (le colonne ensemble non hanno _max/_min)
    ensemble_var_map = {
        "temperature_2m_max": "temperature_2m",
        "temperature_2m_min": "temperature_2m",
        "temperature_2m_mean": "temperature_2m",
        "precipitation_sum": "precipitation",
        "wind_speed_10m_max": "wind_speed_10m",
    }
    ens_var = ensemble_var_map.get(variable, variable)

    # Per temperature_2m_max prendiamo il MAX per membro; per _min il MIN
    agg = "MAX" if "max" in variable or variable == "temperature_2m_mean" else "MIN"
    if "sum" in variable:
        agg = "SUM"

    # Get ensemble values for target date (aggregati per membro)
    ensemble_rows = db.execute(
        f"""SELECT {agg}({ens_var}) FROM ensemble_members
        WHERE city_slug = ? AND time::DATE = ? AND {ens_var} IS NOT NULL
        GROUP BY model, member_id""",
        [city_slug, parsed_date],
    ).fetchall()

    if not ensemble_rows:
        raise HTTPException(404, f"No ensemble data for {city_slug} on {target_date}")

    ensemble_values = np.array([r[0] for r in ensemble_rows])

    # Get historical values for same day of year
    historical_values = get_historical_values_for_doy(db, city_slug, parsed_date, variable)

    # Get deterministic forecast
    det_row = db.execute(
        f"""SELECT {variable} FROM forecasts_daily
        WHERE city_slug = ? AND date = ? AND {variable} IS NOT NULL
        ORDER BY model_run DESC LIMIT 1""",
        [city_slug, parsed_date],
    ).fetchone()
    deterministic = det_row[0] if det_row else None

    estimate = estimate_probability_kde(
        ensemble_values, threshold_low, threshold_high,
        historical_values if len(historical_values) > 0 else None,
        deterministic,
        f"{threshold_low}-{threshold_high}",
    )

    return {
        "city": city,
        "target_date": target_date,
        "variable": variable,
        "threshold_low": threshold_low,
        "threshold_high": threshold_high,
        "n_ensemble_members": len(ensemble_values),
        "n_historical_samples": len(historical_values),
        "estimate": estimate.model_dump(),
    }


@router.get("/{city_slug}/anomaly")
def get_anomalies(city_slug: str, variable: str = "temperature_2m_max", days: int = 90):
    db = get_db()
    city = get_city(city_slug, db)
    if city is None:
        raise HTTPException(404, f"City '{city_slug}' not found")

    anomalies = detect_anomalies_from_db(db, city_slug, variable, days)
    return {
        "city": city,
        "variable": variable,
        "n_anomalies": len(anomalies),
        "anomalies": [a.model_dump() for a in anomalies],
    }


@router.get("/{city_slug}/accuracy")
def get_accuracy(city_slug: str, variable: str = "temperature_2m_max"):
    db = get_db()
    city = get_city(city_slug, db)
    if city is None:
        raise HTTPException(404, f"City '{city_slug}' not found")

    metrics = compute_accuracy_from_db(db, city_slug, variable)
    if metrics is None:
        return {
            "city": city,
            "variable": variable,
            "message": "Insufficient data for accuracy computation. Need overlapping forecast and observation dates.",
        }

    return {"city": city, "accuracy": metrics.model_dump()}


@router.get("/{city_slug}/climate-normal")
def get_climate_normal_endpoint(city_slug: str, target_date: str, variable: str = "temperature_2m_max"):
    db = get_db()
    city = get_city(city_slug, db)
    if city is None:
        raise HTTPException(404, f"City '{city_slug}' not found")

    parsed_date = date.fromisoformat(target_date)
    normal = get_climate_normal(db, city_slug, parsed_date, variable)

    return {"city": city, **normal}


@router.post("/verify")
def run_verify(lookback_days: int = 7):
    db = get_db()
    result = run_verification(db, lookback_days)
    return result


@router.get("/verification/summary")
def verification_summary(city_slug: str | None = None):
    db = get_db()
    summary = get_verification_summary(db, city_slug)
    return {"summary": summary}


@router.post("/backtest")
def run_backtest(
    city_slug: str,
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    variable: str = "temperature_2m_max",
    min_edge: float = 0.05,
    min_confidence: float = 0.6,
    kelly_multiplier: float = 0.25,
    initial_bankroll: float = 1000.0,
):
    """Run a backtest simulation."""
    from weather_engine.backtesting.engine import BacktestEngine

    db = get_db()
    city = get_city(city_slug, db)
    if city is None:
        raise HTTPException(404, f"City '{city_slug}' not found")

    engine = BacktestEngine(db)
    result = engine.run(
        city_slug=city_slug,
        start_date=date.fromisoformat(start_date),
        end_date=date.fromisoformat(end_date),
        variable=variable,
        min_edge=min_edge,
        min_confidence=min_confidence,
        kelly_multiplier=kelly_multiplier,
        initial_bankroll=initial_bankroll,
    )

    return {
        "city": city,
        "period": {"start": start_date, "end": end_date},
        "params": {
            "variable": variable,
            "min_edge": min_edge,
            "min_confidence": min_confidence,
            "kelly_multiplier": kelly_multiplier,
            "initial_bankroll": initial_bankroll,
        },
        "summary": {
            "n_trades": result.n_trades,
            "total_pnl": round(result.total_pnl, 2),
            "win_rate": round(result.win_rate, 3),
            "sharpe_ratio": round(result.sharpe_ratio, 2),
            "max_drawdown": round(result.max_drawdown, 3),
            "profit_factor": round(result.profit_factor, 2) if result.profit_factor != float('inf') else None,
            "avg_edge": round(result.avg_edge, 4),
        },
        "equity_curve": result.equity_curve,
        "trades": [
            {
                "date": str(t.date),
                "city": t.city_slug,
                "outcome": t.outcome,
                "our_prob": round(t.our_prob, 3),
                "market_price": round(t.market_price, 3),
                "edge": round(t.edge, 4),
                "stake": t.stake,
                "odds": t.odds,
                "won": t.won,
                "pnl": t.pnl,
            }
            for t in result.trades
        ],
    }


@router.get("/{city_slug}/crps")
def get_crps(city_slug: str, variable: str = "temperature_2m_max", days: int = 30):
    """Compute CRPS (Continuous Ranked Probability Score) for ensemble forecasts."""
    db = get_db()
    city = get_city(city_slug, db)
    if city is None:
        raise HTTPException(404, f"City '{city_slug}' not found")

    from weather_engine.analysis.crps import crps_batch
    result = crps_batch(db, city_slug, variable, days)
    return {"city": city, **result}


@router.get("/{city_slug}/skill-score")
def get_skill_score(city_slug: str, variable: str = "temperature_2m_max", days: int = 30):
    """Compute CRPS Skill Score vs climatology."""
    db = get_db()
    city = get_city(city_slug, db)
    if city is None:
        raise HTTPException(404, f"City '{city_slug}' not found")

    from weather_engine.analysis.crps import skill_score_vs_climatology
    result = skill_score_vs_climatology(db, city_slug, variable, days)
    if result is None:
        return {"city": city, "message": "Insufficient data for skill score computation"}
    return {"city": city, **result}


@router.get("/{city_slug}/spread-skill")
def get_spread_skill(city_slug: str, days: int = 30):
    """Compute spread-skill relationship for ensemble calibration analysis."""
    db = get_db()
    city = get_city(city_slug, db)
    if city is None:
        raise HTTPException(404, f"City '{city_slug}' not found")

    from weather_engine.analysis.calibration import spread_skill
    result = spread_skill(db, city_slug, days)
    return {"city": city, **result}


@router.get("/{city_slug}/reliability")
def get_reliability(city_slug: str, variable: str = "temperature_2m_max", n_bins: int = 10):
    """Compute reliability diagram data for probabilistic forecast calibration."""
    db = get_db()
    city = get_city(city_slug, db)
    if city is None:
        raise HTTPException(404, f"City '{city_slug}' not found")

    from weather_engine.analysis.calibration import reliability_diagram
    result = reliability_diagram(db, city_slug, variable, n_bins)
    return {"city": city, **result}
