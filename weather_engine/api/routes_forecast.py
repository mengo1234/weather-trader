from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from weather_engine.db import get_db, get_city, get_cities

router = APIRouter(prefix="/forecast", tags=["forecast"])


@router.get("/sources/status")
def get_sources_status():
    """Stato di tutte le fonti dati — freshness, coverage, reliability."""
    db = get_db()

    sources = []

    # Define all source tables and their metadata
    source_defs = [
        ("ensemble", "ensemble_members", "model", "time"),
        ("deterministic", "deterministic_forecasts", "model", "date"),
        ("forecast_hourly", "forecasts_hourly", None, "time"),
        ("forecast_daily", "forecasts_daily", None, "date"),
        ("observations", "observations", None, "date"),
        ("climate_normals", "climate_normals", None, None),
        ("air_quality", "air_quality", None, "forecast_date"),
        ("seasonal", "seasonal_forecast", None, "collected_at"),
        ("climate_indicators", "climate_indicators", "model", "date"),
        ("marine", "marine_data", None, "date"),
        ("flood", "flood_data", None, "date"),
        ("model_accuracy", "model_accuracy", "model", "date"),
        ("cross_reference", "cross_reference_scores", None, "target_date"),
    ]

    for name, table, model_col, date_col in source_defs:
        try:
            row_count = db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            last_update = None
            if date_col:
                lu_row = db.execute(f"SELECT MAX({date_col}) FROM {table}").fetchone()
                last_update = str(lu_row[0]) if lu_row and lu_row[0] else None

            n_models = None
            if model_col:
                nm_row = db.execute(f"SELECT COUNT(DISTINCT {model_col}) FROM {table}").fetchone()
                n_models = nm_row[0] if nm_row else 0

            # Get reliability from collection_log
            log_row = db.execute(
                """SELECT status, finished_at FROM collection_log
                WHERE collector = ? ORDER BY finished_at DESC LIMIT 1""",
                [name],
            ).fetchone()

            status = "active" if row_count > 0 else "empty"
            if log_row and log_row[0] == "error":
                status = "error"

            sources.append({
                "source": name,
                "table": table,
                "rows": row_count,
                "last_update": last_update,
                "n_models": n_models,
                "status": status,
            })
        except Exception as e:
            sources.append({
                "source": name,
                "table": table,
                "rows": 0,
                "last_update": None,
                "n_models": None,
                "status": "error",
                "error": str(e),
            })

    return {
        "sources": sources,
        "total_sources": len(sources),
        "active_sources": sum(1 for s in sources if s["status"] == "active"),
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/{city_slug}")
def get_forecast(city_slug: str, days: int = 7):
    db = get_db()
    city = get_city(city_slug, db)
    if city is None:
        raise HTTPException(404, f"City '{city_slug}' not found")

    # Get latest daily forecast
    daily = db.execute(
        """SELECT date, temperature_2m_max, temperature_2m_min, temperature_2m_mean,
                precipitation_sum, snowfall_sum, precipitation_hours,
                wind_speed_10m_max, wind_gusts_10m_max, uv_index_max,
                precipitation_probability_max
        FROM forecasts_daily
        WHERE city_slug = ?
        ORDER BY model_run DESC, date ASC
        LIMIT ?""",
        [city_slug, days],
    ).fetchall()

    # Get ensemble spread for each day
    ensemble_spread = db.execute(
        """SELECT time::DATE as date,
                AVG(temperature_2m) as mean_temp,
                STDDEV(temperature_2m) as std_temp,
                MIN(temperature_2m) as min_temp,
                MAX(temperature_2m) as max_temp,
                COUNT(DISTINCT member_id) as n_members
        FROM ensemble_members
        WHERE city_slug = ?
        GROUP BY time::DATE
        ORDER BY date ASC""",
        [city_slug],
    ).fetchall()

    spread_by_date = {}
    for r in ensemble_spread:
        spread_by_date[str(r[0])] = {
            "ensemble_mean": round(r[1], 1) if r[1] else None,
            "ensemble_std": round(r[2], 1) if r[2] else None,
            "ensemble_min": round(r[3], 1) if r[3] else None,
            "ensemble_max": round(r[4], 1) if r[4] else None,
            "n_members": r[5],
        }

    return {
        "city": city,
        "forecast": [
            {
                "date": str(r[0]),
                "temp_max": r[1],
                "temp_min": r[2],
                "temp_mean": r[3],
                "precipitation_sum": r[4],
                "snowfall_sum": r[5],
                "precipitation_hours": r[6],
                "wind_max": r[7],
                "wind_gusts_max": r[8],
                "uv_max": r[9],
                "precip_probability": r[10],
                "ensemble": spread_by_date.get(str(r[0]), {}),
            }
            for r in daily
        ],
    }


@router.get("/overview/all")
def get_overview():
    """Batch overview for all cities - replaces 24 separate API calls from the dashboard."""
    db = get_db()
    cities = get_cities(db)

    # Latest daily forecast (next 3 days) for each city
    daily_rows = db.execute(
        """WITH ranked AS (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY city_slug, date ORDER BY model_run DESC) as rn
            FROM forecasts_daily
            WHERE date >= CURRENT_DATE AND date <= CURRENT_DATE + INTERVAL '3 days'
        )
        SELECT city_slug, date, temperature_2m_max, temperature_2m_min,
               precipitation_sum, wind_speed_10m_max, wind_gusts_10m_max,
               uv_index_max, precipitation_probability_max
        FROM ranked WHERE rn = 1
        ORDER BY city_slug, date"""
    ).fetchall()

    # Ensemble spread per city (next 3 days)
    ens_rows = db.execute(
        """SELECT city_slug, time::DATE as date,
                AVG(temperature_2m) as mean_temp,
                STDDEV(temperature_2m) as std_temp,
                MIN(temperature_2m) as min_temp,
                MAX(temperature_2m) as max_temp,
                COUNT(DISTINCT member_id) as n_members
        FROM ensemble_members
        WHERE time::DATE >= CURRENT_DATE AND time::DATE <= CURRENT_DATE + INTERVAL '3 days'
        GROUP BY city_slug, time::DATE
        ORDER BY city_slug, date"""
    ).fetchall()

    # Build response grouped by city
    city_map = {c["slug"]: c for c in cities}
    daily_by_city: dict[str, list] = {}
    for r in daily_rows:
        daily_by_city.setdefault(r[0], []).append({
            "date": str(r[1]),
            "temp_max": r[2], "temp_min": r[3],
            "precipitation_sum": r[4],
            "wind_max": r[5], "wind_gusts_max": r[6],
            "uv_max": r[7], "precip_probability": r[8],
        })

    ens_by_city: dict[str, dict] = {}
    for r in ens_rows:
        ens_by_city.setdefault(r[0], {})[str(r[1])] = {
            "ensemble_mean": round(r[2], 1) if r[2] else None,
            "ensemble_std": round(r[3], 1) if r[3] else None,
            "ensemble_min": round(r[4], 1) if r[4] else None,
            "ensemble_max": round(r[5], 1) if r[5] else None,
            "n_members": r[6],
        }

    result = []
    for slug, city_info in city_map.items():
        forecasts = daily_by_city.get(slug, [])
        ensemble = ens_by_city.get(slug, {})
        # Merge ensemble into forecasts
        for f in forecasts:
            f["ensemble"] = ensemble.get(f["date"], {})
        result.append({
            "city": city_info,
            "forecast": forecasts,
        })

    return {"cities": result}


@router.get("/{city_slug}/convergence")
def get_convergence(city_slug: str):
    """Get forecast convergence analysis for a city."""
    from datetime import date as date_type
    from weather_engine.analysis.convergence import check_convergence

    db = get_db()
    city = get_city(city_slug, db)
    if city is None:
        raise HTTPException(404, f"City '{city_slug}' not found")

    result = check_convergence(db, city_slug, target_date=date_type.today())
    return {
        "city": city,
        "convergence": result,
    }
