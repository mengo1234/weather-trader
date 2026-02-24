import threading
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query

from weather_engine.db import get_db, get_city, get_cities

router = APIRouter(tags=["health"])


@router.get("/health")
def health():
    db = get_db()
    try:
        db.execute("SELECT 1").fetchone()
        db_ok = True
    except Exception:
        db_ok = False

    return {
        "status": "ok" if db_ok else "degraded",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "database": "connected" if db_ok else "error",
    }


@router.get("/metrics")
def metrics():
    db = get_db()
    stats = {}

    for table in ["forecasts_hourly", "forecasts_daily", "ensemble_members", "observations", "climate_normals"]:
        try:
            count = db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            stats[table] = count
        except Exception:
            stats[table] = -1

    # Latest collection
    try:
        latest = db.execute(
            "SELECT collector, city_slug, finished_at, status FROM collection_log ORDER BY finished_at DESC LIMIT 5"
        ).fetchall()
        stats["recent_collections"] = [
            {"collector": r[0], "city": r[1], "at": str(r[2]), "status": r[3]} for r in latest
        ]
    except Exception:
        stats["recent_collections"] = []

    return stats


@router.post("/admin/seed-historical")
def seed_historical(
    city: str | None = Query(None, description="Slug citta' (default: tutte)"),
    start_date: str = Query("2020-01-01"),
    end_date: str | None = Query(None),
):
    """Avvia seed storico in background (non blocca il server)."""
    from weather_engine.collectors.historical import HistoricalCollector

    db = get_db()
    if city:
        cities = [get_city(city, db)]
        if cities[0] is None:
            raise HTTPException(404, f"Citta' '{city}' non trovata")
    else:
        cities = get_cities(db)

    def _run():
        h = HistoricalCollector(get_db())
        for c in cities:
            try:
                n = h.collect(c, start_date=start_date, end_date=end_date)
                h.compute_climate_normals(c["slug"])
            except Exception as e:
                import logging
                logging.getLogger(__name__).error("Seed %s fallito: %s", c["slug"], e)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return {"status": "avviato", "cities": [c["slug"] for c in cities], "start_date": start_date}
