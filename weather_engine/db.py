import json
import logging
from pathlib import Path

import duckdb

from weather_engine.config import settings

logger = logging.getLogger(__name__)

_connection: duckdb.DuckDBPyConnection | None = None

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS cities (
    slug VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    latitude DOUBLE NOT NULL,
    longitude DOUBLE NOT NULL,
    timezone VARCHAR NOT NULL,
    country VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS forecasts_hourly (
    city_slug VARCHAR NOT NULL,
    time TIMESTAMP NOT NULL,
    model_run TIMESTAMP NOT NULL,
    temperature_2m DOUBLE,
    relative_humidity_2m DOUBLE,
    dew_point_2m DOUBLE,
    apparent_temperature DOUBLE,
    pressure_msl DOUBLE,
    surface_pressure DOUBLE,
    cloud_cover DOUBLE,
    cloud_cover_low DOUBLE,
    cloud_cover_mid DOUBLE,
    cloud_cover_high DOUBLE,
    wind_speed_10m DOUBLE,
    wind_direction_10m DOUBLE,
    wind_gusts_10m DOUBLE,
    precipitation DOUBLE,
    rain DOUBLE,
    snowfall DOUBLE,
    snow_depth DOUBLE,
    visibility DOUBLE,
    cape DOUBLE,
    shortwave_radiation DOUBLE,
    direct_radiation DOUBLE,
    diffuse_radiation DOUBLE,
    uv_index DOUBLE,
    PRIMARY KEY (city_slug, time, model_run)
);

CREATE TABLE IF NOT EXISTS forecasts_daily (
    city_slug VARCHAR NOT NULL,
    date DATE NOT NULL,
    model_run TIMESTAMP NOT NULL,
    temperature_2m_max DOUBLE,
    temperature_2m_min DOUBLE,
    temperature_2m_mean DOUBLE,
    apparent_temperature_max DOUBLE,
    apparent_temperature_min DOUBLE,
    precipitation_sum DOUBLE,
    rain_sum DOUBLE,
    snowfall_sum DOUBLE,
    precipitation_hours DOUBLE,
    precipitation_probability_max DOUBLE,
    wind_speed_10m_max DOUBLE,
    wind_gusts_10m_max DOUBLE,
    wind_direction_10m_dominant DOUBLE,
    shortwave_radiation_sum DOUBLE,
    uv_index_max DOUBLE,
    sunrise VARCHAR,
    sunset VARCHAR,
    PRIMARY KEY (city_slug, date, model_run)
);

CREATE TABLE IF NOT EXISTS ensemble_members (
    city_slug VARCHAR NOT NULL,
    time TIMESTAMP NOT NULL,
    model VARCHAR NOT NULL,
    member_id INTEGER NOT NULL,
    temperature_2m DOUBLE,
    precipitation DOUBLE,
    wind_speed_10m DOUBLE,
    wind_gusts_10m DOUBLE,
    cloud_cover DOUBLE,
    pressure_msl DOUBLE,
    relative_humidity_2m DOUBLE,
    dew_point_2m DOUBLE,
    shortwave_radiation DOUBLE,
    cape DOUBLE,
    surface_pressure DOUBLE,
    soil_temperature DOUBLE,
    soil_moisture DOUBLE,
    visibility DOUBLE,
    snow_depth DOUBLE,
    PRIMARY KEY (city_slug, time, model, member_id)
);

CREATE TABLE IF NOT EXISTS deterministic_forecasts (
    city_slug VARCHAR NOT NULL,
    date DATE NOT NULL,
    model VARCHAR NOT NULL,
    model_run TIMESTAMP,
    temp_max DOUBLE,
    temp_min DOUBLE,
    precip_sum DOUBLE,
    wind_max DOUBLE,
    wind_gusts_max DOUBLE,
    pressure_msl DOUBLE,
    cape DOUBLE,
    snow_depth DOUBLE,
    soil_temp DOUBLE,
    soil_moisture DOUBLE,
    visibility DOUBLE,
    PRIMARY KEY (city_slug, date, model, model_run)
);

CREATE TABLE IF NOT EXISTS observations (
    city_slug VARCHAR NOT NULL,
    date DATE NOT NULL,
    temperature_2m_max DOUBLE,
    temperature_2m_min DOUBLE,
    temperature_2m_mean DOUBLE,
    precipitation_sum DOUBLE,
    rain_sum DOUBLE,
    snowfall_sum DOUBLE,
    wind_speed_10m_max DOUBLE,
    wind_gusts_10m_max DOUBLE,
    wind_direction_10m_dominant DOUBLE,
    shortwave_radiation_sum DOUBLE,
    pressure_msl_mean DOUBLE,
    PRIMARY KEY (city_slug, date)
);

CREATE TABLE IF NOT EXISTS climate_normals (
    city_slug VARCHAR NOT NULL,
    day_of_year INTEGER NOT NULL,
    temperature_2m_max_mean DOUBLE,
    temperature_2m_max_std DOUBLE,
    temperature_2m_min_mean DOUBLE,
    temperature_2m_min_std DOUBLE,
    temperature_2m_mean_mean DOUBLE,
    precipitation_sum_mean DOUBLE,
    precipitation_sum_std DOUBLE,
    wind_speed_10m_max_mean DOUBLE,
    PRIMARY KEY (city_slug, day_of_year)
);

CREATE TABLE IF NOT EXISTS forecast_verification (
    city_slug VARCHAR NOT NULL,
    target_date DATE NOT NULL,
    forecast_date TIMESTAMP NOT NULL,
    horizon_hours INTEGER NOT NULL,
    variable VARCHAR NOT NULL,
    forecast_value DOUBLE NOT NULL,
    observed_value DOUBLE NOT NULL,
    error DOUBLE NOT NULL,
    abs_error DOUBLE NOT NULL,
    squared_error DOUBLE NOT NULL,
    PRIMARY KEY (city_slug, target_date, forecast_date, variable)
);

CREATE TABLE IF NOT EXISTS market_predictions (
    id INTEGER PRIMARY KEY DEFAULT nextval('market_pred_seq'),
    market_id VARCHAR,
    question VARCHAR NOT NULL,
    city_slug VARCHAR NOT NULL,
    variable VARCHAR NOT NULL,
    target_date DATE NOT NULL,
    predicted_at TIMESTAMP NOT NULL,
    outcomes_json VARCHAR NOT NULL,
    best_bet VARCHAR,
    edge DOUBLE,
    kelly_fraction DOUBLE,
    result VARCHAR
);

CREATE TABLE IF NOT EXISTS air_quality (
    city_slug VARCHAR NOT NULL,
    forecast_date DATE NOT NULL,
    hour INTEGER NOT NULL,
    pm10 DOUBLE,
    pm25 DOUBLE,
    ozone DOUBLE,
    no2 DOUBLE,
    aqi_european INTEGER,
    co DOUBLE,
    so2 DOUBLE,
    us_aqi INTEGER,
    dust DOUBLE,
    uv_index DOUBLE,
    ammonia DOUBLE,
    collected_at TIMESTAMP DEFAULT now(),
    PRIMARY KEY (city_slug, forecast_date, hour)
);

CREATE TABLE IF NOT EXISTS marine_data (
    city_slug VARCHAR NOT NULL,
    date DATE NOT NULL,
    wave_height_max DOUBLE,
    wave_period_max DOUBLE,
    wave_direction_dominant INTEGER,
    swell_wave_height_max DOUBLE,
    ocean_current_velocity DOUBLE,
    collected_at TIMESTAMP DEFAULT now(),
    PRIMARY KEY (city_slug, date)
);

CREATE TABLE IF NOT EXISTS flood_data (
    city_slug VARCHAR NOT NULL,
    date DATE NOT NULL,
    river_discharge DOUBLE,
    river_discharge_mean DOUBLE,
    river_discharge_max DOUBLE,
    collected_at TIMESTAMP DEFAULT now(),
    PRIMARY KEY (city_slug, date)
);

CREATE TABLE IF NOT EXISTS seasonal_forecast (
    city_slug VARCHAR NOT NULL,
    month INTEGER NOT NULL,
    year INTEGER NOT NULL,
    temp_anomaly DOUBLE,
    precip_anomaly DOUBLE,
    collected_at TIMESTAMP DEFAULT now(),
    PRIMARY KEY (city_slug, month, year)
);

CREATE TABLE IF NOT EXISTS climate_indicators (
    city_slug VARCHAR NOT NULL,
    date DATE NOT NULL,
    model VARCHAR NOT NULL,
    temp_max DOUBLE,
    temp_min DOUBLE,
    temp_mean DOUBLE,
    precip_sum DOUBLE,
    wind_max DOUBLE,
    collected_at TIMESTAMP DEFAULT now(),
    PRIMARY KEY (city_slug, date, model)
);

CREATE TABLE IF NOT EXISTS model_accuracy (
    model VARCHAR NOT NULL,
    city_slug VARCHAR NOT NULL,
    variable VARCHAR NOT NULL,
    horizon_days INTEGER NOT NULL,
    date DATE NOT NULL,
    forecast_value DOUBLE,
    observed_value DOUBLE,
    error DOUBLE,
    abs_error DOUBLE,
    PRIMARY KEY (model, city_slug, variable, horizon_days, date)
);

CREATE TABLE IF NOT EXISTS cross_reference_scores (
    city_slug VARCHAR NOT NULL,
    target_date DATE NOT NULL,
    computed_at TIMESTAMP DEFAULT now(),
    model_agreement DOUBLE,
    atmospheric_stability DOUBLE,
    pressure_patterns DOUBLE,
    soil_moisture_bias DOUBLE,
    cross_variable_consistency DOUBLE,
    marine_influence DOUBLE,
    flood_precip_consistency DOUBLE,
    climate_trend_alignment DOUBLE,
    aqi_weather_correlation DOUBLE,
    deterministic_agreement DOUBLE,
    composite_score DOUBLE,
    source_count INTEGER,
    PRIMARY KEY (city_slug, target_date)
);

CREATE TABLE IF NOT EXISTS collection_log (
    id INTEGER PRIMARY KEY DEFAULT nextval('collection_log_seq'),
    collector VARCHAR NOT NULL,
    city_slug VARCHAR NOT NULL,
    started_at TIMESTAMP NOT NULL,
    finished_at TIMESTAMP,
    status VARCHAR NOT NULL,
    rows_inserted INTEGER DEFAULT 0,
    error_message VARCHAR
);

CREATE TABLE IF NOT EXISTS bets (
    id INTEGER PRIMARY KEY DEFAULT nextval('bets_seq'),
    timestamp TIMESTAMP NOT NULL,
    market_question VARCHAR NOT NULL,
    outcome VARCHAR NOT NULL,
    stake DOUBLE DEFAULT 0,
    odds DOUBLE DEFAULT 0,
    our_prob DOUBLE DEFAULT 0,
    edge DOUBLE DEFAULT 0,
    confidence DOUBLE DEFAULT 0,
    city_slug VARCHAR,
    target_date DATE,
    status VARCHAR DEFAULT 'pending',
    pnl DOUBLE DEFAULT 0,
    resolved_at TIMESTAMP,
    resolution_source VARCHAR
);

CREATE TABLE IF NOT EXISTS teleconnection_indices (
    index_name VARCHAR NOT NULL,
    date DATE NOT NULL,
    value DOUBLE NOT NULL,
    collected_at TIMESTAMP DEFAULT now(),
    PRIMARY KEY (index_name, date)
);

CREATE TABLE IF NOT EXISTS model_crps (
    model VARCHAR NOT NULL,
    city_slug VARCHAR NOT NULL,
    variable VARCHAR NOT NULL,
    horizon_days INTEGER NOT NULL,
    date DATE NOT NULL,
    crps_value DOUBLE,
    PRIMARY KEY (model, city_slug, variable, horizon_days, date)
);

CREATE TABLE IF NOT EXISTS horizon_profiles (
    city_slug VARCHAR NOT NULL,
    variable VARCHAR NOT NULL,
    horizon_days INTEGER NOT NULL,
    n_samples INTEGER,
    mae DOUBLE,
    rmse DOUBLE,
    bias DOUBLE,
    crps DOUBLE,
    updated_at TIMESTAMP DEFAULT now(),
    PRIMARY KEY (city_slug, variable, horizon_days)
);

CREATE TABLE IF NOT EXISTS prediction_scores (
    bet_id INTEGER PRIMARY KEY,
    market_prediction_id INTEGER,
    city_slug VARCHAR,
    variable VARCHAR,
    target_date DATE,
    horizon_days INTEGER,
    our_prob DOUBLE NOT NULL,
    market_prob DOUBLE,
    outcome_binary INTEGER NOT NULL,
    brier_score DOUBLE NOT NULL,
    log_score DOUBLE NOT NULL,
    market_brier DOUBLE,
    scored_at TIMESTAMP NOT NULL,
    ensemble_prob DOUBLE,
    historical_prob DOUBLE,
    deterministic_prob DOUBLE,
    analog_prob DOUBLE,
    bma_prob DOUBLE,
    ensemble_brier DOUBLE,
    historical_brier DOUBLE,
    deterministic_brier DOUBLE,
    analog_brier DOUBLE,
    bma_brier DOUBLE
);

CREATE TABLE IF NOT EXISTS calibration_bins (
    variable VARCHAR NOT NULL,
    bin_index INTEGER NOT NULL,
    bin_low DOUBLE NOT NULL,
    bin_high DOUBLE NOT NULL,
    n_predictions INTEGER DEFAULT 0,
    n_positive INTEGER DEFAULT 0,
    sum_predicted DOUBLE DEFAULT 0,
    avg_predicted DOUBLE,
    observed_frequency DOUBLE,
    updated_at TIMESTAMP DEFAULT now(),
    PRIMARY KEY (variable, bin_index)
);

CREATE TABLE IF NOT EXISTS calibration_models (
    variable VARCHAR PRIMARY KEY,
    model_type VARCHAR NOT NULL,
    model_blob BLOB,
    n_samples INTEGER NOT NULL,
    brier_before DOUBLE,
    brier_after DOUBLE,
    fitted_at TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS spread_snapshots (
    city_slug VARCHAR NOT NULL,
    target_date DATE NOT NULL,
    variable VARCHAR NOT NULL,
    collected_at TIMESTAMP NOT NULL,
    ensemble_mean DOUBLE,
    ensemble_std DOUBLE,
    ensemble_min DOUBLE,
    ensemble_max DOUBLE,
    n_members INTEGER,
    iqr DOUBLE,
    PRIMARY KEY (city_slug, target_date, variable, collected_at)
);

CREATE TABLE IF NOT EXISTS feedback_reports (
    id INTEGER PRIMARY KEY DEFAULT nextval('feedback_report_seq'),
    report_type VARCHAR NOT NULL,
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,
    report_json VARCHAR NOT NULL,
    generated_at TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS learned_weights (
    weight_group VARCHAR NOT NULL,
    variable VARCHAR NOT NULL,
    weights_json VARCHAR NOT NULL,
    n_samples INTEGER NOT NULL,
    brier_before DOUBLE,
    brier_after DOUBLE,
    fitted_at TIMESTAMP NOT NULL,
    PRIMARY KEY (weight_group, variable)
);

CREATE TABLE IF NOT EXISTS market_price_snapshots (
    market_id VARCHAR NOT NULL,
    condition_id VARCHAR NOT NULL,
    outcome VARCHAR NOT NULL,
    price DOUBLE NOT NULL,
    volume DOUBLE,
    liquidity DOUBLE,
    collected_at TIMESTAMP NOT NULL,
    PRIMARY KEY (condition_id, outcome, collected_at)
);

CREATE TABLE IF NOT EXISTS circuit_breaker_state (
    source VARCHAR PRIMARY KEY,
    consecutive_failures INTEGER DEFAULT 0,
    last_failure_at TIMESTAMP,
    last_success_at TIMESTAMP,
    state VARCHAR DEFAULT 'closed',
    cooldown_until TIMESTAMP
);

CREATE TABLE IF NOT EXISTS line_movement (
    condition_id VARCHAR NOT NULL,
    city_slug VARCHAR,
    variable VARCHAR,
    target_date DATE,
    our_prob DOUBLE,
    market_price DOUBLE,
    edge DOUBLE,
    confidence DOUBLE,
    signal VARCHAR,
    collected_at TIMESTAMP NOT NULL,
    PRIMARY KEY (condition_id, collected_at)
);
"""


def _safe_add_column(db: duckdb.DuckDBPyConnection, table: str, column: str, col_type: str, default: str = "") -> None:
    """Add a column to a table if it doesn't already exist."""
    try:
        default_clause = f" DEFAULT {default}" if default else ""
        db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}{default_clause}")
        logger.debug("Added column %s.%s", table, column)
    except Exception:
        pass  # Column already exists


def get_db() -> duckdb.DuckDBPyConnection:
    """Ritorna una connessione DuckDB. Thread-safe: ogni thread ottiene un cursore separato."""
    global _connection
    if _connection is None:
        db_path = Path(settings.db_path).expanduser()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        _connection = duckdb.connect(str(db_path))
        _init_schema(_connection)
        _seed_cities(_connection)
    # DuckDB: .cursor() crea un cursore thread-safe dallo stesso database
    return _connection.cursor()


def _init_schema(db: duckdb.DuckDBPyConnection) -> None:
    db.execute("CREATE SEQUENCE IF NOT EXISTS market_pred_seq START 1")
    db.execute("CREATE SEQUENCE IF NOT EXISTS collection_log_seq START 1")
    db.execute("CREATE SEQUENCE IF NOT EXISTS bets_seq START 1")
    db.execute("CREATE SEQUENCE IF NOT EXISTS feedback_report_seq START 1")
    for stmt in SCHEMA_SQL.split(";"):
        stmt = stmt.strip()
        if stmt:
            db.execute(stmt)
    # Migrations: add new columns to existing tables
    _safe_add_column(db, "cross_reference_scores", "teleconnection_alignment", "DOUBLE", "50")
    _safe_add_column(db, "cross_reference_scores", "ensemble_regime_score", "DOUBLE", "50")
    _safe_add_column(db, "cross_reference_scores", "extreme_value_score", "DOUBLE", "50")
    _safe_add_column(db, "market_predictions", "source_probs_json", "VARCHAR")
    _safe_add_column(db, "bets", "market_prediction_id", "INTEGER")
    _safe_add_column(db, "bets", "variable", "VARCHAR")
    _safe_add_column(db, "bets", "confidence_scores_json", "VARCHAR")
    _safe_add_column(db, "bets", "cross_ref_json", "VARCHAR")
    logger.info("Database schema initialized")


def _seed_cities(db: duckdb.DuckDBPyConnection) -> None:
    # Try multiple paths: relative to package, or relative to project root
    pkg_dir = Path(__file__).parent
    candidates = [
        pkg_dir.parent.parent.parent.parent / "config" / "cities.json",  # from src/weather_engine/
        Path.cwd().parent / "config" / "cities.json",  # from weather-engine/
        Path.cwd() / "config" / "cities.json",  # from Bot/
    ]
    cities_file = None
    for candidate in candidates:
        if candidate.exists():
            cities_file = candidate
            break
    if cities_file is None:
        cities_file = candidates[0]  # fallback for error message
    if not cities_file.exists():
        logger.warning("cities.json not found at %s", cities_file)
        return

    with open(cities_file) as f:
        cities = json.load(f)

    for slug, info in cities.items():
        db.execute(
            """INSERT OR REPLACE INTO cities (slug, name, latitude, longitude, timezone, country)
            VALUES (?, ?, ?, ?, ?, ?)""",
            [slug, info["name"], info["latitude"], info["longitude"], info["timezone"], info["country"]],
        )
    logger.info("Seeded %d cities", len(cities))


def get_cities(db: duckdb.DuckDBPyConnection | None = None) -> list[dict]:
    db = db or get_db()
    rows = db.execute("SELECT slug, name, latitude, longitude, timezone, country FROM cities").fetchall()
    return [
        {"slug": r[0], "name": r[1], "latitude": r[2], "longitude": r[3], "timezone": r[4], "country": r[5]}
        for r in rows
    ]


def get_city(slug: str, db: duckdb.DuckDBPyConnection | None = None) -> dict | None:
    db = db or get_db()
    row = db.execute(
        "SELECT slug, name, latitude, longitude, timezone, country FROM cities WHERE slug = ?",
        [slug],
    ).fetchone()
    if row is None:
        return None
    return {"slug": row[0], "name": row[1], "latitude": row[2], "longitude": row[3], "timezone": row[4], "country": row[5]}
