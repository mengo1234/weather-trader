"""Verifica automatica previsioni vs osservazioni reali.

Confronta le previsioni giornaliere con le osservazioni per popolare
la tabella forecast_verification e calcolare bias correction.
"""
import logging
from datetime import date, timedelta

logger = logging.getLogger(__name__)

VERIFY_VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "shortwave_radiation_sum",
    "pressure_msl_mean",
]


def run_verification(db, lookback_days: int = 7) -> dict:
    """Verifica le previsioni degli ultimi N giorni contro le osservazioni.

    Per ogni giorno passato con osservazioni, confronta con la previsione
    più recente che era disponibile prima di quel giorno.
    """
    today = date.today()
    start = today - timedelta(days=lookback_days)
    total_verified = 0
    total_new = 0

    cities = db.execute("SELECT slug FROM cities").fetchall()

    for (city_slug,) in cities:
        for variable in VERIFY_VARIABLES:
            n_new = _verify_city_variable(db, city_slug, variable, start, today)
            total_new += n_new
            total_verified += 1

    logger.info("Verification complete: %d new records from %d checks", total_new, total_verified)
    return {"new_records": total_new, "checks": total_verified}


def _verify_city_variable(
    db, city_slug: str, variable: str, start: date, end: date
) -> int:
    """Verifica una singola variabile per una città."""
    # Trova giorni con osservazioni ma senza verifica
    obs_rows = db.execute(
        f"""SELECT o.date, o.{variable}
        FROM observations o
        WHERE o.city_slug = ? AND o.date BETWEEN ? AND ?
        AND o.{variable} IS NOT NULL
        AND NOT EXISTS (
            SELECT 1 FROM forecast_verification fv
            WHERE fv.city_slug = o.city_slug
            AND fv.target_date = o.date
            AND fv.variable = ?
        )
        ORDER BY o.date""",
        [city_slug, start, end, variable],
    ).fetchall()

    if not obs_rows:
        return 0

    n_inserted = 0
    for obs_date, obs_value in obs_rows:
        # Trova la previsione più recente fatta PRIMA della data target
        fc_row = db.execute(
            f"""SELECT model_run, {variable}
            FROM forecasts_daily
            WHERE city_slug = ? AND date = ? AND {variable} IS NOT NULL
            AND model_run < ?::TIMESTAMP
            ORDER BY model_run DESC LIMIT 1""",
            [city_slug, obs_date, obs_date],
        ).fetchone()

        if fc_row is None:
            continue

        fc_run, fc_value = fc_row
        horizon_hours = int((obs_date - fc_run.date()).total_seconds() / 3600) if hasattr(fc_run, 'date') else 24
        error = float(fc_value - obs_value)

        db.execute(
            """INSERT OR REPLACE INTO forecast_verification
            (city_slug, target_date, forecast_date, horizon_hours, variable,
             forecast_value, observed_value, error, abs_error, squared_error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                city_slug, obs_date, fc_run, horizon_hours, variable,
                float(fc_value), float(obs_value), error, abs(error), error ** 2,
            ],
        )
        n_inserted += 1

    return n_inserted


def check_calibration(db, city_slug: str, variable: str = "temperature_2m_max") -> dict:
    """Multi-source verification: compare ensemble spread vs actual error.

    If ensemble spread was tight and forecast was accurate → well calibrated.
    If ensemble spread was tight but forecast was wrong → overconfident.
    If ensemble spread was wide and forecast was wrong → correctly uncertain.
    """
    try:
        # Get recent verifications with their corresponding ensemble spread
        rows = db.execute(
            """SELECT fv.abs_error, fv.horizon_hours,
                      em_stats.ens_std
            FROM forecast_verification fv
            LEFT JOIN (
                SELECT city_slug, time::DATE as target_date,
                       STDDEV(temperature_2m) as ens_std
                FROM ensemble_members
                WHERE temperature_2m IS NOT NULL
                GROUP BY city_slug, time::DATE
            ) em_stats ON em_stats.city_slug = fv.city_slug
                       AND em_stats.target_date = fv.target_date
            WHERE fv.city_slug = ? AND fv.variable = ?
            AND fv.horizon_hours <= 72
            ORDER BY fv.target_date DESC LIMIT 50""",
            [city_slug, variable],
        ).fetchall()

        if len(rows) < 5:
            return {"calibration": "insufficient_data", "n_samples": len(rows)}

        tight_correct = 0   # low spread, low error
        tight_wrong = 0     # low spread, high error (overconfident!)
        wide_correct = 0    # high spread, low error
        wide_wrong = 0      # high spread, high error

        for abs_error, _, ens_std in rows:
            if ens_std is None:
                continue
            tight = ens_std < 3.0  # < 3°F spread
            correct = abs_error < 3.0  # < 3°F error

            if tight and correct:
                tight_correct += 1
            elif tight and not correct:
                tight_wrong += 1
            elif not tight and correct:
                wide_correct += 1
            else:
                wide_wrong += 1

        total = tight_correct + tight_wrong + wide_correct + wide_wrong
        if total == 0:
            return {"calibration": "insufficient_data", "n_samples": 0}

        overconfidence_rate = tight_wrong / max(1, tight_correct + tight_wrong)

        if overconfidence_rate < 0.2:
            calibration = "well_calibrated"
        elif overconfidence_rate < 0.4:
            calibration = "slightly_overconfident"
        else:
            calibration = "overconfident"

        return {
            "calibration": calibration,
            "overconfidence_rate": round(overconfidence_rate, 3),
            "tight_correct": tight_correct,
            "tight_wrong": tight_wrong,
            "wide_correct": wide_correct,
            "wide_wrong": wide_wrong,
            "n_samples": total,
            "penalty_suggestion": round(overconfidence_rate * 10, 1),  # 0-10 point penalty
        }

    except Exception as e:
        logger.warning("Calibration check failed for %s: %s", city_slug, e)
        return {"calibration": "error", "n_samples": 0}


def get_verification_summary(db, city_slug: str | None = None) -> list[dict]:
    """Riassunto delle metriche di verifica per variabile."""
    where = "WHERE city_slug = ?" if city_slug else ""
    params = [city_slug] if city_slug else []

    rows = db.execute(
        f"""SELECT
            variable,
            COUNT(*) as n,
            AVG(error) as bias,
            AVG(abs_error) as mae,
            SQRT(AVG(squared_error)) as rmse,
            MIN(target_date) as from_date,
            MAX(target_date) as to_date
        FROM forecast_verification
        {where}
        GROUP BY variable
        ORDER BY variable""",
        params,
    ).fetchall()

    return [
        {
            "variable": r[0],
            "n_samples": r[1],
            "bias": round(r[2], 3),
            "mae": round(r[3], 3),
            "rmse": round(r[4], 3),
            "from_date": str(r[5]),
            "to_date": str(r[6]),
        }
        for r in rows
    ]


def compute_horizon_profiles(db, city_slug: str | None = None, variables: list[str] | None = None) -> dict:
    """Compute forecast accuracy profiles by horizon bucket (day 1, 2, ..., 14).

    Groups forecast_verification by horizon, computes MAE/RMSE/bias per bucket,
    and saves to horizon_profiles table.
    """
    import math

    variables = variables or VERIFY_VARIABLES
    where_parts = []
    params = []

    if city_slug:
        where_parts.append("city_slug = ?")
        params.append(city_slug)

    where_clause = "WHERE " + " AND ".join(where_parts) if where_parts else ""

    total = 0

    # Get all city slugs
    if city_slug:
        city_slugs = [city_slug]
    else:
        city_slugs = [r[0] for r in db.execute("SELECT slug FROM cities").fetchall()]

    for slug in city_slugs:
        for variable in variables:
            rows = db.execute(
                """SELECT
                    CAST(CEIL(horizon_hours / 24.0) AS INTEGER) as horizon_days,
                    COUNT(*) as n,
                    AVG(error) as bias,
                    AVG(abs_error) as mae,
                    SQRT(AVG(squared_error)) as rmse
                FROM forecast_verification
                WHERE city_slug = ? AND variable = ?
                GROUP BY horizon_days
                ORDER BY horizon_days""",
                [slug, variable],
            ).fetchall()

            for row in rows:
                horizon_d = int(row[0])
                if horizon_d < 1 or horizon_d > 14:
                    continue

                db.execute(
                    """INSERT OR REPLACE INTO horizon_profiles
                    (city_slug, variable, horizon_days, n_samples, mae, rmse, bias, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, now())""",
                    [slug, variable, horizon_d, row[1], round(row[3], 3), round(row[4], 3), round(row[2], 3)],
                )
                total += 1

    logger.info("Horizon profiles computed: %d profiles", total)
    return {"profiles": total}
