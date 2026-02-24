"""Prediction scoring: Brier, log score, source attribution.

Scores resolved bets against actual outcomes to measure prediction quality.
"""

import json
import logging
import math
from datetime import date, datetime, timezone

logger = logging.getLogger(__name__)

_EPSILON = 1e-6


def score_prediction(our_prob: float, outcome_binary: int, market_prob: float | None = None) -> dict:
    """Compute Brier and log score for a single prediction.

    Args:
        our_prob: Our predicted probability for the outcome.
        outcome_binary: 1 if outcome occurred, 0 if not.
        market_prob: Market-implied probability (optional).

    Returns:
        Dict with brier_score, log_score, market_brier.
    """
    p = max(_EPSILON, min(1 - _EPSILON, our_prob))
    o = outcome_binary

    brier = (p - o) ** 2
    log_score = -(o * math.log(p) + (1 - o) * math.log(1 - p))

    market_brier = None
    if market_prob is not None:
        mp = max(_EPSILON, min(1 - _EPSILON, market_prob))
        market_brier = (mp - o) ** 2

    return {
        "brier_score": brier,
        "log_score": log_score,
        "market_brier": market_brier,
    }


def score_source(source_prob: float | None, outcome_binary: int) -> float | None:
    """Compute Brier score for a single source probability.

    Returns None if source_prob is None.
    """
    if source_prob is None:
        return None
    p = max(_EPSILON, min(1 - _EPSILON, source_prob))
    return (p - outcome_binary) ** 2


def score_resolved_bet(db, bet_id: int) -> dict | None:
    """Score a resolved bet by linking it to its market prediction.

    Computes Brier/log for us, for market, and for each source.
    Inserts result into prediction_scores table.

    Returns score dict or None if bet cannot be scored.
    """
    bet = db.execute(
        """SELECT id, market_question, outcome, stake, odds, our_prob, edge,
                  confidence, city_slug, target_date, status, pnl,
                  market_prediction_id, variable
        FROM bets WHERE id = ?""",
        [bet_id],
    ).fetchone()

    if bet is None:
        return None

    (bid, question, outcome, stake, odds, our_prob, edge,
     confidence, city_slug, target_date, status, pnl,
     market_prediction_id, variable) = bet

    if status not in ("won", "lost"):
        return None

    outcome_binary = 1 if status == "won" else 0

    # Get market_prob from odds if available
    market_prob = 1.0 / odds if odds and odds > 0 else None

    # Source probabilities from market_prediction
    source_probs = {}
    if market_prediction_id is not None:
        mp_row = db.execute(
            "SELECT source_probs_json, outcomes_json FROM market_predictions WHERE id = ?",
            [market_prediction_id],
        ).fetchone()
        if mp_row:
            # Try to extract per-source probs for the specific outcome
            if mp_row[0]:
                try:
                    all_source_probs = json.loads(mp_row[0])
                    for sp in all_source_probs:
                        if sp.get("outcome") == outcome:
                            source_probs = sp
                            break
                except (json.JSONDecodeError, TypeError):
                    pass

            # Try to get market_prob from outcomes_json if not from odds
            if market_prob is None and mp_row[1]:
                try:
                    outcomes_data = json.loads(mp_row[1])
                    for od in outcomes_data:
                        if od.get("outcome") == outcome:
                            market_prob = od.get("market_price")
                            break
                except (json.JSONDecodeError, TypeError):
                    pass

    # Compute horizon_days
    horizon_days = None
    if target_date:
        try:
            td = target_date if isinstance(target_date, date) else date.fromisoformat(str(target_date))
            # Approximate: scored_at - bet creation would be more accurate,
            # but target_date - today at scoring time is good enough
            horizon_days = max(0, (td - date.today()).days)
        except (ValueError, TypeError):
            pass

    # Core scores
    scores = score_prediction(our_prob, outcome_binary, market_prob)

    # Per-source scores
    ensemble_prob = source_probs.get("ensemble_prob")
    historical_prob = source_probs.get("historical_prob")
    deterministic_prob = source_probs.get("deterministic_prob")
    analog_prob = source_probs.get("analog_prob")
    bma_prob = source_probs.get("bma_prob")

    ensemble_brier = score_source(ensemble_prob, outcome_binary)
    historical_brier = score_source(historical_prob, outcome_binary)
    deterministic_brier = score_source(deterministic_prob, outcome_binary)
    analog_brier = score_source(analog_prob, outcome_binary)
    bma_brier = score_source(bma_prob, outcome_binary)

    now = datetime.now(timezone.utc)

    # Insert into prediction_scores
    try:
        db.execute(
            """INSERT INTO prediction_scores
            (bet_id, market_prediction_id, city_slug, variable, target_date,
             horizon_days, our_prob, market_prob, outcome_binary,
             brier_score, log_score, market_brier, scored_at,
             ensemble_prob, historical_prob, deterministic_prob, analog_prob, bma_prob,
             ensemble_brier, historical_brier, deterministic_brier, analog_brier, bma_brier)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                bid, market_prediction_id, city_slug, variable, target_date,
                horizon_days, our_prob, market_prob, outcome_binary,
                scores["brier_score"], scores["log_score"], scores["market_brier"], now,
                ensemble_prob, historical_prob, deterministic_prob, analog_prob, bma_prob,
                ensemble_brier, historical_brier, deterministic_brier, analog_brier, bma_brier,
            ],
        )
    except Exception as e:
        logger.warning("Failed to insert prediction_score for bet %d: %s", bid, e)
        return None

    result = {
        "bet_id": bid,
        "outcome_binary": outcome_binary,
        **scores,
        "ensemble_brier": ensemble_brier,
        "historical_brier": historical_brier,
        "deterministic_brier": deterministic_brier,
        "analog_brier": analog_brier,
        "bma_brier": bma_brier,
    }
    logger.debug("Scored bet %d: brier=%.4f, log=%.4f", bid, scores["brier_score"], scores["log_score"])
    return result


def score_all_unscored(db) -> dict:
    """Score all resolved bets that haven't been scored yet.

    Returns:
        Dict with scored, skipped, errors counts.
    """
    rows = db.execute(
        """SELECT b.id FROM bets b
        WHERE b.status IN ('won', 'lost')
        AND b.id NOT IN (SELECT bet_id FROM prediction_scores)"""
    ).fetchall()

    scored = 0
    skipped = 0
    errors = 0

    for (bet_id,) in rows:
        try:
            result = score_resolved_bet(db, bet_id)
            if result is not None:
                scored += 1
            else:
                skipped += 1
        except Exception as e:
            logger.warning("Error scoring bet %d: %s", bet_id, e)
            errors += 1

    logger.info("Scoring complete: %d scored, %d skipped, %d errors", scored, skipped, errors)
    return {"scored": scored, "skipped": skipped, "errors": errors}


def get_aggregate_scores(db, city_slug: str | None = None, variable: str | None = None, days_back: int = 90) -> dict:
    """Get aggregate Brier/log scores with breakdowns.

    Args:
        db: Database connection.
        city_slug: Filter by city (optional).
        variable: Filter by variable (optional).
        days_back: How many days to look back.

    Returns:
        Dict with overall scores, breakdowns, and BSS vs market.
    """
    from datetime import timedelta
    cutoff = date.today() - timedelta(days=days_back)

    conditions = ["scored_at >= ?"]
    params: list = [cutoff]

    if city_slug:
        conditions.append("city_slug = ?")
        params.append(city_slug)
    if variable:
        conditions.append("variable = ?")
        params.append(variable)

    where = " AND ".join(conditions)

    # Overall scores
    row = db.execute(
        f"""SELECT
            COUNT(*) as n,
            AVG(brier_score) as avg_brier,
            AVG(log_score) as avg_log,
            AVG(market_brier) as avg_market_brier,
            AVG(outcome_binary) as win_rate
        FROM prediction_scores WHERE {where}""",
        params,
    ).fetchone()

    n, avg_brier, avg_log, avg_market_brier, win_rate = row

    if n == 0:
        return {"n": 0, "avg_brier": None, "avg_log": None, "bss": None}

    # Brier Skill Score vs market
    bss = None
    if avg_market_brier and avg_market_brier > 0:
        bss = 1 - avg_brier / avg_market_brier

    # Breakdown by city
    city_rows = db.execute(
        f"""SELECT city_slug, COUNT(*), AVG(brier_score), AVG(log_score)
        FROM prediction_scores WHERE {where}
        GROUP BY city_slug ORDER BY AVG(brier_score) ASC""",
        params,
    ).fetchall()

    # Breakdown by variable
    var_rows = db.execute(
        f"""SELECT variable, COUNT(*), AVG(brier_score), AVG(log_score)
        FROM prediction_scores WHERE {where}
        GROUP BY variable ORDER BY AVG(brier_score) ASC""",
        params,
    ).fetchall()

    # Breakdown by horizon
    horizon_rows = db.execute(
        f"""SELECT horizon_days, COUNT(*), AVG(brier_score), AVG(log_score)
        FROM prediction_scores WHERE {where} AND horizon_days IS NOT NULL
        GROUP BY horizon_days ORDER BY horizon_days ASC""",
        params,
    ).fetchall()

    return {
        "n": n,
        "avg_brier": round(avg_brier, 4) if avg_brier is not None else None,
        "avg_log": round(avg_log, 4) if avg_log is not None else None,
        "avg_market_brier": round(avg_market_brier, 4) if avg_market_brier is not None else None,
        "bss": round(bss, 4) if bss is not None else None,
        "win_rate": round(win_rate, 3) if win_rate is not None else None,
        "by_city": [
            {"city_slug": r[0], "n": r[1], "avg_brier": round(r[2], 4), "avg_log": round(r[3], 4)}
            for r in city_rows
        ],
        "by_variable": [
            {"variable": r[0], "n": r[1], "avg_brier": round(r[2], 4), "avg_log": round(r[3], 4)}
            for r in var_rows
        ],
        "by_horizon": [
            {"horizon_days": r[0], "n": r[1], "avg_brier": round(r[2], 4), "avg_log": round(r[3], 4)}
            for r in horizon_rows
        ],
    }


def get_source_attribution(db, days_back: int = 90) -> dict:
    """Get per-source Brier scores to identify which source is most accurate.

    Returns:
        Dict with avg brier per source and ranking.
    """
    from datetime import timedelta
    cutoff = date.today() - timedelta(days=days_back)

    row = db.execute(
        """SELECT
            COUNT(*) as n,
            AVG(ensemble_brier) as ensemble,
            AVG(historical_brier) as historical,
            AVG(deterministic_brier) as deterministic,
            AVG(analog_brier) as analog,
            AVG(bma_brier) as bma
        FROM prediction_scores
        WHERE scored_at >= ?""",
        [cutoff],
    ).fetchone()

    n = row[0]
    if n == 0:
        return {"n": 0, "sources": {}, "ranking": []}

    sources = {}
    source_names = ["ensemble", "historical", "deterministic", "analog", "bma"]
    for i, name in enumerate(source_names):
        val = row[i + 1]
        if val is not None:
            sources[name] = round(val, 4)

    # Ranking: lower brier = better
    ranking = sorted(sources.items(), key=lambda x: x[1])

    return {
        "n": n,
        "sources": sources,
        "ranking": [{"source": name, "avg_brier": score} for name, score in ranking],
    }
