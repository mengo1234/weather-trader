import logging
from datetime import date, datetime, timezone

from weather_engine.db import get_db

logger = logging.getLogger(__name__)


def auto_resolve_bets(db=None) -> dict:
    """Auto-resolve pending bets whose target_date has passed by comparing with observations."""
    db = db or get_db()
    today = date.today()

    pending = db.execute(
        """SELECT id, market_question, outcome, stake, odds, city_slug, target_date
        FROM bets WHERE status = 'pending' AND target_date < ?""",
        [today],
    ).fetchall()

    resolved = 0
    won = 0
    lost = 0

    for bet in pending:
        bet_id, question, outcome, stake, odds, city_slug, target_date = bet
        if not city_slug or not target_date:
            continue

        # Get actual observation for the target date
        obs = db.execute(
            """SELECT temperature_2m_max, temperature_2m_min, precipitation_sum,
                    wind_speed_10m_max
            FROM observations WHERE city_slug = ? AND date = ?""",
            [city_slug, target_date],
        ).fetchone()

        if obs is None:
            # Try to get from forecast verification
            continue

        actual_temp_max = obs[0]
        if actual_temp_max is None:
            continue

        # Determine win/loss based on outcome range
        is_win = _check_outcome(outcome, actual_temp_max)

        if is_win is None:
            continue

        status = "won" if is_win else "lost"
        pnl = (stake * (odds - 1)) if is_win else -stake

        db.execute(
            """UPDATE bets SET status = ?, pnl = ?, resolved_at = ?, resolution_source = 'auto'
            WHERE id = ?""",
            [status, pnl, datetime.now(timezone.utc), bet_id],
        )

        resolved += 1
        if is_win:
            won += 1
        else:
            lost += 1

    logger.info("Auto-resolved %d bets: %d won, %d lost", resolved, won, lost)

    if resolved > 0:
        try:
            from weather_engine.analysis.scoring import score_all_unscored
            score_all_unscored(db)
        except Exception as e:
            logger.warning("Scoring failed: %s", e)

    return {"resolved": resolved, "won": won, "lost": lost}


def _check_outcome(outcome: str, actual_value: float) -> bool | None:
    """Check if an outcome matches the actual value.

    Supports formats like:
    - "32-33°F" -> check if actual is between 32 and 33
    - "31°F or below" -> check if actual <= 31
    - "40°F or above" -> check if actual >= 40
    """
    import re

    outcome = outcome.strip()

    # "X°F or below/above"
    m = re.match(r"(\d+(?:\.\d+)?)\s*°?\s*[FfCc]?\s+or\s+(below|above|lower|higher)", outcome, re.IGNORECASE)
    if m:
        val = float(m.group(1))
        direction = m.group(2).lower()
        if direction in ("below", "lower"):
            return actual_value <= val
        else:
            return actual_value >= val

    # "X-Y°F"
    m = re.match(r"(\d+(?:\.\d+)?)\s*[-\u2013]\s*(\d+(?:\.\d+)?)", outcome)
    if m:
        low = float(m.group(1))
        high = float(m.group(2))
        return low <= actual_value <= high

    return None


def get_bet_stats(db=None) -> dict:
    """Get overall betting statistics."""
    db = db or get_db()

    stats = db.execute("""
        SELECT
            COUNT(*) as total,
            COUNT(CASE WHEN status = 'pending' THEN 1 END) as pending,
            COUNT(CASE WHEN status = 'won' THEN 1 END) as won,
            COUNT(CASE WHEN status = 'lost' THEN 1 END) as lost,
            COALESCE(SUM(pnl), 0) as net_pnl,
            COALESCE(SUM(stake), 0) as total_staked
        FROM bets
    """).fetchone()

    total, pending, won_count, lost_count, net_pnl, total_staked = stats
    resolved = won_count + lost_count
    win_rate = won_count / resolved if resolved > 0 else 0
    roi = net_pnl / total_staked if total_staked > 0 else 0

    return {
        "total": total,
        "pending": pending,
        "won": won_count,
        "lost": lost_count,
        "win_rate": round(win_rate, 3),
        "net_pnl": round(net_pnl, 2),
        "total_staked": round(total_staked, 2),
        "roi": round(roi, 4),
    }
