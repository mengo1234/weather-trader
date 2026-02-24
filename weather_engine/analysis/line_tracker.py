"""Line movement tracking: record market price snapshots over time.

Tracks our probability vs market price for each condition, enabling
timing signals (SCOMMETTI_ORA, ASPETTA, SBRIGATI, TROPPO_TARDI).
"""

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def record_line_snapshot(
    db,
    condition_id: str,
    city_slug: str,
    variable: str,
    target_date,
    our_prob: float,
    market_price: float,
    edge: float,
    confidence: float,
    signal: str,
) -> None:
    """Record a line movement snapshot for a condition."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    try:
        db.execute(
            """INSERT INTO line_movement
            (condition_id, city_slug, variable, target_date,
             our_prob, market_price, edge, confidence, signal, collected_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [condition_id, city_slug, variable, target_date,
             our_prob, market_price, edge, confidence, signal, now],
        )
    except Exception as e:
        logger.warning("Failed to record line snapshot for %s: %s", condition_id, e)


def get_line_history(db, condition_id: str) -> list[dict]:
    """Get all line movement snapshots for a condition, ordered by time."""
    rows = db.execute(
        """SELECT condition_id, city_slug, variable, target_date,
                  our_prob, market_price, edge, confidence, signal, collected_at
        FROM line_movement
        WHERE condition_id = ?
        ORDER BY collected_at ASC""",
        [condition_id],
    ).fetchall()

    cols = ["condition_id", "city_slug", "variable", "target_date",
            "our_prob", "market_price", "edge", "confidence", "signal", "collected_at"]
    return [dict(zip(cols, r)) for r in rows]


def analyze_line_movement(db, condition_id: str) -> dict:
    """Analyze line movement for a condition to produce a timing signal.

    Requires at least 2 snapshots.

    Returns:
        Dict with edge_trend, edge_velocity, market_direction, timing_signal,
        n_snapshots, hours_tracked.
    """
    history = get_line_history(db, condition_id)

    if len(history) < 2:
        return {
            "edge_trend": "insufficient_data",
            "edge_velocity": 0.0,
            "market_direction": "unknown",
            "timing_signal": "SCOMMETTI_ORA",
            "n_snapshots": len(history),
            "hours_tracked": 0.0,
        }

    first = history[0]
    last = history[-1]

    first_edge = first["edge"] or 0.0
    last_edge = last["edge"] or 0.0
    first_price = first["market_price"] or 0.0
    last_price = last["market_price"] or 0.0
    last_confidence = last["confidence"] or 0.0

    # Time span
    first_time = first["collected_at"]
    last_time = last["collected_at"]
    if hasattr(first_time, "timestamp") and hasattr(last_time, "timestamp"):
        delta_hours = (last_time.timestamp() - first_time.timestamp()) / 3600
    else:
        delta_hours = 0.0
    delta_hours = max(delta_hours, 0.001)  # avoid division by zero

    # Edge trend
    edge_delta = last_edge - first_edge
    if edge_delta > 0.01:
        edge_trend = "increasing"
    elif edge_delta < -0.01:
        edge_trend = "decreasing"
    else:
        edge_trend = "stable"

    # Edge velocity (change per hour)
    edge_velocity = edge_delta / delta_hours

    # Market direction (relative to us)
    price_delta = last_price - first_price
    if price_delta < -0.01:
        # Market price dropped → moving toward us (our edge increases)
        market_direction = "toward_us"
    elif price_delta > 0.01:
        # Market price rose → moving away from us
        market_direction = "away_from_us"
    else:
        market_direction = "stable"

    # Timing signal
    if last_edge < 0.02 and edge_trend == "decreasing":
        timing_signal = "TROPPO_TARDI"
    elif edge_trend == "increasing" and edge_velocity > 0.01:
        # Edge growing fast — market moving toward us, wait for peak
        timing_signal = "ASPETTA"
    elif edge_trend == "decreasing" and last_edge >= 0.02:
        # Edge shrinking — bet now or miss it
        timing_signal = "SBRIGATI"
    elif (edge_trend in ("stable", "increasing")) and last_confidence >= 50:
        timing_signal = "SCOMMETTI_ORA"
    else:
        timing_signal = "SCOMMETTI_ORA"

    return {
        "edge_trend": edge_trend,
        "edge_velocity": round(edge_velocity, 6),
        "market_direction": market_direction,
        "timing_signal": timing_signal,
        "n_snapshots": len(history),
        "hours_tracked": round(delta_hours, 2),
    }
