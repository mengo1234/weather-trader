"""Circuit breaker pattern for data collectors.

States: closed (ok) -> open (blocked) -> half_open (probing) -> closed.
After FAILURE_THRESHOLD consecutive failures, the circuit opens.
After COOLDOWN_SECONDS, it transitions to half_open (allows one probe request).
On success in half_open, it closes again. On failure, it reopens.
"""

import logging
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

FAILURE_THRESHOLD = 5
COOLDOWN_SECONDS = 300  # 5 minutes
HALF_OPEN_AFTER = 600  # 10 minutes


def _utcnow() -> datetime:
    """Return current UTC time as a naive datetime (for DuckDB TIMESTAMP columns)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _ensure_row(db, source: str) -> None:
    """Ensure a row exists for this source."""
    existing = db.execute(
        "SELECT source FROM circuit_breaker_state WHERE source = ?", [source]
    ).fetchone()
    if not existing:
        db.execute(
            """INSERT INTO circuit_breaker_state (source, consecutive_failures, state)
            VALUES (?, 0, 'closed')""",
            [source],
        )


def record_success(db, source: str) -> None:
    """Record a successful request. Resets failure count and closes circuit."""
    _ensure_row(db, source)
    now = _utcnow()
    db.execute(
        """UPDATE circuit_breaker_state
        SET consecutive_failures = 0, state = 'closed', last_success_at = ?
        WHERE source = ?""",
        [now, source],
    )


def record_failure(db, source: str) -> None:
    """Record a failed request. Opens circuit after FAILURE_THRESHOLD consecutive failures."""
    _ensure_row(db, source)
    now = _utcnow()

    row = db.execute(
        "SELECT consecutive_failures, state FROM circuit_breaker_state WHERE source = ?",
        [source],
    ).fetchone()

    failures = (row[0] or 0) + 1
    current_state = row[1] or "closed"

    if failures >= FAILURE_THRESHOLD:
        new_state = "open"
        cooldown = now + timedelta(seconds=COOLDOWN_SECONDS)
        db.execute(
            """UPDATE circuit_breaker_state
            SET consecutive_failures = ?, state = ?, last_failure_at = ?, cooldown_until = ?
            WHERE source = ?""",
            [failures, new_state, now, cooldown, source],
        )
        logger.warning("Circuit breaker OPEN for %s after %d consecutive failures", source, failures)
    elif current_state == "half_open":
        # Failed during probe — reopen
        cooldown = now + timedelta(seconds=COOLDOWN_SECONDS)
        db.execute(
            """UPDATE circuit_breaker_state
            SET consecutive_failures = ?, state = 'open', last_failure_at = ?, cooldown_until = ?
            WHERE source = ?""",
            [failures, now, cooldown, source],
        )
        logger.warning("Circuit breaker re-OPENED for %s (half_open probe failed)", source)
    else:
        db.execute(
            """UPDATE circuit_breaker_state
            SET consecutive_failures = ?, last_failure_at = ?
            WHERE source = ?""",
            [failures, now, source],
        )


def is_available(db, source: str) -> bool:
    """Check if requests should be allowed for this source."""
    _ensure_row(db, source)
    row = db.execute(
        "SELECT state, cooldown_until FROM circuit_breaker_state WHERE source = ?",
        [source],
    ).fetchone()

    state = row[0] or "closed"
    cooldown_until = row[1]

    if state == "closed":
        return True

    if state == "open":
        now = _utcnow()
        if cooldown_until is not None:
            if now >= cooldown_until:
                # Check if enough time has passed for half-open probe
                half_open_time = cooldown_until + timedelta(seconds=HALF_OPEN_AFTER - COOLDOWN_SECONDS)
                if now >= half_open_time:
                    db.execute(
                        "UPDATE circuit_breaker_state SET state = 'half_open' WHERE source = ?",
                        [source],
                    )
                    logger.info("Circuit breaker HALF_OPEN for %s (probing)", source)
                    return True
        return False

    if state == "half_open":
        return True

    return False


def get_all_states(db) -> list[dict]:
    """Get all circuit breaker states for monitoring."""
    rows = db.execute(
        """SELECT source, consecutive_failures, last_failure_at, last_success_at, state, cooldown_until
        FROM circuit_breaker_state
        ORDER BY source"""
    ).fetchall()

    return [
        {
            "source": r[0],
            "consecutive_failures": r[1],
            "last_failure_at": str(r[2]) if r[2] else None,
            "last_success_at": str(r[3]) if r[3] else None,
            "state": r[4],
            "cooldown_until": str(r[5]) if r[5] else None,
        }
        for r in rows
    ]
