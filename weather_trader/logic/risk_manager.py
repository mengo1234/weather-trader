"""Portfolio-level risk management.

Analyzes current bets to enforce exposure limits, concentration limits,
drawdown monitoring, and loss streak detection.
"""

import logging

from weather_trader.constants import INITIAL_BANKROLL

logger = logging.getLogger("weather_trader.risk_manager")

# Risk limits
MAX_EXPOSURE_PCT = 0.50      # Max 50% of bankroll exposed
MAX_CITY_PCT = 0.15          # Max 15% of bankroll on a single city
MAX_DRAWDOWN = 0.25          # Block new bets if drawdown > 25%
MAX_LOSS_STREAK = 5          # Reduce Kelly if 5+ consecutive losses


def check_portfolio_risk(bets: list[dict], bankroll: float) -> dict:
    """Analyze portfolio risk and return limits.

    Args:
        bets: All bets (pending + resolved)
        bankroll: Current bankroll

    Returns:
        Dict with exposure metrics, warnings, and can_bet flag.
    """
    pending = [b for b in bets if b.get("status") == "pending"]
    resolved = [b for b in bets if b.get("status") in ("won", "lost")]

    # --- Exposure ---
    exposure = sum(b.get("stake", 0) for b in pending)
    exposure_pct = exposure / bankroll if bankroll > 0 else 1.0

    # --- City concentration ---
    city_exposure = {}
    for b in pending:
        city = b.get("city", "unknown") or "unknown"
        city_exposure[city] = city_exposure.get(city, 0) + b.get("stake", 0)
    max_city_pct = max(city_exposure.values(), default=0) / bankroll if bankroll > 0 else 0
    max_city_name = max(city_exposure, key=city_exposure.get) if city_exposure else ""

    # --- Drawdown ---
    running = INITIAL_BANKROLL
    peak = INITIAL_BANKROLL
    max_dd = 0.0
    current_dd = 0.0
    for b in sorted(resolved, key=lambda x: str(x.get("timestamp", ""))):
        running += b.get("pnl", 0)
        peak = max(peak, running)
        dd = (peak - running) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)
    # Current drawdown from current bankroll
    if peak > 0:
        current_dd = (peak - bankroll) / peak if bankroll < peak else 0.0

    # --- Loss streak ---
    recent = sorted(resolved, key=lambda x: str(x.get("timestamp", "")), reverse=True)[:20]
    loss_streak = 0
    for b in recent:
        if b.get("status") == "lost":
            loss_streak += 1
        else:
            break

    # --- Win streak (for display) ---
    win_streak = 0
    for b in recent:
        if b.get("status") == "won":
            win_streak += 1
        else:
            break

    # --- Kelly reduction factor ---
    kelly_reduction = 1.0
    if loss_streak >= MAX_LOSS_STREAK:
        kelly_reduction = 0.5  # Half Kelly during loss streaks

    # --- Can bet? ---
    can_bet = (
        exposure_pct < MAX_EXPOSURE_PCT
        and max_dd < MAX_DRAWDOWN
        and loss_streak < MAX_LOSS_STREAK
    )

    warnings = _build_warnings(
        exposure_pct, max_city_pct, max_city_name,
        max_dd, current_dd, loss_streak, bankroll,
    )

    return {
        "exposure": exposure,
        "exposure_pct": exposure_pct,
        "city_exposure": city_exposure,
        "max_city_pct": max_city_pct,
        "max_city_name": max_city_name,
        "max_drawdown": max_dd,
        "current_drawdown": current_dd,
        "loss_streak": loss_streak,
        "win_streak": win_streak,
        "kelly_reduction": kelly_reduction,
        "can_bet": can_bet,
        "warnings": warnings,
        "n_pending": len(pending),
        "n_resolved": len(resolved),
    }


def _build_warnings(
    exposure_pct: float,
    max_city_pct: float,
    max_city_name: str,
    max_dd: float,
    current_dd: float,
    loss_streak: int,
    bankroll: float,
) -> list[dict]:
    """Build list of risk warnings with severity levels."""
    warnings = []

    # Exposure warnings
    if exposure_pct >= MAX_EXPOSURE_PCT:
        warnings.append({
            "level": "critical",
            "message": f"Esposizione massima raggiunta ({exposure_pct:.0%} >= {MAX_EXPOSURE_PCT:.0%})",
            "icon": "block",
        })
    elif exposure_pct >= MAX_EXPOSURE_PCT * 0.8:
        warnings.append({
            "level": "warning",
            "message": f"Esposizione alta ({exposure_pct:.0%}), vicino al limite {MAX_EXPOSURE_PCT:.0%}",
            "icon": "warning_amber",
        })

    # City concentration
    if max_city_pct >= MAX_CITY_PCT:
        warnings.append({
            "level": "warning",
            "message": f"Troppo concentrato su {max_city_name} ({max_city_pct:.0%} >= {MAX_CITY_PCT:.0%})",
            "icon": "location_city",
        })

    # Drawdown
    if max_dd >= MAX_DRAWDOWN:
        warnings.append({
            "level": "critical",
            "message": f"Max drawdown {max_dd:.0%} raggiunto — scommesse bloccate fino a recovery",
            "icon": "trending_down",
        })
    elif current_dd >= MAX_DRAWDOWN * 0.7:
        warnings.append({
            "level": "warning",
            "message": f"Drawdown corrente {current_dd:.0%}, vicino al limite {MAX_DRAWDOWN:.0%}",
            "icon": "trending_down",
        })

    # Loss streak
    if loss_streak >= MAX_LOSS_STREAK:
        warnings.append({
            "level": "critical",
            "message": f"{loss_streak} perdite consecutive — Kelly ridotto a metà",
            "icon": "sentiment_very_dissatisfied",
        })
    elif loss_streak >= 3:
        warnings.append({
            "level": "warning",
            "message": f"{loss_streak} perdite consecutive — cautela",
            "icon": "sentiment_dissatisfied",
        })

    # Low bankroll
    if bankroll < INITIAL_BANKROLL * 0.3:
        warnings.append({
            "level": "critical",
            "message": f"Bankroll molto basso (${bankroll:.0f}), rischio rovina",
            "icon": "account_balance",
        })

    return warnings
