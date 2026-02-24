"""P&L tracking: API-backed with local JSON fallback."""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

from weather_trader.api_client import api_get, api_post
from weather_trader.constants import INITIAL_BANKROLL

logger = logging.getLogger("weather_trader.pnl")

PNL_FILE = Path(__file__).parent.parent.parent / "pnl_data.json"


# ---------------------------------------------------------------------------
# Local JSON fallback helpers
# ---------------------------------------------------------------------------

def _load_local() -> dict:
    if PNL_FILE.exists():
        try:
            return json.loads(PNL_FILE.read_text())
        except Exception as e:
            logger.error("Failed to load local P&L data: %s", e)
    return {"bets": [], "bankroll": INITIAL_BANKROLL, "initial_bankroll": INITIAL_BANKROLL}


def _save_local(data: dict):
    try:
        PNL_FILE.write_text(json.dumps(data, indent=2, default=str))
    except Exception as e:
        logger.error("Failed to save local P&L data: %s", e)


def _api_bet_to_local(b: dict) -> dict:
    """Convert backend bet row to local format."""
    return {
        "id": b.get("id", 0),
        "timestamp": str(b.get("timestamp", "")),
        "market": b.get("market_question", ""),
        "outcome": b.get("outcome", ""),
        "stake": b.get("stake", 0) or 0,
        "odds": b.get("odds", 0) or 0,
        "our_prob": b.get("our_prob", 0) or 0,
        "edge": b.get("edge", 0) or 0,
        "confidence": b.get("confidence", 0) or 0,
        "city": b.get("city_slug", ""),
        "target_date": str(b.get("target_date", "")),
        "status": b.get("status", "pending"),
        "pnl": b.get("pnl", 0) or 0,
    }


# ---------------------------------------------------------------------------
# Public API (same interface as before)
# ---------------------------------------------------------------------------

def load_pnl() -> dict:
    """Load P&L data, preferring backend API."""
    result = api_get("/market/bets/list?limit=200")
    if result and "bets" in result:
        bets = [_api_bet_to_local(b) for b in result["bets"]]
        # Compute bankroll from bets
        pending_stakes = sum(b["stake"] for b in bets if b["status"] == "pending")
        resolved_pnl = sum(b["pnl"] for b in bets if b["status"] in ("won", "lost"))
        bankroll = INITIAL_BANKROLL + resolved_pnl - pending_stakes
        return {
            "bets": bets,
            "bankroll": bankroll,
            "initial_bankroll": INITIAL_BANKROLL,
        }
    # Fallback to local JSON
    logger.info("API unavailable, falling back to local pnl_data.json")
    return _load_local()


def save_pnl(data: dict):
    """Save P&L data locally (kept for backward compat, API handles persistence)."""
    _save_local(data)


def record_bet(market_question: str, outcome: str, stake: float,
               odds: float, our_prob: float, edge: float,
               city: str = "", date: str = "", confidence: float = 0):
    """Record a bet via backend API, with local fallback."""
    result = api_post("/market/bets/record", {
        "market_question": market_question,
        "outcome": outcome,
        "stake": stake,
        "odds": odds,
        "our_prob": our_prob,
        "edge": edge,
        "confidence": confidence,
        "city_slug": city,
        "target_date": date,
    })
    if result and result.get("status") == "recorded":
        logger.info("Bet recorded via API: %s %.0f$ on %s", city, stake, outcome)
        return {"id": 0, "status": "recorded"}

    # Fallback to local
    logger.warning("API record failed, saving locally")
    data = _load_local()
    bet = {
        "id": len(data["bets"]) + 1,
        "timestamp": datetime.now().isoformat(),
        "market": market_question,
        "outcome": outcome,
        "stake": stake,
        "odds": odds,
        "our_prob": our_prob,
        "edge": edge,
        "confidence": confidence,
        "city": city,
        "target_date": date,
        "status": "pending",
        "pnl": 0.0,
    }
    data["bets"].append(bet)
    data["bankroll"] -= stake
    _save_local(data)
    return bet


def resolve_bet(bet_id: int, won: bool):
    """Resolve a bet via backend API, with local fallback."""
    result = api_post(f"/market/bets/{bet_id}/resolve", {"won": won})
    if result and "status" in result:
        logger.info("Bet #%d resolved via API: %s (P&L: %.2f)",
                     bet_id, result["status"], result.get("pnl", 0))
        return

    # Fallback to local
    logger.warning("API resolve failed, resolving locally")
    data = _load_local()
    for bet in data["bets"]:
        if bet["id"] == bet_id and bet["status"] == "pending":
            if won:
                payout = bet["stake"] * bet["odds"]
                bet["pnl"] = payout - bet["stake"]
                bet["status"] = "won"
                data["bankroll"] += payout
            else:
                bet["pnl"] = -bet["stake"]
                bet["status"] = "lost"
            break
    _save_local(data)


def get_pnl_stats() -> dict:
    """Compute P&L statistics, preferring backend API."""
    stats = api_get("/market/bets/stats")
    if stats and "total" in stats:
        # Base stats from API
        total = stats["total"]
        pending = stats["pending"]
        won_count = stats["won"]
        lost_count = stats["lost"]
        net_pnl = stats["net_pnl"]
        total_staked = stats["total_staked"]
        win_rate = stats["win_rate"]
        roi = stats.get("roi", 0) or 0

        # Extra fields need bets list
        pnl_data = load_pnl()
        bets = pnl_data.get("bets", [])
        bankroll = pnl_data.get("bankroll", INITIAL_BANKROLL)

        resolved = [b for b in bets if b["status"] in ("won", "lost")]
        pnls = [b["pnl"] for b in resolved]

        today_str = datetime.now().strftime("%Y-%m-%d")
        today_pnl = sum(b["pnl"] for b in resolved if str(b["timestamp"])[:10] == today_str)

        week_ago = (datetime.now() - timedelta(days=7)).isoformat()
        week_pnl = sum(b["pnl"] for b in resolved if str(b["timestamp"]) >= week_ago)

        return {
            "total_bets": total,
            "pending": pending,
            "won": won_count,
            "lost": lost_count,
            "total_pnl": net_pnl,
            "bankroll": bankroll,
            "initial_bankroll": INITIAL_BANKROLL,
            "win_rate": win_rate,
            "avg_edge": sum(b["edge"] for b in bets) / len(bets) if bets else 0,
            "avg_stake": sum(b["stake"] for b in bets) / len(bets) if bets else 0,
            "best_bet": max(pnls) if pnls else 0,
            "worst_bet": min(pnls) if pnls else 0,
            "roi": roi,
            "today_pnl": today_pnl,
            "week_pnl": week_pnl,
        }

    # Fallback: compute from local data
    logger.info("API stats unavailable, computing from local data")
    data = _load_local()
    bets = data["bets"]
    if not bets:
        return {
            "total_bets": 0, "pending": 0, "won": 0, "lost": 0,
            "total_pnl": 0, "bankroll": data["bankroll"],
            "initial_bankroll": data.get("initial_bankroll", INITIAL_BANKROLL),
            "win_rate": 0, "avg_edge": 0, "avg_stake": 0,
            "best_bet": 0, "worst_bet": 0, "roi": 0,
            "today_pnl": 0, "week_pnl": 0,
        }

    resolved = [b for b in bets if b["status"] != "pending"]
    won = [b for b in resolved if b["status"] == "won"]
    lost = [b for b in resolved if b["status"] == "lost"]
    pending_bets = [b for b in bets if b["status"] == "pending"]

    total_pnl = sum(b["pnl"] for b in resolved)
    total_staked = sum(b["stake"] for b in resolved) or 1

    today_str = datetime.now().strftime("%Y-%m-%d")
    today_pnl = sum(b["pnl"] for b in resolved if b["timestamp"][:10] == today_str)

    week_ago = (datetime.now() - timedelta(days=7)).isoformat()
    week_pnl = sum(b["pnl"] for b in resolved if b["timestamp"] >= week_ago)

    pnls = [b["pnl"] for b in resolved]

    return {
        "total_bets": len(bets),
        "pending": len(pending_bets),
        "won": len(won),
        "lost": len(lost),
        "total_pnl": total_pnl,
        "bankroll": data["bankroll"],
        "initial_bankroll": data.get("initial_bankroll", INITIAL_BANKROLL),
        "win_rate": len(won) / len(resolved) if resolved else 0,
        "avg_edge": sum(b["edge"] for b in bets) / len(bets),
        "avg_stake": sum(b["stake"] for b in bets) / len(bets),
        "best_bet": max(pnls) if pnls else 0,
        "worst_bet": min(pnls) if pnls else 0,
        "roi": total_pnl / total_staked,
        "today_pnl": today_pnl,
        "week_pnl": week_pnl,
    }
