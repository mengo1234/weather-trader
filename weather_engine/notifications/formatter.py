from datetime import datetime, timezone

def format_opportunity(rec: dict, meta: dict) -> str:
    """Format a betting opportunity for Telegram."""
    city = meta.get("city", {}).get("name", "?")
    date_str = meta.get("target_date", "?")
    best = rec.get("best_bet", "N/A")
    edge = rec.get("expected_value", 0)
    kelly = rec.get("kelly_fraction", 0)

    lines = [
        f"🎯 *Opportunity Found*",
        f"📍 {city} — {date_str}",
        f"💰 Best bet: *{best}*",
        f"📊 Edge: {edge:+.1%} | Kelly: {kelly:.1%}",
        "",
    ]
    for o in rec.get("outcomes", [])[:5]:
        emoji = "✅" if o.get("edge", 0) > 0.05 else "➖"
        lines.append(f"{emoji} {o['outcome']}: mkt={o['market_price']:.0%} vs ours={o['our_probability']:.0%}")

    return "\n".join(lines)


def format_daily_report(opportunities: list, stats: dict | None = None) -> str:
    """Format daily summary report for Telegram."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"📋 *Daily Weather Trading Report*",
        f"🕐 {now}",
        "",
    ]

    if stats:
        lines.append(f"📊 *Portfolio Stats*")
        lines.append(f"  Total bets: {stats.get('total', 0)}")
        lines.append(f"  Win rate: {stats.get('win_rate', 0):.0%}")
        lines.append(f"  Net P&L: ${stats.get('net_pnl', 0):.2f}")
        lines.append("")

    if opportunities:
        lines.append(f"🎯 *Top Opportunities ({len(opportunities)})*")
        for i, opp in enumerate(opportunities[:5], 1):
            rec = opp.get("recommendation", {})
            meta = opp.get("metadata", {})
            city = meta.get("city", {}).get("name", "?")
            best = rec.get("best_bet", "N/A")
            edge_val = max((o.get("edge", 0) for o in rec.get("outcomes", [])), default=0)
            lines.append(f"  {i}. {city}: {best} ({edge_val:+.1%})")
    else:
        lines.append("No significant opportunities today.")

    return "\n".join(lines)


def format_bet_proposal(opportunity: dict, signal: dict, stake: float, timing_signal: str) -> str:
    """Format a bet proposal for Telegram interactive flow."""
    rec = opportunity.get("recommendation", {})
    meta = opportunity.get("metadata", {})
    city = meta.get("city", {}).get("name", "?") if isinstance(meta.get("city"), dict) else meta.get("city", "?")
    date_str = meta.get("target_date") or rec.get("date", "?")
    best_bet = rec.get("best_bet", "N/A")

    # Find best outcome details
    best_row = None
    for o in rec.get("outcomes", []):
        if o.get("outcome") == best_bet:
            best_row = o
            break

    edge_pct = (best_row.get("edge", 0) * 100) if best_row else 0
    our_prob = (best_row.get("our_probability", 0) * 100) if best_row else 0
    mkt_price = (best_row.get("market_price", 0) * 100) if best_row else 0
    conf_pct = (best_row.get("confidence", 0) * 100) if best_row else 0

    sig_label = signal.get("signal", "?")
    eff_edge = signal.get("effective_edge", 0) * 100

    lines = [
        "PROPOSTA SCOMMESSA",
        f"Citta': {city}",
        f"Data: {date_str}",
        f"Outcome: {best_bet}",
        "",
        f"Edge: {edge_pct:.1f}% | Eff. edge: {eff_edge:.1f}%",
        f"Prob: {our_prob:.1f}% vs Mercato: {mkt_price:.1f}%",
        f"Confidenza: {conf_pct:.0f}%",
        f"Signal: {sig_label}",
        f"Timing: {timing_signal}",
        "",
        f"Stake proposto: ${stake:.2f}",
        "",
        "Rispondi /confirm per confermare (valido 5 min)",
    ]
    return "\n".join(lines)


def format_bet_resolved(bet: dict) -> str:
    """Format a resolved bet notification."""
    status = "✅ WON" if bet.get("status") == "won" else "❌ LOST"
    lines = [
        f"{status} — *Bet Resolved*",
        f"📍 {bet.get('city_slug', '?')} — {bet.get('target_date', '?')}",
        f"🎯 {bet.get('market_question', '?')}",
        f"💵 Stake: ${bet.get('stake', 0):.2f} → P&L: ${bet.get('pnl', 0):+.2f}",
    ]
    return "\n".join(lines)
