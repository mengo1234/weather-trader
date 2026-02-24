"""Feedback report generation: post-mortem analysis of prediction quality.

Generates periodic reports combining scoring, calibration, source attribution,
and trend analysis.
"""

import json
import logging
from datetime import date, datetime, timedelta, timezone

logger = logging.getLogger(__name__)


def generate_feedback_report(db, period_start: date, period_end: date, report_type: str = "weekly") -> dict:
    """Generate a comprehensive feedback report for the given period.

    Sections:
      1. Performance: Brier, log score, win rate, ROI, BSS vs market
      2. Source attribution: ranking of 5 sources by Brier
      3. Confidence calibration: win rate per confidence bin
      4. Breakdown by city, variable, horizon
      5. Trend: comparison with previous period
      6. Calibration: model status

    Returns:
        Complete report dict (also saved to feedback_reports table).
    """
    # 1. Performance
    perf_row = db.execute(
        """SELECT
            COUNT(*) as n,
            AVG(brier_score) as avg_brier,
            AVG(log_score) as avg_log,
            AVG(market_brier) as avg_market_brier,
            AVG(outcome_binary) as win_rate
        FROM prediction_scores
        WHERE target_date >= ? AND target_date <= ?""",
        [period_start, period_end],
    ).fetchone()

    n_scored = perf_row[0]
    performance = {
        "n_scored": n_scored,
        "avg_brier": round(perf_row[1], 4) if perf_row[1] is not None else None,
        "avg_log": round(perf_row[2], 4) if perf_row[2] is not None else None,
        "avg_market_brier": round(perf_row[3], 4) if perf_row[3] is not None else None,
        "win_rate": round(perf_row[4], 3) if perf_row[4] is not None else None,
    }

    # BSS
    if performance["avg_brier"] is not None and performance["avg_market_brier"] and performance["avg_market_brier"] > 0:
        performance["bss"] = round(1 - performance["avg_brier"] / performance["avg_market_brier"], 4)
    else:
        performance["bss"] = None

    # ROI from bets table
    roi_row = db.execute(
        """SELECT COALESCE(SUM(pnl), 0), COALESCE(SUM(stake), 0)
        FROM bets WHERE status IN ('won', 'lost')
        AND target_date >= ? AND target_date <= ?""",
        [period_start, period_end],
    ).fetchone()
    net_pnl, total_staked = roi_row
    performance["net_pnl"] = round(net_pnl, 2)
    performance["roi"] = round(net_pnl / total_staked, 4) if total_staked > 0 else None

    # 2. Source attribution
    src_row = db.execute(
        """SELECT
            AVG(ensemble_brier) as ensemble,
            AVG(historical_brier) as historical,
            AVG(deterministic_brier) as deterministic,
            AVG(analog_brier) as analog,
            AVG(bma_brier) as bma
        FROM prediction_scores
        WHERE target_date >= ? AND target_date <= ?""",
        [period_start, period_end],
    ).fetchone()

    sources = {}
    source_names = ["ensemble", "historical", "deterministic", "analog", "bma"]
    for i, name in enumerate(source_names):
        val = src_row[i]
        if val is not None:
            sources[name] = round(val, 4)
    source_ranking = sorted(sources.items(), key=lambda x: x[1])
    source_attribution = {
        "sources": sources,
        "ranking": [{"source": s, "avg_brier": b} for s, b in source_ranking],
    }

    # 3. Confidence calibration
    confidence_calibration = confidence_calibration_analysis(db, period_start=period_start, period_end=period_end)

    # 4. Breakdown by city
    city_rows = db.execute(
        """SELECT city_slug, COUNT(*), AVG(brier_score), AVG(outcome_binary)
        FROM prediction_scores
        WHERE target_date >= ? AND target_date <= ? AND city_slug IS NOT NULL
        GROUP BY city_slug ORDER BY AVG(brier_score) ASC""",
        [period_start, period_end],
    ).fetchall()
    by_city = [
        {"city_slug": r[0], "n": r[1], "avg_brier": round(r[2], 4), "win_rate": round(r[3], 3)}
        for r in city_rows
    ]

    # Breakdown by variable
    var_rows = db.execute(
        """SELECT variable, COUNT(*), AVG(brier_score), AVG(outcome_binary)
        FROM prediction_scores
        WHERE target_date >= ? AND target_date <= ? AND variable IS NOT NULL
        GROUP BY variable ORDER BY AVG(brier_score) ASC""",
        [period_start, period_end],
    ).fetchall()
    by_variable = [
        {"variable": r[0], "n": r[1], "avg_brier": round(r[2], 4), "win_rate": round(r[3], 3)}
        for r in var_rows
    ]

    # Breakdown by horizon
    horizon_rows = db.execute(
        """SELECT horizon_days, COUNT(*), AVG(brier_score), AVG(outcome_binary)
        FROM prediction_scores
        WHERE target_date >= ? AND target_date <= ? AND horizon_days IS NOT NULL
        GROUP BY horizon_days ORDER BY horizon_days ASC""",
        [period_start, period_end],
    ).fetchall()
    by_horizon = [
        {"horizon_days": r[0], "n": r[1], "avg_brier": round(r[2], 4), "win_rate": round(r[3], 3)}
        for r in horizon_rows
    ]

    # 5. Trend: compare with previous period
    period_days = (period_end - period_start).days
    prev_start = period_start - timedelta(days=period_days)
    prev_end = period_start - timedelta(days=1)

    prev_row = db.execute(
        """SELECT COUNT(*), AVG(brier_score), AVG(outcome_binary)
        FROM prediction_scores
        WHERE target_date >= ? AND target_date <= ?""",
        [prev_start, prev_end],
    ).fetchone()

    trend = {
        "previous_period": {"start": str(prev_start), "end": str(prev_end)},
        "previous_n": prev_row[0],
        "previous_brier": round(prev_row[1], 4) if prev_row[1] is not None else None,
        "previous_win_rate": round(prev_row[2], 3) if prev_row[2] is not None else None,
    }
    if performance["avg_brier"] is not None and trend["previous_brier"] is not None:
        trend["brier_change"] = round(performance["avg_brier"] - trend["previous_brier"], 4)
        trend["improved"] = trend["brier_change"] < 0
    else:
        trend["brier_change"] = None
        trend["improved"] = None

    # 6. Calibration model status
    cal_rows = db.execute(
        "SELECT variable, model_type, n_samples, brier_before, brier_after FROM calibration_models"
    ).fetchall()
    calibration_status = [
        {
            "variable": r[0],
            "model_type": r[1],
            "n_samples": r[2],
            "brier_before": round(r[3], 4) if r[3] is not None else None,
            "brier_after": round(r[4], 4) if r[4] is not None else None,
        }
        for r in cal_rows
    ]

    report = {
        "report_type": report_type,
        "period_start": str(period_start),
        "period_end": str(period_end),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "performance": performance,
        "source_attribution": source_attribution,
        "confidence_calibration": confidence_calibration,
        "by_city": by_city,
        "by_variable": by_variable,
        "by_horizon": by_horizon,
        "trend": trend,
        "calibration_status": calibration_status,
    }

    # Save to feedback_reports
    try:
        db.execute(
            """INSERT INTO feedback_reports (report_type, period_start, period_end, report_json, generated_at)
            VALUES (?, ?, ?, ?, ?)""",
            [report_type, period_start, period_end, json.dumps(report), datetime.now(timezone.utc)],
        )
    except Exception as e:
        logger.warning("Failed to save feedback report: %s", e)

    return report


def confidence_calibration_analysis(
    db,
    days_back: int = 90,
    period_start: date | None = None,
    period_end: date | None = None,
) -> dict:
    """Analyze win rate by confidence bin.

    Groups bets by confidence level (0-20, 20-40, ..., 80-100)
    and computes win rate per bin.

    Returns:
        Dict with bins and whether high-confidence bets win more.
    """
    if period_start is not None and period_end is not None:
        rows = db.execute(
            """SELECT confidence, status FROM bets
            WHERE status IN ('won', 'lost')
            AND target_date >= ? AND target_date <= ?""",
            [period_start, period_end],
        ).fetchall()
    else:
        cutoff = date.today() - timedelta(days=days_back)
        rows = db.execute(
            """SELECT confidence, status FROM bets
            WHERE status IN ('won', 'lost')
            AND timestamp >= ?""",
            [cutoff],
        ).fetchall()

    if not rows:
        return {"bins": [], "n_bets": 0, "high_confidence_wins_more": None}

    bin_edges = [0, 20, 40, 60, 80, 100]
    bins = []

    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = [(conf, status) for conf, status in rows if lo <= (conf or 0) * 100 < hi or (i == len(bin_edges) - 2 and (conf or 0) * 100 == hi)]
        n_total = len(in_bin)
        n_won = sum(1 for _, s in in_bin if s == "won")
        win_rate = n_won / n_total if n_total > 0 else None

        bins.append({
            "confidence_range": f"{lo}-{hi}%",
            "n_bets": n_total,
            "n_won": n_won,
            "win_rate": round(win_rate, 3) if win_rate is not None else None,
        })

    # Check if high-confidence bets win more
    low_bins = [b for b in bins if b["win_rate"] is not None and b["confidence_range"] in ("0-20%", "20-40%")]
    high_bins = [b for b in bins if b["win_rate"] is not None and b["confidence_range"] in ("60-80%", "80-100%")]

    low_wr = sum(b["win_rate"] for b in low_bins) / len(low_bins) if low_bins else None
    high_wr = sum(b["win_rate"] for b in high_bins) / len(high_bins) if high_bins else None

    high_wins_more = None
    if low_wr is not None and high_wr is not None:
        high_wins_more = high_wr > low_wr

    return {
        "bins": bins,
        "n_bets": len(rows),
        "high_confidence_wins_more": high_wins_more,
    }


def format_report_telegram(report: dict) -> str:
    """Format a feedback report for Telegram with emoji and markdown."""
    perf = report.get("performance", {})
    trend = report.get("trend", {})
    src = report.get("source_attribution", {})

    lines = [
        f"📊 *Feedback Report* ({report.get('report_type', 'weekly')})",
        f"📅 {report.get('period_start')} → {report.get('period_end')}",
        "",
        "📈 *Performance*",
        f"  Bets scored: {perf.get('n_scored', 0)}",
        f"  Win rate: {perf.get('win_rate', 'N/A')}",
        f"  Brier score: {perf.get('avg_brier', 'N/A')}",
        f"  BSS vs market: {perf.get('bss', 'N/A')}",
        f"  ROI: {perf.get('roi', 'N/A')}",
        f"  Net PnL: ${perf.get('net_pnl', 0):.2f}",
    ]

    # Trend
    if trend.get("brier_change") is not None:
        emoji = "✅" if trend["improved"] else "⚠️"
        lines.append(f"\n{emoji} *Trend*: Brier change {trend['brier_change']:+.4f}")

    # Source attribution
    ranking = src.get("ranking", [])
    if ranking:
        lines.append("\n🏆 *Source Ranking* (lower Brier = better)")
        for i, r in enumerate(ranking):
            medal = ["🥇", "🥈", "🥉"][i] if i < 3 else f" {i+1}."
            lines.append(f"  {medal} {r['source']}: {r['avg_brier']:.4f}")

    # Confidence calibration
    conf = report.get("confidence_calibration", {})
    if conf.get("high_confidence_wins_more") is not None:
        emoji = "✅" if conf["high_confidence_wins_more"] else "⚠️"
        lines.append(f"\n{emoji} High-confidence bets win more: {conf['high_confidence_wins_more']}")

    # Calibration status
    cal = report.get("calibration_status", [])
    if cal:
        lines.append("\n🔧 *Calibration Models*")
        for c in cal:
            lines.append(f"  {c['variable']}: {c['model_type']} (n={c['n_samples']})")

    return "\n".join(lines)
