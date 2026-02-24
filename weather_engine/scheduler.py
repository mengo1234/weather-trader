import logging

from apscheduler.schedulers.background import BackgroundScheduler

from weather_engine.collectors.air_quality import AirQualityCollector
from weather_engine.collectors.climate import ClimateCollector
from weather_engine.collectors.deterministic import DeterministicMultiCollector
from weather_engine.collectors.ensemble import EnsembleCollector
from weather_engine.collectors.flood import FloodCollector
from weather_engine.collectors.forecast import ForecastCollector
from weather_engine.collectors.marine import MarineCollector
from weather_engine.collectors.seasonal import SeasonalCollector
from weather_engine.analysis.verification import run_verification
from weather_engine.config import settings
from weather_engine.db import get_cities, get_db

logger = logging.getLogger(__name__)

_scheduler: BackgroundScheduler | None = None


def _run_collector(collector_cls, name: str, **kwargs):
    try:
        db = get_db()
        cities = get_cities(db)
        collector = collector_cls(db, **kwargs)
        n = collector.collect_all(cities)
        logger.info("Scheduled %s: %d rows total", name, n)
    except Exception as e:
        logger.error("Scheduled %s failed: %s", name, e)


def start_scheduler():
    global _scheduler
    if _scheduler is not None:
        return _scheduler

    _scheduler = BackgroundScheduler()

    _scheduler.add_job(
        _run_collector,
        "interval",
        seconds=settings.forecast_interval,
        args=[ForecastCollector, "forecast"],
        id="forecast",
        next_run_time=None,  # Don't run immediately on startup
    )

    # Ensemble Batch 1 (core models) — every 4 hours
    _scheduler.add_job(
        _run_collector,
        "interval",
        seconds=14400,  # 4h
        args=[EnsembleCollector, "ensemble_batch1"],
        kwargs={"batch": 1},
        id="ensemble_batch1",
        next_run_time=None,
    )

    # Ensemble Batch 2 (secondary models) — every 6 hours
    _scheduler.add_job(
        _run_collector,
        "interval",
        seconds=21600,  # 6h
        args=[EnsembleCollector, "ensemble_batch2"],
        kwargs={"batch": 2},
        id="ensemble_batch2",
        next_run_time=None,
    )

    # Ensemble Batch 3 (regional/supplementary) — every 8 hours
    _scheduler.add_job(
        _run_collector,
        "interval",
        seconds=28800,  # 8h
        args=[EnsembleCollector, "ensemble_batch3"],
        kwargs={"batch": 3},
        id="ensemble_batch3",
        next_run_time=None,
    )

    # Deterministic multi-model — every 3 hours
    _scheduler.add_job(
        _run_collector,
        "interval",
        seconds=10800,  # 3h
        args=[DeterministicMultiCollector, "deterministic_multi"],
        id="deterministic_multi",
        next_run_time=None,
    )

    _scheduler.add_job(
        _run_collector,
        "interval",
        seconds=settings.air_quality_interval,
        args=[AirQualityCollector, "air_quality"],
        id="air_quality",
        next_run_time=None,
    )

    # Marine — every 6 hours (coastal cities only, filtered in collector)
    _scheduler.add_job(
        _run_collector,
        "interval",
        seconds=settings.marine_interval,
        args=[MarineCollector, "marine"],
        id="marine",
        next_run_time=None,
    )

    # Flood — every 6 hours
    _scheduler.add_job(
        _run_collector,
        "interval",
        seconds=21600,  # 6h
        args=[FloodCollector, "flood"],
        id="flood",
        next_run_time=None,
    )

    # Seasonal forecast — run every 12 hours (data updates slowly)
    _scheduler.add_job(
        _run_collector,
        "interval",
        seconds=43200,
        args=[SeasonalCollector, "seasonal"],
        id="seasonal",
        next_run_time=None,
    )

    # Climate indicators — run once daily at 06:00 UTC
    _scheduler.add_job(
        _run_collector,
        "cron",
        hour=6,
        args=[ClimateCollector, "climate"],
        id="climate",
    )

    # Verifica previsioni giornaliera alle 08:00 UTC
    _scheduler.add_job(
        _run_verification,
        "cron",
        hour=8,
        id="verification",
    )

    # Model accuracy update — daily at 08:15 UTC (after verification)
    _scheduler.add_job(
        _run_model_accuracy_update,
        "cron",
        hour=8,
        minute=15,
        id="model_accuracy",
    )

    # Cross-reference compute — every 3 hours
    _scheduler.add_job(
        _run_cross_reference_update,
        "interval",
        seconds=10800,  # 3h
        id="cross_reference",
        next_run_time=None,
    )

    # Telegram: check opportunities every 30 min
    _scheduler.add_job(
        _check_opportunities_and_notify,
        "interval",
        seconds=1800,
        id="telegram_opportunities",
        next_run_time=None,
    )

    # Telegram: daily report at 08:30 UTC
    _scheduler.add_job(
        _send_daily_telegram,
        "cron",
        hour=8,
        minute=30,
        id="telegram_daily",
    )

    # Telegram bot commands: poll updates frequently (lightweight, no-op if disabled)
    _scheduler.add_job(
        _poll_telegram_commands,
        "interval",
        seconds=15,
        id="telegram_bot_poll",
        next_run_time=None,
    )

    # Auto-resolve bets at 12:00 UTC daily
    _scheduler.add_job(
        _auto_resolve_bets,
        "cron",
        hour=12,
        id="auto_resolve_bets",
    )

    # Teleconnection indices — every 12 hours
    _scheduler.add_job(
        _run_teleconnection_update,
        "interval",
        seconds=settings.teleconnection_interval,
        id="teleconnection",
        next_run_time=None,
    )

    # Horizon profiles — daily at 08:30 UTC (after verification + model accuracy)
    _scheduler.add_job(
        _run_horizon_profiles,
        "cron",
        hour=8,
        minute=30,
        id="horizon_profiles",
    )

    # Drift detection — daily at 09:00 UTC
    _scheduler.add_job(
        _run_drift_detection,
        "cron",
        hour=9,
        id="drift_detection",
    )

    # Prediction scoring — daily at 12:15 UTC (after auto-resolve at 12:00)
    _scheduler.add_job(
        _run_prediction_scoring,
        "cron",
        hour=12,
        minute=15,
        id="prediction_scoring",
    )

    # Calibration update — weekly on Sunday at 10:00 UTC
    _scheduler.add_job(
        _run_calibration_update,
        "cron",
        day_of_week="sun",
        hour=10,
        id="calibration_update",
    )

    # Spread snapshots — every 4 hours
    _scheduler.add_job(
        _run_spread_snapshots,
        "interval",
        seconds=14400,  # 4h
        id="spread_snapshots",
        next_run_time=None,
    )

    # Weekly feedback report — Monday at 09:00 UTC
    _scheduler.add_job(
        _run_weekly_feedback_report,
        "cron",
        day_of_week="mon",
        hour=9,
        id="weekly_feedback_report",
    )

    # Weight learning — weekly on Sunday at 11:00 UTC (after calibration at 10:00)
    _scheduler.add_job(
        _run_weight_learning,
        "cron",
        day_of_week="sun",
        hour=11,
        id="weight_learning",
    )

    # Market price snapshots — every 30 minutes
    _scheduler.add_job(
        _snapshot_market_prices,
        "interval",
        seconds=1800,
        id="market_price_snapshots",
        next_run_time=None,
    )

    _scheduler.start()
    logger.info("Scheduler started with %d jobs", len(_scheduler.get_jobs()))
    return _scheduler


def _run_verification():
    try:
        db = get_db()
        result = run_verification(db)
        logger.info("Verification: %d new records", result["new_records"])
    except Exception as e:
        logger.error("Verification failed: %s", e)


def _run_model_accuracy_update():
    """Update per-model accuracy after daily verification."""
    try:
        from weather_engine.analysis.model_tracker import run_model_accuracy_update
        db = get_db()
        result = run_model_accuracy_update(db)
        logger.info("Model accuracy update: %d records", result["records"])
    except Exception as e:
        logger.error("Model accuracy update failed: %s", e)


def _run_cross_reference_update():
    """Compute cross-reference scores for all cities."""
    try:
        from weather_engine.analysis.cross_reference import run_cross_reference_update
        db = get_db()
        result = run_cross_reference_update(db)
        logger.info("Cross-reference update: %d computed", result["computed"])
    except Exception as e:
        logger.error("Cross-reference update failed: %s", e)


def _check_opportunities_and_notify():
    """Check for high-edge opportunities and send Telegram alert."""
    try:
        from weather_engine.notifications.telegram import TelegramNotifier
        from weather_engine.notifications.formatter import format_opportunity

        notifier = TelegramNotifier()
        if not notifier.enabled:
            return

        # Import and call the live Polymarket scan logic
        from weather_engine.api.routes_market import scan_markets
        result = scan_markets(min_edge=0.0)
        opportunities = result.get("markets", [])

        for opp in opportunities[:3]:
            rec = opp.get("recommendation", {})
            if rec.get("best_bet") and rec.get("expected_value", 0) > 0.05:
                msg = format_opportunity(rec, opp.get("metadata", {}))
                notifier.send(msg)
    except Exception as e:
        logger.error("Telegram opportunity check failed: %s", e)


def _send_daily_telegram():
    """Send daily summary report via Telegram."""
    try:
        from weather_engine.notifications.telegram import TelegramNotifier
        from weather_engine.notifications.formatter import format_daily_report

        notifier = TelegramNotifier()
        if not notifier.enabled:
            return

        from weather_engine.api.routes_market import scan_markets
        result = scan_markets(min_edge=0.0)
        opportunities = result.get("markets", [])

        # Get bet stats if available
        stats = None
        try:
            from weather_engine.market_bridge.pnl_resolver import get_bet_stats
            stats = get_bet_stats()
        except Exception:
            pass

        msg = format_daily_report(opportunities, stats)
        notifier.send(msg)
    except Exception as e:
        logger.error("Telegram daily report failed: %s", e)


def _poll_telegram_commands():
    """Poll Telegram updates and respond to bot commands."""
    try:
        from weather_engine.notifications.telegram_bot import poll_telegram_commands

        result = poll_telegram_commands()
        processed = int(result.get("processed", 0) or 0)
        skipped = int(result.get("skipped", 0) or 0)
        if processed > 0 or skipped > 0:
            logger.info("Telegram bot poll: processed=%d skipped=%d", processed, skipped)
    except Exception as e:
        logger.error("Telegram bot poll failed: %s", e)


def _auto_resolve_bets():
    """Auto-resolve pending bets and notify via Telegram."""
    try:
        from weather_engine.market_bridge.pnl_resolver import auto_resolve_bets
        result = auto_resolve_bets()

        if result.get("resolved", 0) > 0:
            try:
                from weather_engine.notifications.telegram import TelegramNotifier
                notifier = TelegramNotifier()
                if notifier.enabled:
                    msg = f"🔄 *Auto-Resolved {result['resolved']} bets*\n✅ Won: {result['won']} | ❌ Lost: {result['lost']}"
                    notifier.send(msg)
            except Exception:
                pass
    except Exception as e:
        logger.error("Auto-resolve bets failed: %s", e)


def _run_teleconnection_update():
    """Fetch teleconnection indices from NOAA CPC."""
    try:
        from weather_engine.collectors.teleconnection import collect_teleconnections
        db = get_db()
        n = collect_teleconnections(db, config=settings)
        logger.info("Teleconnection update: %d records", n)
    except Exception as e:
        logger.error("Teleconnection update failed: %s", e)


def _run_horizon_profiles():
    """Compute horizon profiles for all cities."""
    try:
        from weather_engine.analysis.verification import compute_horizon_profiles
        db = get_db()
        result = compute_horizon_profiles(db)
        logger.info("Horizon profiles: %d profiles computed", result["profiles"])
    except Exception as e:
        logger.error("Horizon profiles failed: %s", e)


def _run_drift_detection():
    """Run drift detection and alert via Telegram if any model is in alert status."""
    try:
        from weather_engine.analysis.drift import detect_drift_all_cities, get_worst_drift
        db = get_db()
        all_drifts = detect_drift_all_cities(db)
        alerts = get_worst_drift(db)

        if alerts:
            logger.warning("Drift alerts: %d models degrading/alert", len(alerts))
            try:
                from weather_engine.notifications.telegram import TelegramNotifier
                notifier = TelegramNotifier()
                if notifier.enabled:
                    alert_lines = [f"⚠️ *Model Drift Alert* — {len(alerts)} issues"]
                    for d in alerts[:5]:
                        alert_lines.append(
                            f"• {d.model} / {d.variable} / {d.city_slug}: "
                            f"{d.status.upper()} (ratio={d.drift_ratio:.2f}, "
                            f"MAE 7d={d.mae_7d:.2f} vs 30d={d.mae_30d:.2f})"
                        )
                    notifier.send("\n".join(alert_lines))
            except Exception:
                pass
    except Exception as e:
        logger.error("Drift detection failed: %s", e)


def _run_prediction_scoring():
    """Score all unscored resolved bets."""
    try:
        from weather_engine.analysis.scoring import score_all_unscored
        db = get_db()
        result = score_all_unscored(db)
        logger.info("Prediction scoring: %d scored, %d skipped, %d errors",
                     result["scored"], result["skipped"], result["errors"])
    except Exception as e:
        logger.error("Prediction scoring failed: %s", e)


def _run_calibration_update():
    """Update calibration bins and fit calibration models."""
    try:
        from weather_engine.analysis.recalibration import update_calibration_bins, fit_calibration_model
        db = get_db()

        # Update bins
        bins_result = update_calibration_bins(db, variable="all")
        logger.info("Calibration bins updated: %d variables", bins_result.get("updated", 0))

        # Fit models for each variable
        for var in bins_result.get("variables", []):
            try:
                fit_result = fit_calibration_model(db, var)
                logger.info("Calibration model for %s: %s (n=%d)",
                            var, fit_result["model_type"], fit_result["n_samples"])
            except Exception as e:
                logger.warning("Calibration model fit failed for %s: %s", var, e)
    except Exception as e:
        logger.error("Calibration update failed: %s", e)


def _run_spread_snapshots():
    """Take spread snapshots for all cities."""
    try:
        from weather_engine.analysis.spread_tracker import snapshot_all_cities
        db = get_db()
        result = snapshot_all_cities(db)
        logger.info("Spread snapshots: %d collected", result["snapshots"])
    except Exception as e:
        logger.error("Spread snapshots failed: %s", e)


def _run_weekly_feedback_report():
    """Generate weekly feedback report and send via Telegram."""
    try:
        from datetime import timedelta
        from weather_engine.analysis.feedback_report import generate_feedback_report, format_report_telegram
        db = get_db()

        from datetime import date as date_cls
        period_end = date_cls.today()
        period_start = period_end - timedelta(days=7)
        report = generate_feedback_report(db, period_start, period_end, report_type="weekly")

        # Send via Telegram
        try:
            from weather_engine.notifications.telegram import TelegramNotifier
            notifier = TelegramNotifier()
            if notifier.enabled:
                msg = format_report_telegram(report)
                notifier.send(msg)
        except Exception:
            pass

        logger.info("Weekly feedback report generated: %d scored predictions", report["performance"]["n_scored"])
    except Exception as e:
        logger.error("Weekly feedback report failed: %s", e)


def _run_weight_learning():
    """Learn optimal weights from resolved bets."""
    try:
        from weather_engine.analysis.weight_learner import (
            learn_confidence_weights, learn_blending_weights, learn_cross_ref_weights,
        )
        db = get_db()

        # Confidence weights
        learn_confidence_weights(db)

        # Blending weights for each variable and horizon band
        for var in ["temperature_2m_max"]:
            for band in ["short", "medium", "long", "extended"]:
                learn_blending_weights(db, variable=var, horizon_band=band)

        # Cross-ref weights
        learn_cross_ref_weights(db)

        logger.info("Weight learning completed")
    except Exception as e:
        logger.error("Weight learning failed: %s", e)


def _snapshot_market_prices():
    """Snapshot current market prices from Polymarket for backtesting."""
    try:
        from weather_engine.api.routes_market import _fetch_gamma_weather_markets
        import json as _json
        from datetime import datetime, timezone

        db = get_db()
        raw_markets = _fetch_gamma_weather_markets()
        if not raw_markets:
            return

        n = 0
        now = datetime.now(timezone.utc)
        for market in raw_markets:
            condition_id = market.get("condition_id", "")
            market_id = market.get("market_slug", "") or market.get("event_title", "")
            volume = float(market.get("volume", 0) or 0)
            liquidity = float(market.get("liquidity", 0) or 0)

            outcomes = market.get("outcomes", [])
            raw_prices = market.get("outcomePrices", [])
            if isinstance(raw_prices, str):
                try:
                    raw_prices = _json.loads(raw_prices)
                except (ValueError, TypeError):
                    raw_prices = []

            for i, outcome in enumerate(outcomes):
                price = float(raw_prices[i]) if i < len(raw_prices) else 0.0
                if price <= 0:
                    continue
                try:
                    db.execute(
                        """INSERT INTO market_price_snapshots
                        (market_id, condition_id, outcome, price, volume, liquidity, collected_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        [market_id, condition_id, outcome, price, volume, liquidity, now],
                    )
                    n += 1
                except Exception:
                    pass

        logger.info("Market price snapshots: %d prices from %d markets", n, len(raw_markets))
    except Exception as e:
        logger.error("Market price snapshots failed: %s", e)


def stop_scheduler():
    global _scheduler
    if _scheduler:
        _scheduler.shutdown()
        _scheduler = None
        logger.info("Scheduler stopped")
