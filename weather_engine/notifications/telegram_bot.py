import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

import httpx

from weather_engine.config import settings

logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}"

# Pending bet expiration in seconds
_PENDING_BET_TTL = 300  # 5 minutes


def _parse_id_set(raw: str) -> set[str]:
    return {part.strip() for part in str(raw or "").split(",") if part.strip()}


class TelegramCommandBot:
    """Simple polling Telegram bot for basic operational commands."""

    def __init__(self, token: str = "", allowed_chat_id: str = ""):
        self.token = token or settings.telegram_bot_token
        self.allowed_chat_id = str(allowed_chat_id or settings.telegram_chat_id or "").strip()
        self.allowed_chat_ids = _parse_id_set(settings.telegram_allowed_chat_ids)
        if self.allowed_chat_id:
            self.allowed_chat_ids.add(self.allowed_chat_id)
        self.allowed_user_ids = _parse_id_set(settings.telegram_allowed_user_ids)
        # Private / semi-private mode: require token plus at least one chat/user allowlist rule.
        self.enabled = bool(self.token and (self.allowed_chat_ids or self.allowed_user_ids))
        self._offset: int | None = None
        self._bootstrapped = False
        self._pending_bet: dict | None = None

    def poll_once(self) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False, "processed": 0}

        updates = self._get_updates()
        if not updates:
            self._bootstrapped = True
            return {"enabled": True, "processed": 0}

        # Avoid replaying old messages after a restart.
        if not self._bootstrapped:
            self._offset = max(int(u.get("update_id", 0)) for u in updates) + 1
            self._bootstrapped = True
            logger.info("Telegram bot initialized, skipped %d pending updates", len(updates))
            return {"enabled": True, "processed": 0, "skipped": len(updates)}

        processed = 0
        for update in updates:
            update_id = update.get("update_id")
            if isinstance(update_id, int):
                self._offset = update_id + 1

            message = update.get("message") or update.get("edited_message")
            if not isinstance(message, dict):
                continue

            chat = message.get("chat") or {}
            chat_id = str(chat.get("id", ""))
            if not chat_id:
                continue
            from_user = message.get("from") or {}
            user_id = str(from_user.get("id", "")).strip()
            allowed_by_chat = chat_id in self.allowed_chat_ids if self.allowed_chat_ids else False
            allowed_by_user = user_id in self.allowed_user_ids if (user_id and self.allowed_user_ids) else False
            if (self.allowed_chat_ids or self.allowed_user_ids) and not (allowed_by_chat or allowed_by_user):
                continue

            text = (message.get("text") or "").strip()
            if not text.startswith("/"):
                continue

            reply = self._handle_command(text)
            if reply:
                self._send_message(chat_id, reply)
                processed += 1

        return {"enabled": True, "processed": processed}

    def _api_post(self, method: str, payload: dict[str, Any]) -> Any:
        url = TELEGRAM_API.format(token=self.token) + f"/{method}"
        resp = httpx.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if not data.get("ok"):
            raise RuntimeError(f"Telegram API {method} failed: {data}")
        return data.get("result")

    def _get_updates(self) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {"timeout": 0, "limit": 20, "allowed_updates": ["message", "edited_message"]}
        if self._offset is not None:
            payload["offset"] = self._offset
        result = self._api_post("getUpdates", payload)
        return result if isinstance(result, list) else []

    def _send_message(self, chat_id: str, text: str) -> None:
        try:
            self._api_post("sendMessage", {"chat_id": chat_id, "text": text})
        except Exception as e:
            logger.error("Telegram bot reply failed: %s", e)

    def _handle_command(self, raw_text: str) -> str:
        command, args = self._parse_command(raw_text)

        try:
            if command in ("start", "help"):
                return self._help_message()
            if command == "status":
                return self._status_message()
            if command == "top":
                return self._top_message(limit=3)
            if command == "pnl":
                return self._pnl_message()
            if command == "city":
                if not args:
                    return "Uso: /city <slug o nome>\nEsempio: /city roma"
                return self._top_message(limit=5, city_filter=" ".join(args))
            if command == "bet":
                city_filter = " ".join(args) if args else None
                return self._bet_message(city_filter=city_filter)
            if command == "confirm":
                return self._confirm_message()
            if command == "portfolio":
                return self._portfolio_message()
            if command == "health":
                return self._health_message()
            return "Comando non riconosciuto.\nUsa /help"
        except Exception as e:
            logger.exception("Telegram bot command failed (%s): %s", command, e)
            return f"Errore durante il comando /{command}. Controlla i log del backend."

    def _parse_command(self, raw_text: str) -> tuple[str, list[str]]:
        parts = raw_text.split()
        head = (parts[0] if parts else "/help").strip()
        if head.startswith("/"):
            head = head[1:]
        if "@" in head:
            head = head.split("@", 1)[0]
        return head.lower(), parts[1:]

    def _help_message(self) -> str:
        return (
            "Weather Trader Bot\n"
            "\n"
            "Comandi:\n"
            "/status - stato engine e raccolte\n"
            "/top - top opportunita' LIVE Polymarket\n"
            "/city <nome|slug> - opportunita' LIVE per citta'\n"
            "/pnl - statistiche scommesse\n"
            "/bet - migliore opportunita' con sizing\n"
            "/bet <citta'> - opportunita' per citta' specifica\n"
            "/confirm - conferma ultima bet proposta\n"
            "/portfolio - posizioni aperte e P&L\n"
            "/health - stato circuit breaker e staleness\n"
            "/help - questa guida"
        )

    def _status_message(self) -> str:
        from weather_engine.api.routes_health import health, metrics

        h = health()
        m = metrics()

        recent = m.get("recent_collections", [])[:3]
        recent_lines = []
        for row in recent:
            recent_lines.append(
                f"- {row.get('collector', '?')} {row.get('city', '?')} [{row.get('status', '?')}]"
            )

        return "\n".join(
            [
                "Status Engine",
                f"Status: {h.get('status', 'unknown')}",
                f"DB: {h.get('database', 'unknown')}",
                f"Forecast hourly: {m.get('forecasts_hourly', 'n/a')}",
                f"Forecast daily: {m.get('forecasts_daily', 'n/a')}",
                f"Observations: {m.get('observations', 'n/a')}",
                "",
                "Ultime raccolte:",
                *(recent_lines or ["- nessuna"]),
            ]
        )

    def _top_message(self, limit: int = 3, city_filter: str | None = None) -> str:
        from weather_engine.api.routes_market import scan_markets

        result = scan_markets(min_edge=0.0)
        opportunities = result.get("markets", [])

        if city_filter:
            needle = city_filter.strip().lower()
            filtered = []
            for opp in opportunities:
                meta = opp.get("metadata", {}) or {}
                city = meta.get("city", {}) or {}
                rec = opp.get("recommendation", {}) or {}
                hay = " ".join(
                    str(v)
                    for v in [
                        city.get("slug"),
                        city.get("name"),
                        rec.get("city"),
                    ]
                    if v
                ).lower()
                if needle in hay:
                    filtered.append(opp)
            opportunities = filtered

        lines = ["Top opportunita' LIVE"]
        if city_filter:
            lines[0] += f" ({city_filter})"

        shown = 0
        for opp in opportunities:
            if shown >= limit:
                break

            rec = opp.get("recommendation", {}) or {}
            best_bet = rec.get("best_bet")
            if not best_bet:
                continue

            outcomes = rec.get("outcomes", []) or []
            best_row = next((o for o in outcomes if o.get("outcome") == best_bet), None)
            if not best_row:
                continue

            meta = opp.get("metadata", {}) or {}
            city = meta.get("city", {}) or {}
            city_name = city.get("name") or rec.get("city") or "?"
            date_str = rec.get("date") or meta.get("target_date") or "?"
            edge_pct = float(best_row.get("edge", 0) or 0) * 100
            prob_pct = float(best_row.get("our_probability", 0) or 0) * 100
            price_pct = float(best_row.get("market_price", 0) or 0) * 100
            conf_pct = float(best_row.get("confidence", 0) or 0) * 100
            ev_pct = float(rec.get("expected_value", 0) or 0) * 100

            shown += 1
            lines.extend(
                [
                    "",
                    f"{shown}. {city_name} ({date_str})",
                    f"   Bet: {best_bet}",
                    f"   Edge: {edge_pct:.1f}% | EV: {ev_pct:.1f}% | Conf: {conf_pct:.0f}%",
                    f"   Prob: {prob_pct:.1f}% | Prezzo: {price_pct:.1f}%",
                ]
            )

        if shown == 0:
            lines.append("")
            if city_filter:
                lines.append("Nessuna opportunita' LIVE trovata per questa citta'.")
            else:
                lines.append("Nessuna opportunita' LIVE disponibile su Polymarket al momento.")

        return "\n".join(lines)

    def _pnl_message(self) -> str:
        from weather_engine.api.routes_market import bet_stats

        s = bet_stats()
        return "\n".join(
            [
                "Betting Stats",
                f"Totali: {s.get('total', 0)} | Pending: {s.get('pending', 0)}",
                f"Won: {s.get('won', 0)} | Lost: {s.get('lost', 0)}",
                f"Win rate: {float(s.get('win_rate', 0) or 0) * 100:.1f}%",
                f"Net PnL: {float(s.get('net_pnl', 0) or 0):.2f}",
                f"Staked: {float(s.get('total_staked', 0) or 0):.2f}",
                f"ROI: {float(s.get('roi', 0) or 0) * 100:.2f}%",
            ]
        )

    def _bet_message(self, city_filter: str | None = None) -> str:
        """Find the best opportunity and propose a bet with sizing."""
        from weather_engine.api.routes_market import scan_markets
        from weather_engine.market_bridge.ev_calculator import adaptive_kelly
        from weather_engine.notifications.formatter import format_bet_proposal

        result = scan_markets(min_edge=0.0)
        opportunities = result.get("markets", [])

        if city_filter:
            needle = city_filter.strip().lower()
            filtered = []
            for opp in opportunities:
                meta = opp.get("metadata", {}) or {}
                city = meta.get("city", {}) or {}
                rec = opp.get("recommendation", {}) or {}
                hay = " ".join(
                    str(v) for v in [city.get("slug"), city.get("name"), rec.get("city")] if v
                ).lower()
                if needle in hay:
                    filtered.append(opp)
            opportunities = filtered

        # Find best opportunity with a best_bet
        best_opp = None
        best_edge = 0.0
        for opp in opportunities:
            rec = opp.get("recommendation", {}) or {}
            if not rec.get("best_bet"):
                continue
            outcomes = rec.get("outcomes", [])
            best_row = next((o for o in outcomes if o.get("outcome") == rec["best_bet"]), None)
            if best_row and float(best_row.get("edge", 0)) > best_edge:
                best_edge = float(best_row.get("edge", 0))
                best_opp = opp

        if best_opp is None:
            if city_filter:
                return f"Nessuna opportunita' trovata per '{city_filter}'."
            return "Nessuna opportunita' disponibile al momento."

        rec = best_opp.get("recommendation", {})
        meta = best_opp.get("metadata", {})
        market_info = best_opp.get("market", {})
        best_bet = rec["best_bet"]
        best_row = next((o for o in rec.get("outcomes", []) if o.get("outcome") == best_bet), {})

        our_prob = float(best_row.get("our_probability", 0))
        market_price = float(best_row.get("market_price", 0))
        conf_pct = float(best_row.get("confidence", 0)) * 100
        liquidity = float(market_info.get("liquidity", 50000))

        # Build a simple signal dict for the formatter
        signal = {
            "signal": "SCOMMETTI" if best_edge >= 0.05 and conf_pct >= 50 else "CAUTELA",
            "effective_edge": best_edge,
        }

        # Adaptive Kelly sizing
        kelly_frac = adaptive_kelly(
            our_prob, market_price,
            confidence_total=conf_pct,
            signal=signal["signal"],
            liquidity=liquidity,
        )
        bankroll = 1000.0  # Default bankroll
        stake = round(kelly_frac * bankroll, 2)

        # Line movement timing signal
        timing_signal = "SCOMMETTI_ORA"
        try:
            from weather_engine.analysis.line_tracker import analyze_line_movement
            from weather_engine.db import get_db
            db = get_db()
            cond_id = market_info.get("condition_id", "")
            if cond_id:
                lm = analyze_line_movement(db, cond_id)
                timing_signal = lm.get("timing_signal", "SCOMMETTI_ORA")
        except Exception:
            pass

        # Format the proposal
        msg = format_bet_proposal(best_opp, signal, stake, timing_signal)

        # Store pending bet
        self._pending_bet = {
            "created_at": time.time(),
            "market_question": rec.get("market_question", ""),
            "outcome": best_bet,
            "stake": stake,
            "odds": 1.0 / market_price if market_price > 0 else 0,
            "our_prob": our_prob,
            "edge": best_edge,
            "confidence": conf_pct,
            "city_slug": meta.get("city", {}).get("slug", "") if isinstance(meta.get("city"), dict) else rec.get("city", ""),
            "target_date": meta.get("target_date") or rec.get("date", ""),
            "variable": meta.get("variable", ""),
        }

        return msg

    def _confirm_message(self) -> str:
        """Confirm and record the pending bet."""
        if self._pending_bet is None:
            return "Nessuna bet in attesa. Usa /bet per generare una proposta."

        # Check TTL
        elapsed = time.time() - self._pending_bet.get("created_at", 0)
        if elapsed > _PENDING_BET_TTL:
            self._pending_bet = None
            return "Proposta scaduta (> 5 min). Usa /bet per una nuova proposta."

        bet = self._pending_bet
        self._pending_bet = None

        try:
            from weather_engine.db import get_db
            db = get_db()
            from datetime import date

            td = date.fromisoformat(bet["target_date"]) if bet.get("target_date") else None
            db.execute(
                """INSERT INTO bets (timestamp, market_question, outcome, stake, odds,
                   our_prob, edge, confidence, city_slug, target_date, variable)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    datetime.now(timezone.utc), bet["market_question"], bet["outcome"],
                    bet["stake"], bet["odds"], bet["our_prob"], bet["edge"],
                    bet["confidence"], bet["city_slug"], td, bet.get("variable", ""),
                ],
            )
            return (
                f"Bet registrata!\n"
                f"Outcome: {bet['outcome']}\n"
                f"Stake: ${bet['stake']:.2f}\n"
                f"Edge: {bet['edge'] * 100:.1f}%\n"
                f"Piazza manualmente su Polymarket."
            )
        except Exception as e:
            logger.exception("Failed to record bet: %s", e)
            return f"Errore nella registrazione della bet: {e}"

    def _portfolio_message(self) -> str:
        """Show open positions, total exposure, and P&L."""
        try:
            from weather_engine.db import get_db
            db = get_db()

            # Open positions
            pending = db.execute(
                "SELECT COUNT(*), COALESCE(SUM(stake), 0) FROM bets WHERE status = 'pending'"
            ).fetchone()
            n_open = pending[0] if pending else 0
            exposure = pending[1] if pending else 0

            # Resolved P&L
            resolved = db.execute(
                """SELECT COUNT(*), COALESCE(SUM(pnl), 0), COALESCE(SUM(stake), 0)
                FROM bets WHERE status IN ('won', 'lost')"""
            ).fetchone()
            n_resolved = resolved[0] if resolved else 0
            total_pnl = resolved[1] if resolved else 0
            total_staked = resolved[2] if resolved else 0

            roi = (total_pnl / total_staked * 100) if total_staked > 0 else 0

            return "\n".join([
                "Portfolio",
                f"Posizioni aperte: {n_open}",
                f"Esposizione: ${exposure:.2f}",
                "",
                f"Risolte: {n_resolved}",
                f"P&L: ${total_pnl:.2f}",
                f"ROI: {roi:.1f}%",
            ])
        except Exception as e:
            logger.exception("Portfolio command failed: %s", e)
            return f"Errore nel calcolo portfolio: {e}"

    def _health_message(self) -> str:
        """Show circuit breaker states and data staleness."""
        lines = ["Health Check"]

        try:
            from weather_engine.db import get_db
            from weather_engine.resilience.circuit_breaker import get_all_states
            from weather_engine.resilience.staleness import check_data_staleness

            db = get_db()

            # Circuit breakers
            states = get_all_states(db)
            if states:
                lines.append("")
                lines.append("Circuit Breakers:")
                for s in states[:5]:
                    icon = "OK" if s["state"] == "closed" else "APERTO"
                    lines.append(f"  {s['source']}: {icon} (fail: {s['consecutive_failures']})")
            else:
                lines.append("Circuit Breakers: nessuno registrato")

            # Staleness for top 3 cities
            from weather_engine.db import get_cities
            cities = get_cities(db)
            lines.append("")
            lines.append("Staleness:")
            for city in cities[:3]:
                try:
                    stal = check_data_staleness(db, city["slug"])
                    score = stal.get("staleness_score", 0)
                    ens_h = stal.get("ensemble_hours", 999)
                    lines.append(f"  {city['name']}: {score}/100 (ens: {ens_h:.0f}h)")
                except Exception:
                    lines.append(f"  {city['name']}: errore")

        except Exception as e:
            logger.exception("Health command failed: %s", e)
            lines.append(f"Errore: {e}")

        return "\n".join(lines)


_BOT: TelegramCommandBot | None = None


def poll_telegram_commands() -> dict[str, Any]:
    global _BOT
    if _BOT is None:
        _BOT = TelegramCommandBot()
    return _BOT.poll_once()
