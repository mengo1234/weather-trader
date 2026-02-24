"""Dashboard section: hero card, stats, verdict, action plan, P&L, cities overview."""

from datetime import datetime

import flet as ft

from weather_trader.constants import (
    ACCENT, ACCENT2, BG, CITY_MAP, CITY_NAMES, CITY_SLUGS,
    GREEN, INITIAL_BANKROLL, ORANGE, RED, TEXT, TEXT_DIM, YELLOW, f2c,
    TYPE_2XS, TYPE_XS, TYPE_SM, TYPE_MD, TYPE_LG, TYPE_XL, TYPE_DISPLAY, TYPE_DISPLAY_COMPACT, TYPE_METRIC,
)
from weather_trader.api_client import api_get
from weather_trader.app_state import AppState
from weather_trader.logic.pnl_tracker import load_pnl, get_pnl_stats
from weather_trader.logic.risk_manager import check_portfolio_risk
from weather_trader.widgets.factory import (
    make_badge, make_card, make_empty_state, make_info_box,
    make_kv_row, make_loading_indicator, make_section_title,
    make_stat_chip, make_verdict_banner, pad, uv_color, z_score_color,
)
from weather_trader.widgets.charts import build_sparkline
from weather_trader.widgets.distribution import build_consensus_indicator
from weather_trader.widgets.confidence import calculate_confidence, build_confidence_meter
from weather_trader.widgets.pnl_widgets import build_pnl_sparkline


def create_dashboard(page: ft.Page, state: AppState, safe_update):
    """Create dashboard section. Returns (build, load) tuple."""

    # Altezze uniformi per card sulla stessa riga (desktop)
    ROW_ANALYTICS_H = 330   # P&L / Calibrazione / Sparkline
    ROW_INSIGHTS_H = 290    # Alert / Top opportunità / Miglior confidenza
    ROW_ANALYTICS_H_TIGHT = 350
    ROW_INSIGHTS_H_TIGHT = 305

    # --- Containers ---
    dash_hero = ft.Container(content=ft.Text("—", color=TEXT_DIM), height=180)
    dash_stats_row1 = ft.ResponsiveRow([], spacing=8, run_spacing=8)
    dash_stats_row2 = ft.ResponsiveRow([], spacing=8, run_spacing=8)
    dash_sparkline = ft.Container(height=60)
    dash_alerts = ft.Container(
        content=ft.Text("Nessun alert", color=TEXT_DIM, size=TYPE_MD),
        padding=10,
    )
    dash_top_opps = ft.Column([], spacing=6)
    dash_pnl_card = ft.Container(padding=10)
    dash_calibration_card = ft.Container(padding=10)
    dash_best_confidence = ft.Container(padding=10)
    dash_convergence = ft.Container(padding=10)
    dash_verdict = ft.Container()
    dash_action_plan = ft.Container(padding=10)
    dash_risk_card = ft.Container(padding=10)
    dash_cities_overview = ft.Container(padding=10)

    def _viewport_width() -> int:
        try:
            if getattr(page, "width", None):
                return int(page.width)
        except Exception:
            pass
        try:
            return int(page.window.width)
        except Exception:
            return 1320

    def _row_heights() -> tuple[int | None, int | None]:
        w = _viewport_width()
        if w < 980:
            return None, None
        if w < 1260:
            return ROW_ANALYTICS_H_TIGHT, ROW_INSIGHTS_H_TIGHT
        return ROW_ANALYTICS_H, ROW_INSIGHTS_H

    def build():
        row_analytics_h, row_insights_h = _row_heights()
        return ft.Column([
            make_section_title("Dashboard", ft.Icons.DASHBOARD, ACCENT),
            make_info_box(
                "Panoramica generale: meteo di oggi, qualità previsioni, e migliori opportunità di scommessa. "
                "Verde = positivo, Rosso = negativo, Giallo = attenzione.",
            ),
            ft.Container(height=6),
            dash_verdict,
            ft.Container(height=6),
            # Cosa fare oggi
            make_card(ft.Column([
                ft.Row([
                    ft.Icon(ft.Icons.CHECKLIST, color=GREEN, size=18),
                    ft.Text("Cosa Fare Oggi", size=TYPE_LG, weight=ft.FontWeight.BOLD, color=TEXT),
                ], spacing=6),
                ft.Container(height=4),
                dash_action_plan,
            ]), border_color=ft.Colors.with_opacity(0.15, GREEN)),
            ft.Container(height=8),
            # Hero + stats
            make_card(dash_hero),
            ft.Container(height=10),
            dash_stats_row1,
            ft.Container(height=6),
            dash_stats_row2,
            ft.Container(height=10),
            # P&L + Calibrazione + Sparkline
            ft.ResponsiveRow([
                ft.Column([
                    make_card(ft.Column([
                        ft.Row([
                            ft.Icon(ft.Icons.ACCOUNT_BALANCE_WALLET, color=GREEN, size=18),
                            ft.Text("P&L Scommesse", size=TYPE_LG, weight=ft.FontWeight.BOLD, color=TEXT),
                        ], spacing=6),
                        make_info_box("Il tuo portafoglio scommesse: quanto hai vinto/perso, quante scommesse attive.", GREEN),
                        ft.Container(height=4),
                        dash_pnl_card,
                    ]), border_color=ft.Colors.with_opacity(0.15, GREEN), height=row_analytics_h),
                ], col={"xs": 12, "md": 4}),
                ft.Column([
                    make_card(ft.Column([
                        ft.Row([
                            ft.Icon(ft.Icons.VERIFIED, color=ACCENT, size=18),
                            ft.Text("Calibrazione Modello", size=TYPE_LG, weight=ft.FontWeight.BOLD, color=TEXT),
                        ], spacing=6),
                        make_info_box("Quanto sono precise le nostre previsioni? MAE basso = previsioni accurate.", ACCENT),
                        ft.Container(height=4),
                        dash_calibration_card,
                    ]), height=row_analytics_h),
                ], col={"xs": 12, "md": 4}),
                ft.Column([
                    make_card(ft.Column([
                        ft.Text("Temperatura 7 giorni", size=TYPE_MD, color=TEXT_DIM,
                                 weight=ft.FontWeight.BOLD),
                        ft.Container(height=4),
                        dash_sparkline,
                    ]), height=row_analytics_h),
                ], col={"xs": 12, "md": 4}),
            ]),
            ft.Container(height=10),
            # Rischio Portfolio
            make_card(ft.Column([
                ft.Row([
                    ft.Icon(ft.Icons.SHIELD, color=ACCENT2, size=18),
                    ft.Text("Rischio Portfolio", size=TYPE_LG, weight=ft.FontWeight.BOLD, color=TEXT),
                ], spacing=6),
                make_info_box("Gestione del rischio: esposizione, concentrazione, drawdown e streak.", ACCENT2),
                ft.Container(height=4),
                dash_risk_card,
            ]), border_color=ft.Colors.with_opacity(0.15, ACCENT2)),
            ft.Container(height=10),
            # Panoramica tutte le città
            make_card(ft.Column([
                ft.Row([
                    ft.Icon(ft.Icons.PUBLIC, color=ACCENT, size=18),
                    ft.Text("Panoramica Città", size=TYPE_LG, weight=ft.FontWeight.BOLD, color=TEXT),
                    ft.Container(expand=True),
                    ft.Text("Temp • Precip • Spread • Vento", size=TYPE_XS, color=TEXT_DIM),
                ], spacing=6),
                ft.Container(height=4),
                dash_cities_overview,
            ])),
            ft.Container(height=10),
            # Alerts + Top opps + Best confidence
            ft.ResponsiveRow([
                ft.Column([
                    make_card(ft.Column([
                        ft.Row([
                            ft.Icon(ft.Icons.WARNING_AMBER, color=YELLOW, size=18),
                            ft.Text("Alert & Anomalie", size=TYPE_LG, weight=ft.FontWeight.BOLD, color=TEXT),
                        ], spacing=6),
                        make_info_box("Temperature anomale rispetto alla media stagionale. Anomalie = opportunità di scommessa.", YELLOW),
                        ft.Container(height=4),
                        dash_alerts,
                    ]), height=row_insights_h),
                ], col={"xs": 12, "md": 4}),
                ft.Column([
                    make_card(ft.Column([
                        ft.Row([
                            ft.Icon(ft.Icons.TRENDING_UP, color=GREEN, size=18),
                            ft.Text("Top 3 Opportunità", size=TYPE_LG, weight=ft.FontWeight.BOLD, color=TEXT),
                            ft.Container(expand=True),
                            ft.Text("LIVE + SIM", size=TYPE_XS, color=TEXT_DIM),
                        ], spacing=6),
                        make_info_box("Le scommesse con il vantaggio più alto. Badge LIVE=Polymarket reale, SIM=analisi simulata.", GREEN),
                        ft.Container(height=4),
                        dash_top_opps,
                    ]), height=row_insights_h),
                ], col={"xs": 12, "md": 4}),
                ft.Column([
                    make_card(ft.Column([
                        ft.Row([
                            ft.Icon(ft.Icons.SHIELD, color=ACCENT2, size=18),
                            ft.Text("Miglior Confidenza", size=TYPE_LG, weight=ft.FontWeight.BOLD, color=TEXT),
                        ], spacing=6),
                        make_info_box("La scommessa in cui siamo più sicuri della nostra analisi.", ACCENT2),
                        ft.Container(height=4),
                        dash_best_confidence,
                    ]), height=row_insights_h),
                ], col={"xs": 12, "md": 4}),
            ]),
            ft.Container(height=10),
            # Convergenza Forecast
            make_card(ft.Column([
                ft.Row([
                    ft.Icon(ft.Icons.COMPARE_ARROWS, color=ACCENT, size=18),
                    ft.Text("Convergenza Forecast", size=TYPE_LG, weight=ft.FontWeight.BOLD, color=TEXT),
                ], spacing=6),
                make_info_box(
                    "Se i modelli successivi convergono (spread diminuisce), la previsione è più affidabile. "
                    "Divergente = meno affidabile.", ACCENT,
                ),
                ft.Container(height=4),
                dash_convergence,
            ])),
        ], scroll=ft.ScrollMode.AUTO, spacing=0)

    def load():
        """Carica dati dashboard."""
        try:
            _load_impl()
        except Exception as e:
            import logging
            logging.getLogger("weather_trader.dashboard").error("Dashboard load error: %s", e)
            dash_verdict.content = make_empty_state(
                ft.Icons.ERROR_OUTLINE, f"Errore caricamento dashboard: {e}")
            safe_update()

    def _load_impl():
        # Health check
        health = api_get("/health")
        if health and health.get("status") == "ok":
            if hasattr(state, "status_dot"):
                state.status_dot.bgcolor = GREEN
            if hasattr(state, "status_text"):
                state.status_text.value = "Connesso"
            if hasattr(state, "status_hint_text") and state.status_hint_text is not None:
                state.status_hint_text.value = "Dashboard sincronizzata"
        else:
            if hasattr(state, "status_dot"):
                state.status_dot.bgcolor = RED
            if hasattr(state, "status_text"):
                state.status_text.value = "Offline"
            if hasattr(state, "status_hint_text") and state.status_hint_text is not None:
                state.status_hint_text.value = "Engine non raggiungibile"
            safe_update()
            return

        # Forecast
        data = api_get(f"/forecast/{state.current_city}?days=14")
        state.cache["forecast"] = data
        forecasts = (data or {}).get("forecast", [])
        city_info = (data or {}).get("city", {})
        city_name = city_info.get("name", CITY_MAP.get(state.current_city, state.current_city))

        # Metrics
        metrics = api_get("/metrics")
        state.cache["metrics"] = metrics

        if metrics:
            fc_h = metrics.get("forecasts_hourly", 0)
            ens = metrics.get("ensemble_members", 0)
            obs = metrics.get("observations", 0)
            nrm = metrics.get("climate_normals", 0)
            if hasattr(state, "sb_db_text"):
                state.sb_db_text.value = f"DB: {fc_h:,} forecast  |  {ens:,} ensemble  |  {obs:,} osservazioni  |  {nrm:,} normals"
            if hasattr(state, "sb_update_text"):
                state.sb_update_text.value = f"Ultimo aggiornamento: {datetime.now().strftime('%H:%M:%S')}"

        if not forecasts:
            dash_hero.content = make_empty_state(ft.Icons.CLOUD_OFF, "Nessun forecast disponibile")
            safe_update()
            return

        today = forecasts[0]
        t_max = today.get("temp_max", 0)
        t_min = today.get("temp_min", 0)
        precip = today.get("precipitation_sum", 0) or 0
        wind = today.get("wind_max", 0) or 0
        uv = today.get("uv_max", 0) or 0
        ens_data = today.get("ensemble", {})
        ens_n = ens_data.get("n_members", "—")
        ens_spread = (ens_data.get("ensemble_max", 0) or 0) - (ens_data.get("ensemble_min", 0) or 0)
        ens_std = ens_data.get("ensemble_std", 0) or 0
        obs_count = (metrics or {}).get("observations", 0)

        # Condizioni meteo
        condition = "Sereno"
        cond_icon = ft.Icons.WB_SUNNY
        cond_color = YELLOW
        if precip > 5:
            condition = "Pioggia"
            cond_icon = ft.Icons.WATER_DROP
            cond_color = "#26c6da"
        elif precip > 0:
            condition = "Pioggia leggera"
            cond_icon = ft.Icons.GRAIN
            cond_color = "#26c6da"
        elif wind > 30:
            condition = "Ventoso"
            cond_icon = ft.Icons.AIR
            cond_color = YELLOW

        # Valori derivati usati da hero e stat chips
        temp_max_c = f2c(t_max)
        temp_min_c = f2c(t_min)
        spread_c = ens_spread * 5 / 9

        # Hero Card
        confidence_hint = "Alta" if spread_c < 3 else ("Media" if spread_c < 6 else "Bassa")
        conf_color = GREEN if spread_c < 3 else (YELLOW if spread_c < 6 else RED)

        hero_left = ft.Column([
            ft.Row([
                ft.Text(city_name, size=TYPE_LG, color=TEXT, weight=ft.FontWeight.W_700),
                ft.Container(width=6),
                make_badge(
                    confidence_hint,
                    ft.Colors.with_opacity(0.18, conf_color),
                    conf_color,
                ),
            ], spacing=0, wrap=True),
            ft.Row([
                ft.Text(
                    f"{f2c(t_max):.0f}°C",
                    size=TYPE_DISPLAY if _viewport_width() >= 1180 else TYPE_DISPLAY_COMPACT,
                    color=TEXT,
                    weight=ft.FontWeight.BOLD,
                ),
                ft.Column([
                    ft.Text(f"/{f2c(t_min):.0f}°C", size=TYPE_XL, color=TEXT_DIM),
                    ft.Row([
                        ft.Icon(cond_icon, color=cond_color, size=18),
                        ft.Text(condition, size=TYPE_MD, color=cond_color, weight=ft.FontWeight.W_600),
                    ], spacing=4),
                ], spacing=3),
            ], spacing=8, vertical_alignment=ft.CrossAxisAlignment.END),
            ft.Text(today.get("date", datetime.now().strftime("%Y-%m-%d")), size=TYPE_MD, color=TEXT_DIM),
            ft.Container(height=4),
            ft.Row([
                make_badge(f"Spread ±{spread_c:.1f}°C", ft.Colors.with_opacity(0.14, ORANGE), ORANGE),
                make_badge(f"Ensemble {ens_n}", ft.Colors.with_opacity(0.14, ACCENT2), ACCENT2),
                make_badge(f"Pioggia {precip:.1f}mm", ft.Colors.with_opacity(0.14, "#26c6da"), "#26c6da"),
            ], spacing=6, wrap=True),
        ], spacing=3, expand=True)

        hero_right = ft.Container(
            content=ft.Column([
                ft.Container(
                    content=ft.Icon(cond_icon, size=58, color=cond_color),
                    width=88,
                    height=88,
                    border_radius=18,
                    bgcolor=ft.Colors.with_opacity(0.10, cond_color),
                    border=ft.Border.all(1, ft.Colors.with_opacity(0.18, cond_color)),
                    alignment=ft.Alignment(0, 0),
                ),
                ft.Container(height=4),
                build_consensus_indicator(ens_std, width=190),
                ft.Text("Consensus ensemble", size=TYPE_XS, color=TEXT_DIM),
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=3),
            padding=pad(h=6, v=4),
        )

        dash_hero.content = ft.Container(
            content=ft.ResponsiveRow([
                ft.Column([hero_left], col={"xs": 12, "md": 8}),
                ft.Column([hero_right], col={"xs": 12, "md": 4}),
            ], spacing=8, run_spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            padding=pad(h=4, v=2),
            border_radius=14,
            bgcolor=ft.Colors.with_opacity(0.10, ACCENT),
            border=ft.Border.all(1, ft.Colors.with_opacity(0.10, ACCENT)),
        )
        dash_hero.height = 210 if _viewport_width() >= 980 else None

        # 8 stat chips (2 righe da 4) con sottotitoli esplicativi
        # Contesto temperatura
        if temp_max_c > 30:
            temp_comment = "molto caldo"
        elif temp_max_c > 20:
            temp_comment = "piacevole"
        elif temp_max_c > 10:
            temp_comment = "fresco"
        elif temp_max_c > 0:
            temp_comment = "freddo"
        else:
            temp_comment = "gelido"

        precip_comment = "nessuna" if precip == 0 else ("leggera" if precip < 5 else ("moderata" if precip < 15 else "forte"))
        wind_comment = "calmo" if wind < 10 else ("moderato" if wind < 25 else ("forte" if wind < 40 else "tempesta"))
        uv_comment = "basso" if uv < 3 else ("moderato" if uv < 6 else ("alto" if uv < 8 else "molto alto"))
        ens_comment = f"{ens_n} modelli" if isinstance(ens_n, int) else ""
        spread_comment = "previsione sicura" if spread_c < 3 else ("incertezza media" if spread_c < 6 else "molta incertezza")

        dash_stats_row1.controls = [
            ft.Column([make_stat_chip("Temp Max", f"{temp_max_c:.1f}°C", ACCENT, ft.Icons.THERMOSTAT, temp_comment)], col={"xs": 6, "md": 3}),
            ft.Column([make_stat_chip("Temp Min", f"{temp_min_c:.1f}°C", "#42a5f5", ft.Icons.AC_UNIT)], col={"xs": 6, "md": 3}),
            ft.Column([make_stat_chip("Pioggia", f"{precip:.1f} mm", "#26c6da", ft.Icons.WATER_DROP, precip_comment)], col={"xs": 6, "md": 3}),
            ft.Column([make_stat_chip("Vento", f"{wind:.0f} mph", YELLOW, ft.Icons.AIR, wind_comment)], col={"xs": 6, "md": 3}),
        ]
        dash_stats_row2.controls = [
            ft.Column([make_stat_chip("UV", f"{uv:.1f}", uv_color(uv), ft.Icons.WB_SUNNY, uv_comment)], col={"xs": 6, "md": 3}),
            ft.Column([make_stat_chip("Ensemble", f"{ens_n}", ACCENT2, ft.Icons.GROUPS, ens_comment)], col={"xs": 6, "md": 3}),
            ft.Column([make_stat_chip("Spread", f"±{spread_c:.1f}°", ORANGE, ft.Icons.SWAP_VERT, spread_comment)], col={"xs": 6, "md": 3}),
            ft.Column([make_stat_chip("Osservazioni", f"{obs_count:,}", GREEN, ft.Icons.HISTORY, "dati storici")], col={"xs": 6, "md": 3}),
        ]

        # === Verdetto giornaliero ===
        verdict_parts = []
        if temp_max_c > 30:
            verdict_parts.append("Giornata molto calda")
        elif temp_max_c < 5:
            verdict_parts.append("Giornata fredda")
        else:
            verdict_parts.append(f"Temperatura {temp_comment}")
        if precip > 5:
            verdict_parts.append("con pioggia significativa")
        elif precip > 0:
            verdict_parts.append("con pioggia leggera")
        if wind > 30:
            verdict_parts.append("e vento forte")
        if spread_c > 6:
            verdict_parts.append("— Previsioni incerte, attenzione alle scommesse!")
            v_color, v_icon = YELLOW, ft.Icons.WARNING_AMBER
        elif spread_c < 3:
            verdict_parts.append("— Previsioni affidabili, buon momento per scommettere")
            v_color, v_icon = GREEN, ft.Icons.THUMB_UP
        else:
            verdict_parts.append("— Incertezza nella media")
            v_color, v_icon = ACCENT, ft.Icons.INFO
        dash_verdict.content = make_verdict_banner(
            ". ".join(verdict_parts) + ".", v_color, v_icon)

        # === ANOMALIE — caricamento PRIMA dell'action plan (bugfix) ===
        anomalies = api_get(f"/analysis/{state.current_city}/anomaly?variable=temperature_2m_max&days=30")
        anom_list = (anomalies or {}).get("anomalies", [])

        # === COSA FARE OGGI ===
        action_items = []
        # Step 1: Valuta la situazione
        if spread_c < 3:
            action_items.append(ft.Row([
                ft.Icon(ft.Icons.CHECK_CIRCLE, color=GREEN, size=16),
                ft.Text("Le previsioni sono affidabili oggi — buon momento per cercare scommesse",
                         size=TYPE_SM, color=TEXT, expand=True),
            ], spacing=6))
        elif spread_c < 6:
            action_items.append(ft.Row([
                ft.Icon(ft.Icons.WARNING_AMBER, color=YELLOW, size=16),
                ft.Text("Incertezza moderata — scommetti solo su mercati con segnale SCOMMETTI",
                         size=TYPE_SM, color=TEXT, expand=True),
            ], spacing=6))
        else:
            action_items.append(ft.Row([
                ft.Icon(ft.Icons.BLOCK, color=RED, size=16),
                ft.Text("Troppa incertezza oggi — meglio aspettare domani o scommettere su altre città",
                         size=TYPE_SM, color=TEXT, expand=True),
            ], spacing=6))

        # Step 2: Suggerimento concreto
        action_items.append(ft.Row([
            ft.Icon(ft.Icons.ARROW_FORWARD, color=ACCENT, size=16),
            ft.Text("Vai su Mercati → filtro 'Solo SCOMMETTI' → guarda i mercati con countdown OGGI o DOMANI",
                     size=TYPE_SM, color=TEXT, expand=True),
        ], spacing=6))

        # Step 3: Bankroll check
        pnl_check = get_pnl_stats()
        bk = pnl_check["bankroll"]
        if bk < 100:
            action_items.append(ft.Row([
                ft.Icon(ft.Icons.ERROR, color=RED, size=16),
                ft.Text(f"Bankroll basso (${bk:.0f}) — scommetti molto poco o aspetta di ricaricare",
                         size=TYPE_SM, color=RED, expand=True),
            ], spacing=6))
        elif pnl_check["pending"] > 5:
            action_items.append(ft.Row([
                ft.Icon(ft.Icons.HOURGLASS_BOTTOM, color=YELLOW, size=16),
                ft.Text(f"Hai {pnl_check['pending']} scommesse aperte — prima risolvi quelle (Storico → pulsanti V/X)",
                         size=TYPE_SM, color=TEXT, expand=True),
            ], spacing=6))

        # Step 4: Anomalie = opportunità
        if anom_list:
            action_items.append(ft.Row([
                ft.Icon(ft.Icons.LIGHTBULB, color=YELLOW, size=16),
                ft.Text(f"Ci sono {len(anom_list)} anomalie meteo — controlla se ci sono mercati legati a queste date",
                         size=TYPE_SM, color=TEXT, expand=True),
            ], spacing=6))

        dash_action_plan.content = ft.Column(action_items, spacing=6)

        # Sparkline 7 giorni
        temps_7d = [fc.get("temp_max", 0) for fc in forecasts[:7]]
        dash_sparkline.content = ft.Column([
            build_sparkline(temps_7d, width=250, height=40, color=ACCENT),
            ft.Container(height=2),
            make_info_box("Andamento temperatura massima nei prossimi 7 giorni"),
        ], spacing=2)

        # === Panoramica tutte le città con SEMAFORO ===
        # Batch: single call instead of 24 individual calls
        overview = api_get("/forecast/overview/all")
        city_forecast_map = {}
        if overview and "cities" in overview:
            for entry in overview["cities"]:
                slug_key = entry.get("city", {}).get("slug", "")
                if slug_key:
                    city_forecast_map[slug_key] = entry.get("forecast", [])

        city_rows = []
        for slug, name in zip(CITY_SLUGS, CITY_NAMES):
            fc_list = city_forecast_map.get(slug, [])
            if not fc_list:
                continue
            d = fc_list[0]
            t_max_c = f2c(d.get("temp_max", 0))
            t_min_c = f2c(d.get("temp_min", 0))
            precip_v = d.get("precipitation_sum", 0) or 0
            wind_v = d.get("wind_max", 0) or 0
            ens = d.get("ensemble", {})
            spread_v = ((ens.get("ensemble_max", 0) or 0) - (ens.get("ensemble_min", 0) or 0)) * 5 / 9
            std_v = (ens.get("ensemble_std", 0) or 0) * 5 / 9
            # Colore spread
            sp_col = GREEN if spread_v < 3 else (YELLOW if spread_v < 6 else RED)

            # === SEMAFORO ===
            # Verde = previsione affidabile, puoi scommettere
            # Giallo = incerto, scommetti con cautela
            # Rosso = troppo incerto, non scommettere
            problems = 0
            if std_v > 3:
                problems += 1
            if std_v > 5:
                problems += 1
            if wind_v > 30:
                problems += 1
            if not ens:
                problems += 2  # niente ensemble = pericolo

            if problems == 0:
                sem_icon = ft.Icons.CIRCLE
                sem_color = GREEN
                sem_label = "OK"
            elif problems <= 1:
                sem_icon = ft.Icons.CIRCLE
                sem_color = YELLOW
                sem_label = "ATT"
            else:
                sem_icon = ft.Icons.CIRCLE
                sem_color = RED
                sem_label = "NO"

            is_current = slug == state.current_city
            city_rows.append(ft.Container(
                content=ft.Row([
                    ft.Icon(sem_icon, color=sem_color, size=10),
                    ft.Text(name, size=TYPE_SM, color=ACCENT if is_current else TEXT,
                             weight=ft.FontWeight.BOLD if is_current else ft.FontWeight.NORMAL,
                             width=80),
                    ft.Text(f"{t_max_c:.0f}/{t_min_c:.0f}°C", size=TYPE_SM, color=TEXT, width=62),
                    ft.Text(f"{precip_v:.1f}mm", size=TYPE_XS,
                             color="#26c6da" if precip_v > 0 else TEXT_DIM, width=48),
                    ft.Text(f"{wind_v:.0f}mph", size=TYPE_XS,
                             color=YELLOW if wind_v > 25 else TEXT_DIM, width=48),
                    ft.Container(
                        content=ft.Text(f"±{std_v:.1f}°", size=TYPE_2XS, color=BG,
                                         weight=ft.FontWeight.BOLD),
                        bgcolor=sp_col, border_radius=4, padding=pad(h=4, v=1), width=45,
                    ),
                    ft.Container(
                        content=ft.Text(sem_label, size=TYPE_2XS, color=BG, weight=ft.FontWeight.BOLD),
                        bgcolor=sem_color, border_radius=3, padding=pad(h=4, v=1), width=30,
                    ),
                ], spacing=3),
                bgcolor=ft.Colors.with_opacity(0.06, ACCENT) if is_current else None,
                border_radius=4,
                padding=pad(h=6, v=3),
            ))
        if city_rows:
            # Header
            hdr = ft.Container(
                content=ft.Row([
                    ft.Text("", width=10),
                    ft.Text("Città", size=TYPE_2XS, color=TEXT_DIM, weight=ft.FontWeight.BOLD, width=80),
                    ft.Text("Max/Min", size=TYPE_2XS, color=TEXT_DIM, weight=ft.FontWeight.BOLD, width=62),
                    ft.Text("Pioggia", size=TYPE_2XS, color=TEXT_DIM, weight=ft.FontWeight.BOLD, width=48),
                    ft.Text("Vento", size=TYPE_2XS, color=TEXT_DIM, weight=ft.FontWeight.BOLD, width=48),
                    ft.Text("Spread", size=TYPE_2XS, color=TEXT_DIM, weight=ft.FontWeight.BOLD, width=45),
                    ft.Text("Bet?", size=TYPE_2XS, color=TEXT_DIM, weight=ft.FontWeight.BOLD, width=30),
                ], spacing=3),
                padding=pad(h=6, v=2),
            )
            legend = ft.Row([
                ft.Icon(ft.Icons.CIRCLE, color=GREEN, size=8),
                ft.Text("OK = scommetti", size=TYPE_2XS, color=TEXT_DIM),
                ft.Icon(ft.Icons.CIRCLE, color=YELLOW, size=8),
                ft.Text("ATT = cautela", size=TYPE_2XS, color=TEXT_DIM),
                ft.Icon(ft.Icons.CIRCLE, color=RED, size=8),
                ft.Text("NO = non scommettere", size=TYPE_2XS, color=TEXT_DIM),
            ], spacing=4)
            dash_cities_overview.content = ft.Column([legend, hdr] + city_rows, spacing=1)
        else:
            dash_cities_overview.content = ft.Text("Nessun dato", color=TEXT_DIM, size=TYPE_MD)

        # Alert — anomalie (already loaded above before action plan)
        if anom_list:
            alert_items = [
                ft.Text("Temperature insolite rispetto alla media storica:",
                         size=TYPE_XS, color=TEXT_DIM, italic=True),
            ]
            for a in anom_list[:5]:
                z = a.get("z_score", 0)
                if abs(z) >= 2:
                    desc = "molto insolita"
                elif abs(z) >= 1.5:
                    desc = "insolita"
                else:
                    desc = "lieve deviazione"
                alert_items.append(ft.Row([
                    make_badge(desc, z_score_color(z), TEXT),
                    ft.Text(a.get("date", ""), size=TYPE_SM, color=TEXT_DIM),
                    ft.Text(f"{f2c(a.get('value', 0)):.1f}°C", size=TYPE_SM, color=TEXT),
                ], spacing=6))
            dash_alerts.content = ft.Column(alert_items, spacing=4)
        else:
            dash_alerts.content = ft.Column([
                ft.Icon(ft.Icons.CHECK_CIRCLE, color=GREEN, size=24),
                ft.Text("Nessuna anomalia rilevata", color=GREEN, size=TYPE_MD),
                ft.Text("Il meteo è nella norma per questa stagione", color=TEXT_DIM, size=TYPE_XS),
            ], spacing=4, horizontal_alignment=ft.CrossAxisAlignment.CENTER)

        # Top 3 opportunità
        scan_live = api_get("/market/scan", timeout=20)
        opps = api_get("/market/opportunities", timeout=30)
        live_opps_list = (scan_live or {}).get("markets", [])
        sim_opps_list = (opps or {}).get("opportunities", [])
        opps_list = live_opps_list + sim_opps_list
        dash_top_opps.controls.clear()
        if opps_list:
            for opp in opps_list[:3]:
                rec = opp.get("recommendation", {})
                mkt = opp.get("market", {})
                edge = rec.get("expected_value", 0)
                bet = rec.get("best_bet", "—")
                q = mkt.get("question", "—")
                edge_color = GREEN if edge > 0 else RED
                is_live = bool(mkt.get("volume", 0) or opp in live_opps_list or opp.get("_source") == "polymarket")
                dash_top_opps.controls.append(ft.Container(
                    content=ft.Row([
                        ft.Column([
                            ft.Row([
                                make_badge(
                                    "LIVE" if is_live else "SIM",
                                    GREEN if is_live else ft.Colors.with_opacity(0.18, ORANGE),
                                    BG if is_live else ORANGE,
                                ),
                                ft.Text(q[:44], size=TYPE_SM, color=TEXT,
                                         overflow=ft.TextOverflow.ELLIPSIS, max_lines=1),
                            ], spacing=6, wrap=False),
                            ft.Text(f"Bet: {bet}", size=TYPE_XS, color=TEXT_DIM),
                        ], spacing=2, expand=True),
                        ft.Column([
                            ft.Container(
                                content=ft.Text(f"EV {edge:+.1%}", size=TYPE_SM, color=BG,
                                                 weight=ft.FontWeight.BOLD),
                                bgcolor=edge_color, border_radius=6, padding=pad(h=8, v=3),
                            ),
                            ft.Text(
                                f"Vol ${mkt.get('volume', 0):,.0f}" if is_live and (mkt.get("volume", 0) or 0) > 0 else "Analisi",
                                size=TYPE_2XS,
                                color=TEXT_DIM,
                            ),
                        ], spacing=2, horizontal_alignment=ft.CrossAxisAlignment.END),
                    ], vertical_alignment=ft.CrossAxisAlignment.CENTER),
                    bgcolor=ft.Colors.with_opacity(0.05, GREEN if is_live else ORANGE),
                    border=ft.Border.all(
                        1, ft.Colors.with_opacity(0.14, GREEN if is_live else ORANGE)
                    ),
                    border_radius=10,
                    padding=pad(h=10, v=8),
                ))
        else:
            dash_top_opps.controls.append(
                ft.Text("Nessuna opportunità disponibile", color=TEXT_DIM, size=TYPE_MD))

        # === Risk Card ===
        pnl_risk_data = load_pnl()
        risk_bets = pnl_risk_data.get("bets", [])
        risk_bankroll = pnl_risk_data.get("bankroll", INITIAL_BANKROLL)
        risk = check_portfolio_risk(risk_bets, risk_bankroll)
        state.cache["portfolio_risk"] = risk

        risk_items = []
        # Exposure gauge (text-based)
        exp_pct = risk["exposure_pct"]
        exp_color = GREEN if exp_pct < 0.3 else (YELLOW if exp_pct < 0.5 else RED)
        bar_width = min(200, max(4, int(exp_pct / 0.5 * 200)))
        risk_items.append(ft.Column([
            make_kv_row("Esposizione", f"${risk['exposure']:.0f} ({exp_pct:.0%} / 50%)", exp_color),
            ft.Container(
                content=ft.Stack([
                    ft.Container(width=200, height=8, bgcolor=ft.Colors.with_opacity(0.1, TEXT), border_radius=4),
                    ft.Container(width=bar_width, height=8, bgcolor=exp_color, border_radius=4),
                ]),
                width=200, height=8,
            ),
        ], spacing=2))

        # City concentration
        if risk["city_exposure"]:
            city_items = []
            for city, amount in sorted(risk["city_exposure"].items(), key=lambda x: x[1], reverse=True)[:5]:
                city_pct = amount / risk_bankroll if risk_bankroll > 0 else 0
                c_color = GREEN if city_pct < 0.10 else (YELLOW if city_pct < 0.15 else RED)
                city_items.append(ft.Row([
                    ft.Text(city, size=TYPE_XS, color=TEXT, width=70),
                    ft.Text(f"${amount:.0f} ({city_pct:.0%})", size=TYPE_XS, color=c_color),
                ], spacing=4))
            risk_items.append(ft.Column([
                ft.Text("Concentrazione città", size=TYPE_XS, color=TEXT_DIM, weight=ft.FontWeight.BOLD),
                *city_items,
            ], spacing=2))

        # Drawdown
        dd_color = GREEN if risk["max_drawdown"] < 0.15 else (YELLOW if risk["max_drawdown"] < 0.25 else RED)
        risk_items.append(make_kv_row("Max Drawdown", f"{risk['max_drawdown']:.1%}", dd_color))
        risk_items.append(make_kv_row("Drawdown corrente", f"{risk['current_drawdown']:.1%}",
                                       GREEN if risk["current_drawdown"] < 0.10 else dd_color))

        # Streak
        if risk["loss_streak"] > 0:
            risk_items.append(make_kv_row("Streak perdite", f"{risk['loss_streak']}",
                                           RED if risk["loss_streak"] >= 5 else (YELLOW if risk["loss_streak"] >= 3 else TEXT)))
        if risk["win_streak"] > 0:
            risk_items.append(make_kv_row("Streak vittorie", f"{risk['win_streak']}", GREEN))

        # Warnings
        for w in risk["warnings"]:
            w_color = RED if w["level"] == "critical" else YELLOW
            risk_items.append(ft.Row([
                ft.Icon(ft.Icons.WARNING, color=w_color, size=12),
                ft.Text(w["message"], size=TYPE_2XS, color=w_color),
            ], spacing=4))

        if not risk["can_bet"]:
            risk_items.insert(0, ft.Container(
                content=ft.Row([
                    ft.Icon(ft.Icons.BLOCK, color=RED, size=14),
                    ft.Text("SCOMMESSE BLOCCATE", size=TYPE_SM, color=BG, weight=ft.FontWeight.BOLD),
                ], spacing=4),
                bgcolor=RED, border_radius=6, padding=pad(h=8, v=4),
            ))
        else:
            risk_items.insert(0, ft.Container(
                content=ft.Row([
                    ft.Icon(ft.Icons.CHECK_CIRCLE, color=GREEN, size=14),
                    ft.Text("Puoi scommettere", size=TYPE_SM, color=GREEN),
                ], spacing=4),
            ))

        dash_risk_card.content = ft.Column(risk_items, spacing=4)

        # === P&L Card ===
        pnl = get_pnl_stats()
        pnl_color = GREEN if pnl["total_pnl"] >= 0 else RED
        bankroll = pnl["bankroll"]
        dash_pnl_card.content = ft.Column([
            ft.Row([
                ft.Text("Bankroll", size=TYPE_SM, color=TEXT_DIM),
                ft.Text(f"${bankroll:,.0f}", size=TYPE_METRIC, color=ACCENT, weight=ft.FontWeight.BOLD),
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            ft.Container(height=2),
            make_kv_row("P&L Totale", f"${pnl['total_pnl']:+,.0f}", pnl_color),
            make_kv_row("Oggi", f"${pnl['today_pnl']:+,.0f}",
                          GREEN if pnl["today_pnl"] >= 0 else RED),
            make_kv_row("Settimana", f"${pnl['week_pnl']:+,.0f}",
                          GREEN if pnl["week_pnl"] >= 0 else RED),
            make_kv_row("Scommesse", f"{pnl['total_bets']} ({pnl['pending']} pending)"),
            make_kv_row("Win Rate", f"{pnl['win_rate']:.0%}",
                          GREEN if pnl["win_rate"] > 0.5 else RED),
            make_kv_row("ROI", f"{pnl['roi']:.1%}", pnl_color),
            ft.Container(height=4),
            build_pnl_sparkline(load_pnl().get("bets", []), width=250, height=35),
        ], spacing=3)

        # === Calibrazione modello ===
        acc = api_get(f"/analysis/{state.current_city}/accuracy?variable=temperature_2m_max")
        acc_data = (acc or {}).get("accuracy")
        if acc_data and isinstance(acc_data, dict):
            mae = acc_data.get("mae", 0) or 0
            rmse = acc_data.get("rmse", 0) or 0
            bias = acc_data.get("bias", 0) or 0
            n_samples = acc_data.get("n_samples", 0)
            # Qualità calibrazione: MAE < 2 = ottimo, < 4 = buono, > 4 = scarso
            if mae < 2:
                cal_label, cal_color = "ECCELLENTE", GREEN
            elif mae < 4:
                cal_label, cal_color = "BUONA", YELLOW
            else:
                cal_label, cal_color = "DA MIGLIORARE", RED

            dash_calibration_card.content = ft.Column([
                ft.Row([
                    ft.Container(
                        content=ft.Text(cal_label, size=TYPE_XS, color=BG, weight=ft.FontWeight.BOLD),
                        bgcolor=cal_color, border_radius=4, padding=pad(h=6, v=2)),
                    ft.Text(f"n={n_samples}", size=TYPE_XS, color=TEXT_DIM),
                ], spacing=6),
                ft.Container(height=4),
                make_kv_row("MAE", f"{mae * 5 / 9:.2f}°C", cal_color),
                make_kv_row("RMSE", f"{rmse * 5 / 9:.2f}°C",
                              YELLOW if rmse > 5 else TEXT),
                make_kv_row("Bias", f"{bias * 5 / 9:+.2f}°C",
                              RED if abs(bias) > 2 else TEXT),
                ft.Container(height=4),
                ft.Text("Bias < 0 = sottostima, > 0 = sovrastima",
                         size=TYPE_2XS, color=TEXT_DIM, italic=True),
            ], spacing=3)
        else:
            dash_calibration_card.content = ft.Text(
                "Dati accuratezza non disponibili", color=TEXT_DIM, size=TYPE_MD)

        # === Convergence ===
        conv_data = api_get(f"/forecast/{state.current_city}?days=3")
        conv_forecasts = (conv_data or {}).get("forecast", [])
        if conv_forecasts:
            # Compute simple convergence from ensemble spread trend
            conv_items = []
            prev_std = None
            for fc in conv_forecasts[:3]:
                fc_date = fc.get("date", "?")
                fc_ens = fc.get("ensemble", {})
                std = (fc_ens.get("ensemble_std") or 0) * 5 / 9  # to Celsius
                n_m = fc_ens.get("n_members", 0)
                if prev_std is not None and std > 0:
                    if std < prev_std * 0.9:
                        trend_icon = ft.Icons.TRENDING_DOWN
                        trend_color = GREEN
                        trend_text = "convergente"
                    elif std > prev_std * 1.1:
                        trend_icon = ft.Icons.TRENDING_UP
                        trend_color = RED
                        trend_text = "divergente"
                    else:
                        trend_icon = ft.Icons.TRENDING_FLAT
                        trend_color = YELLOW
                        trend_text = "stabile"
                else:
                    trend_icon = ft.Icons.REMOVE
                    trend_color = TEXT_DIM
                    trend_text = "—"
                conv_items.append(ft.Row([
                    ft.Text(fc_date, size=TYPE_XS, color=TEXT_DIM, width=80),
                    ft.Text(f"±{std:.1f}°C", size=TYPE_XS, color=TEXT, width=55),
                    ft.Icon(trend_icon, color=trend_color, size=14),
                    ft.Text(trend_text, size=TYPE_XS, color=trend_color),
                ], spacing=4))
                prev_std = std
            dash_convergence.content = ft.Column(conv_items, spacing=4)
        else:
            dash_convergence.content = ft.Text(
                "Dati convergenza non disponibili", color=TEXT_DIM, size=TYPE_MD)

        # === Best confidence opportunity ===
        if opps_list:
            best_opp = None
            best_conf = None
            best_score = 0
            for opp in opps_list[:10]:
                rec = opp.get("recommendation", {})
                meta = opp.get("metadata", {})
                outcomes = rec.get("outcomes", [])
                max_edge = max((o.get("edge", 0) for o in outcomes), default=0)
                ens_d = forecasts[0].get("ensemble", {}) if forecasts else {}
                conf = calculate_confidence(ens_d, max_edge, acc_data, len(outcomes))
                if conf["total"] > best_score:
                    best_score = conf["total"]
                    best_conf = conf
                    best_opp = opp
            if best_opp and best_conf:
                rec = best_opp.get("recommendation", {})
                mkt = best_opp.get("market", {})
                is_live_best = bool(mkt.get("volume", 0) or best_opp in live_opps_list)
                dash_best_confidence.content = ft.Column([
                    ft.Row([
                        make_badge(
                            "LIVE" if is_live_best else "SIM",
                            GREEN if is_live_best else ft.Colors.with_opacity(0.18, ORANGE),
                            BG if is_live_best else ORANGE,
                        ),
                        ft.Text(mkt.get("question", "—")[:38], size=TYPE_SM, color=TEXT,
                                 overflow=ft.TextOverflow.ELLIPSIS, max_lines=2, expand=True),
                    ], spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                    ft.Text(f"Bet: {rec.get('best_bet', '—')}", size=TYPE_XS, color=TEXT_DIM),
                    ft.Container(height=4),
                    build_confidence_meter(best_conf, width=250),
                ], spacing=4)
            else:
                dash_best_confidence.content = ft.Text(
                    "Nessun dato", color=TEXT_DIM, size=TYPE_MD)
        else:
            dash_best_confidence.content = ft.Text(
                "Nessuna opportunità", color=TEXT_DIM, size=TYPE_MD)

        safe_update()

    return build, load
