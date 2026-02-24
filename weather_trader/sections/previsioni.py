"""Previsioni (Forecast) section: temp chart, secondary charts, data table, climate, ensemble."""

import logging
import threading
from datetime import datetime

import flet as ft

from weather_trader.constants import (
    ACCENT, ACCENT2, BG, GREEN, ORANGE, RED, TEXT, TEXT_DIM, YELLOW, f2c,
)
from weather_trader.api_client import api_get
from weather_trader.app_state import AppState
from weather_trader.widgets.factory import (
    make_badge, make_card, make_empty_state, make_info_box, make_kv_row,
    make_loading_indicator, make_section_title, make_verdict_banner,
    pad, uv_color,
)
from weather_trader.widgets.charts import (
    build_temp_bar_chart, build_precip_chart, build_precip_probability_chart,
    build_ensemble_spread_chart, build_temp_range_chart, build_snow_chart,
    build_precip_hours_chart, build_wind_chart, build_uv_chart,
)
from weather_trader.widgets.distribution import build_consensus_indicator
from weather_trader.widgets.confidence import calculate_confidence

logger = logging.getLogger("weather_trader.sections.previsioni")


def create_previsioni(page: ft.Page, state: AppState, safe_update):
    """Create previsioni section. Returns (build, load) tuple."""

    ROW_SECONDARY_CHARTS_H = 210
    ROW_MODEL_CARDS_H = 260

    # --- Containers ---
    prev_chart_container = ft.Container(
        content=make_loading_indicator("Caricamento forecast..."),
        height=340,
    )
    prev_secondary_charts = ft.ResponsiveRow([])
    prev_secondary_insights = ft.ResponsiveRow([], spacing=2, run_spacing=2)
    prev_table = ft.DataTable(
        columns=[
            ft.DataColumn(ft.Text("Data", color=TEXT_DIM, size=12, weight=ft.FontWeight.BOLD)),
            ft.DataColumn(ft.Text("Max °C", color=TEXT_DIM, size=12, weight=ft.FontWeight.BOLD)),
            ft.DataColumn(ft.Text("Min °C", color=TEXT_DIM, size=12, weight=ft.FontWeight.BOLD)),
            ft.DataColumn(ft.Text("Media", color=TEXT_DIM, size=12, weight=ft.FontWeight.BOLD)),
            ft.DataColumn(ft.Text("Precip", color=TEXT_DIM, size=12, weight=ft.FontWeight.BOLD)),
            ft.DataColumn(ft.Text("P.Prob", color=TEXT_DIM, size=12, weight=ft.FontWeight.BOLD)),
            ft.DataColumn(ft.Text("Vento", color=TEXT_DIM, size=12, weight=ft.FontWeight.BOLD)),
            ft.DataColumn(ft.Text("Raff.", color=TEXT_DIM, size=12, weight=ft.FontWeight.BOLD)),
            ft.DataColumn(ft.Text("UV", color=TEXT_DIM, size=12, weight=ft.FontWeight.BOLD)),
            ft.DataColumn(ft.Text("Ens ±", color=TEXT_DIM, size=12, weight=ft.FontWeight.BOLD)),
        ],
        border_radius=12,
        heading_row_color=ft.Colors.with_opacity(0.05, TEXT),
        data_row_max_height=40,
    )
    if hasattr(prev_table, "column_spacing"):
        prev_table.column_spacing = 18
    if hasattr(prev_table, "divider_thickness"):
        prev_table.divider_thickness = 0.6
    if hasattr(prev_table, "horizontal_margin"):
        prev_table.horizontal_margin = 12
    prev_climate_card = ft.Container(padding=10)
    prev_ensemble_card = ft.Container(
        padding=10,
        height=170,
        content=make_loading_indicator("Caricamento ensemble..."),
    )
    prev_trading_quality_card = ft.Container(padding=10)
    prev_trading_keydays = ft.Column([], spacing=6, scroll=ft.ScrollMode.AUTO, height=250)
    prev_trading_risk_card = ft.Container(padding=10)
    prev_trading_matrix = ft.Column([], spacing=6, scroll=ft.ScrollMode.AUTO, height=420)
    prev_live_markets_list = ft.Column([], spacing=6, scroll=ft.ScrollMode.AUTO, height=320)
    prev_probability_result = ft.Container(
        padding=10,
        height=220,
        border_radius=12,
        bgcolor=ft.Colors.with_opacity(0.12, TEXT),
        border=ft.Border.all(1, ft.Colors.with_opacity(0.08, TEXT)),
        content=make_loading_indicator("Calcolo probabilità..."),
    )
    prev_verdict = ft.Container()
    prob_forecast_by_date: dict[str, dict] = {}

    prev_load_state_lock = threading.Lock()
    prev_load_running = False
    prev_load_pending = False

    def _threaded_load(fn):
        """Esegue fn in background thread."""
        def _run():
            try:
                fn()
            except Exception as e:
                logger.error("Threaded load error: %s", e)
        threading.Thread(target=_run, daemon=True).start()

    def on_days_toggle(e):
        sel = e.control.selected
        if sel:
            if isinstance(sel, list) and len(sel) > 0:
                val = sel[0]
            else:
                val = str(sel)
            try:
                state.forecast_days = int(val)
            except (ValueError, TypeError):
                state.forecast_days = 7
        _threaded_load(load)

    days_toggle = ft.SegmentedButton(
        segments=[
            ft.Segment(
                value="7",
                label=ft.Container(
                    content=ft.Text("7 giorni", text_align=ft.TextAlign.CENTER),
                    padding=pad(h=16, v=2),
                ),
            ),
            ft.Segment(
                value="14",
                label=ft.Container(
                    content=ft.Text("14 giorni", text_align=ft.TextAlign.CENTER),
                    padding=pad(h=16, v=2),
                ),
            ),
        ],
        selected=["7"],
        on_change=on_days_toggle,
    )

    prob_day_dropdown = ft.Dropdown(
        width=170,
        color=TEXT,
        border_color=ACCENT2,
        text_size=13,
        content_padding=pad(h=10, v=6),
        options=[],
        value=None,
    )
    prob_low_field = ft.TextField(
        width=110,
        label="Soglia bassa °C",
        hint_text="es. 18",
        value="",
        color=TEXT,
    )
    prob_high_field = ft.TextField(
        width=110,
        label="Soglia alta °C",
        hint_text="es. 22",
        value="",
        color=TEXT,
    )
    prob_calc_btn = ft.Button(
        "Calcola",
        icon=ft.Icons.CALCULATE,
        bgcolor=ACCENT2,
        color=BG,
        style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=10), padding=pad(h=10, v=10)),
    )

    def _c2f(c: float) -> float:
        return c * 9 / 5 + 32

    def _refresh_probability_day_options(fcs: list[dict]):
        nonlocal prob_forecast_by_date
        prob_forecast_by_date = {str(fc.get("date", "")): fc for fc in fcs if fc.get("date")}
        prob_day_dropdown.options = [
            ft.dropdown.Option(key=str(fc.get("date", "")), text=str(fc.get("date", ""))[5:])
            for fc in fcs if fc.get("date")
        ]
        if not prob_day_dropdown.options:
            prob_day_dropdown.value = None
            return
        valid_keys = [opt.key for opt in prob_day_dropdown.options]
        if prob_day_dropdown.value not in valid_keys:
            prob_day_dropdown.value = valid_keys[0]

        # Pre-fill suggested threshold band around forecast max for selected day (if empty)
        fc_sel = prob_forecast_by_date.get(prob_day_dropdown.value or "")
        if fc_sel:
            tmax_f = float(fc_sel.get("temp_max", 0) or 0)
            tmax_c = f2c(tmax_f)
            if not (prob_low_field.value or "").strip():
                prob_low_field.value = f"{tmax_c - 1.5:.1f}"
            if not (prob_high_field.value or "").strip():
                prob_high_field.value = f"{tmax_c + 1.5:.1f}"

    def _compute_probability_custom():
        target_date = prob_day_dropdown.value
        if not target_date:
            prev_probability_result.content = make_empty_state(
                ft.Icons.CALENDAR_MONTH, "Seleziona un giorno forecast"
            )
            safe_update()
            return

        try:
            low_c = float(str(prob_low_field.value or "").replace(",", "."))
            high_c = float(str(prob_high_field.value or "").replace(",", "."))
        except Exception:
            prev_probability_result.content = make_empty_state(
                ft.Icons.ERROR_OUTLINE, "Soglie non valide", "Inserisci numeri in °C (es. 18.5)"
            )
            safe_update()
            return

        if high_c <= low_c:
            prev_probability_result.content = make_empty_state(
                ft.Icons.SWAP_VERT, "Intervallo non valido", "La soglia alta deve essere maggiore della soglia bassa"
            )
            safe_update()
            return

        low_f = _c2f(low_c)
        high_f = _c2f(high_c)
        prev_probability_result.content = make_loading_indicator("Calcolo probabilità (KDE + ensemble + storico)...")
        safe_update()

        resp = api_get(
            f"/analysis/{state.current_city}/probability?"
            f"target_date={target_date}&variable=temperature_2m_max&"
            f"threshold_low={low_f:.3f}&threshold_high={high_f:.3f}",
            timeout=20,
        )

        est = (resp or {}).get("estimate") if isinstance(resp, dict) else None
        if not est:
            prev_probability_result.content = make_empty_state(
                ft.Icons.QUERY_STATS, "Probabilità non disponibile",
                "Mancano dati ensemble/storici per il giorno selezionato"
            )
            safe_update()
            return

        p = float(est.get("probability", est.get("blended_prob", 0)) or 0)
        p_lo = float(est.get("confidence_lower", 0) or 0)
        p_hi = float(est.get("confidence_upper", 0) or 0)
        p_ens = float(est.get("ensemble_prob", 0) or 0)
        p_hist = float(est.get("historical_prob", 0) or 0)
        p_det = float(est.get("deterministic_prob", 0) or 0)
        ens_spread_f = float(est.get("ensemble_spread", 0) or 0)
        ens_spread_c = ens_spread_f * 5 / 9

        p_color = GREEN if p >= 0.65 else (YELLOW if p >= 0.35 else RED)
        fc_sel = prob_forecast_by_date.get(target_date, {})
        forecast_tmax_c = f2c(fc_sel.get("temp_max", 0) or 0) if fc_sel else None
        n_ens = (resp or {}).get("n_ensemble_members", 0)
        n_hist = (resp or {}).get("n_historical_samples", 0)

        prev_probability_result.content = ft.Column([
            ft.Row([
                make_badge(f"{target_date[5:]}", ft.Colors.with_opacity(0.14, ACCENT2), ACCENT2),
                make_badge(f"Range {low_c:.1f}–{high_c:.1f}°C", ft.Colors.with_opacity(0.14, ACCENT), ACCENT),
                ft.Container(expand=True),
                make_badge(f"P {p:.0%}", ft.Colors.with_opacity(0.16, p_color), p_color),
            ], spacing=6, wrap=True),
            ft.Container(height=4),
            ft.Text(
                f"Probabilità stimata che Tmax cada nell'intervallo: {p:.1%}",
                size=12, color=TEXT, weight=ft.FontWeight.BOLD,
            ),
            ft.Text(
                f"CI approx: {p_lo:.1%} – {p_hi:.1%} · Ensemble spread: {ens_spread_c:.1f}°C",
                size=10, color=TEXT_DIM,
            ),
            ft.Container(height=4),
            ft.ProgressBar(value=max(0, min(1, p)), color=p_color, bgcolor=ft.Colors.with_opacity(0.10, TEXT)),
            ft.Container(height=6),
            make_kv_row("Forecast Tmax", f"{forecast_tmax_c:.1f}°C" if forecast_tmax_c is not None else "—", ACCENT),
            make_kv_row("Prob. ensemble", f"{p_ens:.1%}", ACCENT2),
            make_kv_row("Prob. storico", f"{p_hist:.1%}", TEXT_DIM),
            make_kv_row("Prob. deterministica", f"{p_det:.1%}", TEXT),
            make_kv_row("Membri ensemble", f"{n_ens}", GREEN if n_ens >= 20 else YELLOW),
            make_kv_row("Campioni storici", f"{n_hist}", GREEN if n_hist >= 50 else YELLOW),
            ft.Container(height=4),
            ft.Text(
                "Uso scommesse: confronta questa probabilità con il prezzo implicito del mercato (YES price). "
                "Edge = Pnostra - Pmercato.",
                size=10, color=TEXT_DIM, italic=True,
            ),
        ], spacing=3, scroll=ft.ScrollMode.AUTO, height=190)
        safe_update()

    def _on_prob_compute(e=None):
        _threaded_load(_compute_probability_custom)

    def _on_prob_day_change(e):
        target_date = prob_day_dropdown.value
        fc_sel = prob_forecast_by_date.get(target_date or "", {})
        if fc_sel:
            tmax_c = f2c(fc_sel.get("temp_max", 0) or 0)
            prob_low_field.value = f"{tmax_c - 1.5:.1f}"
            prob_high_field.value = f"{tmax_c + 1.5:.1f}"
            safe_update()
        _on_prob_compute()

    prob_calc_btn.on_click = _on_prob_compute
    prob_day_dropdown.on_select = _on_prob_day_change

    def build():
        return ft.Column([
            make_section_title("Previsioni", ft.Icons.CALENDAR_MONTH, ACCENT),
            ft.Container(height=4),
            ft.Row([
                ft.Container(
                    content=ft.Row(
                        [days_toggle],
                        alignment=ft.MainAxisAlignment.CENTER,
                        scroll=ft.ScrollMode.AUTO,
                    ),
                    border_radius=14,
                    bgcolor=ft.Colors.with_opacity(0.28, TEXT),
                    border=ft.Border.all(1, ft.Colors.with_opacity(0.10, TEXT)),
                    padding=pad(h=14, v=6),
                    expand=True,
                ),
            ]),
            make_info_box(
                "Previsioni meteo per i prossimi giorni. Le barre blu mostrano la temperatura prevista, "
                "le bande viola l'incertezza (ensemble). Banda stretta = previsione più sicura.",
            ),
            ft.Container(height=6),
            prev_verdict,
            ft.Container(height=6),
            # Grafico principale temperatura
            make_card(ft.Column([
                ft.Row([
                    ft.Text("Temperatura Forecast", size=15, weight=ft.FontWeight.BOLD, color=TEXT),
                    ft.Container(expand=True),
                    ft.Row([
                        ft.Container(width=12, height=12, bgcolor=ACCENT, border_radius=2),
                        ft.Text("Temp prevista", size=11, color=TEXT_DIM),
                        ft.Container(width=8),
                        ft.Container(width=12, height=12,
                                      bgcolor=ft.Colors.with_opacity(0.3, ACCENT2), border_radius=2),
                        ft.Text("Incertezza", size=11, color=TEXT_DIM),
                    ], spacing=4),
                ]),
                ft.Container(height=4),
                prev_chart_container,
                make_info_box("Ogni barra rappresenta un giorno. Altezza = temperatura. La banda viola mostra il range di incertezza dei modelli."),
            ])),
            ft.Container(height=10),
            # 4 grafici secondari
            prev_secondary_charts,
            ft.Container(height=8),
            prev_secondary_insights,
            ft.Container(height=10),
            # Trading desk (forecast diagnostics for betting)
            ft.ResponsiveRow([
                ft.Column([
                    make_card(ft.Column([
                        ft.Row([
                            ft.Icon(ft.Icons.FACT_CHECK, color=ACCENT2, size=18),
                            ft.Text("Qualità Forecast (Betting)", size=14, weight=ft.FontWeight.BOLD, color=TEXT),
                        ], spacing=6),
                        make_info_box("Affidabilità della previsione basata su accuracy storico, spread-skill e incertezza ensemble."),
                        ft.Container(height=4),
                        prev_trading_quality_card,
                    ]), height=380),
                ], col={"xs": 12, "md": 4}),
                ft.Column([
                    make_card(ft.Column([
                        ft.Row([
                            ft.Icon(ft.Icons.EVENT_AVAILABLE, color=GREEN, size=18),
                            ft.Text("Giorni Chiave", size=14, weight=ft.FontWeight.BOLD, color=TEXT),
                        ], spacing=6),
                        make_info_box("Selezione dei giorni più interessanti o rischiosi per mercati meteo temperatura."),
                        ft.Container(height=4),
                        prev_trading_keydays,
                    ]), height=380),
                ], col={"xs": 12, "md": 4}),
                ft.Column([
                    make_card(ft.Column([
                        ft.Row([
                            ft.Icon(ft.Icons.GPP_BAD, color=YELLOW, size=18),
                            ft.Text("Rischi & Flag", size=14, weight=ft.FontWeight.BOLD, color=TEXT),
                        ], spacing=6),
                        make_info_box("Conteggi rapidi dei rischi meteo che rendono le scommesse meno affidabili."),
                        ft.Container(height=4),
                        prev_trading_risk_card,
                    ]), height=380),
                ], col={"xs": 12, "md": 4}),
            ]),
            ft.Container(height=10),
            ft.ResponsiveRow([
                ft.Column([
                    make_card(ft.Column([
                        ft.Row([
                            ft.Icon(ft.Icons.PODCASTS, color=GREEN, size=18),
                            ft.Text("Mercati LIVE Collegati", size=14, weight=ft.FontWeight.BOLD, color=TEXT),
                        ], spacing=6),
                        make_info_box("Mercati Polymarket live trovati per città e date del forecast corrente."),
                        ft.Container(height=4),
                        prev_live_markets_list,
                    ]), height=460),
                ], col={"xs": 12, "md": 6}),
                ft.Column([
                    make_card(ft.Column([
                        ft.Row([
                            ft.Icon(ft.Icons.QUERY_STATS, color=ACCENT2, size=18),
                            ft.Text("Probabilità Custom (Reale)", size=14, weight=ft.FontWeight.BOLD, color=TEXT),
                        ], spacing=6),
                        make_info_box(
                            "Calcola P(Tmax in un intervallo) con ensemble + storico + deterministico "
                            "sul giorno selezionato. Usa poi il prezzo mercato per stimare l'edge."
                        ),
                        ft.Container(height=4),
                        ft.ResponsiveRow([
                            ft.Column([prob_day_dropdown], col={"xs": 12, "sm": 4}),
                            ft.Column([prob_low_field], col={"xs": 6, "sm": 3}),
                            ft.Column([prob_high_field], col={"xs": 6, "sm": 3}),
                            ft.Column([ft.Container(content=prob_calc_btn, padding=pad(v=20))], col={"xs": 12, "sm": 2}),
                        ], spacing=8, run_spacing=8),
                        ft.Container(height=4),
                        prev_probability_result,
                    ]), height=460),
                ], col={"xs": 12, "md": 6}),
            ]),
            ft.Container(height=10),
            make_card(ft.Column([
                ft.Row([
                    ft.Icon(ft.Icons.TABLE_CHART, color=ACCENT, size=18),
                    ft.Text("Matrice Trading Giornaliera", size=14, weight=ft.FontWeight.BOLD, color=TEXT),
                ], spacing=6),
                make_info_box(
                    "Valutazione giorno-per-giorno per scommesse meteo (temperatura): confidenza forecast, "
                    "incertezza ensemble, rischio pioggia/vento e semaforo operativo."
                ),
                ft.Container(height=4),
                prev_trading_matrix,
            ]), height=560),
            ft.Container(height=10),
            # Tabella
            make_card(ft.Column([
                ft.Text("Tabella Forecast Dettagliata", size=15, weight=ft.FontWeight.BOLD, color=TEXT),
                make_info_box("Tutti i numeri giorno per giorno. Precip = pioggia in mm, P.Prob = probabilità di pioggia, Raff. = raffiche di vento, Ens± = incertezza ensemble."),
                ft.Container(height=4),
                ft.Container(
                    content=prev_table,
                    height=420,
                    border_radius=14,
                    bgcolor=ft.Colors.with_opacity(0.18, TEXT),
                    border=ft.Border.all(1, ft.Colors.with_opacity(0.08, TEXT)),
                    padding=pad(h=6, v=6),
                ),
            ])),
            ft.Container(height=10),
            # Climate + Ensemble
            ft.ResponsiveRow([
                ft.Column([
                    make_card(ft.Column([
                        ft.Row([
                            ft.Icon(ft.Icons.THERMOSTAT, color=GREEN, size=18),
                            ft.Text("Climatologia", size=14, weight=ft.FontWeight.BOLD, color=TEXT),
                        ], spacing=6),
                        make_info_box("Confronto tra la temperatura prevista oggi e la media storica per questo periodo dell'anno.", GREEN),
                        ft.Container(height=4),
                        prev_climate_card,
                    ]), height=ROW_MODEL_CARDS_H),
                ], col={"xs": 12, "md": 6}),
                ft.Column([
                    make_card(ft.Column([
                        ft.Row([
                            ft.Icon(ft.Icons.GROUPS, color=ACCENT2, size=18),
                            ft.Text("Ensemble", size=14, weight=ft.FontWeight.BOLD, color=TEXT),
                        ], spacing=6),
                        make_info_box("L'ensemble è un gruppo di modelli meteo diversi. Se concordano, la previsione è più affidabile.", ACCENT2),
                        ft.Container(height=4),
                        prev_ensemble_card,
                    ]), height=ROW_MODEL_CARDS_H),
                ], col={"xs": 12, "md": 6}),
            ]),
        ], scroll=ft.ScrollMode.AUTO, spacing=0)

    def load():
        """Carica dati previsioni."""
        nonlocal prev_load_running, prev_load_pending

        with prev_load_state_lock:
            if prev_load_running:
                prev_load_pending = True
                logger.debug("Previsioni load skipped (already running), queued one rerun")
                return
            prev_load_running = True

        while True:
            try:
                _load_impl()
            except Exception as e:
                logger.error("Previsioni load error: %s", e)
                prev_verdict.content = make_empty_state(
                    ft.Icons.ERROR_OUTLINE, f"Errore caricamento previsioni: {e}")
                safe_update()

            with prev_load_state_lock:
                if prev_load_pending:
                    prev_load_pending = False
                    logger.debug("Previsioni load rerun (queued)")
                    continue
                prev_load_running = False
                break

    def _load_impl():
        days = state.forecast_days
        # Placeholder immediati per pannelli lenti (evita rettangoli vuoti mentre caricano)
        prev_live_markets_list.controls.clear()
        prev_live_markets_list.controls.append(make_loading_indicator("Caricamento mercati LIVE..."))
        prev_trading_matrix.controls.clear()
        prev_trading_matrix.controls.append(make_loading_indicator("Calcolo matrice trading..."))
        prev_trading_keydays.controls.clear()
        prev_trading_keydays.controls.append(ft.Text("Calcolo giorni chiave...", color=TEXT_DIM, size=12))
        prev_trading_quality_card.content = make_loading_indicator("Calcolo qualità forecast...")
        prev_trading_risk_card.content = make_loading_indicator("Calcolo rischi & flag...")
        prev_probability_result.content = make_loading_indicator("Calcolo probabilità...")

        data = state.cache.get("forecast")
        if not data:
            data = api_get(f"/forecast/{state.current_city}?days=14")
            state.cache["forecast"] = data

        forecasts = (data or {}).get("forecast", [])
        city_info = (data or {}).get("city", {})
        logger.info(
            "Previsioni load start city=%s days=%s forecast_rows=%s",
            state.current_city, days, len(forecasts)
        )

        if not forecasts:
            prev_chart_container.content = make_empty_state(
                ft.Icons.CLOUD_OFF, "Nessun forecast disponibile")
            safe_update()
            return

        fcs = forecasts[:days]

        # Verdetto previsioni
        temps_max = [f2c(fc.get("temp_max", 0)) for fc in fcs]
        temps_min = [f2c(fc.get("temp_min", 0)) for fc in fcs]
        avg_precip = sum(fc.get("precipitation_sum", 0) or 0 for fc in fcs) / len(fcs) if fcs else 0
        trend = temps_max[-1] - temps_max[0] if len(temps_max) > 1 else 0
        if trend > 3:
            trend_txt = f"Tendenza in riscaldamento (+{trend:.0f}°C nei prossimi {days} giorni)"
            t_color = RED
        elif trend < -3:
            trend_txt = f"Tendenza in raffreddamento ({trend:.0f}°C nei prossimi {days} giorni)"
            t_color = "#42a5f5"
        else:
            trend_txt = f"Temperatura stabile nei prossimi {days} giorni"
            t_color = GREEN
        if avg_precip > 5:
            trend_txt += ". Periodo piovoso — previsioni meteo più incerte."
        elif avg_precip > 1:
            trend_txt += ". Qualche pioggia prevista."
        else:
            trend_txt += ". Tempo prevalentemente asciutto."
        prev_verdict.content = make_verdict_banner(trend_txt, t_color, ft.Icons.TRENDING_FLAT)

        # Grafico principale
        prev_chart_container.content = build_temp_bar_chart(fcs, chart_height=260, max_days=days)

        # Grafici secondari
        prev_secondary_charts.controls = [
            ft.Column([
                make_card(build_precip_chart(fcs, chart_height=140, max_days=days), height=ROW_SECONDARY_CHARTS_H),
            ], col={"xs": 12, "sm": 6, "lg": 3}),
            ft.Column([
                make_card(build_precip_probability_chart(fcs, chart_height=140, max_days=days), height=ROW_SECONDARY_CHARTS_H),
            ], col={"xs": 12, "sm": 6, "lg": 3}),
            ft.Column([
                make_card(build_wind_chart(fcs, chart_height=140, max_days=days), height=ROW_SECONDARY_CHARTS_H),
            ], col={"xs": 12, "sm": 6, "lg": 3}),
            ft.Column([
                make_card(build_uv_chart(fcs, chart_height=140, max_days=days), height=ROW_SECONDARY_CHARTS_H),
            ], col={"xs": 12, "sm": 6, "lg": 3}),
        ]

        # Insight extra (reali): incertezza ensemble e escursione termica
        prev_secondary_insights.controls = [
            ft.Column([
                make_card(
                    build_ensemble_spread_chart(fcs, chart_height=140, max_days=days),
                    height=ROW_SECONDARY_CHARTS_H,
                ),
            ], col={"xs": 12, "sm": 6, "lg": 3}),
            ft.Column([
                make_card(
                    build_temp_range_chart(fcs, chart_height=140, max_days=days),
                    height=ROW_SECONDARY_CHARTS_H,
                ),
            ], col={"xs": 12, "sm": 6, "lg": 3}),
            ft.Column([
                make_card(
                    build_snow_chart(fcs, chart_height=140, max_days=days),
                    height=ROW_SECONDARY_CHARTS_H,
                ),
            ], col={"xs": 12, "sm": 6, "lg": 3}),
            ft.Column([
                make_card(
                    build_precip_hours_chart(fcs, chart_height=140, max_days=days),
                    height=ROW_SECONDARY_CHARTS_H,
                ),
            ], col={"xs": 12, "sm": 6, "lg": 3}),
        ]

        # Tabella
        prev_table.rows = []
        for fc in fcs:
            ens_data = fc.get("ensemble", {})
            ens_std = ens_data.get("ensemble_std")
            ens_str = f"±{ens_std:.1f}" if ens_std else "—"
            precip = fc.get("precipitation_sum", 0) or 0
            precip_prob = fc.get("precip_probability", 0) or 0
            wind = fc.get("wind_max", 0) or 0
            gusts = fc.get("wind_gusts_max", 0) or 0
            uv_val = fc.get("uv_max", 0) or 0
            t_mean = fc.get("temp_mean", 0) or 0

            prev_table.rows.append(ft.DataRow(cells=[
                ft.DataCell(ft.Text(fc.get("date", "")[5:], color=TEXT, size=12)),
                ft.DataCell(ft.Text(f"{f2c(fc.get('temp_max', 0)):.1f}", color=ACCENT, size=12,
                                     weight=ft.FontWeight.BOLD)),
                ft.DataCell(ft.Text(f"{f2c(fc.get('temp_min', 0)):.1f}", color="#42a5f5", size=12)),
                ft.DataCell(ft.Text(f"{f2c(t_mean):.1f}", color=TEXT, size=12)),
                ft.DataCell(ft.Text(f"{precip:.1f}",
                                     color="#26c6da" if precip > 0 else TEXT_DIM, size=12)),
                ft.DataCell(ft.Text(f"{precip_prob:.0f}%", color=TEXT_DIM, size=12)),
                ft.DataCell(ft.Text(f"{wind:.0f}",
                                     color=YELLOW if wind > 30 else TEXT_DIM, size=12)),
                ft.DataCell(ft.Text(f"{gusts:.0f}",
                                     color=RED if gusts > 40 else TEXT_DIM, size=12)),
                ft.DataCell(ft.Text(f"{uv_val:.1f}", color=uv_color(uv_val), size=12)),
                ft.DataCell(ft.Text(ens_str, color=ACCENT2, size=12)),
            ]))

        # Mostra subito il forecast principale; i pannelli "trading/live" possono richiedere più tempo.
        try:
            safe_update()
            logger.info("safe_update #1 (charts+table): OK")
        except Exception as exc:
            logger.error("safe_update #1 FAILED: %s", exc, exc_info=True)

        # Trading diagnostics (tutto reale: forecast + ensemble + accuracy/spread-skill)
        acc_resp = api_get(
            f"/analysis/{state.current_city}/accuracy?variable=temperature_2m_max",
            timeout=6,
        )
        acc_data = (acc_resp or {}).get("accuracy") if isinstance(acc_resp, dict) else None
        spread_skill_resp = api_get(
            f"/analysis/{state.current_city}/spread-skill?days=30",
            timeout=6,
        )

        day_rows = []
        for idx, fc in enumerate(fcs):
            ens = fc.get("ensemble", {}) or {}
            ens_std_f = ens.get("ensemble_std") or 0
            ens_min_f = ens.get("ensemble_min")
            ens_max_f = ens.get("ensemble_max")
            ens_spread_c = None
            if ens_min_f is not None and ens_max_f is not None:
                ens_spread_c = (float(ens_max_f) - float(ens_min_f)) * 5 / 9
            ens_std_c = float(ens_std_f) * 5 / 9 if ens_std_f else 0

            precip_prob = float(fc.get("precip_probability", 0) or 0)
            precip_mm = float(fc.get("precipitation_sum", 0) or 0)
            wind = float(fc.get("wind_max", 0) or 0)
            gust = float(fc.get("wind_gusts_max", 0) or 0)
            uv_val = float(fc.get("uv_max", 0) or 0)
            tmax_c = f2c(fc.get("temp_max", 0) or 0)
            tmin_c = f2c(fc.get("temp_min", 0) or 0)
            temp_range_c = tmax_c - tmin_c
            days_ahead = idx + 1

            conf = calculate_confidence(ens, edge=0.0, accuracy_data=acc_data, n_outcomes=2, days_ahead=days_ahead)
            conf_total = conf.get("total", 0)

            flags = []
            risk_points = 0
            if ens_spread_c is None:
                flags.append("no ens")
                risk_points += 3
            elif ens_spread_c >= 6:
                flags.append("spread alto")
                risk_points += 2
            elif ens_spread_c >= 3:
                flags.append("spread medio")
                risk_points += 1
            if precip_prob >= 70:
                flags.append("pioggia prob alta")
                risk_points += 2
            elif precip_prob >= 40:
                flags.append("pioggia possibile")
                risk_points += 1
            if gust >= 40:
                flags.append("raffiche forti")
                risk_points += 2
            elif wind >= 30:
                flags.append("vento forte")
                risk_points += 1
            if days_ahead > 7:
                flags.append("orizzonte lungo")
                risk_points += 1
            if uv_val >= 8:
                flags.append("uv alto")

            trade_score = max(0, min(100, conf_total - risk_points * 8))
            if trade_score >= 60 and risk_points <= 1:
                verdict = "FAVOREVOLE"
                verdict_color = GREEN
            elif trade_score >= 40:
                verdict = "CAUTELA"
                verdict_color = YELLOW
            else:
                verdict = "EVITA"
                verdict_color = RED

            day_rows.append({
                "date": fc.get("date", ""),
                "tmax_c": tmax_c,
                "tmin_c": tmin_c,
                "temp_range_c": temp_range_c,
                "precip_mm": precip_mm,
                "precip_prob": precip_prob,
                "wind": wind,
                "gust": gust,
                "uv": uv_val,
                "ens_std_c": ens_std_c,
                "ens_spread_c": ens_spread_c or 0.0,
                "n_members": ens.get("n_members", 0) or 0,
                "conf_total": conf_total,
                "trade_score": trade_score,
                "risk_points": risk_points,
                "flags": flags,
                "verdict": verdict,
                "verdict_color": verdict_color,
                "days_ahead": days_ahead,
            })

        # Mercati LIVE collegati (reali, Polymarket)
        # Usa cache da mercati se disponibile, altrimenti chiama API
        prev_live_markets_list.controls.clear()
        cached_markets = state.cache.get("markets")
        if cached_markets:
            live_markets = [m for m in cached_markets if m.get("_source") == "polymarket"]
            logger.info("Previsioni LIVE: using cached markets (%d)", len(live_markets))
        else:
            scan_live = api_get("/market/scan", timeout=30)
            live_markets = (scan_live or {}).get("markets", []) if isinstance(scan_live, dict) else []
            logger.info("Previsioni LIVE: fresh scan (%d markets, cache miss)", len(live_markets))
        if live_markets:
            forecast_dates = {str(fc.get("date", "")) for fc in fcs if fc.get("date")}
            matched = []
            for m in live_markets:
                rec = m.get("recommendation", {}) or {}
                meta = m.get("metadata", {}) or {}
                mkt = m.get("market", {}) or {}
                city_slug = rec.get("city") or (meta.get("city", {}) or {}).get("slug")
                date_str = rec.get("date") or meta.get("target_date")
                if city_slug != state.current_city:
                    continue
                if date_str not in forecast_dates:
                    continue
                outcomes = rec.get("outcomes", []) or []
                max_edge = max((o.get("edge", 0) for o in outcomes), default=0)
                matched.append((max_edge, m))

            matched.sort(key=lambda x: x[0], reverse=True)
            logger.info("Previsioni LIVE matched: %d entries for city=%s dates=%s",
                         len(matched), state.current_city, forecast_dates)
            if matched:
                for mi, (_, entry) in enumerate(matched[:4]):
                    rec = entry.get("recommendation", {}) or {}
                    mkt = entry.get("market", {}) or {}
                    best = rec.get("best_bet") or "—"
                    ev = float(rec.get("expected_value", 0) or 0)
                    logger.info("  LIVE entry %d: q=%s best=%s ev=%.3f",
                                 mi, mkt.get("question", "?")[:50], best, ev)
                    outcomes = rec.get("outcomes", []) or []
                    best_row = next((o for o in outcomes if o.get("outcome") == best), None)
                    edge = float((best_row or {}).get("edge", 0) or 0)
                    conf = float((best_row or {}).get("confidence", 0) or 0)
                    ev_color = GREEN if ev > 0 else (YELLOW if ev > -0.02 else RED)
                    prev_live_markets_list.controls.append(ft.Container(
                        content=ft.Column([
                            ft.Row([
                                make_badge("LIVE", ft.Colors.with_opacity(0.16, GREEN), GREEN),
                                make_badge((rec.get("date", "") or "—")[5:], ft.Colors.with_opacity(0.14, ACCENT2), ACCENT2),
                                ft.Container(expand=True),
                                make_badge(f"EV {ev:+.1%}", ft.Colors.with_opacity(0.14, ev_color), ev_color),
                            ], spacing=6, wrap=True),
                            ft.Text(mkt.get("question", "—")[:78], size=11, color=TEXT,
                                    max_lines=2, overflow=ft.TextOverflow.ELLIPSIS),
                            ft.Text(
                                f"Bet: {best} · Edge {edge:+.1%} · Conf {conf:.0%} · Vol ${float(mkt.get('volume', 0) or 0):,.0f}",
                                size=10, color=TEXT_DIM,
                            ),
                        ], spacing=2),
                        bgcolor=ft.Colors.with_opacity(0.05, GREEN),
                        border=ft.Border.all(1, ft.Colors.with_opacity(0.12, GREEN)),
                        border_radius=12,
                        padding=pad(h=10, v=8),
                    ))
            else:
                prev_live_markets_list.controls.append(
                    make_empty_state(ft.Icons.PODCASTS, "Nessun mercato LIVE matchato", "Prova un'altra città o un altro giorno")
                )
        else:
            prev_live_markets_list.controls.append(
                make_empty_state(ft.Icons.PODCASTS, "Nessun mercato LIVE disponibile", "Polymarket scan vuoto o temporaneamente non disponibile")
            )
        logger.info(
            "Previsioni live panel city=%s scan_live=%s matched_rendered=%s",
            state.current_city,
            len(live_markets),
            len(prev_live_markets_list.controls),
        )
        try:
            safe_update()
            logger.info("safe_update #2 (live+probability): OK")
        except Exception as exc:
            logger.error("safe_update #2 FAILED: %s", exc, exc_info=True)

        # Setup probabilità custom (giorni disponibili + soglie suggerite) + calcolo iniziale
        _refresh_probability_day_options(fcs[: min(len(fcs), 10)])
        if prob_day_dropdown.value:
            logger.info("Previsioni probability panel city=%s target=%s", state.current_city, prob_day_dropdown.value)
            _compute_probability_custom()

        # Quality card (accuracy + spread-skill + current readiness)
        q_items = []
        if acc_data and isinstance(acc_data, dict):
            mae_c = (acc_data.get("mae", 0) or 0) * 5 / 9
            rmse_c = (acc_data.get("rmse", 0) or 0) * 5 / 9
            bias_c = (acc_data.get("bias", 0) or 0) * 5 / 9
            q_items.extend([
                make_kv_row("MAE storico", f"{mae_c:.2f}°C", GREEN if mae_c < 2 else (YELLOW if mae_c < 4 else RED)),
                make_kv_row("RMSE storico", f"{rmse_c:.2f}°C", TEXT),
                make_kv_row("Bias storico", f"{bias_c:+.2f}°C", RED if abs(bias_c) > 1.5 else TEXT),
                make_kv_row("Campioni", f"{acc_data.get('n_samples', 0)}", TEXT_DIM),
            ])
        else:
            q_items.append(ft.Text("Accuracy storico non disponibile", size=11, color=TEXT_DIM))

        if spread_skill_resp and isinstance(spread_skill_resp, dict) and spread_skill_resp.get("correlation") is not None:
            ms_c = (spread_skill_resp.get("mean_spread", 0) or 0) * 5 / 9
            me_c = (spread_skill_resp.get("mean_error", 0) or 0) * 5 / 9
            ratio = spread_skill_resp.get("spread_skill_ratio", 0) or 0
            corr = spread_skill_resp.get("correlation", 0) or 0
            ratio_color = GREEN if 0.8 <= ratio <= 1.2 else (YELLOW if 0.6 <= ratio <= 1.5 else RED)
            q_items.extend([
                ft.Container(height=4),
                make_kv_row("Spread medio", f"{ms_c:.2f}°C", ACCENT2),
                make_kv_row("Errore medio", f"{me_c:.2f}°C", TEXT),
                make_kv_row("Spread/Skill", f"{ratio:.2f}", ratio_color),
                make_kv_row("Correlazione", f"{corr:+.2f}", GREEN if corr > 0.3 else (YELLOW if corr > 0 else RED)),
                ft.Text(str(spread_skill_resp.get("interpretation", "")), size=10, color=TEXT_DIM, italic=True),
            ])

        if day_rows:
            best_conf = max(day_rows, key=lambda x: x["conf_total"])
            best_trade = max(day_rows, key=lambda x: x["trade_score"])
            q_items.extend([
                ft.Container(height=4),
                make_kv_row("Best conf", f"{best_conf['date'][5:]} ({best_conf['conf_total']:.0f}/100)", best_conf["verdict_color"]),
                make_kv_row("Best trade", f"{best_trade['date'][5:]} ({best_trade['trade_score']:.0f}/100)", best_trade["verdict_color"]),
            ])
        prev_trading_quality_card.content = ft.Column(q_items, spacing=3) if q_items else ft.Text("Nessun dato", color=TEXT_DIM, size=12)

        # Key days card
        prev_trading_keydays.controls.clear()
        if day_rows:
            best_trade = sorted(day_rows, key=lambda x: x["trade_score"], reverse=True)[:2]
            highest_risk = sorted(day_rows, key=lambda x: (x["risk_points"], x["ens_spread_c"], x["gust"], x["precip_prob"]), reverse=True)[:2]
            rainy = max(day_rows, key=lambda x: x["precip_prob"])
            windy = max(day_rows, key=lambda x: x["gust"])
            picks = [
                ("Top setup", best_trade[0], GREEN),
                ("Seconda scelta", best_trade[1], YELLOW) if len(best_trade) > 1 else None,
                ("Più piovoso", rainy, ACCENT2),
                ("Più ventoso", windy, ORANGE),
                ("Rischio max", highest_risk[0], RED) if highest_risk else None,
            ]
            for item in picks:
                if item is None:
                    continue
                label, d, c = item
                prev_trading_keydays.controls.append(ft.Container(
                    content=ft.Column([
                        ft.Row([
                            make_badge(label, ft.Colors.with_opacity(0.16, c), c),
                            ft.Container(expand=True),
                            make_badge(d["verdict"], ft.Colors.with_opacity(0.18, d["verdict_color"]), d["verdict_color"]),
                        ], spacing=6),
                        ft.Text(
                            f"{d['date'][5:]} · {d['tmax_c']:.0f}/{d['tmin_c']:.0f}°C · score {d['trade_score']:.0f}/100",
                            size=11, color=TEXT,
                        ),
                        ft.Text(
                            f"Spread {d['ens_spread_c']:.1f}°C · Pioggia {d['precip_prob']:.0f}%/{d['precip_mm']:.1f}mm · Raffiche {d['gust']:.0f}mph",
                            size=10, color=TEXT_DIM,
                        ),
                    ], spacing=2),
                    bgcolor=ft.Colors.with_opacity(0.06, c),
                    border=ft.Border.all(1, ft.Colors.with_opacity(0.14, c)),
                    border_radius=12,
                    padding=pad(h=10, v=8),
                ))
        else:
            prev_trading_keydays.controls.append(ft.Text("Nessun dato", color=TEXT_DIM, size=12))

        # Risk flags summary
        if day_rows:
            n_fav = sum(1 for d in day_rows if d["verdict"] == "FAVOREVOLE")
            n_cau = sum(1 for d in day_rows if d["verdict"] == "CAUTELA")
            n_avoid = sum(1 for d in day_rows if d["verdict"] == "EVITA")
            n_rain = sum(1 for d in day_rows if d["precip_prob"] >= 40)
            n_wind = sum(1 for d in day_rows if d["gust"] >= 40 or d["wind"] >= 30)
            n_unc = sum(1 for d in day_rows if d["ens_spread_c"] >= 6)
            avg_conf = sum(d["conf_total"] for d in day_rows) / len(day_rows)
            avg_trade = sum(d["trade_score"] for d in day_rows) / len(day_rows)
            prev_trading_risk_card.content = ft.Column([
                ft.Row([
                    make_badge(f"Favorevoli: {n_fav}", ft.Colors.with_opacity(0.16, GREEN), GREEN),
                    make_badge(f"Cautela: {n_cau}", ft.Colors.with_opacity(0.16, YELLOW), YELLOW),
                    make_badge(f"Evita: {n_avoid}", ft.Colors.with_opacity(0.16, RED), RED),
                ], spacing=6, wrap=True),
                ft.Container(height=4),
                make_kv_row("Pioggia (>=40%)", f"{n_rain}/{len(day_rows)} giorni", ACCENT2),
                make_kv_row("Vento critico", f"{n_wind}/{len(day_rows)} giorni", ORANGE),
                make_kv_row("Incertezza alta", f"{n_unc}/{len(day_rows)} giorni", RED if n_unc else TEXT_DIM),
                make_kv_row("Conf media", f"{avg_conf:.0f}/100", GREEN if avg_conf >= 55 else (YELLOW if avg_conf >= 40 else RED)),
                make_kv_row("Trade score medio", f"{avg_trade:.0f}/100", GREEN if avg_trade >= 55 else (YELLOW if avg_trade >= 40 else RED)),
                ft.Container(height=4),
                ft.Text(
                    "Uso rapido: privilegia giorni Favorevoli con spread basso e pioggia/vento contenuti.",
                    size=10, color=TEXT_DIM, italic=True,
                ),
            ], spacing=3)
        else:
            prev_trading_risk_card.content = ft.Text("Nessun dato", color=TEXT_DIM, size=12)

        # Trading matrix rows (dettaglio giorno per giorno)
        prev_trading_matrix.controls.clear()
        if day_rows:
            for d in day_rows:
                flags_txt = ", ".join(d["flags"][:3]) if d["flags"] else "nessun flag critico"
                prev_trading_matrix.controls.append(ft.Container(
                    content=ft.Column([
                        ft.Row([
                            ft.Row([
                                ft.Text(d["date"][5:], size=11, color=TEXT, weight=ft.FontWeight.BOLD),
                                ft.Text(f"(D+{d['days_ahead']})", size=10, color=TEXT_DIM),
                            ], spacing=4),
                            ft.Container(expand=True),
                            make_badge(f"Conf {d['conf_total']:.0f}", ft.Colors.with_opacity(0.14, ACCENT2), ACCENT2),
                            make_badge(f"Score {d['trade_score']:.0f}", ft.Colors.with_opacity(0.14, d["verdict_color"]), d["verdict_color"]),
                            make_badge(d["verdict"], ft.Colors.with_opacity(0.16, d["verdict_color"]), d["verdict_color"]),
                        ], spacing=6, wrap=True),
                        ft.Row([
                            make_kv_row("Temp", f"{d['tmax_c']:.0f}/{d['tmin_c']:.0f}°C"),
                            make_kv_row("ΔT", f"{d['temp_range_c']:.1f}°C", ORANGE if d["temp_range_c"] >= 12 else TEXT),
                        ], spacing=10),
                        ft.Row([
                            make_kv_row("Pioggia", f"{d['precip_mm']:.1f}mm / {d['precip_prob']:.0f}%", ACCENT2 if d["precip_prob"] < 40 else YELLOW),
                            make_kv_row("Vento", f"{d['wind']:.0f} / raff. {d['gust']:.0f} mph", ORANGE if d["gust"] >= 35 else TEXT),
                        ], spacing=10),
                        ft.Row([
                            make_kv_row("Ens spread", f"{d['ens_spread_c']:.1f}°C", RED if d["ens_spread_c"] >= 6 else (YELLOW if d["ens_spread_c"] >= 3 else GREEN)),
                            make_kv_row("UV", f"{d['uv']:.1f}", uv_color(d["uv"])),
                        ], spacing=10),
                        ft.Text(f"Flag: {flags_txt}", size=10, color=TEXT_DIM, italic=True),
                    ], spacing=3),
                    bgcolor=ft.Colors.with_opacity(0.05, d["verdict_color"]),
                    border=ft.Border(
                        left=ft.BorderSide(3, d["verdict_color"]),
                        top=ft.BorderSide(1, ft.Colors.with_opacity(0.12, d["verdict_color"])),
                        right=ft.BorderSide(1, ft.Colors.with_opacity(0.25, TEXT)),
                        bottom=ft.BorderSide(1, ft.Colors.with_opacity(0.25, TEXT)),
                    ),
                    border_radius=12,
                    padding=pad(h=12, v=10),
                ))
        else:
            prev_trading_matrix.controls.append(ft.Text("Nessun dato", color=TEXT_DIM, size=12))
        logger.info(
            "Previsioni trading panels city=%s day_rows=%s keydays_controls=%s matrix_controls=%s",
            state.current_city,
            len(day_rows),
            len(prev_trading_keydays.controls),
            len(prev_trading_matrix.controls),
        )
        try:
            safe_update()
            logger.info("safe_update #3 (trading panels): OK")
        except Exception as exc:
            logger.error("safe_update #3 FAILED: %s", exc, exc_info=True)

        # Climate normal
        today_str = datetime.now().strftime("%Y-%m-%d")
        normal = api_get(f"/analysis/{state.current_city}/climate-normal?target_date={today_str}&variable=temperature_2m_max")
        if normal and "normal_mean" in normal:
            nm = normal["normal_mean"]
            ns = normal.get("normal_std") or 1
            actual = forecasts[0].get("temp_max", 0)
            dev = actual - nm if nm else 0
            dev_color = RED if abs(dev) > ns else TEXT_DIM
            doy = normal.get("day_of_year", "—")

            dev_c = dev * 5 / 9
            if abs(dev_c) < 2:
                clim_verdict = "Nella norma per questa stagione"
                clim_color = GREEN
            elif dev_c > 0:
                clim_verdict = f"Più caldo del solito di {abs(dev_c):.1f}°C"
                clim_color = RED
            else:
                clim_verdict = f"Più freddo del solito di {abs(dev_c):.1f}°C"
                clim_color = "#42a5f5"

            prev_climate_card.content = ft.Column([
                make_kv_row("Normale odierna", f"{f2c(nm):.1f}°C (±{ns * 5 / 9:.1f})", ACCENT),
                make_kv_row("Forecast oggi", f"{f2c(actual):.1f}°C", ACCENT),
                make_kv_row("Deviazione", f"{dev * 5 / 9:+.1f}°C", dev_color),
                make_kv_row("Giorno anno", f"{doy}", TEXT_DIM),
                ft.Container(height=4),
                ft.ProgressBar(
                    value=min(1, max(0, 0.5 + dev / (2 * ns))),
                    color=dev_color, bgcolor=ft.Colors.with_opacity(0.1, TEXT),
                ),
                ft.Container(height=2),
                ft.Text(clim_verdict, size=10, color=clim_color,
                         weight=ft.FontWeight.W_500, italic=True),
            ], spacing=4)
        else:
            prev_climate_card.content = ft.Text(
                "Normale climatica non disponibile", color=TEXT_DIM, size=12)

        # Ensemble info
        ens_today = forecasts[0].get("ensemble", {})
        if ens_today:
            n_members = ens_today.get("n_members", 0)
            ens_mean = ens_today.get("ensemble_mean", 0) or 0
            ens_std_val = ens_today.get("ensemble_std", 0) or 0
            ens_min_val = ens_today.get("ensemble_min", 0) or 0
            ens_max_val = ens_today.get("ensemble_max", 0) or 0
            spread = ens_max_val - ens_min_val

            spread_c_ens = spread * 5 / 9
            if spread_c_ens < 3:
                ens_verdict = f"Modelli molto concordi (spread {spread_c_ens:.1f}°C) — previsione affidabile"
                ens_v_color = GREEN
            elif spread_c_ens < 6:
                ens_verdict = f"Concordanza media (spread {spread_c_ens:.1f}°C) — previsione decente"
                ens_v_color = YELLOW
            else:
                ens_verdict = f"Modelli discordi (spread {spread_c_ens:.1f}°C) — previsione incerta, prudenza!"
                ens_v_color = RED

            prev_ensemble_card.content = ft.Column([
                make_kv_row("N. membri", f"{n_members}", ACCENT2),
                make_kv_row("Media ensemble", f"{f2c(ens_mean):.1f}°C", ACCENT),
                make_kv_row("Std dev", f"{ens_std_val * 5 / 9:.2f}°C", TEXT),
                make_kv_row("Range", f"{f2c(ens_min_val):.1f} — {f2c(ens_max_val):.1f}°C", TEXT),
                make_kv_row("Spread", f"{spread * 5 / 9:.1f}°C", ORANGE if spread > 5 else TEXT),
                ft.Container(height=8),
                build_consensus_indicator(ens_std_val, width=250),
                ft.Container(height=4),
                ft.Text(ens_verdict, size=10, color=ens_v_color,
                         weight=ft.FontWeight.W_500, italic=True),
            ], spacing=4, scroll=ft.ScrollMode.AUTO)
        else:
            prev_ensemble_card.content = ft.Text(
                "Dati ensemble non disponibili", color=TEXT_DIM, size=12)

        try:
            safe_update()
            logger.info("safe_update #4 (climate+ensemble): OK — PREVISIONI LOAD COMPLETE")
        except Exception as exc:
            logger.error("safe_update #4 FAILED: %s", exc, exc_info=True)

    return build, load
