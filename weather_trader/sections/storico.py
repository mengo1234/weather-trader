"""Storico (History) section: statistics, anomalies, accuracy, bet history."""

import logging
import threading
from datetime import datetime

logger = logging.getLogger("weather_trader.sections.storico")

import flet as ft

from weather_trader.constants import (
    ACCENT, ACCENT2, BG, CARD, GREEN, ORANGE, OUTLINE_SOFT, RED, SURFACE_2, TEXT, TEXT_DIM, YELLOW,
    CITY_MAP, f2c,
)
from weather_trader.api_client import api_get
from weather_trader.app_state import AppState
from weather_trader.logic.pnl_tracker import load_pnl, resolve_bet, get_pnl_stats
from weather_trader.widgets.factory import (
    make_badge, make_card, make_empty_state, make_info_box, make_kv_row,
    make_section_title, make_stat_chip, pad, z_score_color,
)
from weather_trader.widgets.charts import build_sparkline
from weather_trader.widgets.distribution import build_distribution_chart
from weather_trader.widgets.pnl_widgets import build_pnl_sparkline


def create_storico(page: ft.Page, state: AppState, safe_update):
    """Factory for the Storico (History) section."""

    ROW_STATS_H = 265
    ROW_CLIMATE_H = 235
    ROW_ACCURACY_H = 305
    ROW_METRICS_H = 305

    def style_select(ctrl, accent=ACCENT):
        if hasattr(ctrl, "border_width"):
            ctrl.border_width = 0
        if hasattr(ctrl, "focused_border_width"):
            ctrl.focused_border_width = 0
        if hasattr(ctrl, "border_color"):
            ctrl.border_color = ft.Colors.TRANSPARENT
        if hasattr(ctrl, "focused_border_color"):
            ctrl.focused_border_color = ft.Colors.TRANSPARENT
        if hasattr(ctrl, "bgcolor"):
            ctrl.bgcolor = ft.Colors.TRANSPARENT
        return ft.Container(
            content=ctrl,
            border_radius=12,
            bgcolor=ft.Colors.with_opacity(0.22, SURFACE_2),
            border=ft.Border.all(1, ft.Colors.with_opacity(0.45, OUTLINE_SOFT)),
            padding=pad(h=4, v=2),
        )

    # ------------------------------------------------------------------
    # local threaded_load helper
    # ------------------------------------------------------------------
    def threaded_load(fn, *args):
        def _run():
            try:
                fn(*args)
            except Exception as e:
                import logging
                logging.getLogger("weather_trader.storico").error("Load error: %s", e)
        threading.Thread(target=_run, daemon=True).start()

    # ------------------------------------------------------------------
    # filter dropdowns
    # ------------------------------------------------------------------
    stor_var_dropdown = ft.Dropdown(
        width=200,
        border_color=ACCENT,
        color=TEXT,
        text_size=13,
        content_padding=pad(h=10, v=6),
        options=[
            ft.dropdown.Option(key="temperature_2m_max", text="Temp Max"),
            ft.dropdown.Option(key="temperature_2m_min", text="Temp Min"),
            ft.dropdown.Option(key="precipitation_sum", text="Precipitazioni"),
            ft.dropdown.Option(key="wind_speed_10m_max", text="Vento Max"),
        ],
        value="temperature_2m_max",
    )
    stor_days_dropdown = ft.Dropdown(
        width=120,
        border_color=ACCENT,
        color=TEXT,
        text_size=13,
        content_padding=pad(h=10, v=6),
        options=[
            ft.dropdown.Option(key="30", text="30 giorni"),
            ft.dropdown.Option(key="60", text="60 giorni"),
            ft.dropdown.Option(key="90", text="90 giorni"),
        ],
        value="90",
    )
    stor_var_dropdown_wrap = style_select(stor_var_dropdown)
    stor_days_dropdown_wrap = style_select(stor_days_dropdown)

    # ------------------------------------------------------------------
    # mutable containers
    # ------------------------------------------------------------------
    stor_stats_panel = ft.Container(padding=10)
    stor_distribution = ft.Container(padding=10)
    stor_anomalies_panel = ft.Container(padding=10)
    stor_climate_card = ft.Container(padding=10)
    stor_sparkline = ft.Container(height=80)
    stor_accuracy_panel = ft.Container(padding=10)
    stor_bias_panel = ft.Container(padding=10)
    stor_bet_history = ft.Container(padding=10)
    stor_roi_city = ft.Container(padding=10)
    stor_metrics_card = ft.Container(padding=10)
    stor_equity_curve = ft.Container(height=80)
    stor_backtest_card = ft.Container(padding=10)

    # Backtest form controls
    bt_city = ft.Dropdown(
        width=140, border_color=ACCENT, color=TEXT, text_size=12,
        content_padding=pad(h=8, v=4),
        options=[ft.dropdown.Option(key=s, text=n)
                 for s, n in zip(
                     ["nyc", "miami", "chicago", "london", "roma"],
                     ["New York", "Miami", "Chicago", "Londra", "Roma"])],
        value="nyc",
    )
    bt_city_wrap = style_select(bt_city)
    bt_min_edge = ft.Slider(min=1, max=15, divisions=14, value=5,
                             label="{value}%", active_color=ACCENT, width=160)
    bt_min_conf = ft.Slider(min=30, max=90, divisions=12, value=60,
                             label="{value}", active_color=GREEN, width=160)
    bt_result = ft.Container(padding=10)

    # ------------------------------------------------------------------
    # filter on_select handlers
    # ------------------------------------------------------------------
    def on_stor_filter_change(e):
        threaded_load(load)

    stor_var_dropdown.on_select = on_stor_filter_change
    stor_days_dropdown.on_select = on_stor_filter_change

    # ------------------------------------------------------------------
    # build
    # ------------------------------------------------------------------
    def build():
        return ft.Column([
            ft.Row([
                make_section_title("Storico", ft.Icons.HISTORY, GREEN),
                ft.Container(expand=True),
                stor_var_dropdown_wrap,
                ft.Container(width=8),
                stor_days_dropdown_wrap,
            ]),
            make_info_box(
                "Dati storici del meteo: statistiche, anomalie e precisione delle previsioni passate. "
                "Questi dati ci aiutano a capire quanto possiamo fidarci delle previsioni future.",
                GREEN,
            ),
            ft.Container(height=10),
            # Stats + Distribution
            ft.ResponsiveRow([
                ft.Column([
                    make_card(ft.Column([
                        ft.Row([
                            ft.Icon(ft.Icons.ANALYTICS, color=ACCENT, size=18),
                            ft.Text("Statistiche Descrittive", size=14,
                                     weight=ft.FontWeight.BOLD, color=TEXT),
                        ], spacing=6),
                        make_info_box(
                            "Riassunto numerico dei dati passati. Media = valore tipico. "
                            "Std Dev = quanto i valori variano. P10/P90 = range in cui cade il 80% dei valori.",
                        ),
                        ft.Container(height=4),
                        stor_stats_panel,
                    ]), height=ROW_STATS_H),
                ], col={"xs": 12, "md": 6}),
                ft.Column([
                    make_card(ft.Column([
                        ft.Row([
                            ft.Icon(ft.Icons.EQUALIZER, color=ACCENT2, size=18),
                            ft.Text("Distribuzione", size=14,
                                     weight=ft.FontWeight.BOLD, color=TEXT),
                        ], spacing=6),
                        make_info_box(
                            "Come sono distribuiti i valori: la scatola mostra il range centrale (25%-75%), "
                            "la linea il range ampio (10%-90%). Se la scatola è stretta, i valori sono costanti.",
                            ACCENT2,
                        ),
                        ft.Container(height=4),
                        stor_distribution,
                    ]), height=ROW_STATS_H),
                ], col={"xs": 12, "md": 6}),
            ]),
            ft.Container(height=10),
            # Anomalies
            make_card(ft.Column([
                ft.Row([
                    ft.Icon(ft.Icons.WARNING_AMBER, color=YELLOW, size=18),
                    ft.Text("Anomalie Rilevate", size=14, weight=ft.FontWeight.BOLD, color=TEXT),
                ], spacing=6),
                make_info_box(
                    "Giorni con valori molto diversi dalla media (z-score). z > 2 = molto insolito. "
                    "Anomalie recenti possono creare opportunità di scommessa perché i mercati reagiscono in ritardo.",
                    YELLOW,
                ),
                ft.Container(height=4),
                stor_anomalies_panel,
            ])),
            ft.Container(height=10),
            # Climate + Sparkline
            ft.ResponsiveRow([
                ft.Column([
                    make_card(ft.Column([
                        ft.Row([
                            ft.Icon(ft.Icons.THERMOSTAT, color=GREEN, size=18),
                            ft.Text("Normale Climatica", size=14,
                                     weight=ft.FontWeight.BOLD, color=TEXT),
                        ], spacing=6),
                        make_info_box(
                            "Il valore 'normale' per oggi basato su anni di dati storici. "
                            "Se il meteo previsto è molto diverso dalla normale, c'è potenziale per scommesse.",
                            GREEN,
                        ),
                        ft.Container(height=4),
                        stor_climate_card,
                    ]), height=ROW_CLIMATE_H),
                ], col={"xs": 12, "md": 6}),
                ft.Column([
                    make_card(ft.Column([
                        ft.Row([
                            ft.Icon(ft.Icons.SHOW_CHART, color=ACCENT, size=18),
                            ft.Text("Osservazioni recenti", size=14,
                                     weight=ft.FontWeight.BOLD, color=TEXT),
                        ], spacing=6),
                        make_info_box("Grafico delle osservazioni recenti — ogni barra è un giorno."),
                        ft.Container(height=4),
                        stor_sparkline,
                    ]), height=ROW_CLIMATE_H),
                ], col={"xs": 12, "md": 6}),
            ]),
            ft.Container(height=10),
            # Accuracy + Bias Analysis
            ft.ResponsiveRow([
                ft.Column([
                    make_card(ft.Column([
                        ft.Row([
                            ft.Icon(ft.Icons.VERIFIED, color=GREEN, size=18),
                            ft.Text("Accuratezza Forecast", size=14,
                                     weight=ft.FontWeight.BOLD, color=TEXT),
                        ], spacing=6),
                        make_info_box(
                            "Quanto le previsioni passate erano corrette. MAE = errore medio in gradi. "
                            "MAE < 2 = eccellente, < 4 = buona, > 6 = scarsa. Più è basso, meglio scommettiamo!",
                            GREEN,
                        ),
                        ft.Container(height=4),
                        stor_accuracy_panel,
                    ]), border_color=ft.Colors.with_opacity(0.1, GREEN), height=ROW_ACCURACY_H),
                ], col={"xs": 12, "md": 6}),
                ft.Column([
                    make_card(ft.Column([
                        ft.Row([
                            ft.Icon(ft.Icons.TUNE, color=ORANGE, size=18),
                            ft.Text("Analisi Bias", size=14,
                                     weight=ft.FontWeight.BOLD, color=TEXT),
                        ], spacing=6),
                        make_info_box(
                            "Il bias indica se le previsioni tendono a sovrastimare o sottostimare. "
                            "Bias vicino a 0 = modello equilibrato. Conoscere il bias aiuta a correggere le scommesse.",
                            ORANGE,
                        ),
                        ft.Container(height=4),
                        stor_bias_panel,
                    ]), border_color=ft.Colors.with_opacity(0.1, ORANGE), height=ROW_ACCURACY_H),
                ], col={"xs": 12, "md": 6}),
            ]),
            ft.Container(height=10),
            # Metriche Avanzate + Equity Curve
            ft.ResponsiveRow([
                ft.Column([
                    make_card(ft.Column([
                        ft.Row([
                            ft.Icon(ft.Icons.INSIGHTS, color=ACCENT2, size=18),
                            ft.Text("Metriche Avanzate", size=14,
                                     weight=ft.FontWeight.BOLD, color=TEXT),
                        ], spacing=6),
                        make_info_box(
                            "Sharpe Ratio, Sortino, Drawdown e altre metriche professionali "
                            "per valutare la qualità della strategia di trading.",
                            ACCENT2,
                        ),
                        ft.Container(height=4),
                        stor_metrics_card,
                    ]), border_color=ft.Colors.with_opacity(0.15, ACCENT2), height=ROW_METRICS_H),
                ], col={"xs": 12, "md": 6}),
                ft.Column([
                    make_card(ft.Column([
                        ft.Row([
                            ft.Icon(ft.Icons.SHOW_CHART, color=GREEN, size=18),
                            ft.Text("Equity Curve", size=14,
                                     weight=ft.FontWeight.BOLD, color=TEXT),
                        ], spacing=6),
                        make_info_box(
                            "Andamento del bankroll nel tempo. Una curva in salita = strategia profittevole.",
                            GREEN,
                        ),
                        ft.Container(height=4),
                        stor_equity_curve,
                    ]), border_color=ft.Colors.with_opacity(0.15, GREEN), height=ROW_METRICS_H),
                ], col={"xs": 12, "md": 6}),
            ]),
            ft.Container(height=10),
            # Backtest
            make_card(ft.Column([
                ft.Row([
                    ft.Icon(ft.Icons.SCIENCE, color=ACCENT, size=18),
                    ft.Text("Backtest", size=14, weight=ft.FontWeight.BOLD, color=TEXT),
                ], spacing=6),
                make_info_box(
                    "Simula la strategia sui dati storici. Scegli città e parametri, "
                    "poi clicca 'Esegui' per vedere come avrebbe performato.",
                ),
                ft.Container(height=4),
                ft.ResponsiveRow([
                    ft.Column([
                        ft.Container(
                            content=ft.Column([
                                ft.Text("Città", size=10, color=TEXT_DIM, weight=ft.FontWeight.W_600),
                                bt_city_wrap,
                            ], spacing=4),
                            padding=pad(h=10, v=8),
                            border_radius=12,
                            bgcolor=ft.Colors.with_opacity(0.18, SURFACE_2),
                            border=ft.Border.all(1, ft.Colors.with_opacity(0.40, OUTLINE_SOFT)),
                        ),
                    ], col={"xs": 12, "md": 4}),
                    ft.Column([
                        ft.Container(
                            content=ft.Column([
                                ft.Text("Min Edge %", size=10, color=TEXT_DIM, weight=ft.FontWeight.W_600),
                                bt_min_edge,
                            ], spacing=4),
                            padding=pad(h=10, v=8),
                            border_radius=12,
                            bgcolor=ft.Colors.with_opacity(0.18, SURFACE_2),
                            border=ft.Border.all(1, ft.Colors.with_opacity(0.40, OUTLINE_SOFT)),
                        ),
                    ], col={"xs": 12, "md": 4}),
                    ft.Column([
                        ft.Container(
                            content=ft.Column([
                                ft.Text("Min Confidenza", size=10, color=TEXT_DIM, weight=ft.FontWeight.W_600),
                                bt_min_conf,
                            ], spacing=4),
                            padding=pad(h=10, v=8),
                            border_radius=12,
                            bgcolor=ft.Colors.with_opacity(0.18, SURFACE_2),
                            border=ft.Border.all(1, ft.Colors.with_opacity(0.40, OUTLINE_SOFT)),
                        ),
                    ], col={"xs": 12, "md": 4}),
                ], spacing=8, run_spacing=8),
                ft.Container(height=4),
                stor_backtest_card,
                bt_result,
            ])),
            ft.Container(height=10),
            # Storico scommesse + ROI per città
            ft.ResponsiveRow([
                ft.Column([
                    make_card(ft.Column([
                        ft.Row([
                            ft.Icon(ft.Icons.RECEIPT_LONG, color=ACCENT, size=18),
                            ft.Text("Storico Scommesse", size=14,
                                     weight=ft.FontWeight.BOLD, color=TEXT),
                        ], spacing=6),
                        make_info_box(
                            "Tutte le scommesse piazzate. Clicca Vinta/Persa per registrare il risultato.",
                        ),
                        ft.Container(height=4),
                        stor_bet_history,
                    ])),
                ], col={"xs": 12, "md": 7}),
                ft.Column([
                    make_card(ft.Column([
                        ft.Row([
                            ft.Icon(ft.Icons.LEADERBOARD, color=GREEN, size=18),
                            ft.Text("ROI per Città", size=14,
                                     weight=ft.FontWeight.BOLD, color=TEXT),
                        ], spacing=6),
                        make_info_box(
                            "Dove guadagni e dove perdi. Concentrati sulle città profittevoli!",
                            GREEN,
                        ),
                        ft.Container(height=4),
                        stor_roi_city,
                    ]), border_color=ft.Colors.with_opacity(0.1, GREEN)),
                ], col={"xs": 12, "md": 5}),
            ]),
        ], scroll=ft.ScrollMode.AUTO, spacing=0)

    # ------------------------------------------------------------------
    # load
    # ------------------------------------------------------------------
    def load():
        """Carica dati storici."""
        try:
            _load_impl()
        except Exception as e:
            import logging
            logging.getLogger("weather_trader.storico").error("Storico load error: %s", e)
            stor_stats_panel.content = make_empty_state(
                ft.Icons.ERROR_OUTLINE, f"Errore caricamento storico: {e}")
            safe_update()

    def _load_impl():
        current_city = state.current_city
        cache = state.cache
        variable = stor_var_dropdown.value or "temperature_2m_max"
        days = int(stor_days_dropdown.value or 90)

        # ---- Statistiche ----
        stats_data = api_get(f"/analysis/{current_city}/stats?variable={variable}&days={days}")
        stats = (stats_data or {}).get("stats", {})
        cache["stats"] = stats

        if stats:
            is_temp = "temperature" in variable
            # Chiavi che sono valori assoluti (vanno convertite con f2c)
            abs_keys = {"mean", "median", "min", "max", "p10", "p25", "p75", "p90"}
            # Chiavi che sono differenze (vanno moltiplicate per 5/9)
            diff_keys = {"std", "iqr"}
            stat_rows = []
            stat_fields = [
                ("N campioni", "count", TEXT),
                ("Media", "mean", ACCENT),
                ("Mediana", "median", ACCENT),
                ("Std Dev", "std", TEXT),
                ("Min", "min", "#42a5f5"),
                ("Max", "max", RED),
                ("Skewness", "skewness", TEXT),
                ("Kurtosis", "kurtosis", TEXT),
                ("P10", "p10", TEXT_DIM),
                ("P25", "p25", TEXT_DIM),
                ("P75", "p75", TEXT_DIM),
                ("P90", "p90", TEXT_DIM),
                ("IQR", "iqr", TEXT_DIM),
            ]
            for label, key, color in stat_fields:
                val = stats.get(key)
                if val is not None:
                    if key == "count":
                        fmt = f"{val:,.0f}"
                    elif is_temp and key in abs_keys:
                        fmt = f"{f2c(val):.2f}°C"
                    elif is_temp and key in diff_keys:
                        fmt = f"{val * 5 / 9:.2f}°C"
                    else:
                        fmt = f"{val:.2f}"
                    stat_rows.append(make_kv_row(label, fmt, color))
            # Interpretazione statistiche
            skew = stats.get("skewness", 0) or 0
            kurt = stats.get("kurtosis", 0) or 0
            interp_parts = []
            if abs(skew) < 0.5:
                interp_parts.append("Distribuzione simmetrica (valori alti e bassi equilibrati)")
            elif skew > 0:
                interp_parts.append("Più valori bassi del solito, con occasionali picchi alti")
            else:
                interp_parts.append("Più valori alti del solito, con occasionali cali")
            if kurt > 1:
                interp_parts.append("Valori estremi più frequenti del normale")
            stat_rows.append(ft.Container(height=4))
            stat_rows.append(ft.Text(
                " — ".join(interp_parts), size=9, color=TEXT_DIM, italic=True))
            stor_stats_panel.content = ft.Column(stat_rows, spacing=3)

            # Distribution box plot — converto in °C se temperatura
            dist_stats = stats
            if is_temp:
                dist_stats = {**stats}
                for k in abs_keys:
                    if k in dist_stats:
                        dist_stats[k] = f2c(dist_stats[k])
            stor_distribution.content = ft.Column([
                build_distribution_chart(dist_stats, width=350, height=60),
                ft.Container(height=8),
                ft.Text(f"Variabile: {variable.replace('_', ' ').title()}", size=11, color=TEXT_DIM),
                ft.Text(f"Ultimi {days} giorni — {CITY_MAP.get(current_city, current_city)}",
                         size=11, color=TEXT_DIM),
            ], spacing=4)
        else:
            stor_stats_panel.content = ft.Text("Dati non disponibili", color=TEXT_DIM, size=12)
            stor_distribution.content = ft.Text("—", color=TEXT_DIM, size=12)

        # ---- Anomalie ----
        anom_data = api_get(f"/analysis/{current_city}/anomaly?variable={variable}&days={days}")
        anom_list = (anom_data or {}).get("anomalies", [])
        cache["anomalies"] = anom_list

        if anom_list:
            is_temp = "temperature" in variable
            anom_rows = []
            for a in anom_list[:15]:
                z = a.get("z_score", 0)
                raw_val = a.get("value", 0)
                disp_val = f2c(raw_val) if is_temp else raw_val
                anom_rows.append(ft.Container(
                    content=ft.Row([
                        make_badge(f"z={z:+.1f}", z_score_color(z), TEXT),
                        ft.Text(a.get("date", ""), size=11, color=TEXT),
                        ft.Text(f"{disp_val:.1f}", size=11, color=ACCENT,
                                 weight=ft.FontWeight.BOLD),
                        ft.Text(a.get("reasoning", "")[:80], size=10, color=TEXT_DIM,
                                 overflow=ft.TextOverflow.ELLIPSIS, expand=True),
                    ], spacing=8),
                    padding=pad(h=4, v=3),
                    border=ft.Border.only(
                        left=ft.BorderSide(2, z_score_color(z))),
                ))
            stor_anomalies_panel.content = ft.Column(anom_rows, spacing=4)
        else:
            stor_anomalies_panel.content = ft.Text(
                "Nessuna anomalia rilevata", color=TEXT_DIM, size=12)

        # ---- Climate normal ----
        today_str = datetime.now().strftime("%Y-%m-%d")
        normal = api_get(
            f"/analysis/{current_city}/climate-normal?target_date={today_str}&variable={variable}")
        if normal and "normal_mean" in normal:
            nm = normal["normal_mean"]
            ns = normal.get("normal_std") or 1
            doy = normal.get("day_of_year", "—")
            is_temp_n = "temperature" in variable
            disp_nm = f2c(nm) if is_temp_n else nm
            disp_ns = ns * 5 / 9 if is_temp_n else ns

            stor_climate_card.content = ft.Column([
                make_kv_row("Normale odierna", f"{disp_nm:.2f}{'°C' if is_temp_n else ''}", ACCENT),
                make_kv_row("Deviazione std", f"±{disp_ns:.2f}{'°C' if is_temp_n else ''}", TEXT),
                make_kv_row("Giorno anno", f"{doy}", TEXT_DIM),
                make_kv_row("Variabile", variable.replace("_", " ").title(), TEXT_DIM),
            ], spacing=4)
        else:
            stor_climate_card.content = ft.Text(
                "Normale climatica non disponibile", color=TEXT_DIM, size=12)

        # ---- Sparkline (forecast come proxy per osservazioni recenti) ----
        fc_data = api_get(f"/forecast/{current_city}?days=14")
        fc_list = (fc_data or {}).get("forecast", [])
        if fc_list:
            if "temp" in variable:
                key = "temp_max" if "max" in variable else "temp_min"
            elif "precip" in variable:
                key = "precipitation_sum"
            elif "wind" in variable:
                key = "wind_max"
            else:
                key = "temp_max"
            vals = [fc.get(key, 0) or 0 for fc in fc_list]
            stor_sparkline.content = ft.Column([
                build_sparkline(vals, width=350, height=60,
                                 color=ACCENT if "temp" in variable else "#26c6da"),
                ft.Text(f"{len(vals)} giorni di forecast", size=10, color=TEXT_DIM),
            ], spacing=4)
        else:
            stor_sparkline.content = ft.Text("Nessun dato", color=TEXT_DIM, size=12)

        # ---- Accuratezza Forecast ----
        acc = api_get(f"/analysis/{current_city}/accuracy?variable={variable}")
        acc_data = (acc or {}).get("accuracy")
        if acc_data and isinstance(acc_data, dict):
            mae = acc_data.get("mae", 0) or 0
            rmse = acc_data.get("rmse", 0) or 0
            bias = acc_data.get("bias", 0) or 0
            n_s = acc_data.get("n_samples", 0)
            is_temp_a = "temperature" in variable

            # Qualità
            if mae < 2:
                q_label, q_color = "ECCELLENTE", GREEN
            elif mae < 4:
                q_label, q_color = "BUONA", YELLOW
            elif mae < 6:
                q_label, q_color = "SUFFICIENTE", ORANGE
            else:
                q_label, q_color = "SCARSA", RED

            mae_disp = mae * 5 / 9 if is_temp_a else mae
            rmse_disp = rmse * 5 / 9 if is_temp_a else rmse
            bias_disp = bias * 5 / 9 if is_temp_a else bias
            unit = "°C" if is_temp_a else ""

            stor_accuracy_panel.content = ft.Column([
                ft.Row([
                    ft.Container(
                        content=ft.Text(q_label, size=10, color=BG, weight=ft.FontWeight.BOLD),
                        bgcolor=q_color, border_radius=4, padding=pad(h=6, v=2)),
                    ft.Text(f"Campioni: {n_s}", size=10, color=TEXT_DIM),
                ], spacing=6),
                ft.Container(height=4),
                make_kv_row("MAE", f"{mae_disp:.2f}{unit}", q_color),
                make_kv_row("RMSE", f"{rmse_disp:.2f}{unit}", YELLOW if rmse > 5 else TEXT),
                make_kv_row("Bias", f"{bias_disp:+.2f}{unit}", RED if abs(bias) > 2 else TEXT),
                ft.Container(height=4),
                # Barra visiva MAE
                ft.Text("MAE (più basso = meglio)", size=9, color=TEXT_DIM),
                ft.Container(
                    content=ft.Stack([
                        ft.Container(width=200, height=8,
                                      bgcolor=ft.Colors.with_opacity(0.1, TEXT), border_radius=4),
                        ft.Container(
                            width=max(4, min(200, mae_disp / 10 * 200)), height=8,
                            bgcolor=q_color, border_radius=4),
                    ]),
                    width=200, height=8,
                ),
                ft.Container(height=4),
                ft.Text(
                    f"In media le previsioni sbagliano di {mae_disp:.1f}{unit}. "
                    + ("Ottimo per scommettere!" if mae < 2 else
                       "Buono, ma tieni un margine di sicurezza." if mae < 4 else
                       "Prudenza: le previsioni non sono molto precise."),
                    size=9, color=q_color, italic=True,
                ),
            ], spacing=3)
        else:
            stor_accuracy_panel.content = ft.Text(
                "Dati accuratezza non disponibili", color=TEXT_DIM, size=12)

        # ---- Analisi Bias ----
        # Carica accuracy per multiple variabili
        bias_vars = [
            ("temperature_2m_max", "Temp Max"),
            ("temperature_2m_min", "Temp Min"),
        ]
        bias_rows = []
        for bvar, blabel in bias_vars:
            b_acc = api_get(f"/analysis/{current_city}/accuracy?variable={bvar}")
            b_data = (b_acc or {}).get("accuracy")
            if b_data and isinstance(b_data, dict):
                b_bias = b_data.get("bias", 0) or 0
                b_mae = b_data.get("mae", 0) or 0
                b_bias_c = b_bias * 5 / 9
                b_mae_c = b_mae * 5 / 9
                bias_color = RED if abs(b_bias) > 2 else (YELLOW if abs(b_bias) > 1 else GREEN)
                direction = "sovrastima" if b_bias > 0 else "sottostima"

                bias_rows.append(ft.Container(
                    content=ft.Column([
                        ft.Row([
                            ft.Text(blabel, size=11, color=TEXT, weight=ft.FontWeight.BOLD, width=80),
                            ft.Container(
                                content=ft.Text(direction, size=9, color=BG,
                                                 weight=ft.FontWeight.BOLD),
                                bgcolor=bias_color, border_radius=3, padding=pad(h=5, v=1)),
                            ft.Text(f"bias: {b_bias_c:+.2f}°C", size=10, color=bias_color),
                            ft.Text(f"MAE: {b_mae_c:.2f}°C", size=10, color=TEXT_DIM),
                        ], spacing=6),
                        # Barra bias (centrata su 0)
                        ft.Container(
                            content=ft.Stack([
                                ft.Container(width=200, height=6,
                                              bgcolor=ft.Colors.with_opacity(0.1, TEXT),
                                              border_radius=3),
                                # Centro
                                ft.Container(width=2, height=10,
                                              bgcolor=TEXT_DIM, left=100, top=-2),
                                # Bias bar
                                ft.Container(
                                    width=max(2, abs(b_bias_c) / 5 * 100),
                                    height=6,
                                    bgcolor=bias_color,
                                    border_radius=3,
                                    left=100 if b_bias > 0 else max(0, 100 - abs(b_bias_c) / 5 * 100),
                                ),
                            ]),
                            width=200, height=10,
                        ),
                    ], spacing=2),
                    padding=pad(h=4, v=4),
                ))

        if bias_rows:
            stor_bias_panel.content = ft.Column([
                *bias_rows,
                ft.Container(height=4),
                ft.Text("Centro = bias 0 (perfetto). Sinistra = sottostima, destra = sovrastima.",
                         size=9, color=TEXT_DIM, italic=True),
            ], spacing=4)
        else:
            stor_bias_panel.content = ft.Text(
                "Dati bias non disponibili", color=TEXT_DIM, size=12)

        # ---- Metriche Avanzate ----
        metrics_data = api_get("/market/bets/metrics", timeout=15)
        if metrics_data and metrics_data.get("n_trades", 0) > 0:
            sharpe = metrics_data.get("sharpe_ratio", 0)
            sortino = metrics_data.get("sortino_ratio", 0)
            md = metrics_data.get("max_drawdown", 0)
            pf = metrics_data.get("profit_factor", 0)
            calmar = metrics_data.get("calmar_ratio", 0)
            avg_w = metrics_data.get("avg_win", 0)
            avg_l = metrics_data.get("avg_loss", 0)
            exp = metrics_data.get("expectancy", 0)

            sharpe_color = GREEN if sharpe > 1 else (YELLOW if sharpe > 0.5 else RED)
            md_color = GREEN if md < 0.15 else (YELLOW if md < 0.25 else RED)
            pf_color = GREEN if pf > 1.5 else (YELLOW if pf > 1 else RED)

            stor_metrics_card.content = ft.Column([
                make_kv_row("Sharpe Ratio", f"{sharpe:.2f}", sharpe_color),
                make_kv_row("Sortino Ratio", f"{sortino:.2f}",
                              GREEN if sortino > 1 else (YELLOW if sortino > 0.5 else RED)),
                make_kv_row("Max Drawdown", f"{md:.1%}", md_color),
                make_kv_row("Profit Factor", f"{pf:.2f}", pf_color),
                make_kv_row("Calmar Ratio", f"{calmar:.2f}", TEXT),
                ft.Container(height=4),
                make_kv_row("Avg Win", f"{avg_w:+.1%}", GREEN),
                make_kv_row("Avg Loss", f"{avg_l:+.1%}", RED),
                make_kv_row("Expectancy", f"{exp:+.1%}",
                              GREEN if exp > 0 else RED),
                ft.Container(height=4),
                ft.Text(f"Basato su {metrics_data.get('n_trades', 0)} trade risolti",
                         size=9, color=TEXT_DIM, italic=True),
            ], spacing=3)

            # Equity curve
            eq_data = metrics_data.get("equity_curve", [])
            if eq_data and len(eq_data) > 1:
                stor_equity_curve.content = ft.Column([
                    build_sparkline(eq_data, width=350, height=60,
                                     color=GREEN if eq_data[-1] >= eq_data[0] else RED),
                    ft.Text(f"${eq_data[0]:.0f} → ${eq_data[-1]:.0f} "
                             f"({(eq_data[-1] - eq_data[0]) / eq_data[0]:+.1%})",
                             size=10, color=GREEN if eq_data[-1] >= eq_data[0] else RED),
                ], spacing=4)
            else:
                stor_equity_curve.content = ft.Text("Dati insufficienti", color=TEXT_DIM, size=12)
        else:
            stor_metrics_card.content = ft.Text(
                "Nessun trade risolto per calcolare le metriche", color=TEXT_DIM, size=12)
            stor_equity_curve.content = ft.Text("Nessun dato", color=TEXT_DIM, size=12)

        # ---- Backtest Button ----
        def _run_backtest(e):
            bt_result.content = ft.Row([
                ft.ProgressRing(width=16, height=16, stroke_width=2, color=ACCENT),
                ft.Text("Eseguendo backtest...", color=TEXT_DIM, size=11),
            ], spacing=6)
            safe_update()
            try:
                city_slug = bt_city.value or "nyc"
                min_e = (bt_min_edge.value or 5) / 100.0
                min_c = (bt_min_conf.value or 60) / 100.0
                bt_data = api_get(
                    f"/market/backtest?city={city_slug}&min_edge={min_e}"
                    f"&min_confidence={min_c}&kelly_fraction=0.25",
                    timeout=60,
                )
                if bt_data and bt_data.get("n_trades", 0) > 0:
                    eq = bt_data.get("equity_curve", [])
                    bt_items = [
                        make_kv_row("P&L Totale", f"${bt_data['total_pnl']:+.0f}",
                                      GREEN if bt_data["total_pnl"] > 0 else RED),
                        make_kv_row("Trade", f"{bt_data['n_trades']}", TEXT),
                        make_kv_row("Win Rate", f"{bt_data['win_rate']:.0%}",
                                      GREEN if bt_data["win_rate"] > 0.5 else RED),
                        make_kv_row("Sharpe", f"{bt_data['sharpe_ratio']:.2f}", TEXT),
                        make_kv_row("Max DD", f"{bt_data['max_drawdown']:.1%}", TEXT),
                        make_kv_row("Profit Factor", f"{bt_data['profit_factor']:.2f}", TEXT),
                    ]
                    if eq and len(eq) > 1:
                        bt_items.append(ft.Container(height=4))
                        bt_items.append(build_sparkline(eq, width=350, height=50,
                                                         color=GREEN if eq[-1] >= eq[0] else RED))
                    bt_result.content = ft.Column(bt_items, spacing=3)
                else:
                    bt_result.content = ft.Text(
                        "Nessun trade generato con questi parametri", color=TEXT_DIM, size=11)
            except Exception as ex:
                bt_result.content = ft.Text(f"Errore: {ex}", color=RED, size=11)
            safe_update()

        stor_backtest_card.content = ft.Button(
            "Esegui Backtest",
            icon=ft.Icons.PLAY_ARROW,
            bgcolor=ACCENT,
            color=BG,
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=12),
                padding=pad(h=14, v=10),
            ),
            on_click=lambda e: threaded_load(_run_backtest, e),
        )

        # ---- Storico Scommesse ----
        pnl_data = load_pnl()
        bets = pnl_data.get("bets", [])
        if bets:
            bet_rows = []
            for b in reversed(bets[-20:]):  # Ultime 20
                status = b.get("status", "pending")
                s_color = GREEN if status == "won" else (RED if status == "lost" else YELLOW)
                s_label = {"won": "VINTA", "lost": "PERSA", "pending": "ATTIVA"}.get(status, "?")
                pnl_val = b.get("pnl", 0)

                bet_row_controls = [
                    ft.Text(f"#{b.get('id', '?')}", size=9, color=TEXT_DIM, width=25),
                    ft.Container(
                        content=ft.Text(s_label, size=8, color=BG, weight=ft.FontWeight.BOLD),
                        bgcolor=s_color, border_radius=3, padding=pad(h=4, v=1), width=50,
                    ),
                    ft.Text(b.get("city", "—"), size=10, color=ACCENT, width=55),
                    ft.Text(b.get("outcome", "—")[:20], size=10, color=TEXT, width=110,
                             overflow=ft.TextOverflow.ELLIPSIS),
                    ft.Text(f"${b.get('stake', 0):.0f}", size=10, color=TEXT_DIM, width=40),
                    ft.Text(f"Conf {b.get('confidence', 0):.0%}", size=9, color=TEXT_DIM, width=55),
                ]
                if status != "pending":
                    bet_row_controls.append(
                        ft.Text(f"${pnl_val:+,.0f}", size=10,
                                 color=GREEN if pnl_val >= 0 else RED,
                                 weight=ft.FontWeight.BOLD, width=55))
                else:
                    # Pulsanti risolvi — closure con default arg per catturare bid
                    bid = b.get("id", 0)

                    def _resolve_won(e, _bid=bid):
                        try:
                            resolve_bet(_bid, True)
                        except Exception as ex:
                            logger.error("Resolve won failed for bet %s: %s", _bid, ex)
                        threaded_load(load)

                    def _resolve_lost(e, _bid=bid):
                        try:
                            resolve_bet(_bid, False)
                        except Exception as ex:
                            logger.error("Resolve lost failed for bet %s: %s", _bid, ex)
                        threaded_load(load)

                    bet_row_controls.append(ft.Row([
                        ft.Button("V", bgcolor=GREEN, color=BG,
                                   style=ft.ButtonStyle(
                                       shape=ft.RoundedRectangleBorder(radius=8),
                                       padding=pad(h=6, v=4),
                                   ),
                                   on_click=_resolve_won, width=28, height=24),
                        ft.Button("X", bgcolor=RED, color=BG,
                                   style=ft.ButtonStyle(
                                       shape=ft.RoundedRectangleBorder(radius=8),
                                       padding=pad(h=6, v=4),
                                   ),
                                   on_click=_resolve_lost, width=28, height=24),
                    ], spacing=2))

                bet_rows.append(ft.Container(
                    content=ft.Row(bet_row_controls, spacing=4),
                    bgcolor=ft.Colors.with_opacity(0.05, s_color),
                    border=ft.Border(
                        left=ft.BorderSide(2, s_color),
                        top=ft.BorderSide(1, ft.Colors.with_opacity(0.10, s_color)),
                        right=ft.BorderSide(1, ft.Colors.with_opacity(0.25, OUTLINE_SOFT)),
                        bottom=ft.BorderSide(1, ft.Colors.with_opacity(0.25, OUTLINE_SOFT)),
                    ),
                    border_radius=8, padding=pad(h=8, v=4),
                ))

            # Header
            hdr = ft.Container(
                content=ft.Row([
                    ft.Text("#", size=9, color=TEXT_DIM, weight=ft.FontWeight.BOLD, width=25),
                    ft.Text("Stato", size=9, color=TEXT_DIM, weight=ft.FontWeight.BOLD, width=50),
                    ft.Text("Città", size=9, color=TEXT_DIM, weight=ft.FontWeight.BOLD, width=55),
                    ft.Text("Outcome", size=9, color=TEXT_DIM, weight=ft.FontWeight.BOLD, width=110),
                    ft.Text("Stake", size=9, color=TEXT_DIM, weight=ft.FontWeight.BOLD, width=40),
                    ft.Text("Conf.", size=9, color=TEXT_DIM, weight=ft.FontWeight.BOLD, width=55),
                    ft.Text("P&L", size=9, color=TEXT_DIM, weight=ft.FontWeight.BOLD, width=55),
                ], spacing=4),
                padding=pad(h=8, v=6),
                border_radius=10,
                bgcolor=ft.Colors.with_opacity(0.18, SURFACE_2),
                border=ft.Border.all(1, ft.Colors.with_opacity(0.35, OUTLINE_SOFT)),
            )
            stor_bet_history.content = ft.Column([hdr] + bet_rows, spacing=2)
        else:
            stor_bet_history.content = ft.Text(
                "Nessuna scommessa registrata", color=TEXT_DIM, size=12)

        # ---- ROI per Città ----
        if bets:
            city_pnl = {}
            for b in bets:
                c = b.get("city", "Sconosciuta") or "Sconosciuta"
                if c not in city_pnl:
                    city_pnl[c] = {"pnl": 0, "bets": 0, "won": 0, "staked": 0}
                city_pnl[c]["bets"] += 1
                city_pnl[c]["staked"] += b.get("stake", 0)
                if b.get("status") == "won":
                    city_pnl[c]["won"] += 1
                    city_pnl[c]["pnl"] += b.get("pnl", 0)
                elif b.get("status") == "lost":
                    city_pnl[c]["pnl"] += b.get("pnl", 0)

            # Ordina per P&L
            sorted_cities = sorted(city_pnl.items(), key=lambda x: x[1]["pnl"], reverse=True)
            roi_rows = []
            for city, d in sorted_cities:
                roi = d["pnl"] / d["staked"] if d["staked"] > 0 else 0
                wr = d["won"] / d["bets"] if d["bets"] > 0 else 0
                roi_color = GREEN if d["pnl"] >= 0 else RED
                roi_rows.append(ft.Container(
                    content=ft.Row([
                        ft.Text(city, size=11, color=TEXT, weight=ft.FontWeight.BOLD, width=80),
                        ft.Text(f"${d['pnl']:+,.0f}", size=11, color=roi_color,
                                 weight=ft.FontWeight.BOLD, width=55),
                        ft.Text(f"ROI {roi:+.0%}", size=10, color=roi_color, width=55),
                        ft.Text(f"{d['bets']} bet", size=10, color=TEXT_DIM, width=40),
                        ft.Text(f"WR {wr:.0%}", size=10,
                                 color=GREEN if wr > 0.5 else RED, width=45),
                    ], spacing=4),
                    bgcolor=ft.Colors.with_opacity(0.05, roi_color),
                    border=ft.Border(
                        left=ft.BorderSide(2, roi_color),
                        top=ft.BorderSide(1, ft.Colors.with_opacity(0.10, roi_color)),
                        right=ft.BorderSide(1, ft.Colors.with_opacity(0.25, OUTLINE_SOFT)),
                        bottom=ft.BorderSide(1, ft.Colors.with_opacity(0.25, OUTLINE_SOFT)),
                    ),
                    border_radius=8, padding=pad(h=8, v=4),
                ))
            stor_roi_city.content = ft.Column(roi_rows, spacing=3)
        else:
            stor_roi_city.content = ft.Text(
                "Nessuna scommessa registrata", color=TEXT_DIM, size=12)

        safe_update()

    return build, load
