"""Sistema (System) section: health, DB stats, collections, verification."""

import time
import threading
from datetime import datetime

import flet as ft

from weather_trader.constants import (
    ACCENT, ACCENT2, BG, CARD, GREEN, ORANGE, RED, TEXT, TEXT_DIM, YELLOW,
)
from weather_trader.api_client import api_get, api_post
from weather_trader.app_state import AppState
from weather_trader.widgets.factory import (
    make_badge, make_card, make_empty_state, make_info_box, make_kv_row,
    make_section_title, pad,
)


def create_sistema(page: ft.Page, state: AppState, safe_update):
    """Factory for the Sistema (System) section."""

    ROW_SYSTEM_TWIN_H = 300

    # ------------------------------------------------------------------
    # local threaded_load helper
    # ------------------------------------------------------------------
    def threaded_load(fn, *args):
        def _run():
            try:
                fn(*args)
            except Exception as e:
                import logging
                logging.getLogger("weather_trader.sistema").error("Load error: %s", e)
        threading.Thread(target=_run, daemon=True).start()

    # ------------------------------------------------------------------
    # mutable containers
    # ------------------------------------------------------------------
    sys_health_container = ft.Container(padding=10)
    sys_db_stats = ft.Column([], spacing=6)
    sys_collections_log = ft.Column([], spacing=4)
    sys_verification_table = ft.DataTable(
        columns=[
            ft.DataColumn(ft.Text("Variabile", color=TEXT_DIM, size=11, weight=ft.FontWeight.BOLD)),
            ft.DataColumn(ft.Text("N", color=TEXT_DIM, size=11, weight=ft.FontWeight.BOLD)),
            ft.DataColumn(ft.Text("Bias", color=TEXT_DIM, size=11, weight=ft.FontWeight.BOLD)),
            ft.DataColumn(ft.Text("MAE", color=TEXT_DIM, size=11, weight=ft.FontWeight.BOLD)),
            ft.DataColumn(ft.Text("RMSE", color=TEXT_DIM, size=11, weight=ft.FontWeight.BOLD)),
        ],
        border_radius=12,
        heading_row_color=ft.Colors.with_opacity(0.05, TEXT),
        data_row_max_height=36,
    )
    if hasattr(sys_verification_table, "column_spacing"):
        sys_verification_table.column_spacing = 20
    if hasattr(sys_verification_table, "divider_thickness"):
        sys_verification_table.divider_thickness = 0.6
    if hasattr(sys_verification_table, "horizontal_margin"):
        sys_verification_table.horizontal_margin = 12
    sys_freshness = ft.Column([], spacing=4)

    # ------------------------------------------------------------------
    # verify button + handler
    # ------------------------------------------------------------------
    def on_verify_click(e):
        threaded_load(_run_verification)

    def _run_verification():
        api_post("/analysis/verify", {"lookback_days": 7})
        time.sleep(2)
        load()

    verify_btn = ft.Button(
        "Verifica Forecast",
        icon=ft.Icons.VERIFIED,
        bgcolor=ACCENT, color=BG,
        style=ft.ButtonStyle(
            shape=ft.RoundedRectangleBorder(radius=14),
            padding=pad(h=14, v=10),
        ),
        on_click=on_verify_click,
    )

    # ------------------------------------------------------------------
    # build
    # ------------------------------------------------------------------
    def build():
        return ft.Column([
            ft.Row([
                make_section_title("Sistema", ft.Icons.SETTINGS, TEXT_DIM),
                ft.Container(expand=True),
                verify_btn,
            ]),
            make_info_box(
                "Stato del sistema: se tutto funziona, i dati saranno aggiornati e le previsioni affidabili. "
                "Se qualcosa è rosso, le scommesse potrebbero basarsi su dati vecchi — aspetta che si risolva.",
            ),
            ft.Container(height=10),
            # Health
            make_card(ft.Column([
                ft.Row([
                    ft.Icon(ft.Icons.MONITOR_HEART, color=GREEN, size=18),
                    ft.Text("Server Health", size=14, weight=ft.FontWeight.BOLD, color=TEXT),
                ], spacing=6),
                make_info_box("Il server raccoglie dati meteo e li analizza. Verde = tutto OK, Rosso = problemi.", GREEN),
                ft.Container(height=4),
                sys_health_container,
            ])),
            ft.Container(height=10),
            # DB Stats
            make_card(ft.Column([
                ft.Row([
                    ft.Icon(ft.Icons.STORAGE, color=ACCENT, size=18),
                    ft.Text("Database Stats", size=14, weight=ft.FontWeight.BOLD, color=TEXT),
                ], spacing=6),
                make_info_box("Quantità di dati nel database. Più dati = previsioni migliori. La barra mostra quanto siamo vicini all'obiettivo."),
                ft.Container(height=4),
                sys_db_stats,
            ])),
            ft.Container(height=10),
            # Collections log + Freshness
            ft.ResponsiveRow([
                ft.Column([
                    make_card(ft.Column([
                        ft.Row([
                            ft.Icon(ft.Icons.LIST_ALT, color=YELLOW, size=18),
                            ft.Text("Log Collezioni", size=14, weight=ft.FontWeight.BOLD, color=TEXT),
                        ], spacing=6),
                        make_info_box("Ultime raccolte dati dal server. 'success' = dati scaricati con successo.", YELLOW),
                        ft.Container(height=4),
                        sys_collections_log,
                    ]), height=ROW_SYSTEM_TWIN_H),
                ], col={"xs": 12, "md": 6}),
                ft.Column([
                    make_card(ft.Column([
                        ft.Row([
                            ft.Icon(ft.Icons.SCHEDULE, color=ACCENT, size=18),
                            ft.Text("Freschezza Dati", size=14, weight=ft.FontWeight.BOLD, color=TEXT),
                        ], spacing=6),
                        make_info_box("Quanto tempo fa sono stati aggiornati i dati. Verde < 1h, Giallo < 6h, Rosso > 6h."),
                        ft.Container(height=4),
                        sys_freshness,
                    ]), height=ROW_SYSTEM_TWIN_H),
                ], col={"xs": 12, "md": 6}),
            ]),
            ft.Container(height=10),
            # Verification
            make_card(ft.Column([
                ft.Row([
                    ft.Icon(ft.Icons.VERIFIED, color=GREEN, size=18),
                    ft.Text("Metriche Verifica", size=14, weight=ft.FontWeight.BOLD, color=TEXT),
                ], spacing=6),
                make_info_box(
                    "Confronto tra previsioni e dati reali. Bias = tendenza a sbagliare in una direzione. "
                    "MAE = errore medio. RMSE = errore medio pesato (penalizza errori grandi).",
                    GREEN,
                ),
                ft.Container(height=4),
                ft.Container(
                    content=sys_verification_table,
                    height=300,
                    border_radius=14,
                    bgcolor=ft.Colors.with_opacity(0.18, TEXT),
                    border=ft.Border.all(1, ft.Colors.with_opacity(0.08, TEXT)),
                    padding=pad(h=6, v=6),
                ),
            ])),
        ], scroll=ft.ScrollMode.AUTO, spacing=0)

    # ------------------------------------------------------------------
    # load
    # ------------------------------------------------------------------
    def load():
        """Carica dati sistema."""
        try:
            _load_impl()
        except Exception as e:
            import logging
            logging.getLogger("weather_trader.sistema").error("Sistema load error: %s", e)
            sys_health_container.content = make_empty_state(
                ft.Icons.ERROR_OUTLINE, f"Errore caricamento sistema: {e}")
            safe_update()

    def _load_impl():
        current_city = state.current_city
        cache = state.cache

        # ---- Health ----
        health = api_get("/health")
        cache["health"] = health
        if health:
            is_ok = health.get("status") == "ok"
            db_ok = health.get("database") == "connected"
            sys_health_container.content = ft.Column([
                ft.Row([
                    ft.Container(
                        width=14, height=14, border_radius=7,
                        bgcolor=GREEN if is_ok else RED,
                    ),
                    ft.Text("Server operativo" if is_ok else "Server degradato",
                             size=13, color=GREEN if is_ok else RED,
                             weight=ft.FontWeight.BOLD),
                ], spacing=8),
                make_kv_row("Status", health.get("status", "—"),
                             GREEN if is_ok else RED),
                make_kv_row("Database", health.get("database", "—"),
                             GREEN if db_ok else RED),
                make_kv_row("Timestamp", health.get("timestamp", "—"), TEXT_DIM),
            ], spacing=4)
        else:
            sys_health_container.content = ft.Row([
                ft.Container(width=14, height=14, border_radius=7, bgcolor=RED),
                ft.Text("Server non raggiungibile", size=13, color=RED),
            ], spacing=8)

        # ---- DB Metrics ----
        metrics = api_get("/metrics")
        cache["metrics"] = metrics
        if metrics:
            db_targets = {
                "forecasts_hourly": 100000,
                "forecasts_daily": 10000,
                "ensemble_members": 50000,
                "observations": 50000,
                "climate_normals": 5000,
            }
            sys_db_stats.controls.clear()
            for key, target in db_targets.items():
                count = metrics.get(key, 0)
                ratio = min(1.0, count / target) if target else 0
                label = key.replace("_", " ").title()
                color = GREEN if ratio > 0.5 else (YELLOW if ratio > 0.2 else RED)

                sys_db_stats.controls.append(ft.Column([
                    ft.Row([
                        ft.Text(label, size=12, color=TEXT, width=160),
                        ft.Text(f"{count:,}", size=12, color=ACCENT, weight=ft.FontWeight.BOLD),
                        ft.Container(expand=True),
                        ft.Text(f"{ratio:.0%}", size=11, color=color),
                    ]),
                    ft.ProgressBar(value=ratio, color=color,
                                    bgcolor=ft.Colors.with_opacity(0.1, TEXT)),
                ], spacing=2))

            # ---- Collections log ----
            collections = metrics.get("recent_collections", [])
            sys_collections_log.controls.clear()
            if collections:
                for c in collections[:10]:
                    status = c.get("status", "unknown")
                    s_color = GREEN if status == "success" else RED
                    ts_val = c.get("at", c.get("timestamp", "—"))
                    sys_collections_log.controls.append(ft.Row([
                        make_badge(status, s_color, BG if status == "success" else TEXT),
                        ft.Text(c.get("collector", "—"), size=11, color=TEXT, width=100,
                                 overflow=ft.TextOverflow.ELLIPSIS),
                        ft.Text(c.get("city", "—"), size=11, color=TEXT_DIM),
                        ft.Text(str(ts_val)[:19], size=10, color=TEXT_DIM),
                    ], spacing=6))
            else:
                sys_collections_log.controls.append(
                    ft.Text("Nessun log disponibile", color=TEXT_DIM, size=12))

            # ---- Data freshness ----
            sys_freshness.controls.clear()
            if collections:
                # Raggruppa per collector e mostra ultimo timestamp
                collectors_seen = {}
                for c in collections:
                    coll_name = c.get("collector", "unknown")
                    if coll_name not in collectors_seen:
                        collectors_seen[coll_name] = c.get("at", c.get("timestamp", "—"))
                for coll_name, ts in collectors_seen.items():
                    try:
                        ts_str = str(ts).replace("Z", "").replace("+00:00", "")
                        ts_dt = datetime.fromisoformat(ts_str)
                        age_min = (datetime.now() - ts_dt).total_seconds() / 60
                        age_str = f"{age_min:.0f} min fa" if age_min < 60 else f"{age_min / 60:.1f}h fa"
                        age_color = GREEN if age_min < 60 else (YELLOW if age_min < 360 else RED)
                    except Exception:
                        age_str = str(ts)[:19]
                        age_color = TEXT_DIM
                    sys_freshness.controls.append(
                        make_kv_row(coll_name.replace("_", " ").title(), age_str, age_color))
            else:
                sys_freshness.controls.append(
                    ft.Text("Nessun dato", color=TEXT_DIM, size=12))

        # ---- Verification ----
        verif = api_get(f"/analysis/verification/summary?city_slug={current_city}")
        summary = (verif or {}).get("summary", {})
        cache["verification"] = summary

        sys_verification_table.rows = []
        if summary:
            # summary potrebbe essere un dict con variabili come chiavi, o una lista
            if isinstance(summary, dict):
                for var_name, var_stats in summary.items():
                    if isinstance(var_stats, dict):
                        sys_verification_table.rows.append(ft.DataRow(cells=[
                            ft.DataCell(ft.Text(var_name.replace("_", " ")[:20],
                                                 color=TEXT, size=11)),
                            ft.DataCell(ft.Text(str(var_stats.get("n", "—")),
                                                 color=TEXT, size=11)),
                            ft.DataCell(ft.Text(f"{var_stats.get('bias', 0):.2f}",
                                                 color=TEXT, size=11)),
                            ft.DataCell(ft.Text(f"{var_stats.get('mae', 0):.2f}",
                                                 color=ACCENT, size=11)),
                            ft.DataCell(ft.Text(f"{var_stats.get('rmse', 0):.2f}",
                                                 color=YELLOW if var_stats.get('rmse', 0) > 5 else TEXT,
                                                 size=11)),
                        ]))
            elif isinstance(summary, list):
                for item in summary:
                    if isinstance(item, dict):
                        sys_verification_table.rows.append(ft.DataRow(cells=[
                            ft.DataCell(ft.Text(str(item.get("variable", "—"))[:20],
                                                 color=TEXT, size=11)),
                            ft.DataCell(ft.Text(str(item.get("n", "—")),
                                                 color=TEXT, size=11)),
                            ft.DataCell(ft.Text(f"{item.get('bias', 0):.2f}",
                                                 color=TEXT, size=11)),
                            ft.DataCell(ft.Text(f"{item.get('mae', 0):.2f}",
                                                 color=ACCENT, size=11)),
                            ft.DataCell(ft.Text(f"{item.get('rmse', 0):.2f}",
                                                 color=YELLOW if item.get('rmse', 0) > 5 else TEXT,
                                                 size=11)),
                        ]))

        if not sys_verification_table.rows:
            sys_verification_table.rows.append(ft.DataRow(cells=[
                ft.DataCell(ft.Text("—", color=TEXT_DIM, size=11)),
                ft.DataCell(ft.Text("—", color=TEXT_DIM, size=11)),
                ft.DataCell(ft.Text("—", color=TEXT_DIM, size=11)),
                ft.DataCell(ft.Text("—", color=TEXT_DIM, size=11)),
                ft.DataCell(ft.Text("—", color=TEXT_DIM, size=11)),
            ]))

        # Accuracy per variable
        accuracy = api_get(f"/analysis/{current_city}/accuracy?variable=temperature_2m_max")
        acc_data = (accuracy or {}).get("accuracy")
        if acc_data and isinstance(acc_data, dict):
            mae = acc_data.get("mae", 0) or 0
            rmse = acc_data.get("rmse", 0) or 0
            bias = acc_data.get("bias", 0) or 0
            n_samples = acc_data.get("n_samples", "—")
            if sys_verification_table.rows and sys_verification_table.rows[0].cells[0].content.value == "—":
                sys_verification_table.rows.clear()
            sys_verification_table.rows.append(ft.DataRow(cells=[
                ft.DataCell(ft.Text("temp 2m max", color=GREEN, size=11)),
                ft.DataCell(ft.Text(str(n_samples), color=TEXT, size=11)),
                ft.DataCell(ft.Text(f"{bias:.2f}", color=TEXT, size=11)),
                ft.DataCell(ft.Text(f"{mae:.2f}", color=ACCENT, size=11)),
                ft.DataCell(ft.Text(f"{rmse:.2f}",
                                     color=YELLOW if rmse > 5 else TEXT, size=11)),
            ]))

        safe_update()

    return build, load
