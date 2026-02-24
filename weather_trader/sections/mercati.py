"""Mercati (Markets) section: filter controls, market cards, best bets, risk, P&L."""

import logging
import threading
from datetime import datetime

import flet as ft

from weather_trader.constants import (
    ACCENT, ACCENT2, BG, CITY_NAMES, CITY_SLUGS,
    EDGE_GOOD_THRESHOLD, INITIAL_BANKROLL, KELLY_FRACTION, MAX_BET_PCT,
    SIGNAL_BET_THRESHOLD, SIGNAL_CAUTION_THRESHOLD,
    GREEN, ORANGE, OUTLINE_SOFT, RED, SURFACE_2, TEXT, TEXT_DIM, YELLOW, f2c,
)
from weather_trader.api_client import api_get, api_post
from weather_trader.app_state import AppState
from weather_trader.logic.pnl_tracker import load_pnl, get_pnl_stats, record_bet
from weather_trader.logic.risk_manager import check_portfolio_risk
from weather_trader.widgets.factory import (
    make_badge, make_card, make_empty_state, make_info_box, make_kv_row,
    make_loading_indicator, make_section_title, pad,
)
from weather_trader.widgets.confidence import (
    calculate_confidence, build_confidence_meter,
    calculate_betting_signal, build_betting_signal_widget,
)
from weather_trader.widgets.pnl_widgets import build_pnl_sparkline, build_risk_gauge
from weather_trader.widgets.distribution import build_edge_bar_chart

logger = logging.getLogger("weather_trader.sections.mercati")


def create_mercati(page: ft.Page, state: AppState, safe_update):
    """Create mercati section. Returns (build, load) tuple."""

    ROW_TOP_H = 360

    def style_select(ctrl, border_color=ACCENT):
        """Make dropdowns look like MD3 filled fields (compat-friendly)."""
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
        if hasattr(ctrl, "content_padding"):
            ctrl.content_padding = pad(h=10, v=8)
        return ft.Container(
            content=ctrl,
            border_radius=12,
            bgcolor=ft.Colors.with_opacity(0.24, SURFACE_2),
            border=ft.Border.all(1, ft.Colors.with_opacity(0.55, OUTLINE_SOFT if border_color == ACCENT else border_color)),
            padding=pad(h=4, v=2),
        )

    def field_shell(label, control, hint=None, min_height=None):
        body = [
            ft.Text(label, size=10, color=TEXT_DIM, weight=ft.FontWeight.W_600),
            control,
        ]
        if hint:
            body.append(ft.Text(hint, size=9, color=TEXT_DIM))
        return ft.Container(
            content=ft.Column(body, spacing=4),
            padding=pad(h=10, v=8),
            border_radius=14,
            bgcolor=ft.Colors.with_opacity(0.18, SURFACE_2),
            border=ft.Border.all(1, ft.Colors.with_opacity(0.45, OUTLINE_SOFT)),
            height=min_height,
        )

    mkt_load_state_lock = threading.Lock()
    mkt_load_running = False
    mkt_load_pending = False
    _load_generation = [0]  # Incremented on each new load request; stale loads check this.

    def _threaded_load(fn):
        """Esegue fn in background thread. Se un load e' gia' in corso, mostra
        subito il loading indicator e accoda il nuovo load."""
        nonlocal mkt_load_running, mkt_load_pending

        _load_generation[0] += 1

        with mkt_load_state_lock:
            if mkt_load_running:
                mkt_load_pending = True
                # Mostra subito il loading indicator cosi' l'utente vede feedback
                mkt_cards_list.controls = [make_loading_indicator("Cambio filtro in corso...")]
                mkt_best_bets.controls = [ft.Text("Aggiornamento...", color=TEXT_DIM, size=12)]
                safe_update()
                return
            mkt_load_running = True

        def _run():
            nonlocal mkt_load_running, mkt_load_pending
            while True:
                try:
                    fn()
                except Exception as e:
                    logger.error("Threaded load error: %s", e)

                with mkt_load_state_lock:
                    if mkt_load_pending:
                        mkt_load_pending = False
                        continue
                    mkt_load_running = False
                    break

        threading.Thread(target=_run, daemon=True).start()

    # --- Filter controls ---
    mkt_filter_edge = ft.Slider(
        min=0, max=20, divisions=20, value=0,
        label="{value}%",
        active_color=ACCENT,
        width=200,
    )
    mkt_filter_city = ft.Dropdown(
        width=160,
        border_color=ACCENT,
        color=TEXT,
        text_size=13,
        content_padding=pad(h=10, v=6),
        options=[ft.dropdown.Option(key="all", text="Tutte le città")]
                + [ft.dropdown.Option(key=s, text=n) for s, n in zip(CITY_SLUGS, CITY_NAMES)],
        value="all",
    )
    mkt_filter_sort = ft.Dropdown(
        width=160,
        border_color=ACCENT,
        color=TEXT,
        text_size=13,
        content_padding=pad(h=10, v=6),
        options=[
            ft.dropdown.Option(key="edge_desc", text="Edge ↓"),
            ft.dropdown.Option(key="edge_asc", text="Edge ↑"),
            ft.dropdown.Option(key="ev_desc", text="EV ↓"),
        ],
        value="edge_desc",
    )

    mkt_best_bets = ft.Column([], spacing=6)
    mkt_cards_list = ft.Column([], spacing=8)
    mkt_stats = ft.Container(padding=8)
    mkt_status_text = ft.Text("", color=TEXT_DIM, size=12)
    mkt_risk_panel = ft.Container(padding=8)
    mkt_pnl_mini = ft.Container(padding=8)
    mkt_sort_confidence = ft.Dropdown(
        width=160,
        border_color=ACCENT,
        color=TEXT,
        text_size=13,
        content_padding=pad(h=10, v=6),
        options=[
            ft.dropdown.Option(key="confidence_desc", text="Confidenza ↓"),
            ft.dropdown.Option(key="edge_desc", text="Edge ↓"),
            ft.dropdown.Option(key="edge_asc", text="Edge ↑"),
            ft.dropdown.Option(key="ev_desc", text="EV ↓"),
            ft.dropdown.Option(key="stake_desc", text="Stake ↓"),
        ],
        value="confidence_desc",
    )
    mkt_filter_signal = ft.Dropdown(
        width=160,
        border_color=GREEN,
        color=TEXT,
        text_size=13,
        content_padding=pad(h=10, v=6),
        options=[
            ft.dropdown.Option(key="all", text="Tutti i segnali"),
            ft.dropdown.Option(key="SCOMMETTI", text="Solo SCOMMETTI"),
            ft.dropdown.Option(key="CAUTELA", text="Solo CAUTELA"),
            ft.dropdown.Option(key="NO", text="Solo NON SCOMMETTERE"),
        ],
        value="all",
    )
    mkt_filter_source = ft.Dropdown(
        width=160,
        border_color=ACCENT2,
        color=TEXT,
        text_size=13,
        content_padding=pad(h=10, v=6),
        options=[
            ft.dropdown.Option(key="all", text="Tutte le fonti"),
            ft.dropdown.Option(key="live", text="Solo LIVE (Polymarket)"),
            ft.dropdown.Option(key="simulated", text="Solo Simulate (Analisi)"),
        ],
        value="all",
    )
    mkt_filter_open_only = ft.Checkbox(
        label="Solo mercati aperti", value=True,
        label_style=ft.TextStyle(color=TEXT_DIM, size=11),
        active_color=GREEN,
    )
    mkt_filter_min_conf = ft.Slider(
        min=0, max=80, divisions=16, value=0,
        label="{value}",
        active_color=ACCENT2,
        width=160,
    )
    mkt_source_tab_all_label = ft.Text("Tutti", size=11, weight=ft.FontWeight.BOLD)
    mkt_source_tab_live_label = ft.Text("LIVE", size=11, weight=ft.FontWeight.BOLD)
    mkt_source_tab_sim_label = ft.Text("Simulati", size=11, weight=ft.FontWeight.BOLD)

    def _make_source_tab(key: str, label_ctrl: ft.Text, color: str, icon_name):
        def _on_click(e):
            mkt_filter_source.value = key
            refresh_source_tabs()
            safe_update()
            _threaded_load(load)

        return ft.Container(
            content=ft.Row([
                ft.Icon(icon_name, size=14, color=color),
                label_ctrl,
            ], spacing=5, alignment=ft.MainAxisAlignment.CENTER),
            padding=pad(h=12, v=8),
            border_radius=999,
            border=ft.Border.all(1, ft.Colors.with_opacity(0.35, OUTLINE_SOFT)),
            bgcolor=ft.Colors.with_opacity(0.10, SURFACE_2),
            on_click=_on_click,
        )

    mkt_source_tab_all = _make_source_tab("all", mkt_source_tab_all_label, ACCENT2, ft.Icons.WIDGETS)
    mkt_source_tab_live = _make_source_tab("live", mkt_source_tab_live_label, GREEN, ft.Icons.PODCASTS)
    mkt_source_tab_sim = _make_source_tab("simulated", mkt_source_tab_sim_label, ORANGE, ft.Icons.SCIENCE)
    mkt_source_tabs = ft.Row(
        [mkt_source_tab_all, mkt_source_tab_live, mkt_source_tab_sim],
        spacing=6,
        wrap=True,
    )

    def refresh_source_tabs():
        selected = mkt_filter_source.value or "all"
        tabs = [
            ("all", mkt_source_tab_all, mkt_source_tab_all_label, ACCENT2),
            ("live", mkt_source_tab_live, mkt_source_tab_live_label, GREEN),
            ("simulated", mkt_source_tab_sim, mkt_source_tab_sim_label, ORANGE),
        ]
        for key, tab, label_ctrl, color in tabs:
            is_selected = key == selected
            tab.bgcolor = (
                color if is_selected else ft.Colors.with_opacity(0.10, SURFACE_2)
            )
            tab.border = ft.Border.all(
                1,
                ft.Colors.with_opacity(0.22 if is_selected else 0.35, BG if is_selected else OUTLINE_SOFT),
            )
            label_ctrl.color = BG if is_selected else (TEXT if key != "all" else TEXT_DIM)
            row = tab.content
            if isinstance(row, ft.Row) and row.controls and isinstance(row.controls[0], ft.Icon):
                row.controls[0].color = BG if is_selected else color

    style_select(mkt_filter_sort)  # legacy control not currently shown, keep styling aligned
    mkt_sort_confidence_wrap = style_select(mkt_sort_confidence)
    mkt_filter_signal_wrap = style_select(mkt_filter_signal, border_color=GREEN)
    mkt_filter_city_wrap = style_select(mkt_filter_city)
    mkt_filter_source_wrap = style_select(mkt_filter_source, border_color=ACCENT2)

    def on_filter_change(e):
        refresh_source_tabs()
        _threaded_load(load)

    mkt_filter_edge.on_change = on_filter_change
    mkt_filter_city.on_select = on_filter_change
    mkt_filter_sort.on_select = on_filter_change
    mkt_sort_confidence.on_select = on_filter_change
    mkt_filter_signal.on_select = on_filter_change
    mkt_filter_source.on_select = on_filter_change
    mkt_filter_open_only.on_change = on_filter_change
    mkt_filter_min_conf.on_change = on_filter_change

    def build():
        refresh_source_tabs()
        return ft.Column([
            make_section_title("Mercati", ft.Icons.CASINO, ACCENT2),
            ft.Container(height=6),
            # Guida rapida scommesse
            make_card(ft.Column([
                ft.Row([
                    ft.Icon(ft.Icons.SCHOOL, color=ACCENT, size=18),
                    ft.Text("Guida Rapida alle Scommesse", size=14, weight=ft.FontWeight.BOLD, color=TEXT),
                ], spacing=6),
                ft.Container(height=6),
                ft.ResponsiveRow([
                    ft.Column([
                        ft.Text("Edge (Vantaggio)", size=11, color=ACCENT, weight=ft.FontWeight.BOLD),
                        ft.Text(
                            "La differenza tra la nostra stima e il prezzo del mercato. "
                            "Edge > 0% = il mercato sottovaluta questa possibilità, è un'opportunità.",
                            size=10, color=TEXT_DIM,
                        ),
                    ], col={"xs": 12, "md": 4}),
                    ft.Column([
                        ft.Text("EV (Valore Atteso)", size=11, color=GREEN, weight=ft.FontWeight.BOLD),
                        ft.Text(
                            "Quanto guadagneresti in media per ogni euro scommesso. "
                            "EV +5% = guadagno atteso di 5 centesimi per euro nel lungo periodo.",
                            size=10, color=TEXT_DIM,
                        ),
                    ], col={"xs": 12, "md": 4}),
                    ft.Column([
                        ft.Text("Kelly (Dimensione)", size=11, color=YELLOW, weight=ft.FontWeight.BOLD),
                        ft.Text(
                            "Quanto del tuo bankroll scommettere. Usiamo Kelly frazionario (1/4) "
                            "per limitare il rischio. Non scommettere mai più del suggerito!",
                            size=10, color=TEXT_DIM,
                        ),
                    ], col={"xs": 12, "md": 4}),
                ]),
                ft.Container(height=4),
                ft.ResponsiveRow([
                    ft.Column([
                        ft.Text("Confidenza", size=11, color=ACCENT2, weight=ft.FontWeight.BOLD),
                        ft.Text(
                            "Quanto siamo sicuri dell'analisi (0-100). Considera ensemble, "
                            "edge, precisione modello e complessità. Sopra 55 = buona.",
                            size=10, color=TEXT_DIM,
                        ),
                    ], col={"xs": 12, "md": 6}),
                    ft.Column([
                        ft.Text("Regola d'Oro", size=11, color=YELLOW, weight=ft.FontWeight.BOLD),
                        ft.Text(
                            "Scommetti SOLO quando: Edge > 3%, Confidenza > 50, "
                            "e i modelli concordano. Non inseguire le perdite!",
                            size=10, color=TEXT_DIM,
                        ),
                    ], col={"xs": 12, "md": 6}),
                ]),
            ]), border_color=ft.Colors.with_opacity(0.15, ACCENT)),
            ft.Container(height=8),
            # Filtri + Risk + P&L mini
            ft.ResponsiveRow([
                ft.Column([
                    make_card(ft.Column([
                        ft.Text("Filtri", size=13, color=TEXT_DIM, weight=ft.FontWeight.BOLD),
                        make_info_box("Filtra per vantaggio minimo, città e ordina i risultati."),
                        ft.Container(height=4),
                        ft.ResponsiveRow([
                            ft.Column([
                                field_shell(
                                    "Min Edge %",
                                    ft.Container(content=mkt_filter_edge, padding=pad(h=2, v=0)),
                                    hint="Taglia opportunità deboli",
                                ),
                            ], col={"xs": 12, "md": 6}),
                            ft.Column([
                                field_shell("Città", mkt_filter_city_wrap),
                            ], col={"xs": 12, "md": 6}),
                            ft.Column([
                                field_shell("Ordina per", mkt_sort_confidence_wrap),
                            ], col={"xs": 12, "md": 6}),
                            ft.Column([
                                field_shell("Segnale", mkt_filter_signal_wrap),
                            ], col={"xs": 12, "md": 6}),
                            ft.Column([
                                field_shell(
                                    "Fonte dati",
                                    mkt_filter_source_wrap,
                                    hint="Separa LIVE Polymarket da analisi simulate",
                                ),
                            ], col={"xs": 12, "md": 6}),
                            ft.Column([
                                field_shell(
                                    "Min Confidenza",
                                    ft.Container(content=mkt_filter_min_conf, padding=pad(h=2, v=0)),
                                    hint="0-80",
                                ),
                            ], col={"xs": 12, "md": 6}),
                            ft.Column([
                                field_shell(
                                    "Stato mercato",
                                    ft.Container(
                                        content=ft.Row([mkt_filter_open_only], spacing=0),
                                        padding=pad(h=2, v=6),
                                    ),
                                    hint="Nasconde mercati scaduti",
                                ),
                            ], col={"xs": 12, "md": 6}),
                        ], run_spacing=8, spacing=8),
                    ])),
                ], col={"xs": 12, "md": 4}),
                ft.Column([
                    make_card(ft.Column([
                        ft.Row([
                            ft.Icon(ft.Icons.SHIELD, color=ACCENT2, size=18),
                            ft.Text("Risk Management", size=14, weight=ft.FontWeight.BOLD, color=TEXT),
                        ], spacing=6),
                        make_info_box("Gestione del rischio: quanto scommettere per non perdere tutto.", ACCENT2),
                        ft.Container(height=4),
                        mkt_risk_panel,
                    ]), border_color=ft.Colors.with_opacity(0.15, ACCENT2), height=ROW_TOP_H),
                ], col={"xs": 12, "md": 4}),
                ft.Column([
                    make_card(ft.Column([
                        ft.Row([
                            ft.Icon(ft.Icons.ACCOUNT_BALANCE_WALLET, color=GREEN, size=18),
                            ft.Text("P&L Rapido", size=14, weight=ft.FontWeight.BOLD, color=TEXT),
                        ], spacing=6),
                        ft.Container(height=4),
                        mkt_pnl_mini,
                    ]), border_color=ft.Colors.with_opacity(0.15, GREEN), height=ROW_TOP_H),
                ], col={"xs": 12, "md": 4}),
            ]),
            ft.Container(height=8),
            # Stats riassunto
            mkt_stats,
            ft.Container(height=8),
            # Best Bets
            make_card(ft.Column([
                ft.Row([
                    ft.Icon(ft.Icons.STAR, color=YELLOW, size=18),
                    ft.Text("Best Bets (edge > 5%)", size=14, weight=ft.FontWeight.BOLD, color=TEXT),
                ], spacing=6),
                make_info_box("Le migliori scommesse con vantaggio superiore al 5%. Ordinate per confidenza.", YELLOW),
                ft.Container(height=4),
                mkt_best_bets,
            ])),
            ft.Container(height=10),
            # Market cards
            ft.Container(
                content=ft.Row([
                    ft.Icon(ft.Icons.TUNE, color=ACCENT2, size=16),
                    ft.Text("Vista rapida fonti", size=11, color=TEXT_DIM, weight=ft.FontWeight.BOLD),
                    mkt_source_tabs,
                ], spacing=8, wrap=True, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                padding=pad(h=4, v=4),
            ),
            mkt_status_text,
            ft.Container(height=4),
            mkt_cards_list,
        ], scroll=ft.ScrollMode.AUTO, spacing=0)

    def load():
        """Carica e filtra mercati."""
        try:
            _load_impl()
            logger.info("=== MERCATI _load_impl DONE OK ===")
        except Exception as e:
            logger.error("Mercati load error: %s", e, exc_info=True)
            mkt_cards_list.controls = [make_empty_state(
                ft.Icons.ERROR_OUTLINE, f"Errore caricamento mercati: {e}")]
            safe_update()

    def _load_impl():
        logger.info("=== MERCATI _load_impl START ===")
        mkt_cards_list.controls = [make_loading_indicator("Scansione mercati...")]
        mkt_best_bets.controls = [ft.Text("Caricamento...", color=TEXT_DIM, size=12)]
        safe_update()

        min_edge = (mkt_filter_edge.value or 0) / 100.0
        filter_city = mkt_filter_city.value
        sort_by = mkt_sort_confidence.value or "confidence_desc"
        filter_signal = mkt_filter_signal.value or "all"
        filter_source = mkt_filter_source.value or "all"
        logger.info("Filtri: edge=%.2f city=%s sort=%s signal=%s source=%s",
                     min_edge, filter_city, sort_by, filter_signal, filter_source)

        # Carica solo le fonti necessarie (evita attese inutili quando si seleziona LIVE/SIM)
        need_live = filter_source in ("all", "live")
        need_sim = filter_source in ("all", "simulated")

        scan_result = api_get("/market/scan", timeout=45) if need_live else {"markets": [], "n_scanned": 0}
        opps_result = api_get("/market/opportunities", timeout=90) if need_sim else {"opportunities": []}
        scan_failed = need_live and scan_result is None
        opps_failed = need_sim and opps_result is None
        logger.info("API results: scan=%s (failed=%s), opps=%s (failed=%s)",
                     len((scan_result or {}).get("markets", [])), scan_failed,
                     len((opps_result or {}).get("opportunities", [])), opps_failed)

        # Accuracy per confidence scoring — per città del mercato, non solo corrente
        mkt_accuracy = api_get(f"/analysis/{state.current_city}/accuracy?variable=temperature_2m_max")
        mkt_acc_data = (mkt_accuracy or {}).get("accuracy")
        # Bias per correzione
        mkt_bias = mkt_acc_data.get("bias", 0) if mkt_acc_data and isinstance(mkt_acc_data, dict) else 0

        opportunities = (opps_result or {}).get("opportunities", [])
        real_markets = (scan_result or {}).get("markets", [])

        # Combina
        all_markets = []
        for m in real_markets:
            m["_source"] = "polymarket"
            all_markets.append(m)
        for m in opportunities:
            m["_source"] = "analysis"
            all_markets.append(m)

        n_live_total = sum(1 for m in all_markets if m.get("_source") == "polymarket")
        n_sim_total = sum(1 for m in all_markets if m.get("_source") == "analysis")

        # Cache per dashboard
        state.cache["markets"] = all_markets

        # Filtra + calcola confidence CON VALIDAZIONE
        filtered = []
        today_str = datetime.now().strftime("%Y-%m-%d")
        my_generation = _load_generation[0]
        source_candidates = 0
        reject_counts = {
            "edge": 0,
            "city": 0,
            "source": 0,
            "signal": 0,
            "open_only": 0,
            "min_conf": 0,
        }

        # Pre-filter pass: apply cheap filters BEFORE any API calls
        logger.info("all_markets count: %d (live=%d, sim=%d)", len(all_markets), n_live_total, n_sim_total)
        pre_filtered = []
        for entry in all_markets:
            rec = entry.get("recommendation", {})
            meta = entry.get("metadata", {})
            outcomes = rec.get("outcomes", [])
            max_edge = max((o.get("edge", 0) for o in outcomes), default=0)

            if max_edge < min_edge:
                reject_counts["edge"] += 1
                continue

            if filter_city != "all":
                entry_city = meta.get("city", {}).get("slug", rec.get("city", ""))
                if entry_city != filter_city:
                    reject_counts["city"] += 1
                    continue

            entry_source = entry.get("_source", "")
            if filter_source == "live" and entry_source != "polymarket":
                reject_counts["source"] += 1
                continue
            if filter_source == "simulated" and entry_source != "analysis":
                reject_counts["source"] += 1
                continue
            source_candidates += 1

            # Filtro solo mercati aperti (escludi scaduti)
            mkt_data_pre = entry.get("market", {})
            if mkt_filter_open_only.value:
                mkt_end = mkt_data_pre.get("end_date", "")
                if mkt_end and mkt_end[:10] < today_str:
                    reject_counts["open_only"] += 1
                    continue

            entry["_max_edge"] = max_edge
            pre_filtered.append(entry)

        # Compute confidence + signal for pre-filtered markets
        # Use /market/analyze only for the top 8 (by edge), local calc for the rest
        logger.info("pre_filtered count: %d (rejected: %s)", len(pre_filtered), reject_counts)
        pre_filtered.sort(key=lambda x: x.get("_max_edge", 0), reverse=True)

        analyze_failures = 0  # Circuit breaker: skip server calls after 2 consecutive failures
        analyze_server_used = 0
        for idx, entry in enumerate(pre_filtered):
            rec = entry.get("recommendation", {})
            meta = entry.get("metadata", {})
            outcomes = rec.get("outcomes", [])
            max_edge = entry.get("_max_edge", 0)
            mkt_data = entry.get("market", {})

            target_date = rec.get("date", today_str)
            try:
                td = datetime.strptime(target_date[:10], "%Y-%m-%d")
                days_ahead = max(0, (td - datetime.now()).days)
            except Exception:
                days_ahead = 1

            mkt_question = mkt_data.get("question", "")
            outcome_names = [o.get("outcome", "") for o in outcomes]
            outcome_prices = [o.get("market_price", 0) for o in outcomes]
            conf = None
            bet_signal = None
            city_bias = mkt_bias

            # Server-side analyze only for top 4 markets (by edge) to stay fast
            # Circuit breaker: skip if 2+ consecutive failures (server too slow)
            use_server = (idx < 4 and analyze_failures < 2
                          and mkt_question and outcome_names and outcome_prices)
            if use_server:
                analyze_result = api_post("/market/analyze", {
                    "question": mkt_question,
                    "outcomes": outcome_names,
                    "outcome_prices": outcome_prices,
                }, timeout=3)
                if analyze_result and "confidence" in analyze_result:
                    analyze_failures = 0  # Reset on success
                    analyze_server_used += 1
                    srv_conf_raw = analyze_result.get("confidence", 0)
                    srv_signal_raw = analyze_result.get("betting_signal", 0)
                    srv_rec = analyze_result.get("recommendation", {})
                    srv_meta = analyze_result.get("metadata", {})

                    if isinstance(srv_conf_raw, dict):
                        srv_conf_total = float(srv_conf_raw.get("total", 0) or 0)
                        conf = {
                            **srv_conf_raw,
                            "total": srv_conf_total,
                            "rating": srv_conf_raw.get("rating", (
                                "Ottima" if srv_conf_total > 70 else (
                                    "Buona" if srv_conf_total > 50 else (
                                        "Discreta" if srv_conf_total > 35 else "Scarsa"
                                    )
                                )
                            )),
                            "rating_color": srv_conf_raw.get("rating_color", (
                                "#4caf50" if srv_conf_total > 50 else (
                                    "#ff9800" if srv_conf_total > 35 else "#f44336"
                                )
                            )),
                        }
                    else:
                        try:
                            srv_conf_val = float(srv_conf_raw or 0)
                        except Exception:
                            srv_conf_val = 0.0
                        conf = {
                            "total": srv_conf_val * 100 if srv_conf_val <= 1 else srv_conf_val,
                            "rating": "Ottima" if srv_conf_val > 0.7 else (
                                "Buona" if srv_conf_val > 0.5 else (
                                    "Discreta" if srv_conf_val > 0.35 else "Scarsa")),
                            "rating_color": "#4caf50" if srv_conf_val > 0.5 else (
                                "#ff9800" if srv_conf_val > 0.35 else "#f44336"),
                        }

                    eff_edge = srv_rec.get("expected_value", max_edge)
                    if isinstance(srv_signal_raw, dict):
                        sig_type = srv_signal_raw.get("signal", "NON SCOMMETTERE")
                        sig_color = (
                            GREEN if sig_type == "SCOMMETTI" else (
                                YELLOW if sig_type == "CAUTELA" else RED
                            )
                        )
                        sig_icon = (
                            ft.Icons.CHECK_CIRCLE if sig_type == "SCOMMETTI" else (
                                ft.Icons.WARNING_AMBER if sig_type == "CAUTELA" else ft.Icons.BLOCK
                            )
                        )
                        kelly_mult = float(srv_signal_raw.get("kelly_multiplier", 0) or 0)
                        eff_edge = srv_signal_raw.get("effective_edge", eff_edge)
                    else:
                        try:
                            srv_signal_val = float(srv_signal_raw or 0)
                        except Exception:
                            srv_signal_val = 0.0
                        if srv_signal_val > SIGNAL_BET_THRESHOLD:
                            sig_type = "SCOMMETTI"
                            sig_color = GREEN
                            sig_icon = ft.Icons.CHECK_CIRCLE
                            kelly_mult = KELLY_FRACTION
                        elif srv_signal_val > SIGNAL_CAUTION_THRESHOLD:
                            sig_type = "CAUTELA"
                            sig_color = YELLOW
                            sig_icon = ft.Icons.WARNING_AMBER
                            kelly_mult = KELLY_FRACTION * 0.4
                        else:
                            sig_type = "NON SCOMMETTERE"
                            sig_color = RED
                            sig_icon = ft.Icons.BLOCK
                            kelly_mult = 0
                    bet_signal = {
                        "signal": sig_type,
                        "signal_color": sig_color,
                        "signal_icon": sig_icon,
                        "kelly_multiplier": kelly_mult,
                        "effective_edge": float(eff_edge or 0),
                    }
                    city_bias = srv_meta.get("bias", mkt_bias)
                else:
                    analyze_failures += 1
                    if analyze_failures >= 2:
                        logger.warning("Analyze circuit breaker: %d failures, skipping server calls", analyze_failures)

            # Local confidence + signal (fast, no API call)
            if conf is None or bet_signal is None:
                conf = calculate_confidence(
                    {}, max_edge, mkt_acc_data, len(outcomes), days_ahead=days_ahead)
                bet_signal = calculate_betting_signal(
                    conf, max_edge, bias=mkt_bias, ens_std=5, days_ahead=days_ahead)

            entry["_ev"] = rec.get("expected_value", 0)
            entry["_confidence"] = conf
            entry["_betting_signal"] = bet_signal
            entry["_days_ahead"] = days_ahead
            entry["_bias"] = city_bias

            # Filtro segnale
            if filter_signal != "all":
                sig = bet_signal.get("signal", "")
                if filter_signal == "NO" and sig != "NON SCOMMETTERE":
                    reject_counts["signal"] += 1
                    continue
                elif filter_signal == "SCOMMETTI" and sig != "SCOMMETTI":
                    reject_counts["signal"] += 1
                    continue
                elif filter_signal == "CAUTELA" and sig != "CAUTELA":
                    reject_counts["signal"] += 1
                    continue

            # Filtro confidenza minima
            min_conf_val = float(mkt_filter_min_conf.value or 0)
            if min_conf_val > 0:
                conf_total = conf.get("total", 0) if isinstance(conf, dict) else 0
                if conf_total < min_conf_val:
                    reject_counts["min_conf"] += 1
                    continue

            filtered.append(entry)

        logger.info("After confidence loop: filtered=%d, server_used=%d, analyze_failures=%d, reject_counts=%s",
                     len(filtered), analyze_server_used, analyze_failures, reject_counts)

        # Abort if a newer load was requested
        if _load_generation[0] != my_generation:
            logger.info("ABORT: generation changed (%d != %d)", _load_generation[0], my_generation)
            return

        # Ordina
        if sort_by == "confidence_desc":
            filtered.sort(key=lambda x: x.get("_confidence", {}).get("total", 0), reverse=True)
        elif sort_by == "edge_desc":
            filtered.sort(key=lambda x: x.get("_max_edge", 0), reverse=True)
        elif sort_by == "edge_asc":
            filtered.sort(key=lambda x: x.get("_max_edge", 0))
        elif sort_by == "ev_desc":
            filtered.sort(key=lambda x: x.get("_ev", 0), reverse=True)
        elif sort_by == "stake_desc":
            filtered.sort(key=lambda x: x.get("recommendation", {}).get("suggested_size_pct", 0), reverse=True)

        # Stats
        n_scanned = (scan_result or {}).get("n_scanned", 0)
        n_total = len(all_markets)
        n_filtered = len(filtered)
        avg_edge = sum(e.get("_max_edge", 0) for e in filtered) / n_filtered if n_filtered else 0
        ts = datetime.now().strftime("%H:%M:%S")

        mkt_stats.content = ft.Container(
            content=ft.Row([
                make_badge(f"Scansionati: {n_scanned}", ft.Colors.with_opacity(0.15, ACCENT), ACCENT),
                make_badge(f"Trovati: {n_total}", ft.Colors.with_opacity(0.15, GREEN), GREEN),
                make_badge(f"LIVE: {n_live_total}", ft.Colors.with_opacity(0.14, GREEN), GREEN),
                make_badge(f"SIMULATI: {n_sim_total}", ft.Colors.with_opacity(0.14, ORANGE), ORANGE),
                make_badge(f"Filtrati: {n_filtered}", ft.Colors.with_opacity(0.15, YELLOW), YELLOW),
                make_badge(f"Avg Edge: {avg_edge:.1%}", ft.Colors.with_opacity(0.15, ACCENT2), ACCENT2),
            ], spacing=8, wrap=True),
            border_radius=14,
            bgcolor=ft.Colors.with_opacity(0.22, SURFACE_2),
            border=ft.Border.all(1, ft.Colors.with_opacity(0.45, OUTLINE_SOFT)),
            padding=pad(h=10, v=8),
        )

        source_label = {
            "all": "Tutti",
            "live": "Solo LIVE",
            "simulated": "Solo Simulati",
        }.get(filter_source, "Tutti")
        status_bits = [
            f"Fonte: {source_label}",
            f"LIVE: {len(real_markets)}",
            f"Simulati: {len(opportunities)}",
            f"Scan: {ts}",
        ]
        if scan_failed:
            status_bits.append("LIVE API timeout/errore")
        if opps_failed:
            status_bits.append("Analisi API timeout/errore")
        mkt_status_text.value = " | ".join(status_bits)

        # === Risk Management Panel — portfolio risk analysis ===
        pnl = get_pnl_stats()
        bankroll = pnl["bankroll"]
        pnl_all = load_pnl()
        risk = check_portfolio_risk(pnl_all.get("bets", []), bankroll)
        # Store risk data in state cache for dashboard use
        state.cache["portfolio_risk"] = risk

        if filtered:
            n_go = sum(1 for e in filtered
                        if e.get("_betting_signal", {}).get("signal") == "SCOMMETTI")
            n_caution = sum(1 for e in filtered
                             if e.get("_betting_signal", {}).get("signal") == "CAUTELA")
            n_stop = sum(1 for e in filtered
                          if e.get("_betting_signal", {}).get("signal") == "NON SCOMMETTERE")
            avg_conf = sum(
                e.get("_confidence", {}).get("total", 0) for e in filtered
            ) / len(filtered)
            avg_eff_edge = sum(
                e.get("_betting_signal", {}).get("effective_edge", 0) for e in filtered
            ) / len(filtered)

            risk_items = [
                make_kv_row("Bankroll", f"${bankroll:,.0f}", ACCENT),
                ft.Container(height=4),
                ft.Row([
                    make_badge(f"SCOMMETTI: {n_go}", GREEN, BG),
                    make_badge(f"CAUTELA: {n_caution}", YELLOW, BG),
                    make_badge(f"STOP: {n_stop}", RED, BG if n_stop > 0 else TEXT_DIM),
                ], spacing=4, wrap=True),
                ft.Container(height=4),
                # Portfolio exposure gauge
                make_kv_row("Esposizione", f"{risk['exposure_pct']:.0%} / {50}%",
                              GREEN if risk["exposure_pct"] < 0.3 else (
                                  YELLOW if risk["exposure_pct"] < 0.5 else RED)),
                make_kv_row("Max Drawdown", f"{risk['max_drawdown']:.0%}",
                              GREEN if risk["max_drawdown"] < 0.15 else (
                                  YELLOW if risk["max_drawdown"] < 0.25 else RED)),
            ]
            if risk["loss_streak"] >= 3:
                risk_items.append(
                    make_kv_row("Streak perdite", f"{risk['loss_streak']}",
                                  RED if risk["loss_streak"] >= 5 else YELLOW))
            if risk["max_city_pct"] > 0.10:
                risk_items.append(
                    make_kv_row("Max città", f"{risk['max_city_name']} ({risk['max_city_pct']:.0%})",
                                  YELLOW if risk["max_city_pct"] < 0.15 else RED))
            risk_items.extend([
                ft.Container(height=4),
                make_kv_row("Confidenza media", f"{avg_conf:.0f}/100",
                              GREEN if avg_conf >= 50 else (YELLOW if avg_conf >= 35 else RED)),
            ])

            # Warnings
            for w in risk["warnings"]:
                w_color = RED if w["level"] == "critical" else YELLOW
                risk_items.append(ft.Row([
                    ft.Icon(ft.Icons.WARNING, color=w_color, size=12),
                    ft.Text(w["message"], size=9, color=w_color),
                ], spacing=4))

            if not risk["can_bet"]:
                risk_items.insert(0, ft.Container(
                    content=ft.Row([
                        ft.Icon(ft.Icons.BLOCK, color=RED, size=14),
                        ft.Text("SCOMMESSE BLOCCATE", size=11, color=BG,
                                 weight=ft.FontWeight.BOLD),
                    ], spacing=4),
                    bgcolor=RED, border_radius=6, padding=pad(h=8, v=4),
                ))

            mkt_risk_panel.content = ft.Column(risk_items, spacing=3)
        else:
            mkt_risk_panel.content = ft.Text("Nessun dato", color=TEXT_DIM, size=12)

        # === P&L Mini ===
        pnl_color = GREEN if pnl["total_pnl"] >= 0 else RED
        mkt_pnl_mini.content = ft.Column([
            ft.Row([
                ft.Text("Totale", size=11, color=TEXT_DIM),
                ft.Text(f"${pnl['total_pnl']:+,.0f}", size=16, color=pnl_color,
                         weight=ft.FontWeight.BOLD),
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            make_kv_row("Vinte/Perse", f"{pnl['won']}/{pnl['lost']}", TEXT),
            make_kv_row("Win Rate", f"{pnl['win_rate']:.0%}",
                          GREEN if pnl["win_rate"] > 0.5 else RED),
            make_kv_row("Pending", f"{pnl['pending']}", YELLOW),
            ft.Container(height=4),
            build_pnl_sparkline(load_pnl().get("bets", []), width=200, height=30),
        ], spacing=3)

        # Best Bets — solo SCOMMETTI e CAUTELA con edge > 3%
        best = [e for e in filtered
                if e.get("_betting_signal", {}).get("signal") in ("SCOMMETTI", "CAUTELA")
                and e.get("_betting_signal", {}).get("effective_edge", 0) > 0.03]
        # Ordina: prima SCOMMETTI poi CAUTELA, poi per confidence
        best.sort(key=lambda x: (
            0 if x.get("_betting_signal", {}).get("signal") == "SCOMMETTI" else 1,
            -x.get("_confidence", {}).get("total", 0),
        ))
        mkt_best_bets.controls.clear()
        if best:
            for entry in best[:5]:
                rec = entry.get("recommendation", {})
                mkt_data = entry.get("market", {})
                e_sig = entry.get("_betting_signal", {})
                e_eff_edge = e_sig.get("effective_edge", 0)
                ev = entry.get("_ev", 0)
                bet = rec.get("best_bet", "—")
                q = mkt_data.get("question", "—")
                conf = entry.get("_confidence", {})
                conf_total = conf.get("total", 0)
                e_sig_type = e_sig.get("signal", "?")
                e_sig_color = e_sig.get("signal_color", TEXT_DIM)
                is_live = entry.get("_source") == "polymarket"

                mkt_best_bets.controls.append(ft.Container(
                    content=ft.Row([
                        ft.Icon(
                            ft.Icons.CHECK_CIRCLE if e_sig_type == "SCOMMETTI" else ft.Icons.WARNING_AMBER,
                            color=e_sig_color, size=14),
                        ft.Column([
                            ft.Text(q[:50], size=11, color=TEXT,
                                     overflow=ft.TextOverflow.ELLIPSIS, max_lines=1),
                            ft.Text(f"Bet: {bet}", size=10, color=TEXT_DIM),
                        ], spacing=1, expand=True),
                        ft.Column([
                            ft.Row([
                                ft.Container(
                                    content=ft.Text(
                                        "LIVE" if is_live else "SIM",
                                        size=7,
                                        color=BG if is_live else TEXT,
                                        weight=ft.FontWeight.BOLD,
                                    ),
                                    bgcolor=GREEN if is_live else ft.Colors.with_opacity(0.18, ORANGE),
                                    border=ft.Border.all(
                                        1, ft.Colors.with_opacity(0.25, GREEN if is_live else ORANGE)
                                    ),
                                    border_radius=3,
                                    padding=pad(h=4, v=1),
                                ),
                                ft.Container(
                                    content=ft.Text(e_sig_type, size=8, color=BG,
                                                     weight=ft.FontWeight.BOLD),
                                    bgcolor=e_sig_color, border_radius=3, padding=pad(h=4, v=1)),
                            ], spacing=3, wrap=True),
                            ft.Text(f"{e_eff_edge:+.1%}", size=12, color=GREEN,
                                     weight=ft.FontWeight.BOLD),
                            ft.Text(f"Conf {conf_total:.0f}", size=10, color=TEXT_DIM),
                        ], spacing=1, horizontal_alignment=ft.CrossAxisAlignment.END),
                    ], spacing=8),
                    bgcolor=ft.Colors.with_opacity(0.08, e_sig_color),
                    border=ft.Border.all(1, ft.Colors.with_opacity(0.18, e_sig_color)),
                    border_radius=12,
                    padding=pad(h=12, v=8),
                ))
        else:
            mkt_best_bets.controls.append(
                ft.Text("Nessuna scommessa raccomandata al momento", color=TEXT_DIM, size=12))

        # Market Cards
        logger.info("Building market cards for %d filtered markets...", len(filtered))
        mkt_cards_list.controls.clear()
        if not filtered:
            reject_parts = []
            if reject_counts["open_only"] > 0:
                reject_parts.append(f"scaduti: {reject_counts['open_only']}")
            if reject_counts["signal"] > 0:
                reject_parts.append(f"segnale: {reject_counts['signal']}")
            if reject_counts["min_conf"] > 0:
                reject_parts.append(f"confidenza: {reject_counts['min_conf']}")
            if reject_counts["city"] > 0:
                reject_parts.append(f"città: {reject_counts['city']}")
            if reject_counts["edge"] > 0:
                reject_parts.append(f"edge: {reject_counts['edge']}")

            reject_hint = ""
            if source_candidates > 0 and reject_parts:
                reject_hint = f" | Scartati da filtri: {', '.join(reject_parts)}"

            if filter_source == "live" and scan_failed:
                empty_msg = "Errore fonte LIVE: timeout/errore API (/market/scan)"
                empty_icon = ft.Icons.CLOUD_OFF
            elif filter_source == "live" and not real_markets:
                empty_msg = "Nessun mercato meteo LIVE trovato su Polymarket in questo momento"
                empty_icon = ft.Icons.PODCASTS
            elif filter_source == "simulated" and opps_failed:
                empty_msg = "Errore fonte Simulata: timeout/errore API (/market/opportunities)"
                empty_icon = ft.Icons.CLOUD_OFF
            else:
                empty_msg = "Nessun mercato corrisponde ai filtri"
                if reject_hint:
                    empty_msg += reject_hint
                empty_icon = ft.Icons.ANALYTICS

            mkt_cards_list.controls.append(make_empty_state(empty_icon, empty_msg))
            safe_update()
            return

        def _source_header(title: str, subtitle: str, color: str):
            return ft.Container(
                content=ft.Row([
                    ft.Container(width=8, height=8, border_radius=999, bgcolor=color),
                    ft.Text(title, size=12, color=TEXT, weight=ft.FontWeight.BOLD),
                    ft.Text(subtitle, size=10, color=TEXT_DIM),
                ], spacing=6, wrap=True),
                padding=pad(h=4, v=4),
            )

        def _safe_build_card(entry, idx):
            try:
                card = _build_market_card(entry)
                logger.debug("Card %d built OK: %s", idx, entry.get("market", {}).get("question", "?")[:40])
                return card
            except Exception as exc:
                logger.error("Card %d FAILED: %s — %s", idx,
                             entry.get("market", {}).get("question", "?")[:40], exc, exc_info=True)
                return ft.Container(
                    content=ft.Text(f"Errore card: {exc}", size=10, color=RED),
                    bgcolor=ft.Colors.with_opacity(0.08, RED),
                    border_radius=8, padding=8,
                )

        if filter_source == "all":
            live_entries = [e for e in filtered if e.get("_source") == "polymarket"]
            sim_entries = [e for e in filtered if e.get("_source") == "analysis"]
            logger.info("Source split: live=%d, sim=%d", len(live_entries), len(sim_entries))

            if live_entries:
                mkt_cards_list.controls.append(
                    _source_header("Mercati LIVE (Polymarket)", f"{len(live_entries)} risultati", GREEN)
                )
                for i, entry in enumerate(live_entries):
                    mkt_cards_list.controls.append(_safe_build_card(entry, i))

            if sim_entries:
                if live_entries:
                    mkt_cards_list.controls.append(ft.Container(height=8))
                mkt_cards_list.controls.append(
                    _source_header("Analisi Simulate", f"{len(sim_entries)} risultati", ORANGE)
                )
                for i, entry in enumerate(sim_entries):
                    mkt_cards_list.controls.append(_safe_build_card(entry, i))
        else:
            for i, entry in enumerate(filtered):
                mkt_cards_list.controls.append(_safe_build_card(entry, i))

        logger.info("Cards built: %d controls in mkt_cards_list", len(mkt_cards_list.controls))

        try:
            safe_update()
            logger.info("safe_update() after cards: OK (controls=%d)", len(mkt_cards_list.controls))
        except Exception as exc:
            logger.error("safe_update() after cards FAILED: %s", exc, exc_info=True)

    def _build_market_card(entry):
        """Costruisce una card mercato con ExpansionTile."""
        mkt = entry.get("market", {})
        rec = entry.get("recommendation", {})
        meta = entry.get("metadata", {})

        question = mkt.get("question", "?")
        city_name = meta.get("city", {}).get("name", rec.get("city", ""))
        target_date = rec.get("date", "")
        best_bet = rec.get("best_bet")
        ev = rec.get("expected_value", 0)
        reasoning = rec.get("reasoning", "")
        outcomes_list = rec.get("outcomes", [])
        is_real = entry.get("_source") == "polymarket"
        max_edge = entry.get("_max_edge", 0)

        # === BETTING SIGNAL (il verdetto finale) ===
        bet_signal = entry.get("_betting_signal", {})
        sig_type = bet_signal.get("signal", "NON SCOMMETTERE")
        kelly_mult = bet_signal.get("kelly_multiplier", 0)
        eff_edge = bet_signal.get("effective_edge", max_edge)

        # Kelly e sizing basati sul segnale (non sul raw recommendation)
        # raw_kelly dal backend e' gia' il Kelly frazionario completo
        # Applichiamo solo il moltiplicatore del segnale (0.25 SCOMMETTI, 0.10 CAUTELA)
        raw_kelly = rec.get("kelly_fraction", 0)
        kelly = raw_kelly * kelly_mult if kelly_mult > 0 else 0
        sz = kelly  # gia' scalato, niente doppia moltiplicazione

        edge_color = GREEN if max_edge > EDGE_GOOD_THRESHOLD else (YELLOW if max_edge > 0 else RED)

        # Outcome rows
        outcome_rows = []
        for o in outcomes_list:
            edge = o.get("edge", 0)
            o_color = GREEN if edge > EDGE_GOOD_THRESHOLD else (YELLOW if edge > 0 else RED)
            is_best = o.get("outcome") == best_bet

            outcome_rows.append(ft.Container(
                content=ft.Row([
                    ft.Row([
                        ft.Icon(ft.Icons.STAR, color=YELLOW, size=12) if is_best else ft.Container(width=12),
                        ft.Text(
                            o.get("outcome", ""), size=12, color=TEXT,
                            weight=ft.FontWeight.BOLD if is_best else ft.FontWeight.NORMAL,
                            width=140, overflow=ft.TextOverflow.ELLIPSIS,
                        ),
                    ], spacing=4),
                    ft.Text(f"{o.get('market_price', 0):.0%}", size=11, color=TEXT_DIM, width=50),
                    ft.Text(f"{o.get('our_probability', 0):.0%}", size=11, color=ACCENT,
                             weight=ft.FontWeight.BOLD, width=50),
                    ft.Text(f"{edge:+.1%}", size=11, color=o_color,
                             weight=ft.FontWeight.BOLD, width=55),
                    ft.Text(f"{o.get('confidence', 0):.0%}", size=10, color=TEXT_DIM, width=40),
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                bgcolor=ft.Colors.with_opacity(0.07, GREEN) if is_best else ft.Colors.with_opacity(0.02, TEXT),
                border=ft.Border.all(
                    1,
                    ft.Colors.with_opacity(0.14, GREEN if is_best else OUTLINE_SOFT),
                ),
                border_radius=10,
                padding=pad(h=10, v=5),
            ))

        # Bet banner — colorato in base al segnale
        sig_color = bet_signal.get("signal_color", TEXT_DIM)
        sig_icon = bet_signal.get("signal_icon", ft.Icons.INFO)
        if best_bet and sig_type != "NON SCOMMETTERE":
            banner_bg = GREEN if sig_type == "SCOMMETTI" else YELLOW
            bet_banner = ft.Container(
                content=ft.Row([
                    ft.Icon(sig_icon, color=BG, size=16),
                    ft.Text(f"{sig_type}: {best_bet}", size=12, color=BG,
                             weight=ft.FontWeight.BOLD),
                    ft.Text(f"Edge eff. {eff_edge:+.1%}", size=11, color=BG),
                    ft.Text(f"K {kelly:.1%}", size=11, color=BG),
                ], spacing=6, wrap=True),
                bgcolor=banner_bg,
                border=ft.Border.all(1, ft.Colors.with_opacity(0.18, BG)),
                border_radius=12,
                padding=pad(h=12, v=8),
            )
        elif best_bet and sig_type == "NON SCOMMETTERE":
            bet_banner = ft.Container(
                content=ft.Row([
                    ft.Icon(ft.Icons.BLOCK, color=RED, size=14),
                    ft.Text(f"NON SCOMMETTERE — {best_bet}", size=11, color=RED),
                    ft.Text(f"(edge grezzo {max_edge:+.1%} ma rischio troppo alto)",
                             size=9, color=TEXT_DIM),
                ], spacing=6),
                bgcolor=ft.Colors.with_opacity(0.08, RED),
                border=ft.Border.all(1, ft.Colors.with_opacity(0.18, RED)),
                border_radius=12, padding=pad(h=12, v=8),
            )
        else:
            bet_banner = ft.Container(
                content=ft.Text("Nessun edge significativo", size=11, color=TEXT_DIM),
                bgcolor=ft.Colors.with_opacity(0.03, TEXT),
                border=ft.Border.all(1, ft.Colors.with_opacity(0.10, OUTLINE_SOFT)),
                border_radius=10,
                padding=pad(h=10, v=6),
            )

        # Confidence meter
        conf = entry.get("_confidence", {})
        conf_total = conf.get("total", 0)
        conf_color = conf.get("rating_color", TEXT_DIM)
        conf_rating = conf.get("rating", "?")

        # Pulsante Registra Scommessa — gateato dal segnale
        pnl_data = load_pnl()
        bankroll_val = pnl_data.get("bankroll", 1000)
        # Stake basato su Kelly del segnale (non raw)
        stake_val = bankroll_val * kelly_mult * max(0, eff_edge) if kelly_mult > 0 else 0
        stake_val = min(stake_val, bankroll_val * MAX_BET_PCT)

        # Enforce portfolio exposure limit (max 50% bankroll)
        pending_exposure = sum(
            b["stake"] for b in pnl_data.get("bets", []) if b.get("status") == "pending"
        )
        remaining_capacity = max(0, bankroll_val * 0.50 - pending_exposure)
        stake_val = min(stake_val, remaining_capacity)
        exposure_exceeded = pending_exposure >= bankroll_val * 0.50

        def _on_bet_click(e, q=question, o=best_bet or "", s=stake_val,
                           od=1 / max(0.01, eff_edge) if eff_edge > 0 else 1,
                           op=0, ed=eff_edge,
                           cn=city_name, dt=target_date, cf=conf_total):
            if o and s > 0:
                try:
                    record_bet(q, o, s, od, op, ed, cn, dt, cf / 100)
                    e.control.content = ft.Text("Registrata!", color=BG, size=13)
                    e.control.bgcolor = GREEN
                    e.control.disabled = True
                except Exception as ex:
                    logger.error("Bet record failed: %s", ex)
                    e.control.content = ft.Text("Errore!", color=BG, size=13)
                    e.control.bgcolor = RED
                safe_update()

        bet_btn = ft.Container()
        if exposure_exceeded:
            bet_btn = ft.Container(
                content=ft.Row([
                    ft.Icon(ft.Icons.BLOCK, color=RED, size=14),
                    ft.Text("Esposizione massima raggiunta", size=10, color=RED),
                ], spacing=4),
                bgcolor=ft.Colors.with_opacity(0.08, RED),
                border=ft.Border.all(1, ft.Colors.with_opacity(0.18, RED)),
                border_radius=10, padding=pad(h=10, v=6),
            )
        elif best_bet and sig_type == "SCOMMETTI" and stake_val > 0:
            bet_btn = ft.Button(
                f"Scommetti ${stake_val:.0f}",
                icon=ft.Icons.ADD_CARD,
                bgcolor=GREEN,
                color=BG,
                style=ft.ButtonStyle(
                    shape=ft.RoundedRectangleBorder(radius=12),
                    padding=pad(h=12, v=10),
                ),
                on_click=_on_bet_click,
            )
        elif best_bet and sig_type == "CAUTELA" and stake_val > 0:
            bet_btn = ft.Button(
                f"Cautela ${stake_val:.0f}",
                icon=ft.Icons.WARNING_AMBER,
                bgcolor=YELLOW,
                color=BG,
                style=ft.ButtonStyle(
                    shape=ft.RoundedRectangleBorder(radius=12),
                    padding=pad(h=12, v=10),
                ),
                on_click=_on_bet_click,
            )

        # Dettagli espandibili (segnale + reasoning + edge chart + confidence + risk)
        detail_controls = []
        # Segnale scommessa prominente
        if bet_signal:
            detail_controls.append(build_betting_signal_widget(bet_signal, width=380))
        if conf:
            detail_controls.append(ft.ResponsiveRow([
                ft.Column([build_confidence_meter(conf, width=220)], col={"xs": 12, "md": 7}),
                ft.Column([build_risk_gauge(kelly, kelly_mult, bankroll_val, width=180)], col={"xs": 12, "md": 5}),
            ], spacing=8, run_spacing=8))
        if reasoning:
            detail_controls.append(ft.Text(reasoning, size=11, color=TEXT_DIM, max_lines=4,
                                            overflow=ft.TextOverflow.ELLIPSIS))
        if outcomes_list:
            detail_controls.append(build_edge_bar_chart(
                [{"recommendation": {"expected_value": o.get("edge", 0),
                                      "best_bet": o.get("outcome", "")}}
                 for o in outcomes_list],
                width=350,
            ))

        # Header badge row
        # Countdown ore alla risoluzione
        days_ahead = entry.get("_days_ahead", 0)
        if days_ahead <= 0:
            countdown_text = "OGGI"
            countdown_color = RED
        elif days_ahead == 1:
            countdown_text = "DOMANI"
            countdown_color = ORANGE
        elif days_ahead <= 3:
            countdown_text = f"{days_ahead}gg"
            countdown_color = YELLOW
        else:
            countdown_text = f"{days_ahead}gg"
            countdown_color = TEXT_DIM

        header_badges = ft.Row([
            make_badge(
                "LIVE POLYMARKET" if is_real else "SIMULATO / ANALISI",
                GREEN if is_real else ft.Colors.with_opacity(0.20, ORANGE),
                BG if is_real else TEXT,
            ),
            ft.Container(
                content=ft.Text(city_name, size=10, color=ACCENT),
                bgcolor=ft.Colors.with_opacity(0.1, ACCENT),
                border=ft.Border.all(1, ft.Colors.with_opacity(0.16, ACCENT)),
                border_radius=999, padding=pad(h=7, v=2),
            ) if city_name else ft.Container(),
            ft.Container(
                content=ft.Row([
                    ft.Icon(ft.Icons.TIMER, color=countdown_color, size=10),
                    ft.Text(countdown_text, size=10, color=countdown_color,
                             weight=ft.FontWeight.BOLD),
                ], spacing=2),
                bgcolor=ft.Colors.with_opacity(0.1, countdown_color),
                border=ft.Border.all(1, ft.Colors.with_opacity(0.14, countdown_color)),
                border_radius=999, padding=pad(h=6, v=2),
            ),
            ft.Text(target_date, size=10, color=TEXT_DIM),
            ft.Text(f"Vol: ${mkt.get('volume', 0):,.0f}", size=10, color=TEXT_DIM)
            if mkt.get("volume", 0) > 0 else ft.Container(),
        ], spacing=6, wrap=True)

        # Column header for outcomes
        outcome_header = ft.Container(
            content=ft.Row([
                ft.Text("", width=12),
                ft.Text("Outcome", size=10, color=TEXT_DIM, weight=ft.FontWeight.BOLD, width=140),
                ft.Text("Mkt", size=10, color=TEXT_DIM, weight=ft.FontWeight.BOLD, width=50),
                ft.Text("Nostra", size=10, color=TEXT_DIM, weight=ft.FontWeight.BOLD, width=50),
                ft.Text("Edge", size=10, color=TEXT_DIM, weight=ft.FontWeight.BOLD, width=55),
                ft.Text("Conf", size=10, color=TEXT_DIM, weight=ft.FontWeight.BOLD, width=40),
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN, spacing=4),
            padding=pad(h=10, v=5),
            border_radius=10,
            bgcolor=ft.Colors.with_opacity(0.18, SURFACE_2),
            border=ft.Border.all(1, ft.Colors.with_opacity(0.45, OUTLINE_SOFT)),
        )

        card_content = ft.Column([
            # Header con confidence badge
            ft.Row([
                ft.Column([
                    ft.Text(question, size=13, color=TEXT, weight=ft.FontWeight.BOLD,
                             max_lines=2, overflow=ft.TextOverflow.ELLIPSIS),
                    header_badges,
                ], expand=True, spacing=4),
                ft.Column([
                    ft.Container(
                        content=ft.Text(sig_type, size=10, color=BG,
                                         weight=ft.FontWeight.BOLD),
                        bgcolor=sig_color, border_radius=999, padding=pad(h=9, v=3),
                    ),
                    ft.Container(
                        content=ft.Text(f"{max_edge:+.1%}", size=11, color=BG,
                                         weight=ft.FontWeight.BOLD),
                        bgcolor=edge_color, border_radius=999, padding=pad(h=9, v=3),
                    ),
                    ft.Container(
                        content=ft.Text(f"{conf_rating} {conf_total:.0f}",
                                         size=9, color=BG, weight=ft.FontWeight.BOLD),
                        bgcolor=conf_color, border_radius=999, padding=pad(h=7, v=2),
                    ),
                ], spacing=3, horizontal_alignment=ft.CrossAxisAlignment.END),
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            ft.Container(height=4),
            outcome_header,
            *outcome_rows,
            ft.Row([
                bet_banner,
                ft.Container(expand=True),
                bet_btn,
            ], vertical_alignment=ft.CrossAxisAlignment.CENTER) if best_bet else bet_banner,
        ], spacing=4)

        # Se ci sono dettagli, usa ExpansionTile
        if detail_controls:
            card_content.controls.append(ft.Container(height=4))
            card_content.controls.append(
                ft.ExpansionTile(
                    title=ft.Text("Dettagli analisi", size=11, color=TEXT_DIM),
                    controls=[ft.Container(
                        content=ft.Column(detail_controls, spacing=8),
                        padding=pad(h=8, v=8),
                        border_radius=12,
                        bgcolor=ft.Colors.with_opacity(0.14, SURFACE_2),
                        border=ft.Border.all(1, ft.Colors.with_opacity(0.35, OUTLINE_SOFT)),
                    )],
                    expanded=False,
                    collapsed_icon_color=TEXT_DIM,
                    icon_color=ACCENT,
                )
            )

        return ft.Container(
            content=card_content,
            bgcolor=ft.Colors.with_opacity(0.24, SURFACE_2),
            border=ft.Border(
                left=ft.BorderSide(3, edge_color),
                top=ft.BorderSide(1, ft.Colors.with_opacity(0.14, edge_color)),
                right=ft.BorderSide(1, ft.Colors.with_opacity(0.45, OUTLINE_SOFT)),
                bottom=ft.BorderSide(1, ft.Colors.with_opacity(0.45, OUTLINE_SOFT)),
            ),
            border_radius=14,
            padding=14,
            shadow=ft.BoxShadow(
                spread_radius=0,
                blur_radius=12,
                color=ft.Colors.with_opacity(0.12, "#01040a"),
                offset=ft.Offset(0, 4),
            ),
        )

    return build, load
