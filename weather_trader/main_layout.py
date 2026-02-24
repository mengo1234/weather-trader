"""Main layout: header, navigation rail, status bar, section wiring."""

import inspect
import logging
import threading
import time
from datetime import datetime

import flet as ft

from weather_trader.api_client import api_get, api_post
from weather_trader.app_state import AppState
from weather_trader.constants import (
    ACCENT, ACCENT2, BG, CARD, CARD_HOVER, CITY_NAMES, CITY_SLUGS, GREEN, ORANGE, RED, TEXT, TEXT_DIM, YELLOW,
    FONT_UI, TYPE_XS, TYPE_SM, TYPE_MD, TYPE_LG, TYPE_XL, TYPE_2XL, SURFACE_1,
)
from weather_trader.logging_config import setup_logging
from weather_trader.widgets.factory import pad

logger = logging.getLogger("weather_trader.layout")


def _build_theme():
    """Build Flet theme with Material 3 when supported by runtime version."""
    try:
        params = inspect.signature(ft.Theme).parameters
        kwargs = {}
        if "color_scheme_seed" in params:
            kwargs["color_scheme_seed"] = ACCENT
        if "use_material3" in params:
            kwargs["use_material3"] = True
        if "font_family" in params:
            kwargs["font_family"] = FONT_UI
        return ft.Theme(**kwargs)
    except Exception:
        return ft.Theme(color_scheme_seed=ACCENT)


def main(page: ft.Page):
    setup_logging()
    logger.info("Weather Trader starting")

    page.title = "Weather Trader"
    page.bgcolor = BG
    page.padding = 0
    page.theme_mode = ft.ThemeMode.DARK
    page.theme = _build_theme()
    if hasattr(page, "dark_theme"):
        page.dark_theme = _build_theme()
    # Window sizing: avoid opening larger than the user's screen on smaller laptops.
    # Prefer maximized if supported by the installed Flet version.
    if hasattr(page, "window"):
        try:
            if hasattr(page.window, "maximized"):
                page.window.maximized = True
        except Exception:
            pass
        try:
            page.window.width = 1180
            page.window.height = 760
            page.window.min_width = 980
            page.window.min_height = 640
        except Exception:
            pass

    state = AppState()
    section_labels = ["Dashboard", "Previsioni", "Mercati", "Storico", "Sistema", "Mappa", "Guida"]

    # ========================================================
    # SAFE UPDATE
    # ========================================================
    def safe_update():
        # Try thread-safe APIs first, but fall back progressively.
        try:
            if hasattr(page, "call_from_thread"):
                try:
                    page.call_from_thread(page.update)
                    return
                except Exception as e1:
                    logger.warning("safe_update call_from_thread failed: %s", e1, exc_info=True)
            if hasattr(page, "schedule_update"):
                try:
                    page.schedule_update()
                    return
                except Exception as e2:
                    logger.warning("safe_update schedule_update failed: %s", e2, exc_info=True)
            page.update()
        except Exception as e:
            logger.error("safe_update FAILED: %s", e, exc_info=True)

    def threaded_load(fn, *args):
        """Esegue fn in background thread."""
        def _run():
            try:
                fn(*args)
            except Exception as e:
                logger.error("Threaded load error in %s: %s", fn.__name__, e)
        threading.Thread(target=_run, daemon=True).start()

    def pill(content, bgcolor, border_color=None, padding_h=12, padding_v=8):
        return ft.Container(
            content=content,
            bgcolor=bgcolor,
            border=ft.Border.all(1, border_color or ft.Colors.with_opacity(0.08, TEXT)),
            border_radius=999,
            padding=pad(h=padding_h, v=padding_v),
        )

    # ========================================================
    # HEADER
    # ========================================================
    status_dot = ft.Container(width=10, height=10, border_radius=5, bgcolor=RED)
    status_text = ft.Text("Connessione...", size=TYPE_MD, color=TEXT_DIM)
    status_hint_text = ft.Text("Health check in corso", size=TYPE_XS, color=TEXT_DIM)
    hdr_section_text = ft.Text("Dashboard", size=TYPE_SM, color=TEXT, weight=ft.FontWeight.W_700)
    hdr_city_text = ft.Text("New York", size=TYPE_SM, color=TEXT, weight=ft.FontWeight.W_700)
    sb_context_text = ft.Text("Sezione: Dashboard | Città: New York", size=TYPE_SM, color=TEXT_DIM)
    sb_refresh_text = ft.Text("Auto refresh: 60s", size=TYPE_SM, color=TEXT_DIM)
    btn_refresh_label = ft.Text("Aggiorna", size=TYPE_SM, color=TEXT, weight=ft.FontWeight.W_600)

    def _refresh_shell_context():
        city_name = dict(zip(CITY_SLUGS, CITY_NAMES)).get(state.current_city, state.current_city)
        hdr_city_text.value = city_name
        hdr_section_text.value = section_labels[state.current_section]
        sb_context_text.value = f"Sezione: {section_labels[state.current_section]} | Città: {city_name}"

    def on_city_change(e):
        state.current_city = e.control.value
        state.invalidate_cache()
        _refresh_shell_context()
        load_current_section()

    city_dropdown = ft.Dropdown(
        width=220,
        border_width=0,
        focused_border_width=0,
        border_color=ft.Colors.TRANSPARENT,
        focused_border_color=ft.Colors.TRANSPARENT,
        bgcolor=ft.Colors.TRANSPARENT,
        color=TEXT,
        text_size=TYPE_LG,
        content_padding=pad(h=12, v=8),
        on_select=on_city_change,
        options=[ft.dropdown.Option(key=s, text=n) for s, n in zip(CITY_SLUGS, CITY_NAMES)],
        value="nyc",
    )

    refresh_button = ft.Container(
        content=ft.Row([
            ft.Icon(ft.Icons.REFRESH, size=16, color=ACCENT),
            btn_refresh_label,
        ], spacing=6),
        border_radius=999,
        bgcolor=ft.Colors.with_opacity(0.12, ACCENT),
        border=ft.Border.all(1, ft.Colors.with_opacity(0.22, ACCENT)),
        padding=pad(h=12, v=8),
    )

    header = ft.Container(
        content=ft.Row([
            ft.Row([
                ft.Container(
                    width=44,
                    height=44,
                    border_radius=14,
                    bgcolor=ft.Colors.with_opacity(0.10, ACCENT),
                    border=ft.Border.all(1, ft.Colors.with_opacity(0.16, ACCENT)),
                    content=ft.Icon(ft.Icons.THUNDERSTORM, color=ACCENT, size=24),
                    alignment=ft.Alignment(0, 0),
                ),
                ft.Column([
                    ft.Row([
                        ft.Text("Weather Trader", size=TYPE_2XL, weight=ft.FontWeight.BOLD, color=TEXT),
                        ft.Container(width=8),
                        ft.Container(
                            content=ft.Text("PRO", size=TYPE_XS, color=BG, weight=ft.FontWeight.BOLD),
                            bgcolor=ACCENT2,
                            border_radius=999,
                            padding=pad(h=8, v=3),
                        ),
                    ], spacing=0),
                    ft.Text(
                        "Forecast intelligence + market execution dashboard",
                        size=TYPE_SM,
                        color=TEXT_DIM,
                    ),
                    ft.Row([
                        pill(
                            ft.Row([
                                ft.Icon(ft.Icons.APPS, size=14, color=ACCENT2),
                                hdr_section_text,
                            ], spacing=6),
                            bgcolor=ft.Colors.with_opacity(0.10, ACCENT2),
                            border_color=ft.Colors.with_opacity(0.18, ACCENT2),
                            padding_h=10,
                            padding_v=5,
                        ),
                        pill(
                            ft.Row([
                                ft.Icon(ft.Icons.PIN_DROP_OUTLINED, size=14, color=ACCENT),
                                hdr_city_text,
                            ], spacing=6),
                            bgcolor=ft.Colors.with_opacity(0.08, ACCENT),
                            border_color=ft.Colors.with_opacity(0.14, ACCENT),
                            padding_h=10,
                            padding_v=5,
                        ),
                    ], spacing=8, wrap=True),
                ], spacing=2),
            ], spacing=12),
            ft.Row([
                refresh_button,
                pill(
                    ft.Row([
                        ft.Icon(ft.Icons.LOCATION_CITY, size=16, color=ACCENT),
                        city_dropdown,
                    ], spacing=6),
                    bgcolor=ft.Colors.with_opacity(0.42, CARD),
                    border_color=ft.Colors.with_opacity(0.12, TEXT),
                    padding_h=6,
                    padding_v=4,
                ),
                pill(
                    ft.Column([
                        ft.Row([
                            status_dot,
                            status_text,
                        ], spacing=8),
                        status_hint_text,
                    ], spacing=2),
                    bgcolor=ft.Colors.with_opacity(0.04, GREEN),
                    border_color=ft.Colors.with_opacity(0.10, GREEN),
                    padding_h=12,
                    padding_v=7,
                ),
            ], spacing=10),
        ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN, vertical_alignment=ft.CrossAxisAlignment.CENTER),
        padding=pad(h=22, v=16),
        margin=pad(h=16, v=14),
        border_radius=20,
        gradient=ft.LinearGradient(
            begin=ft.Alignment(-1, -1),
            end=ft.Alignment(1, 1),
            colors=[
                ft.Colors.with_opacity(0.96, CARD_HOVER),
                ft.Colors.with_opacity(0.92, CARD),
            ],
        ),
        border=ft.Border.all(1, ft.Colors.with_opacity(0.10, TEXT)),
        shadow=ft.BoxShadow(
            spread_radius=0,
            blur_radius=20,
            color=ft.Colors.with_opacity(0.22, "#01040a"),
            offset=ft.Offset(0, 10),
        ),
    )

    # ========================================================
    # STATUS BAR (bottom)
    # ========================================================
    sb_db_text = ft.Text("DB: —", size=TYPE_SM, color=TEXT_DIM)
    sb_update_text = ft.Text("Ultimo aggiornamento: —", size=TYPE_SM, color=TEXT_DIM)

    status_bar = ft.Container(
        content=ft.Row([
            pill(
                ft.Row([
                    ft.Icon(ft.Icons.STORAGE, size=14, color=ACCENT),
                    sb_db_text,
                ], spacing=8),
                bgcolor=ft.Colors.with_opacity(0.35, CARD),
                padding_h=10,
                padding_v=6,
            ),
            ft.Container(expand=True),
            pill(
                ft.Row([
                    ft.Icon(ft.Icons.SCHEDULE, size=14, color=YELLOW),
                    sb_refresh_text,
                ], spacing=8),
                bgcolor=ft.Colors.with_opacity(0.35, CARD),
                padding_h=10,
                padding_v=6,
            ),
            ft.Container(width=8),
            pill(
                ft.Row([
                    ft.Icon(ft.Icons.UPDATE, size=14, color=TEXT_DIM),
                    sb_update_text,
                ], spacing=8),
                bgcolor=ft.Colors.with_opacity(0.35, CARD),
                padding_h=10,
                padding_v=6,
            ),
        ], vertical_alignment=ft.CrossAxisAlignment.CENTER),
        padding=pad(h=16, v=0),
        margin=pad(h=16, v=0),
        height=52,
    )

    # Wire status widgets into state so sections can update them
    state.status_dot = status_dot
    state.status_text = status_text
    state.status_hint_text = status_hint_text
    state.sb_db_text = sb_db_text
    state.sb_update_text = sb_update_text
    state.sb_context_text = sb_context_text

    # ========================================================
    # SECTIONS — lazy imports to avoid circular deps
    # ========================================================
    from weather_trader.sections.dashboard import create_dashboard
    from weather_trader.sections.previsioni import create_previsioni
    from weather_trader.sections.mercati import create_mercati
    from weather_trader.sections.storico import create_storico
    from weather_trader.sections.sistema import create_sistema
    from weather_trader.sections.guida import create_guida
    from weather_trader.sections.mappa import create_mappa

    dash_build, dash_load = create_dashboard(page, state, safe_update)
    prev_build, prev_load = create_previsioni(page, state, safe_update)
    mkt_build, mkt_load = create_mercati(page, state, safe_update)
    stor_build, stor_load = create_storico(page, state, safe_update)
    sys_build, sys_load = create_sistema(page, state, safe_update)
    map_build, map_load = create_mappa(page, state, safe_update)
    guide_build, guide_load = create_guida(page, state, safe_update)

    section_builders = [dash_build, prev_build, mkt_build, stor_build, sys_build, map_build, guide_build]
    section_loaders = [dash_load, prev_load, mkt_load, stor_load, sys_load, map_load, guide_load]

    # Navigation callback — used by sections (e.g. mappa → previsioni) to navigate programmatically
    def _navigate_to(idx):
        if 0 <= idx < len(section_builders):
            nav_rail.selected_index = idx
            state.current_section = idx
            _refresh_shell_context()
            content_area.content = section_builders[idx]()
            safe_update()
            threaded_load(section_loaders[idx])
    state.navigate_to = _navigate_to

    def _manual_refresh(e=None):
        btn_refresh_label.value = "Aggiorno..."
        status_hint_text.value = f"Refresh manuale · {datetime.now().strftime('%H:%M:%S')}"
        state.cache["health"] = None
        state.cache["metrics"] = None
        safe_update()

        def _run():
            try:
                section_loaders[state.current_section]()
            finally:
                btn_refresh_label.value = "Aggiorna"
                safe_update()

        threaded_load(_run)

    refresh_button.on_click = _manual_refresh

    # ========================================================
    # CONTENT AREA
    # ========================================================
    content_area = ft.Container(
        content=dash_build(),
        expand=True,
        padding=pad(h=18, v=16),
        border_radius=22,
        bgcolor=ft.Colors.with_opacity(0.55, CARD),
        border=ft.Border.all(1, ft.Colors.with_opacity(0.08, TEXT)),
        shadow=ft.BoxShadow(
            spread_radius=0,
            blur_radius=20,
            color=ft.Colors.with_opacity(0.16, "#02050b"),
            offset=ft.Offset(0, 8),
        ),
    )

    # ========================================================
    # AI ASSISTANT FAB + CHAT OVERLAY (always available)
    # ========================================================
    ai_ui = {"open": False, "busy": False}
    ai_history: list[dict[str, str]] = []

    ai_panel_title = ft.Text("Assistente AI", size=TYPE_LG, color=TEXT, weight=ft.FontWeight.W_700)
    ai_panel_subtitle = ft.Text("Forecast + mercati + sistema (contesto reale)", size=TYPE_XS, color=TEXT_DIM)
    ai_status_text = ft.Text("Pronto", size=TYPE_XS, color=TEXT_DIM)
    ai_context_text = ft.Text("Città: New York · Sezione: Dashboard", size=TYPE_XS, color=TEXT_DIM)
    ai_backend_text = ft.Text("AI backend: check non eseguito", size=TYPE_XS, color=TEXT_DIM)
    ai_model_label = ft.Text("Modello AI (free consigliati):", size=TYPE_XS, color=TEXT_DIM)
    ai_model_hint_text = ft.Text("Default: qwen3:8b (ottimo equilibrio qualità/velocità)", size=TYPE_2XS, color=TEXT_DIM)
    ai_messages = ft.Column(
        spacing=8,
        scroll=ft.ScrollMode.AUTO,
        height=330,
    )
    ai_install_logs = ft.Column(spacing=4, scroll=ft.ScrollMode.AUTO, height=96)
    ai_install_status_text = ft.Text("Installazione: non avviata", size=TYPE_XS, color=TEXT_DIM)
    ai_install_confirm_text = ft.Text(
        "Confermi installazione automatica di Ollama + modello selezionato?",
        size=TYPE_XS,
        color=TEXT_DIM,
    )
    ai_quick_prompt = ft.Text("Prompt rapidi:", size=TYPE_XS, color=TEXT_DIM)
    ai_input = ft.TextField(
        hint_text="Chiedi qualcosa (es. setup migliore oggi?)",
        expand=True,
        border_radius=12,
        text_size=TYPE_SM,
    )
    ai_send_label = ft.Text("Invia", size=TYPE_SM, color=TEXT, weight=ft.FontWeight.W_600)
    ai_install_button_label = ft.Text("Installa Ollama", size=TYPE_XS, color=TEXT, weight=ft.FontWeight.W_600)
    ai_install_refresh_label = ft.Text("Aggiorna stato", size=TYPE_XS, color=TEXT_DIM, weight=ft.FontWeight.W_600)
    ai_selected_model = ft.Dropdown(
        width=230,
        value="qwen3:8b",
        text_size=TYPE_SM,
        border_radius=10,
        options=[
            ft.dropdown.Option("qwen3:8b", "Qwen3 8B (Recommended)"),
            ft.dropdown.Option("qwen3:4b", "Qwen3 4B (PC più leggeri)"),
            ft.dropdown.Option("gemma3:4b", "Gemma 3 4B"),
            ft.dropdown.Option("llama3.2:3b", "Llama 3.2 3B (molto leggero)"),
            ft.dropdown.Option("qwen2.5:7b-instruct", "Qwen2.5 7B Instruct (fallback stabile)"),
            ft.dropdown.Option("qwen3.5:cloud", "Qwen3.5 Cloud (novità, non locale)"),
        ],
    )

    def _ai_refresh_context_line():
        city_name = dict(zip(CITY_SLUGS, CITY_NAMES)).get(state.current_city, state.current_city)
        sec = section_labels[state.current_section]
        ai_context_text.value = f"Città: {city_name} · Sezione: {sec}"

    def _ai_get_selected_model() -> str:
        return str(ai_selected_model.value or "qwen3:8b").strip()

    def _ai_refresh_model_hint():
        model = _ai_get_selected_model()
        if model == "qwen3:8b":
            ai_model_hint_text.value = "Default: qwen3:8b (ottimo equilibrio qualità/velocità)"
        elif model == "qwen3:4b":
            ai_model_hint_text.value = "Più leggero e veloce, qualità inferiore a 8B"
        elif model == "gemma3:4b":
            ai_model_hint_text.value = "Alternativa compatta, buona per chat/riassunti"
        elif model == "llama3.2:3b":
            ai_model_hint_text.value = "Molto leggero, ideale se hai poca RAM/VRAM"
        elif model == "qwen3.5:cloud":
            ai_model_hint_text.value = "Modalità cloud Ollama: non locale, potrebbe non essere gratis"
        else:
            ai_model_hint_text.value = f"Modello selezionato: {model}"

    def _ai_refresh_install_confirm_text():
        model = _ai_get_selected_model()
        ai_install_confirm_text.value = (
            "Conferma installazione automatica.\n"
            f"- Modello: {model}\n"
            "- Scarica file da internet (può essere pesante)\n"
            "- Occupa spazio disco\n"
            "- Su Linux l'installer può richiedere permessi amministratore"
        )

    def _ai_message_bubble(role: str, text: str) -> tuple[ft.Control, ft.Text]:
        is_user = role == "user"
        txt = ft.Text(
            text,
            size=TYPE_SM,
            color=TEXT if is_user else TEXT,
            selectable=True,
        )
        bubble = ft.Container(
            content=txt,
            padding=pad(h=10, v=8),
            border_radius=14,
            bgcolor=ft.Colors.with_opacity(0.18, ACCENT if is_user else CARD_HOVER),
            border=ft.Border.all(
                1,
                ft.Colors.with_opacity(0.20, ACCENT if is_user else TEXT_DIM),
            ),
            width=360,
        )
        row = ft.Row(
            [bubble],
            alignment=ft.MainAxisAlignment.END if is_user else ft.MainAxisAlignment.START,
        )
        return row, txt

    def _ai_append_message(role: str, text: str) -> ft.Text:
        row, txt = _ai_message_bubble(role, text)
        ai_messages.controls.append(row)
        return txt

    def _ai_set_busy(is_busy: bool):
        ai_ui["busy"] = is_busy
        ai_send_label.value = "Invio..." if is_busy else "Invia"
        ai_status_text.value = "Sto analizzando..." if is_busy else "Pronto"
        try:
            ai_input.disabled = is_busy
        except Exception:
            pass

    def _ai_send_message(message_text: str | None = None):
        text = (message_text if message_text is not None else ai_input.value or "").strip()
        if not text:
            return
        if ai_ui["busy"]:
            ai_status_text.value = "Attendi la risposta in corso..."
            safe_update()
            return

        if message_text is None:
            ai_input.value = ""

        _ai_refresh_context_line()
        _ai_append_message("user", text)
        placeholder = _ai_append_message("assistant", "Analizzo forecast, mercati LIVE e sistema...")
        ai_history.append({"role": "user", "content": text})
        ai_history[:] = ai_history[-12:]
        _ai_set_busy(True)
        safe_update()

        payload = {
            "message": text,
            "city_slug": state.current_city,
            "section": section_labels[state.current_section],
            "history": ai_history[-8:],
        }

        def _run():
            resp = api_post("/ai/chat", payload, timeout=75)
            if resp and resp.get("reply"):
                provider = str(resp.get("provider") or "ai").upper()
                model_name = str(resp.get("model") or "").strip()
                provider_line = f"[{provider}{' · ' + model_name if model_name else ''}]\n"
                placeholder.value = provider_line + str(resp.get("reply") or "")
                ai_status_text.value = (
                    f"{provider} · match live {((resp.get('context') or {}).get('live_matches', 0))}"
                )
                ai_history.append({"role": "assistant", "content": str(resp.get("reply") or "")})
                ai_history[:] = ai_history[-12:]
            else:
                placeholder.value = (
                    "AI non disponibile.\n"
                    "Controlla che il backend sia attivo e, per l'AI locale, che Ollama sia avviato."
                )
                ai_status_text.value = "Errore chiamata /ai/chat"
            _ai_set_busy(False)
            safe_update()

        threaded_load(_run)

    def _ai_refresh_backend_status():
        st = api_get("/ai/status", timeout=4)
        inst = api_get("/ai/install/status", timeout=4)
        if st:
            provider = str(st.get("provider") or "ollama")
            reachable = bool(st.get("reachable"))
            configured = bool(st.get("configured"))
            model = st.get("model") or "—"
            if reachable:
                ai_backend_text.value = f"{provider}: online · modello config: {model}"
            else:
                why = "non raggiungibile" if configured else "non configurato"
                ai_backend_text.value = f"{provider}: {why} · modello: {model}"
            if configured and not ai_ui["busy"]:
                try:
                    ai_selected_model.value = model
                    _ai_refresh_model_hint()
                except Exception:
                    pass
        else:
            ai_backend_text.value = "AI backend: endpoint /ai/status non raggiungibile"

        ai_install_logs.controls.clear()
        if inst:
            ai_install_status_text.value = (
                f"Installazione: {inst.get('status', 'n/a')} · step={inst.get('step') or '-'}"
            )
            if inst.get("error"):
                ai_install_status_text.value += f" · errore: {inst.get('error')}"
            logs = inst.get("logs") or []
            for line in logs[-10:]:
                ai_install_logs.controls.append(ft.Text(str(line), size=TYPE_2XS, color=TEXT_DIM))
        else:
            ai_install_status_text.value = "Installazione: stato non disponibile"
            ai_install_logs.controls.append(ft.Text("Impossibile leggere /ai/install/status", size=TYPE_2XS, color=TEXT_DIM))
        safe_update()

    def _ai_on_submit(e):
        _ai_send_message()

    def _ai_toggle_panel(e=None):
        ai_ui["open"] = not ai_ui["open"]
        ai_panel.visible = ai_ui["open"]
        _ai_refresh_context_line()
        if ai_ui["open"]:
            threaded_load(_ai_refresh_backend_status)
        safe_update()

    def _ai_quick_action(prompt: str):
        def _handler(e):
            _ai_send_message(prompt)
        return _handler

    def _ai_show_install_confirm(e=None):
        _ai_refresh_install_confirm_text()
        ai_install_confirm_box.visible = True
        safe_update()

    def _ai_cancel_install(e=None):
        ai_install_confirm_box.visible = False
        safe_update()

    def _ai_confirm_install(e=None):
        model = _ai_get_selected_model()
        ai_install_confirm_box.visible = False
        ai_install_status_text.value = f"Installazione: avvio in corso ({model})..."
        ai_install_logs.controls.clear()
        ai_install_logs.controls.append(ft.Text(f"Richiesta inviata al backend per {model}...", size=TYPE_2XS, color=TEXT_DIM))
        safe_update()

        def _run():
            resp = api_post("/ai/install/ollama", {"confirm": True, "model": model}, timeout=12)
            if resp and resp.get("started"):
                ai_install_status_text.value = f"Installazione: avviata ({resp.get('model')})"
            elif resp:
                ai_install_status_text.value = str(resp.get("message") or "Installazione non avviata")
            else:
                ai_install_status_text.value = "Installazione: errore chiamata endpoint"
            safe_update()
            # Poll a few times to surface progress without user needing to click.
            for _ in range(5):
                time.sleep(2)
                _ai_refresh_backend_status()

        threaded_load(_run)

    def _ai_refresh_install(e=None):
        threaded_load(_ai_refresh_backend_status)

    def _ai_on_model_change(e=None):
        _ai_refresh_model_hint()
        _ai_refresh_install_confirm_text()
        safe_update()

    ai_input.on_submit = _ai_on_submit
    ai_selected_model.on_change = _ai_on_model_change
    _ai_refresh_model_hint()
    _ai_refresh_install_confirm_text()

    ai_send_button = ft.Container(
        content=ft.Row([
            ft.Icon(ft.Icons.SEND, size=14, color=ACCENT),
            ai_send_label,
        ], spacing=6),
        padding=pad(h=12, v=10),
        border_radius=12,
        bgcolor=ft.Colors.with_opacity(0.12, ACCENT),
        border=ft.Border.all(1, ft.Colors.with_opacity(0.24, ACCENT)),
        on_click=lambda e: _ai_send_message(),
    )

    ai_install_button = ft.Container(
        content=ft.Row([
            ft.Icon(ft.Icons.DOWNLOAD, size=13, color=ACCENT2),
            ai_install_button_label,
        ], spacing=6),
        padding=pad(h=10, v=8),
        border_radius=10,
        bgcolor=ft.Colors.with_opacity(0.08, ACCENT2),
        border=ft.Border.all(1, ft.Colors.with_opacity(0.16, ACCENT2)),
        on_click=_ai_show_install_confirm,
    )

    ai_install_refresh_button = ft.Container(
        content=ft.Row([
            ft.Icon(ft.Icons.REFRESH, size=12, color=TEXT_DIM),
            ai_install_refresh_label,
        ], spacing=5),
        padding=pad(h=10, v=8),
        border_radius=10,
        bgcolor=ft.Colors.with_opacity(0.05, TEXT),
        border=ft.Border.all(1, ft.Colors.with_opacity(0.10, TEXT)),
        on_click=_ai_refresh_install,
    )

    ai_install_confirm_box = ft.Container(
        visible=False,
        padding=pad(h=10, v=8),
        border_radius=12,
        bgcolor=ft.Colors.with_opacity(0.32, CARD_HOVER),
        border=ft.Border.all(1, ft.Colors.with_opacity(0.12, ORANGE)),
        content=ft.Column([
            ft.Row([
                ft.Icon(ft.Icons.WARNING_AMBER_OUTLINED, size=14, color=ORANGE),
                ai_install_confirm_text,
            ], spacing=6),
            ft.Row([
                ft.TextButton("Annulla", on_click=_ai_cancel_install),
                ft.FilledButton("Conferma e installa", on_click=_ai_confirm_install),
            ], spacing=8, alignment=ft.MainAxisAlignment.END),
        ], spacing=6),
    )

    ai_quick_actions = ft.Row(
        [
            ft.TextButton("Analizza oggi", on_click=_ai_quick_action("Analizza il setup migliore oggi per la città corrente.")),
            ft.TextButton("Rischi 7 giorni", on_click=_ai_quick_action("Riassumi i principali rischi dei prossimi 7 giorni per scommesse meteo.")),
            ft.TextButton("Top LIVE", on_click=_ai_quick_action("Mostrami i migliori mercati LIVE collegati e spiegami il perché.")),
        ],
        wrap=True,
        spacing=0,
        run_spacing=0,
    )

    ai_panel = ft.Container(
        right=18,
        bottom=104,
        width=440,
        height=600,
        visible=False,
        border_radius=18,
        bgcolor=ft.Colors.with_opacity(0.96, SURFACE_1),
        border=ft.Border.all(1, ft.Colors.with_opacity(0.14, TEXT)),
        shadow=ft.BoxShadow(
            blur_radius=24,
            spread_radius=0,
            color=ft.Colors.with_opacity(0.34, "#01040a"),
            offset=ft.Offset(0, 10),
        ),
        padding=pad(h=14, v=14),
        content=ft.Column([
            ft.Row([
                ft.Row([
                    ft.Container(
                        width=34,
                        height=34,
                        border_radius=10,
                        bgcolor=ft.Colors.with_opacity(0.12, ACCENT2),
                        border=ft.Border.all(1, ft.Colors.with_opacity(0.18, ACCENT2)),
                        alignment=ft.Alignment(0, 0),
                        content=ft.Icon(ft.Icons.CHAT_BUBBLE_OUTLINE, size=18, color=ACCENT2),
                    ),
                    ft.Column([ai_panel_title, ai_panel_subtitle], spacing=1),
                ], spacing=10),
                ft.Container(expand=True),
                ft.IconButton(
                    icon=ft.Icons.CLOSE,
                    icon_color=TEXT_DIM,
                    icon_size=18,
                    tooltip="Chiudi",
                    on_click=_ai_toggle_panel,
                ),
            ], vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ft.Container(height=4),
            ft.Row([
                pill(ai_context_text, bgcolor=ft.Colors.with_opacity(0.35, CARD), padding_h=10, padding_v=6),
            ], wrap=True),
            ft.Container(height=4),
            ft.Container(
                content=ai_messages,
                padding=pad(h=6, v=6),
                border_radius=14,
                bgcolor=ft.Colors.with_opacity(0.30, CARD),
                border=ft.Border.all(1, ft.Colors.with_opacity(0.08, TEXT)),
            ),
            ft.Container(
                content=ft.Column([
                    ai_quick_prompt,
                    ai_quick_actions,
                ], spacing=0),
                padding=pad(h=6, v=2),
            ),
            ft.Row([
                ai_input,
                ai_send_button,
            ], spacing=8),
            ft.Row([
                ft.Icon(ft.Icons.INFO_OUTLINE, size=12, color=TEXT_DIM),
                ai_status_text,
            ], spacing=6),
            ft.Row([
                ft.Icon(ft.Icons.COMPUTER, size=12, color=TEXT_DIM),
                ai_backend_text,
            ], spacing=6),
            ft.Row([
                ft.Icon(ft.Icons.TUNE, size=12, color=TEXT_DIM),
                ai_model_label,
            ], spacing=6),
            ft.Row([
                ai_selected_model,
            ], spacing=6),
            ai_model_hint_text,
            ft.Row([
                ai_install_button,
                ai_install_refresh_button,
            ], spacing=8, wrap=True),
            ai_install_confirm_box,
            ft.Container(
                content=ft.Column([
                    ai_install_status_text,
                    ft.Container(
                        content=ai_install_logs,
                        padding=pad(h=6, v=6),
                        border_radius=10,
                        bgcolor=ft.Colors.with_opacity(0.18, BG),
                        border=ft.Border.all(1, ft.Colors.with_opacity(0.08, TEXT)),
                    ),
                ], spacing=4),
                padding=pad(h=2, v=2),
            ),
            ft.Container(height=8),
        ], spacing=6),
    )

    ai_fab = ft.Container(
        right=20,
        bottom=78,
        width=58,
        height=58,
        border_radius=29,
        gradient=ft.LinearGradient(
            begin=ft.Alignment(-1, -1),
            end=ft.Alignment(1, 1),
            colors=[ACCENT2, ACCENT],
        ),
        shadow=ft.BoxShadow(
            blur_radius=18,
            spread_radius=0,
            color=ft.Colors.with_opacity(0.35, ACCENT2),
            offset=ft.Offset(0, 8),
        ),
        border=ft.Border.all(1, ft.Colors.with_opacity(0.20, TEXT)),
        alignment=ft.Alignment(0, 0),
        on_click=_ai_toggle_panel,
        content=ft.Icon(ft.Icons.CHAT, color=BG, size=28),
    )

    _ai_append_message(
        "assistant",
        "Ciao. Sono l'assistente della piattaforma.\nPosso leggere forecast, mercati LIVE e stato sistema della città selezionata.",
    )

    def on_nav_change(e):
        idx = e.control.selected_index
        if idx == state.current_section:
            return
        state.current_section = idx
        _refresh_shell_context()
        content_area.content = section_builders[idx]()
        safe_update()
        threaded_load(section_loaders[idx])

    def load_current_section():
        """Ricostruisce e carica la sezione corrente."""
        _refresh_shell_context()
        content_area.content = section_builders[state.current_section]()
        safe_update()
        threaded_load(section_loaders[state.current_section])

    # ========================================================
    # NAVIGATION RAIL
    # ========================================================
    nav_rail = ft.NavigationRail(
        selected_index=0,
        label_type=ft.NavigationRailLabelType.SELECTED,
        min_width=80,
        min_extended_width=80,
        bgcolor=ft.Colors.TRANSPARENT,
        indicator_color=ft.Colors.with_opacity(0.18, ACCENT),
        destinations=[
            ft.NavigationRailDestination(
                icon=ft.Icons.DASHBOARD_OUTLINED,
                selected_icon=ft.Icons.DASHBOARD,
                label="Dashboard",
            ),
            ft.NavigationRailDestination(
                icon=ft.Icons.CALENDAR_MONTH_OUTLINED,
                selected_icon=ft.Icons.CALENDAR_MONTH,
                label="Previsioni",
            ),
            ft.NavigationRailDestination(
                icon=ft.Icons.CASINO_OUTLINED,
                selected_icon=ft.Icons.CASINO,
                label="Mercati",
            ),
            ft.NavigationRailDestination(
                icon=ft.Icons.HISTORY_OUTLINED,
                selected_icon=ft.Icons.HISTORY,
                label="Storico",
            ),
            ft.NavigationRailDestination(
                icon=ft.Icons.SETTINGS_OUTLINED,
                selected_icon=ft.Icons.SETTINGS,
                label="Sistema",
            ),
            ft.NavigationRailDestination(
                icon=ft.Icons.MAP_OUTLINED,
                selected_icon=ft.Icons.MAP,
                label="Mappa",
            ),
            ft.NavigationRailDestination(
                icon=ft.Icons.SCHOOL_OUTLINED,
                selected_icon=ft.Icons.SCHOOL,
                label="Guida",
            ),
        ],
        on_change=on_nav_change,
    )

    sidebar = ft.Container(
        width=112,
        margin=pad(h=16, v=0),
        padding=pad(h=10, v=12),
        border_radius=22,
        gradient=ft.LinearGradient(
            begin=ft.Alignment(0, -1),
            end=ft.Alignment(0, 1),
            colors=[
                ft.Colors.with_opacity(0.92, CARD_HOVER),
                ft.Colors.with_opacity(0.88, CARD),
            ],
        ),
        border=ft.Border.all(1, ft.Colors.with_opacity(0.10, TEXT)),
        shadow=ft.BoxShadow(
            spread_radius=0,
            blur_radius=18,
            color=ft.Colors.with_opacity(0.20, "#02050b"),
            offset=ft.Offset(0, 8),
        ),
        content=ft.Column([
            ft.Container(
                content=ft.Icon(ft.Icons.CLOUD_SYNC, color=ACCENT, size=20),
                width=42,
                height=42,
                border_radius=12,
                bgcolor=ft.Colors.with_opacity(0.10, ACCENT),
                border=ft.Border.all(1, ft.Colors.with_opacity(0.18, ACCENT)),
                alignment=ft.Alignment(0, 0),
            ),
            ft.Container(height=6),
            ft.Text("WT", size=TYPE_SM, color=TEXT_DIM, weight=ft.FontWeight.W_600),
            ft.Container(height=10),
            ft.Container(
                content=nav_rail,
                expand=True,
                alignment=ft.Alignment(0, -1),
            ),
            ft.Container(height=8),
            ft.Container(
                content=ft.Column([
                    ft.Icon(ft.Icons.BOLT, size=14, color=ACCENT2),
                    ft.Text("Live", size=TYPE_XS, color=TEXT_DIM),
                ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=4),
                padding=pad(h=8, v=10),
                border_radius=12,
                bgcolor=ft.Colors.with_opacity(0.04, ACCENT2),
                border=ft.Border.all(1, ft.Colors.with_opacity(0.10, ACCENT2)),
                alignment=ft.Alignment(0, 0),
            ),
        ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=0, expand=True),
    )

    # ========================================================
    # LAYOUT PRINCIPALE
    # ========================================================
    root_stack = ft.Stack(
        controls=[
                ft.Container(
                    left=0,
                    top=0,
                    right=0,
                    bottom=0,
                    gradient=ft.LinearGradient(
                        begin=ft.Alignment(-1, -1),
                        end=ft.Alignment(1, 1),
                        colors=[
                            BG,
                            "#0a1020",
                            "#0a0e1a",
                        ],
                    ),
                ),
                ft.Container(
                    left=-80,
                    top=-40,
                    width=260,
                    height=260,
                    border_radius=130,
                    bgcolor=ft.Colors.with_opacity(0.05, ACCENT),
                ),
                ft.Container(
                    right=40,
                    top=40,
                    width=220,
                    height=220,
                    border_radius=110,
                    bgcolor=ft.Colors.with_opacity(0.04, ACCENT2),
                ),
                ft.Container(
                    left=0,
                    top=0,
                    right=0,
                    bottom=0,
                    content=ft.Column([
                        header,
                        ft.Row([
                            sidebar,
                            content_area,
                            ft.Container(width=16),
                        ], expand=True, spacing=0),
                        status_bar,
                    ], expand=True, spacing=0),
                ),
                ai_panel,
                ai_fab,
        ],
        expand=True,
    )
    page.add(root_stack)

    _refresh_shell_context()

    # ========================================================
    # INITIAL LOAD
    # ========================================================
    def initial_load():
        time.sleep(0.5)
        try:
            dash_load()
            logger.info("Dashboard loaded successfully")
        except Exception as e:
            logger.error("Dashboard load failed: %s", e)

    threading.Thread(target=initial_load, daemon=True).start()

    # ========================================================
    # AUTO-REFRESH
    # ========================================================
    def auto_refresh():
        cycle = 0
        while True:
            time.sleep(60)
            cycle += 1
            try:
                t_metrics = time.monotonic()
                metrics = api_get("/metrics")
                metrics_ms = (time.monotonic() - t_metrics) * 1000
                if metrics:
                    state.cache["metrics"] = metrics
                    fc_h = metrics.get("forecasts_hourly", 0)
                    ens = metrics.get("ensemble_members", 0)
                    obs = metrics.get("observations", 0)
                    nrm = metrics.get("climate_normals", 0)
                    sb_db_text.value = (f"DB: {fc_h:,} forecast  |  {ens:,} ensemble  |  "
                                        f"{obs:,} osservazioni  |  {nrm:,} normals")
                    sb_update_text.value = f"Ultimo aggiornamento: {datetime.now().strftime('%H:%M:%S')}"
                    sb_refresh_text.value = f"Auto refresh: 60s · API {metrics_ms:.0f}ms"

                t_health = time.monotonic()
                health = api_get("/health")
                health_ms = (time.monotonic() - t_health) * 1000
                if health and health.get("status") == "ok":
                    status_dot.bgcolor = GREEN
                    status_text.value = "Connesso"
                    status_hint_text.value = f"Health OK · {health_ms:.0f}ms"
                else:
                    status_dot.bgcolor = RED
                    status_text.value = "Offline"
                    status_hint_text.value = f"Health check fallito · {health_ms:.0f}ms"

                safe_update()

                if cycle % 5 == 0:
                    state.cache["forecast"] = None
                    # Reload data sections on their own schedule
                    if state.current_section in (0, 1):  # Dashboard, Previsioni
                        section_loaders[state.current_section]()
                    elif state.current_section == 2:  # Mercati — ogni 5 min
                        state.cache["markets"] = None
                        mkt_load()
                    elif state.current_section == 3:  # Storico — ogni 5 min
                        stor_load()

            except Exception as e:
                logger.error("Auto-refresh error: %s", e)

    threading.Thread(target=auto_refresh, daemon=True).start()
