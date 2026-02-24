"""Mappa section - Interactive weather map with city details panel."""

import logging

import flet as ft

from weather_trader.api_client import api_get
from weather_trader.constants import (
    ACCENT, BG, GREEN, OUTLINE_SOFT, RED, SURFACE_2, TEXT, TEXT_DIM, YELLOW, CITY_MAP, f2c,
)
from weather_trader.logic.pnl_tracker import load_pnl
from weather_trader.widgets.factory import (
    make_card, make_kv_row, make_section_title, make_loading_indicator,
    make_empty_state, pad,
)
from weather_trader.widgets.weather_map import build_weather_map

logger = logging.getLogger(__name__)


def create_mappa(page, state, safe_update):
    def style_select(ctrl):
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

    map_container = ft.Container(expand=True)
    detail_panel = ft.Container(
        expand=True,
        height=500,
        bgcolor=ft.Colors.with_opacity(0.28, SURFACE_2),
        border=ft.Border.all(1, ft.Colors.with_opacity(0.45, OUTLINE_SOFT)),
        padding=16,
        border_radius=14,
        content=ft.Text("Clicca su una citta' sulla mappa", color=TEXT_DIM, size=13),
    )
    layer_dropdown = ft.Dropdown(
        width=160,
        value="temperature",
        options=[
            ft.dropdown.Option("temperature", "Temperatura"),
            ft.dropdown.Option("precipitation", "Precipitazioni"),
            ft.dropdown.Option("confidence", "Confidenza"),
            ft.dropdown.Option("exposure", "Esposizione"),
        ],
        border_color=ACCENT,
        color=TEXT,
        text_size=13,
        on_select=lambda _: load(),
    )
    layer_dropdown_wrap = style_select(layer_dropdown)

    city_data: dict[str, dict] = {}

    def _on_city_click(slug):
        data = city_data.get(slug, {})
        name = CITY_MAP.get(slug, slug)

        rows = []
        if data.get("temp_max") is not None:
            rows.append(ft.Row([
                ft.Text("Temp Max:", size=12, color=TEXT_DIM, width=90),
                ft.Text(f"{f2c(data['temp_max']):.1f}°C", size=12, color=TEXT,
                         weight=ft.FontWeight.BOLD),
            ]))
        if data.get("temp_min") is not None:
            rows.append(ft.Row([
                ft.Text("Temp Min:", size=12, color=TEXT_DIM, width=90),
                ft.Text(f"{f2c(data['temp_min']):.1f}°C", size=12, color=TEXT),
            ]))
        if data.get("precipitation") is not None:
            precip = data["precipitation"]
            rows.append(ft.Row([
                ft.Text("Precipitaz.:", size=12, color=TEXT_DIM, width=90),
                ft.Text(f"{precip:.1f} mm", size=12,
                         color="#26c6da" if precip > 0 else TEXT),
            ]))
        if data.get("wind_max") is not None:
            rows.append(ft.Row([
                ft.Text("Vento Max:", size=12, color=TEXT_DIM, width=90),
                ft.Text(f"{data['wind_max']:.0f} km/h", size=12, color=TEXT),
            ]))
        if data.get("ensemble_std") is not None:
            std = data["ensemble_std"]
            spread_color = GREEN if std < 3 else (ACCENT if std < 5 else RED)
            rows.append(ft.Row([
                ft.Text("Spread Ens.:", size=12, color=TEXT_DIM, width=90),
                ft.Text(f"±{std * 5/9:.1f}°C", size=12, color=spread_color,
                         weight=ft.FontWeight.BOLD),
            ]))
        if data.get("confidence") is not None:
            conf = data["confidence"]
            conf_color = GREEN if conf > 0.7 else (ACCENT if conf > 0.4 else RED)
            rows.append(ft.Row([
                ft.Text("Confidenza:", size=12, color=TEXT_DIM, width=90),
                ft.Text(f"{conf:.0%}", size=12, color=conf_color,
                         weight=ft.FontWeight.BOLD),
            ]))

        # Exposure info
        if data.get("exposure") is not None and data["exposure"] > 0:
            exp = data["exposure"]
            exp_pct = data.get("exposure_pct", 0)
            exp_color = GREEN if exp_pct < 0.05 else (YELLOW if exp_pct < 0.10 else RED)
            rows.append(ft.Row([
                ft.Text("Esposizione:", size=12, color=TEXT_DIM, width=90),
                ft.Text(f"${exp:.0f} ({exp_pct:.0%})", size=12, color=exp_color,
                         weight=ft.FontWeight.BOLD),
            ]))

        detail_panel.content = ft.Column([
            ft.Text(name, size=16, weight=ft.FontWeight.BOLD, color=TEXT),
            ft.Divider(height=1, color=ft.Colors.with_opacity(0.10, TEXT)),
            *rows,
            ft.Container(height=8),
            ft.Button(
                "Vai a Previsioni",
                icon=ft.Icons.CALENDAR_MONTH,
                on_click=lambda _, s=slug: _go_to_forecast(s),
                style=ft.ButtonStyle(
                    bgcolor=ACCENT,
                    color=BG,
                    shape=ft.RoundedRectangleBorder(radius=12),
                    padding=pad(h=12, v=10),
                ),
            ),
        ], spacing=8, scroll=ft.ScrollMode.AUTO)
        safe_update()

    def _go_to_forecast(slug):
        state.current_city = slug
        state.invalidate_cache()
        state.current_section = 1
        # Trigger actual navigation via state callback
        if hasattr(state, 'navigate_to') and state.navigate_to:
            state.navigate_to(1)

    def build():
        return ft.Container(
            content=ft.Column([
                ft.Row([
                    make_section_title("Mappa Meteo", ft.Icons.MAP, ACCENT),
                    ft.Container(expand=True),
                    layer_dropdown_wrap,
                ]),
                ft.ResponsiveRow([
                    ft.Column([
                        ft.Container(
                            content=map_container,
                            expand=True,
                            border_radius=16,
                            bgcolor=ft.Colors.with_opacity(0.18, SURFACE_2),
                            border=ft.Border.all(1, ft.Colors.with_opacity(0.45, OUTLINE_SOFT)),
                            padding=pad(h=8, v=8),
                        ),
                    ], col={"xs": 12, "md": 8}),
                    ft.Column([
                        detail_panel,
                    ], col={"xs": 12, "md": 4}),
                ], expand=True, spacing=12, run_spacing=12),
            ], spacing=10, expand=True),
            expand=True, padding=20,
        )

    def load():
        nonlocal city_data
        map_container.content = make_loading_indicator("Caricamento mappa...")
        safe_update()

        try:
            data = api_get("/forecast/overview/all")
            if not data or "cities" not in data:
                map_container.content = make_empty_state(
                    ft.Icons.MAP_OUTLINED, "Nessun dato disponibile")
                safe_update()
                return

            # Load exposure data from bets
            pnl_data = load_pnl()
            bets = pnl_data.get("bets", [])
            bankroll = pnl_data.get("bankroll", 1000)
            city_stakes = {}
            for b in bets:
                if b.get("status") == "pending":
                    c = b.get("city", "") or ""
                    city_stakes[c] = city_stakes.get(c, 0) + b.get("stake", 0)

            city_data = {}
            for entry in data["cities"]:
                city_info = entry.get("city", {})
                slug = city_info.get("slug", "")
                forecasts = entry.get("forecast", [])

                if forecasts:
                    f = forecasts[0]
                    ens = f.get("ensemble", {})
                    exposure = city_stakes.get(slug, 0)
                    city_data[slug] = {
                        "temp_max": f.get("temp_max"),
                        "temp_min": f.get("temp_min"),
                        "precipitation": f.get("precipitation_sum"),
                        "wind_max": f.get("wind_max"),
                        "ensemble_std": ens.get("ensemble_std"),
                        "ensemble_mean": ens.get("ensemble_mean"),
                        "n_members": ens.get("n_members"),
                        "confidence": 1.0 - min(1.0, (ens.get("ensemble_std") or 5) / 10),
                        "exposure": exposure,
                        "exposure_pct": exposure / bankroll if bankroll > 0 else 0,
                    }

            layer = layer_dropdown.value or "temperature"
            win_w = getattr(getattr(page, "window", None), "width", None) or 1320
            map_h = 520 if win_w >= 1350 else (480 if win_w >= 1100 else (430 if win_w >= 900 else 360))
            detail_panel.height = map_h

            map_widget = build_weather_map(
                city_data,
                on_city_click=_on_city_click,
                width=None,
                height=map_h,
                layer=layer,
            )
            map_container.content = map_widget
            safe_update()

        except Exception as e:
            logger.error("Failed to load map: %s", e)
            map_container.content = make_empty_state(
                ft.Icons.ERROR_OUTLINE, f"Errore caricamento mappa: {e}")
            safe_update()

    return build, load
