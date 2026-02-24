"""Interactive weather map using flet_map (OpenStreetMap)."""

import logging

import flet as ft
import flet_map as fm

from weather_trader.constants import (
    ACCENT, ACCENT2, BG, GREEN, ORANGE, OUTLINE_SOFT, RED, SURFACE_2, TEXT, TEXT_DIM, YELLOW,
    CITY_MAP, CITY_SLUGS,
    f2c,
)

logger = logging.getLogger(__name__)

# City coordinates (lat, lon)
CITY_COORDS = {
    "nyc": (40.7128, -74.006),
    "miami": (25.7617, -80.1918),
    "chicago": (41.8781, -87.6298),
    "los_angeles": (34.0522, -118.2437),
    "dallas": (32.7767, -96.797),
    "atlanta": (33.749, -84.388),
    "seattle": (47.6062, -122.3321),
    "london": (51.5074, -0.1278),
    "paris": (48.8566, 2.3522),
    "ankara": (39.9334, 32.8597),
    "seoul": (37.5665, 126.978),
    "toronto": (43.6532, -79.3832),
    "sao_paulo": (-23.5505, -46.6333),
    "buenos_aires": (-34.6037, -58.3816),
    "wellington": (-41.2865, 174.7762),
    "roma": (41.9028, 12.4964),
    "milano": (45.4642, 9.19),
    "napoli": (40.8518, 14.2681),
    "berlin": (52.52, 13.405),
    "madrid": (40.4168, -3.7038),
    "amsterdam": (52.3676, 4.9041),
    "cesena": (44.1396, 12.2464),
    "bologna": (44.4949, 11.3426),
    "vipiteno": (46.8953, 11.4322),
}


def _temp_to_color(temp_f: float | None) -> str:
    """Convert temperature (°F) to a hex color (blue -> yellow -> red)."""
    if temp_f is None:
        return TEXT_DIM
    t = max(0, min(100, temp_f))
    ratio = t / 100.0
    if ratio < 0.33:
        r, g, b = 50, int(100 + ratio * 3 * 155), 255
    elif ratio < 0.66:
        p = (ratio - 0.33) * 3
        r, g, b = int(p * 255), 255, int(255 * (1 - p))
    else:
        p = (ratio - 0.66) * 3
        r, g, b = 255, int(255 * (1 - p)), 50
    return f"#{r:02x}{g:02x}{b:02x}"


def _precip_to_color(precip: float) -> str:
    """Precipitation amount to color."""
    if precip < 1:
        return "#42a5f5"
    elif precip < 5:
        return "#26c6da"
    elif precip < 15:
        return "#ffa726"
    return "#f44336"


def _confidence_to_color(conf: float) -> str:
    """Confidence (0-1) to color."""
    if conf > 0.7:
        return GREEN
    elif conf > 0.4:
        return "#FFA726"
    return RED


def _exposure_to_color(pct: float) -> str:
    """Exposure percentage to color: green < 5%, yellow 5-10%, red > 10%."""
    if pct <= 0:
        return TEXT_DIM
    if pct < 0.05:
        return GREEN
    elif pct < 0.10:
        return "#FFA726"
    return RED


def _legend_items(layer: str) -> list[tuple[str, str]]:
    """Legend labels and colors for selected layer."""
    if layer == "temperature":
        return [
            ("Freddo", "#42a5f5"),
            ("Mite", "#26c6da"),
            ("Caldo", "#ffd166"),
            ("Molto caldo", "#ff6b6b"),
        ]
    if layer == "precipitation":
        return [
            ("0-1 mm", "#42a5f5"),
            ("1-5 mm", "#26c6da"),
            ("5-15 mm", "#ffa726"),
            (">15 mm", "#f44336"),
        ]
    if layer == "confidence":
        return [
            ("Bassa", RED),
            ("Media", ORANGE),
            ("Alta", GREEN),
        ]
    return [
        ("Nessuna", TEXT_DIM),
        ("<5%", GREEN),
        ("5-10%", YELLOW),
        (">10%", RED),
    ]


def _legend_title(layer: str) -> str:
    return {
        "temperature": "Layer: Temperatura max",
        "precipitation": "Layer: Precipitazioni",
        "confidence": "Layer: Confidenza forecast",
        "exposure": "Layer: Esposizione portafoglio",
    }.get(layer, "Layer")


def build_weather_map(
    city_data: dict[str, dict],
    on_city_click: callable = None,
    width: float | None = None,
    height: float = 500,
    layer: str = "temperature",
) -> ft.Container:
    """Build an interactive weather map with OpenStreetMap tiles."""

    # Build circle markers for the heatmap overlay
    circle_markers = []
    markers = []

    for slug, (lat, lon) in CITY_COORDS.items():
        data = city_data.get(slug, {})
        name = CITY_MAP.get(slug, slug)
        coords = fm.MapLatitudeLongitude(lat, lon)

        # Color + radius based on layer
        if layer == "temperature":
            val = data.get("temp_max")
            color = _temp_to_color(val)
            std = data.get("ensemble_std") or 3
            radius = max(8, min(22, 8 + std * 1.5))
            label = f"{f2c(val):.0f}°C" if val is not None else "—"
        elif layer == "precipitation":
            precip = data.get("precipitation", 0) or 0
            color = _precip_to_color(precip)
            radius = max(8, min(22, 8 + precip * 0.8))
            label = f"{precip:.1f}mm"
        elif layer == "confidence":
            conf = data.get("confidence", 0.5)
            color = _confidence_to_color(conf)
            radius = max(8, min(22, 8 + conf * 14))
            label = f"{conf:.0%}"
        else:  # exposure
            exp_pct = data.get("exposure_pct", 0)
            exposure = data.get("exposure", 0)
            color = _exposure_to_color(exp_pct)
            radius = max(8, min(25, 8 + exp_pct * 150)) if exp_pct > 0 else 6
            label = f"${exposure:.0f}" if exposure > 0 else "—"

        # Circle marker (heatmap dot)
        circle_markers.append(fm.CircleMarker(
            radius=radius,
            coordinates=coords,
            color=ft.Colors.with_opacity(0.35, color),
            border_color=color,
            border_stroke_width=2,
        ))

        # Tooltip text
        tip_parts = [name]
        if data.get("temp_max") is not None:
            tip_parts.append(f"Temp: {f2c(data['temp_max']):.1f}°C")
        if data.get("precipitation") is not None:
            tip_parts.append(f"Precip: {data['precipitation']:.1f}mm")
        if data.get("ensemble_std") is not None:
            tip_parts.append(f"Spread: ±{data['ensemble_std'] * 5 / 9:.1f}°C")
        if data.get("exposure", 0) > 0:
            tip_parts.append(f"Exposure: ${data['exposure']:.0f} ({data.get('exposure_pct', 0):.0%})")

        def _make_click(s):
            def handler(e):
                if on_city_click:
                    on_city_click(s)
            return handler

        # Label marker
        markers.append(fm.Marker(
            coordinates=coords,
            width=80,
            height=40,
            content=ft.GestureDetector(
                on_tap=_make_click(slug),
                content=ft.Container(
                    content=ft.Column([
                        ft.Text(name, size=10, weight=ft.FontWeight.BOLD,
                                color=TEXT, text_align=ft.TextAlign.CENTER),
                        ft.Text(label, size=9, color=color,
                                weight=ft.FontWeight.BOLD,
                                text_align=ft.TextAlign.CENTER),
                    ], spacing=0, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                    bgcolor=ft.Colors.with_opacity(0.82, SURFACE_2),
                    border=ft.Border.all(1, ft.Colors.with_opacity(0.55, OUTLINE_SOFT)),
                    border_radius=8,
                    padding=4,
                    tooltip="\n".join(tip_parts),
                ),
            ),
        ))

    # Dark-themed OSM tile layer
    tile_layer = fm.TileLayer(
        url_template="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        subdomains=["a", "b", "c", "d"],
    )

    # Attribution
    attribution = fm.RichAttribution(
        attributions=[fm.TextSourceAttribution(text="CartoDB Dark Matter")],
    )

    map_widget = fm.Map(
        layers=[
            tile_layer,
            fm.CircleLayer(circles=circle_markers),
            fm.MarkerLayer(markers),
            attribution,
        ],
        initial_center=fm.MapLatitudeLongitude(35, 10),
        initial_zoom=3,
        min_zoom=2,
        max_zoom=12,
    )

    legend = ft.Container(
        left=12,
        top=12,
        content=ft.Column([
            ft.Text(_legend_title(layer), size=11, color=TEXT, weight=ft.FontWeight.BOLD),
            ft.Row([
                ft.Container(
                    content=ft.Row([
                        ft.Container(width=8, height=8, border_radius=4, bgcolor=c),
                        ft.Text(lbl, size=9, color=TEXT_DIM),
                    ], spacing=4),
                    border_radius=999,
                    bgcolor=ft.Colors.with_opacity(0.18, SURFACE_2),
                    border=ft.Border.all(1, ft.Colors.with_opacity(0.35, OUTLINE_SOFT)),
                    padding=ft.Padding(left=6, right=6, top=3, bottom=3),
                )
                for lbl, c in _legend_items(layer)
            ], spacing=6, wrap=True),
        ], spacing=6),
        bgcolor=ft.Colors.with_opacity(0.82, SURFACE_2),
        border=ft.Border.all(1, ft.Colors.with_opacity(0.45, OUTLINE_SOFT)),
        border_radius=12,
        padding=ft.Padding(left=10, right=10, top=8, bottom=8),
    )

    helper = ft.Container(
        right=12,
        top=12,
        content=ft.Row([
            ft.Icon(ft.Icons.TOUCH_APP, size=13, color=ACCENT),
            ft.Text("Clicca una città per dettagli", size=10, color=TEXT_DIM),
        ], spacing=6),
        bgcolor=ft.Colors.with_opacity(0.78, SURFACE_2),
        border=ft.Border.all(1, ft.Colors.with_opacity(0.35, OUTLINE_SOFT)),
        border_radius=999,
        padding=ft.Padding(left=8, right=8, top=5, bottom=5),
    )

    map_base = ft.Container(
        content=map_widget,
        width=width,
        height=height,
        border_radius=14,
        clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
    )

    return ft.Container(
        content=ft.Stack(
            controls=[map_base, legend, helper],
            width=width,
            height=height,
        ),
        width=width,
        height=height,
        border_radius=14,
        clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
    )
