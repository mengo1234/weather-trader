"""Chart builders: temperature, precipitation, precip probability, wind, UV, forecast insights."""

from datetime import date, datetime, timedelta

import flet as ft

from weather_trader.constants import (
    ACCENT, ACCENT2, BG, ORANGE, OUTLINE_SOFT, RED, SURFACE_2, TEXT, TEXT_DIM, YELLOW, f2c,
)
from weather_trader.widgets.factory import uv_color


def _chart_shell(content, height, title=None, subtitle=None):
    """Common chart shell with subtle MD3-like surface and border."""
    header = []
    if title:
        header.append(ft.Text(title, size=12, color=TEXT_DIM, weight=ft.FontWeight.BOLD))
    if subtitle:
        header.append(ft.Text(subtitle, size=9, color=TEXT_DIM))

    return ft.Container(
        content=ft.Column(
            [*header, ft.Container(content=content, expand=True)],
            spacing=4,
            expand=True,
        ),
        height=height,
        border_radius=14,
        bgcolor=ft.Colors.with_opacity(0.35, SURFACE_2),
        border=ft.Border.all(1, ft.Colors.with_opacity(0.6, OUTLINE_SOFT)),
        padding=ft.Padding(left=10, right=10, top=8, bottom=8),
    )


def _grid_background(width, height, lines=3):
    """Subtle horizontal guide lines for chart readability."""
    controls = []
    if lines <= 0:
        return controls
    for i in range(lines + 1):
        y = int(i * (height - 1) / max(1, lines))
        controls.append(
            ft.Container(
                left=0,
                right=0,
                top=y,
                height=1,
                bgcolor=ft.Colors.with_opacity(0.05, TEXT),
            )
        )
    return controls


def build_sparkline(values, width=200, height=40, color=ACCENT, show_dots=False):
    """Mini grafico sparkline con barre sottili."""
    if not values:
        return ft.Container(width=width, height=height)
    vmin = min(values)
    vmax = max(values)
    vrange = vmax - vmin if vmax > vmin else 1
    n = len(values)
    bar_w = max(2, (width - n) / n)

    bars = []
    for i, v in enumerate(values):
        h = max(2, (v - vmin) / vrange * (height - 4))
        bar = ft.Container(
            width=bar_w,
            height=h,
            bgcolor=ft.Colors.with_opacity(0.78, color),
            border_radius=2,
        )
        col = ft.Column(
            [ft.Container(expand=True), bar],
            spacing=0,
            height=height,
            width=bar_w,
        )
        bars.append(col)

    chart_stack = ft.Stack(
        controls=[
            *_grid_background(width, height, lines=2),
            ft.Container(
                left=0,
                top=0,
                right=0,
                bottom=0,
                content=ft.Row(bars, spacing=1, alignment=ft.MainAxisAlignment.CENTER),
            ),
        ],
        width=width,
        height=height,
    )

    return ft.Container(
        content=chart_stack,
        width=width,
        height=height,
        padding=ft.Padding(left=6, right=6, top=4, bottom=4),
        border_radius=10,
        bgcolor=ft.Colors.with_opacity(0.18, SURFACE_2),
        border=ft.Border.all(1, ft.Colors.with_opacity(0.45, OUTLINE_SOFT)),
        clip_behavior=ft.ClipBehavior.HARD_EDGE,
    )


def build_temp_bar_chart(forecasts, chart_height=220, max_days=14):
    """Grafico barre temperatura con bande ensemble."""
    fcs = forecasts[:max_days]
    if not fcs:
        return ft.Container(height=chart_height)

    def day_weather_icon(fc):
        """Heuristic daily weather icon from real daily forecast fields."""
        precip_mm = float(fc.get("precipitation_sum", 0) or 0)
        precip_prob = float(fc.get("precip_probability", 0) or 0)
        wind = float(fc.get("wind_max", 0) or 0)
        gust = float(fc.get("wind_gusts_max", 0) or 0)
        uv_val = float(fc.get("uv_max", 0) or 0)

        if (precip_prob >= 70 and precip_mm >= 2) or precip_mm >= 8:
            return ft.Icons.WATER_DROP, "#4fc3f7", "Piovoso"
        if gust >= 40 or wind >= 30:
            return ft.Icons.AIR, ORANGE, "Ventoso"
        if precip_prob >= 35 or precip_mm >= 1:
            return ft.Icons.CLOUD, "#b0bec5", "Nuvoloso/instabile"
        if uv_val >= 6:
            return ft.Icons.WB_SUNNY, "#ffd54f", "Soleggiato"
        return ft.Icons.CLOUD_QUEUE, ft.Colors.with_opacity(0.9, TEXT_DIM), "Variabile"

    def day_role_label(date_str: str):
        """Highlight today/tomorrow labels from forecast ISO date."""
        try:
            d = datetime.strptime(str(date_str)[:10], "%Y-%m-%d").date()
        except Exception:
            return "", TEXT_DIM, False
        today = date.today()
        if d == today:
            return "OGGI", ACCENT2, True
        if d == today + timedelta(days=1):
            return "DOMANI", YELLOW, True
        return "", TEXT_DIM, False

    def pretty_day_label(date_str: str):
        """Italian compact day label, e.g. 'Lun 24'."""
        weekdays = ["Lun", "Mar", "Mer", "Gio", "Ven", "Sab", "Dom"]
        try:
            d = datetime.strptime(str(date_str)[:10], "%Y-%m-%d").date()
            return f"{weekdays[d.weekday()]} {d.day}"
        except Exception:
            raw = str(date_str)
            return raw[5:] if len(raw) >= 10 else raw

    dates = [pretty_day_label(fc.get("date", "")) for fc in fcs]
    all_vals = []
    bar_data = []
    for fc in fcs:
        t_max = fc.get("temp_max", 0) or 0
        t_min = fc.get("temp_min", 0) or 0
        ens = fc.get("ensemble", {})
        ens_hi = ens.get("ensemble_max", t_max) or t_max
        ens_lo = ens.get("ensemble_min", t_min) or t_min
        bar_data.append((t_max, t_min, ens_hi, ens_lo))
        all_vals.extend([t_max, t_min, ens_hi, ens_lo])

    y_min = min(all_vals) - 5 if all_vals else 0
    y_max = max(all_vals) + 5 if all_vals else 100
    y_range = y_max - y_min if y_max > y_min else 1

    # Riserva spazio per etichette dei giorni/valori sotto le barre.
    top_pad = 8
    bottom_pad = 6
    labels_h = 74
    plot_h = max(120, chart_height - top_pad - bottom_pad - labels_h)
    n_days = max(1, len(fcs))
    if n_days <= 7:
        day_slot_w = 104
        row_spacing = 10
    elif n_days <= 10:
        day_slot_w = 74
        row_spacing = 8
    else:
        day_slot_w = 48
        row_spacing = 6
    ens_bar_w = 28
    temp_bar_w = 18
    ens_left = int((day_slot_w - ens_bar_w) / 2)
    temp_left = int((day_slot_w - temp_bar_w) / 2)

    def to_px(val):
        return max(0, (val - y_min) / y_range * plot_h)

    bar_columns = []
    for i, (t_max, t_min, ens_hi, ens_lo) in enumerate(bar_data):
        ens_h = to_px(ens_hi) - to_px(ens_lo)
        ens_bottom = to_px(ens_lo)
        temp_h = to_px(t_max) - to_px(t_min)
        temp_bottom = to_px(t_min)

        bar_col = ft.Container(
            content=ft.Stack([
                ft.Container(
                    width=ens_bar_w, height=max(2, ens_h),
                    bgcolor=ft.Colors.with_opacity(0.22, ACCENT2),
                    border=ft.Border.all(1, ft.Colors.with_opacity(0.20, ACCENT2)),
                    border_radius=4,
                    bottom=ens_bottom, left=ens_left,
                ),
                ft.Container(
                    width=temp_bar_w, height=max(2, temp_h),
                    bgcolor=ACCENT,
                    border_radius=4,
                    bottom=temp_bottom, left=temp_left,
                ),
            ]),
            width=day_slot_w, height=plot_h,
            tooltip=ft.Tooltip(message=f"{dates[i]}: {f2c(t_min):.0f}-{f2c(t_max):.0f}°C"),
        )

        icon_name, icon_color, icon_label = day_weather_icon(fcs[i])
        role_txt, role_color, is_special_day = day_role_label(fcs[i].get("date", ""))
        date_color = role_color if is_special_day else TEXT_DIM
        col = ft.Column([
            bar_col,
            ft.Icon(icon_name, size=14, color=icon_color, tooltip=ft.Tooltip(message=icon_label)),
            ft.Container(
                content=ft.Text(
                    role_txt if role_txt else " ",
                    size=8,
                    color=BG if is_special_day else ft.Colors.TRANSPARENT,
                    text_align=ft.TextAlign.CENTER,
                    weight=ft.FontWeight.BOLD,
                    width=day_slot_w - 10,
                ),
                bgcolor=ft.Colors.with_opacity(0.20, role_color) if is_special_day else ft.Colors.TRANSPARENT,
                border=ft.Border.all(1, ft.Colors.with_opacity(0.20, role_color)) if is_special_day else None,
                border_radius=999,
                padding=ft.Padding(left=4, right=4, top=1, bottom=1),
            ),
            ft.Text(dates[i], size=10, color=date_color, text_align=ft.TextAlign.CENTER, width=day_slot_w,
                    weight=ft.FontWeight.BOLD if is_special_day else ft.FontWeight.W_500),
            ft.Text(f"{f2c(t_max):.0f}", size=10, color=ACCENT, text_align=ft.TextAlign.CENTER, width=day_slot_w),
        ], spacing=2, horizontal_alignment=ft.CrossAxisAlignment.CENTER, width=day_slot_w)
        bar_columns.append(col)

    y_labels = ft.Column([
        ft.Text(f"{f2c(y_max):.0f}°", size=10, color=TEXT_DIM),
        ft.Container(expand=True),
        ft.Text(f"{f2c((y_max + y_min) / 2):.0f}°", size=10, color=TEXT_DIM),
        ft.Container(expand=True),
        ft.Text(f"{f2c(y_min):.0f}°", size=10, color=TEXT_DIM),
    ], height=plot_h, width=40)

    stack_h = plot_h + labels_h
    min_plot_width = 760 if n_days <= 7 else (560 if n_days <= 10 else 300)
    plot_width = max(min_plot_width, len(bar_columns) * (day_slot_w + row_spacing))
    plot_area = ft.Container(
        content=ft.Stack(
            controls=[
                *_grid_background(plot_width, plot_h, lines=3),
                ft.Container(
                    left=0,
                    top=0,
                    right=0,
                    bottom=0,
                    content=ft.Row(
                        bar_columns,
                        spacing=row_spacing,
                        alignment=ft.MainAxisAlignment.START,
                    ),
                ),
            ],
            width=plot_width,
            height=stack_h,
        ),
        width=plot_width,
        height=chart_height,
        border_radius=14,
        bgcolor=ft.Colors.with_opacity(0.30, SURFACE_2),
        border=ft.Border.all(1, ft.Colors.with_opacity(0.55, OUTLINE_SOFT)),
        padding=ft.Padding(left=8, right=8, top=top_pad, bottom=bottom_pad),
        clip_behavior=ft.ClipBehavior.HARD_EDGE,
    )

    plot_scroll = ft.Row(
        [plot_area],
        spacing=0,
        scroll=ft.ScrollMode.AUTO,
        alignment=ft.MainAxisAlignment.CENTER,
        vertical_alignment=ft.CrossAxisAlignment.START,
        expand=True,
    )

    return ft.Container(
        content=ft.Row(
            [
                y_labels,
                ft.Container(content=plot_scroll, expand=True, alignment=ft.Alignment(0, -1)),
            ],
            spacing=8,
            alignment=ft.MainAxisAlignment.START,
            vertical_alignment=ft.CrossAxisAlignment.START,
        ),
        padding=ft.Padding(left=2, right=2, top=2, bottom=2),
    )


def build_precip_chart(forecasts, chart_height=140, max_days=14):
    """Grafico precipitazioni (barre blu)."""
    fcs = forecasts[:max_days]
    if not fcs:
        return ft.Container(height=chart_height)

    dates = [fc.get("date", "")[5:] for fc in fcs]
    values = [fc.get("precipitation_sum", 0) or 0 for fc in fcs]
    vmax = max(values) if values else 1
    if vmax == 0:
        vmax = 1
    plot_h = max(72, chart_height - 30)

    bars = []
    for i, v in enumerate(values):
        h = max(2, v / vmax * (plot_h - 26))
        color = "#26c6da" if v > 0 else ft.Colors.with_opacity(0.1, TEXT_DIM)
        bar = ft.Column([
            ft.Container(expand=True),
            ft.Text(f"{v:.1f}" if v > 0 else "", size=8, color="#26c6da",
                     text_align=ft.TextAlign.CENTER),
            ft.Container(
                width=16, height=h, bgcolor=color, border_radius=4,
                tooltip=ft.Tooltip(message=f"{dates[i]}: {v:.1f}mm"),
            ),
            ft.Text(dates[i], size=8, color=TEXT_DIM, text_align=ft.TextAlign.CENTER),
        ], spacing=1, horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            height=plot_h, width=24)
        bars.append(bar)

    plot_w = max(120, len(bars) * 26)
    plot_canvas = ft.Container(
        content=ft.Stack(
            controls=[
                *_grid_background(plot_w, plot_h, lines=2),
                ft.Container(
                    left=0, top=0, right=0, bottom=0,
                    content=ft.Row(bars, spacing=2, alignment=ft.MainAxisAlignment.CENTER),
                ),
            ],
            width=plot_w,
            height=plot_h,
        ),
        width=plot_w,
        height=plot_h,
        border_radius=12,
        bgcolor=ft.Colors.with_opacity(0.26, SURFACE_2),
        border=ft.Border.all(1, ft.Colors.with_opacity(0.5, OUTLINE_SOFT)),
        padding=ft.Padding(left=6, right=6, top=4, bottom=4),
        clip_behavior=ft.ClipBehavior.HARD_EDGE,
    )
    return _chart_shell(plot_canvas, height=chart_height, title="Precipitazioni (mm)")


def build_precip_probability_chart(forecasts, chart_height=140, max_days=14):
    """Grafico probabilita' di pioggia (0-100%)."""
    fcs = forecasts[:max_days]
    if not fcs:
        return ft.Container(height=chart_height)

    dates = [fc.get("date", "")[5:] for fc in fcs]
    probs = [max(0.0, min(100.0, float(fc.get("precip_probability", 0) or 0))) for fc in fcs]
    vmax = 100.0
    plot_h = max(72, chart_height - 30)

    bars = []
    for i, p in enumerate(probs):
        h = max(2, p / vmax * (plot_h - 26))
        if p >= 70:
            p_color = RED
        elif p >= 40:
            p_color = YELLOW
        elif p > 0:
            p_color = ACCENT
        else:
            p_color = ft.Colors.with_opacity(0.10, TEXT_DIM)

        bar = ft.Column([
            ft.Container(expand=True),
            ft.Text(f"{p:.0f}%" if p > 0 else "", size=8, color=p_color, text_align=ft.TextAlign.CENTER),
            ft.Container(
                width=16, height=h, bgcolor=p_color, border_radius=4,
                tooltip=ft.Tooltip(message=f"{dates[i]}: {p:.0f}%"),
            ),
            ft.Text(dates[i], size=8, color=TEXT_DIM, text_align=ft.TextAlign.CENTER),
        ], spacing=1, horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            height=plot_h, width=24)
        bars.append(bar)

    plot_w = max(120, len(bars) * 26)
    plot_canvas = ft.Container(
        content=ft.Stack(
            controls=[
                *_grid_background(plot_w, plot_h, lines=2),
                ft.Container(
                    left=0, top=0, right=0, bottom=0,
                    content=ft.Row(bars, spacing=2, alignment=ft.MainAxisAlignment.CENTER),
                ),
            ],
            width=plot_w, height=plot_h,
        ),
        width=plot_w,
        height=plot_h,
        border_radius=12,
        bgcolor=ft.Colors.with_opacity(0.26, SURFACE_2),
        border=ft.Border.all(1, ft.Colors.with_opacity(0.5, OUTLINE_SOFT)),
        padding=ft.Padding(left=6, right=6, top=4, bottom=4),
        clip_behavior=ft.ClipBehavior.HARD_EDGE,
    )
    avg_p = sum(probs) / len(probs) if probs else 0
    return _chart_shell(
        plot_canvas,
        height=chart_height,
        title="Prob. Pioggia (%)",
        subtitle=f"Media periodo: {avg_p:.0f}%",
    )


def build_ensemble_spread_chart(forecasts, chart_height=140, max_days=14):
    """Grafico spread ensemble (max-min) in °C: misura incertezza previsione."""
    fcs = forecasts[:max_days]
    if not fcs:
        return ft.Container(height=chart_height)

    dates = [fc.get("date", "")[5:] for fc in fcs]
    spreads_c = []
    for fc in fcs:
        ens = fc.get("ensemble", {}) or {}
        emax = ens.get("ensemble_max")
        emin = ens.get("ensemble_min")
        if emax is None or emin is None:
            spreads_c.append(0.0)
        else:
            spreads_c.append(max(0.0, (float(emax) - float(emin)) * 5 / 9))

    vmax = max(spreads_c) if spreads_c else 1
    vmax = max(1.0, vmax)
    plot_h = max(72, chart_height - 30)

    bars = []
    for i, s in enumerate(spreads_c):
        h = max(2, s / vmax * (plot_h - 26))
        if s >= 6:
            c = RED
        elif s >= 3:
            c = YELLOW
        elif s > 0:
            c = ACCENT2
        else:
            c = ft.Colors.with_opacity(0.10, TEXT_DIM)

        bars.append(ft.Column([
            ft.Container(expand=True),
            ft.Text(f"{s:.1f}" if s > 0 else "", size=8, color=c, text_align=ft.TextAlign.CENTER),
            ft.Container(
                width=16, height=h, bgcolor=c, border_radius=4,
                tooltip=ft.Tooltip(message=f"{dates[i]}: spread {s:.1f}°C"),
            ),
            ft.Text(dates[i], size=8, color=TEXT_DIM, text_align=ft.TextAlign.CENTER),
        ], spacing=1, horizontal_alignment=ft.CrossAxisAlignment.CENTER, height=plot_h, width=24))

    plot_w = max(120, len(bars) * 26)
    plot_canvas = ft.Container(
        content=ft.Stack(
            controls=[
                *_grid_background(plot_w, plot_h, lines=2),
                ft.Container(
                    left=0, top=0, right=0, bottom=0,
                    content=ft.Row(bars, spacing=2, alignment=ft.MainAxisAlignment.CENTER),
                ),
            ],
            width=plot_w, height=plot_h,
        ),
        width=plot_w,
        height=plot_h,
        border_radius=12,
        bgcolor=ft.Colors.with_opacity(0.26, SURFACE_2),
        border=ft.Border.all(1, ft.Colors.with_opacity(0.5, OUTLINE_SOFT)),
        padding=ft.Padding(left=6, right=6, top=4, bottom=4),
        clip_behavior=ft.ClipBehavior.HARD_EDGE,
    )
    avg_s = sum(spreads_c) / len(spreads_c) if spreads_c else 0
    return _chart_shell(plot_canvas, chart_height, title="Spread Ensemble (°C)", subtitle=f"Media: {avg_s:.1f}°C")


def build_temp_range_chart(forecasts, chart_height=140, max_days=14):
    """Grafico escursione termica giornaliera (Tmax-Tmin) in °C."""
    fcs = forecasts[:max_days]
    if not fcs:
        return ft.Container(height=chart_height)

    dates = [fc.get("date", "")[5:] for fc in fcs]
    ranges_c = []
    for fc in fcs:
        tmax = float(fc.get("temp_max", 0) or 0)
        tmin = float(fc.get("temp_min", 0) or 0)
        ranges_c.append(max(0.0, (tmax - tmin) * 5 / 9))

    vmax = max(ranges_c) if ranges_c else 1
    vmax = max(1.0, vmax)
    plot_h = max(72, chart_height - 30)

    bars = []
    for i, r in enumerate(ranges_c):
        h = max(2, r / vmax * (plot_h - 26))
        if r >= 15:
            c = ORANGE
        elif r >= 9:
            c = YELLOW
        else:
            c = ACCENT
        bars.append(ft.Column([
            ft.Container(expand=True),
            ft.Text(f"{r:.1f}", size=8, color=c, text_align=ft.TextAlign.CENTER),
            ft.Container(
                width=16, height=h, bgcolor=c, border_radius=4,
                tooltip=ft.Tooltip(message=f"{dates[i]}: ΔT {r:.1f}°C"),
            ),
            ft.Text(dates[i], size=8, color=TEXT_DIM, text_align=ft.TextAlign.CENTER),
        ], spacing=1, horizontal_alignment=ft.CrossAxisAlignment.CENTER, height=plot_h, width=24))

    plot_w = max(120, len(bars) * 26)
    plot_canvas = ft.Container(
        content=ft.Stack(
            controls=[
                *_grid_background(plot_w, plot_h, lines=2),
                ft.Container(
                    left=0, top=0, right=0, bottom=0,
                    content=ft.Row(bars, spacing=2, alignment=ft.MainAxisAlignment.CENTER),
                ),
            ],
            width=plot_w, height=plot_h,
        ),
        width=plot_w,
        height=plot_h,
        border_radius=12,
        bgcolor=ft.Colors.with_opacity(0.26, SURFACE_2),
        border=ft.Border.all(1, ft.Colors.with_opacity(0.5, OUTLINE_SOFT)),
        padding=ft.Padding(left=6, right=6, top=4, bottom=4),
        clip_behavior=ft.ClipBehavior.HARD_EDGE,
    )
    avg_r = sum(ranges_c) / len(ranges_c) if ranges_c else 0
    return _chart_shell(plot_canvas, chart_height, title="Escursione Termica (°C)", subtitle=f"Media: {avg_r:.1f}°C")


def build_snow_chart(forecasts, chart_height=140, max_days=14):
    """Grafico neve giornaliera (snowfall_sum)."""
    fcs = forecasts[:max_days]
    if not fcs:
        return ft.Container(height=chart_height)

    dates = [fc.get("date", "")[5:] for fc in fcs]
    values = [max(0.0, float(fc.get("snowfall_sum", 0) or 0)) for fc in fcs]
    vmax = max(values) if values else 1
    vmax = max(1.0, vmax)
    plot_h = max(72, chart_height - 30)

    bars = []
    for i, v in enumerate(values):
        h = max(2, v / vmax * (plot_h - 26)) if vmax > 0 else 2
        c = "#b3e5fc" if v > 0 else ft.Colors.with_opacity(0.10, TEXT_DIM)
        bars.append(ft.Column([
            ft.Container(expand=True),
            ft.Text(f"{v:.1f}" if v > 0 else "", size=8, color=c, text_align=ft.TextAlign.CENTER),
            ft.Container(
                width=16, height=h, bgcolor=c, border_radius=4,
                tooltip=ft.Tooltip(message=f"{dates[i]}: neve {v:.1f}"),
                border=ft.Border.all(1, ft.Colors.with_opacity(0.20, "#81d4fa")) if v > 0 else None,
            ),
            ft.Text(dates[i], size=8, color=TEXT_DIM, text_align=ft.TextAlign.CENTER),
        ], spacing=1, horizontal_alignment=ft.CrossAxisAlignment.CENTER, height=plot_h, width=24))

    plot_w = max(120, len(bars) * 26)
    plot_canvas = ft.Container(
        content=ft.Stack(
            controls=[
                *_grid_background(plot_w, plot_h, lines=2),
                ft.Container(
                    left=0, top=0, right=0, bottom=0,
                    content=ft.Row(bars, spacing=2, alignment=ft.MainAxisAlignment.CENTER),
                ),
            ],
            width=plot_w, height=plot_h,
        ),
        width=plot_w,
        height=plot_h,
        border_radius=12,
        bgcolor=ft.Colors.with_opacity(0.26, SURFACE_2),
        border=ft.Border.all(1, ft.Colors.with_opacity(0.5, OUTLINE_SOFT)),
        padding=ft.Padding(left=6, right=6, top=4, bottom=4),
        clip_behavior=ft.ClipBehavior.HARD_EDGE,
    )
    return _chart_shell(plot_canvas, chart_height, title="Neve", subtitle=f"Totale: {sum(values):.1f}")


def build_precip_hours_chart(forecasts, chart_height=140, max_days=14):
    """Grafico ore di precipitazione giornaliere."""
    fcs = forecasts[:max_days]
    if not fcs:
        return ft.Container(height=chart_height)

    dates = [fc.get("date", "")[5:] for fc in fcs]
    values = [max(0.0, float(fc.get("precipitation_hours", 0) or 0)) for fc in fcs]
    vmax = max(values) if values else 1
    vmax = max(1.0, vmax)
    plot_h = max(72, chart_height - 30)

    bars = []
    for i, v in enumerate(values):
        h = max(2, v / vmax * (plot_h - 26))
        c = ACCENT2 if v > 0 else ft.Colors.with_opacity(0.10, TEXT_DIM)
        bars.append(ft.Column([
            ft.Container(expand=True),
            ft.Text(f"{v:.1f}h" if v > 0 else "", size=8, color=c, text_align=ft.TextAlign.CENTER),
            ft.Container(
                width=16, height=h, bgcolor=c, border_radius=4,
                tooltip=ft.Tooltip(message=f"{dates[i]}: {v:.1f} ore precip."),
            ),
            ft.Text(dates[i], size=8, color=TEXT_DIM, text_align=ft.TextAlign.CENTER),
        ], spacing=1, horizontal_alignment=ft.CrossAxisAlignment.CENTER, height=plot_h, width=26))

    plot_w = max(120, len(bars) * 28)
    plot_canvas = ft.Container(
        content=ft.Stack(
            controls=[
                *_grid_background(plot_w, plot_h, lines=2),
                ft.Container(
                    left=0, top=0, right=0, bottom=0,
                    content=ft.Row(bars, spacing=2, alignment=ft.MainAxisAlignment.CENTER),
                ),
            ],
            width=plot_w, height=plot_h,
        ),
        width=plot_w,
        height=plot_h,
        border_radius=12,
        bgcolor=ft.Colors.with_opacity(0.26, SURFACE_2),
        border=ft.Border.all(1, ft.Colors.with_opacity(0.5, OUTLINE_SOFT)),
        padding=ft.Padding(left=6, right=6, top=4, bottom=4),
        clip_behavior=ft.ClipBehavior.HARD_EDGE,
    )
    avg_h = sum(values) / len(values) if values else 0
    return _chart_shell(plot_canvas, chart_height, title="Ore Precipitazione", subtitle=f"Media: {avg_h:.1f}h")


def build_wind_chart(forecasts, chart_height=140, max_days=14):
    """Grafico vento con barre gialle + raffiche rosse."""
    fcs = forecasts[:max_days]
    if not fcs:
        return ft.Container(height=chart_height)

    dates = [fc.get("date", "")[5:] for fc in fcs]
    winds = [fc.get("wind_max", 0) or 0 for fc in fcs]
    gusts = [fc.get("wind_gusts_max", 0) or 0 for fc in fcs]
    vmax = max(max(winds, default=1), max(gusts, default=1))
    if vmax == 0:
        vmax = 1
    plot_h = max(72, chart_height - 36)

    bars = []
    for i in range(len(fcs)):
        w, g = winds[i], gusts[i]
        wh = max(2, w / vmax * (plot_h - 16))
        gh = max(2, g / vmax * (plot_h - 16)) if g > w else 0

        stack_items = [
            ft.Container(
                width=14, height=wh, bgcolor=YELLOW, border_radius=4,
                bottom=0, left=3,
            ),
        ]
        if gh > 0:
            stack_items.append(ft.Container(
                width=14, height=gh, bgcolor=ft.Colors.with_opacity(0.3, RED),
                border_radius=4, bottom=0, left=3,
            ))

        bar = ft.Column([
            ft.Container(expand=True),
            ft.Container(
                content=ft.Stack(stack_items),
                width=20, height=plot_h - 12,
                tooltip=ft.Tooltip(message=f"{dates[i]}: {w:.0f} mph (raffica {g:.0f})"),
            ),
            ft.Text(dates[i], size=8, color=TEXT_DIM, text_align=ft.TextAlign.CENTER),
        ], spacing=1, horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            height=plot_h, width=24)
        bars.append(bar)

    header = ft.Row([
        ft.Row([
            ft.Text("Vento (mph)", size=12, color=TEXT_DIM, weight=ft.FontWeight.BOLD),
            ft.Container(width=8),
            ft.Container(width=10, height=10, bgcolor=YELLOW, border_radius=2),
            ft.Text("Max", size=10, color=TEXT_DIM),
            ft.Container(width=10, height=10, bgcolor=ft.Colors.with_opacity(0.3, RED), border_radius=2),
            ft.Text("Raffica", size=10, color=TEXT_DIM),
        ], spacing=4),
    ], spacing=4)
    plot_w = max(120, len(bars) * 26)
    plot_canvas = ft.Container(
        content=ft.Stack(
            controls=[
                *_grid_background(plot_w, plot_h, lines=2),
                ft.Container(
                    left=0, top=0, right=0, bottom=0,
                    content=ft.Row(bars, spacing=2, alignment=ft.MainAxisAlignment.CENTER),
                ),
            ],
            width=plot_w, height=plot_h,
        ),
        width=plot_w,
        height=plot_h,
        border_radius=12,
        bgcolor=ft.Colors.with_opacity(0.26, SURFACE_2),
        border=ft.Border.all(1, ft.Colors.with_opacity(0.5, OUTLINE_SOFT)),
        padding=ft.Padding(left=6, right=6, top=4, bottom=4),
        clip_behavior=ft.ClipBehavior.HARD_EDGE,
    )
    return _chart_shell(ft.Column([header, plot_canvas], spacing=4), height=chart_height)


def build_uv_chart(forecasts, chart_height=140, max_days=14):
    """Grafico UV index con barre colorate per livello."""
    fcs = forecasts[:max_days]
    if not fcs:
        return ft.Container(height=chart_height)

    dates = [fc.get("date", "")[5:] for fc in fcs]
    uvs = [fc.get("uv_max", 0) or 0 for fc in fcs]
    vmax = max(uvs) if uvs else 1
    if vmax == 0:
        vmax = 1
    plot_h = max(72, chart_height - 30)

    bars = []
    for i, u in enumerate(uvs):
        h = max(2, u / max(vmax, 11) * (plot_h - 26))
        bar = ft.Column([
            ft.Container(expand=True),
            ft.Text(f"{u:.1f}" if u > 0 else "", size=8, color=uv_color(u),
                     text_align=ft.TextAlign.CENTER),
            ft.Container(
                width=16, height=h, bgcolor=uv_color(u), border_radius=4,
                tooltip=ft.Tooltip(message=f"{dates[i]}: UV {u:.1f}"),
            ),
            ft.Text(dates[i], size=8, color=TEXT_DIM, text_align=ft.TextAlign.CENTER),
        ], spacing=1, horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            height=plot_h, width=24)
        bars.append(bar)

    plot_w = max(120, len(bars) * 26)
    plot_canvas = ft.Container(
        content=ft.Stack(
            controls=[
                *_grid_background(plot_w, plot_h, lines=2),
                ft.Container(
                    left=0, top=0, right=0, bottom=0,
                    content=ft.Row(bars, spacing=2, alignment=ft.MainAxisAlignment.CENTER),
                ),
            ],
            width=plot_w, height=plot_h,
        ),
        width=plot_w,
        height=plot_h,
        border_radius=12,
        bgcolor=ft.Colors.with_opacity(0.26, SURFACE_2),
        border=ft.Border.all(1, ft.Colors.with_opacity(0.5, OUTLINE_SOFT)),
        padding=ft.Padding(left=6, right=6, top=4, bottom=4),
        clip_behavior=ft.ClipBehavior.HARD_EDGE,
    )
    return _chart_shell(plot_canvas, height=chart_height, title="UV Index")
