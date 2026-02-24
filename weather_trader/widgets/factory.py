"""Reusable UI widget factories."""

import flet as ft

from weather_trader.constants import (
    ACCENT, ACCENT2, BG, CARD, CARD_HOVER, OUTLINE_SOFT, SURFACE_2, TEXT, TEXT_DIM, UV_COLORS,
    TYPE_2XS, TYPE_XS, TYPE_SM, TYPE_MD, TYPE_LG, TYPE_XL, TYPE_METRIC_LG,
)


def pad(h=0, v=0):
    return ft.Padding(left=h, right=h, top=v, bottom=v)


def make_card(content, width=None, height=None, border_color=None):
    """Card premium con bordo soft e ombra più profonda."""
    return ft.Container(
        content=ft.Stack([
            ft.Container(
                top=0,
                left=16,
                right=16,
                height=1,
                bgcolor=ft.Colors.with_opacity(0.14, border_color or ACCENT2),
            ),
            ft.Container(content=content),
        ]),
        bgcolor=CARD,
        border_radius=20,
        padding=20,
        width=width,
        height=height,
        border=ft.Border.all(
            1,
            border_color or ft.Colors.with_opacity(0.08, TEXT),
        ),
        gradient=ft.LinearGradient(
            begin=ft.Alignment(-1, -1),
            end=ft.Alignment(1, 1),
            colors=[
                ft.Colors.with_opacity(0.98, CARD_HOVER),
                ft.Colors.with_opacity(0.96, CARD),
            ],
        ),
        shadow=[
            ft.BoxShadow(
                spread_radius=0,
                blur_radius=28,
                color=ft.Colors.with_opacity(0.25, "#02040a"),
                offset=ft.Offset(0, 12),
            ),
            ft.BoxShadow(
                spread_radius=0,
                blur_radius=10,
                color=ft.Colors.with_opacity(0.10, ACCENT),
                offset=ft.Offset(0, 1),
            ),
        ],
    )


def make_stat_chip(label, value, color=ACCENT, icon=None, subtitle=""):
    """Chip statistica compatta con look premium."""
    controls = [
        ft.Row([
            ft.Icon(icon, size=14, color=TEXT_DIM) if icon else ft.Container(width=0),
            ft.Text(label, size=TYPE_SM, color=TEXT_DIM, weight=ft.FontWeight.W_500, expand=True),
        ], spacing=4),
        ft.Text(str(value), size=TYPE_METRIC_LG, color=color, weight=ft.FontWeight.BOLD),
    ]
    if subtitle:
        controls.append(ft.Text(subtitle, size=TYPE_2XS, color=TEXT_DIM, italic=True))
    return ft.Container(
        content=ft.Column(controls, spacing=4, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
        bgcolor=ft.Colors.with_opacity(0.06, color),
        border_radius=16,
        border=ft.Border.all(1, ft.Colors.with_opacity(0.14, color)),
        padding=pad(h=16, v=14),
        expand=True,
    )


def make_info_box(text, color=ACCENT):
    """Box informativo con icona info per spiegazioni."""
    return ft.Container(
        content=ft.Row([
            ft.Container(
                width=4,
                height=32,
                border_radius=4,
                bgcolor=ft.Colors.with_opacity(0.85, color),
            ),
            ft.Container(
                width=24,
                height=24,
                border_radius=8,
                bgcolor=ft.Colors.with_opacity(0.12, color),
                border=ft.Border.all(1, ft.Colors.with_opacity(0.18, color)),
                content=ft.Icon(ft.Icons.INFO_OUTLINE, size=14, color=color),
                alignment=ft.Alignment(0, 0),
            ),
            ft.Text(text, size=TYPE_XS, color=TEXT_DIM, italic=True, expand=True),
        ], spacing=6),
        bgcolor=ft.Colors.with_opacity(0.06, color),
        border=ft.Border.all(1, ft.Colors.with_opacity(0.14, color)),
        border_radius=14,
        padding=pad(h=12, v=8),
    )


def make_verdict_banner(text, color, icon=ft.Icons.LIGHTBULB):
    """Banner con verdetto/consiglio in linguaggio semplice."""
    return ft.Container(
        content=ft.Row([
            ft.Icon(icon, size=18, color=color),
            ft.Text(text, size=TYPE_MD, color=TEXT, weight=ft.FontWeight.W_500, expand=True),
        ], spacing=8),
        bgcolor=ft.Colors.with_opacity(0.07, color),
        border=ft.Border.all(1, ft.Colors.with_opacity(0.20, color)),
        border_radius=14,
        padding=pad(h=14, v=12),
    )


def make_badge(text, bg_color, text_color=None):
    """Badge piccolo colorato."""
    return ft.Container(
        content=ft.Text(text, size=TYPE_XS, color=text_color or BG, weight=ft.FontWeight.BOLD),
        bgcolor=bg_color,
        border=ft.Border.all(1, ft.Colors.with_opacity(0.18, text_color or BG)),
        border_radius=999,
        padding=pad(h=8, v=3),
    )


def make_section_title(text, icon=None, icon_color=ACCENT):
    """Titolo sezione con icona."""
    controls = []
    if icon:
        controls.append(
            ft.Container(
                content=ft.Icon(icon, color=icon_color, size=18),
                width=34,
                height=34,
                border_radius=10,
                bgcolor=ft.Colors.with_opacity(0.10, icon_color),
                border=ft.Border.all(1, ft.Colors.with_opacity(0.18, icon_color)),
                alignment=ft.Alignment(0, 0),
            )
        )
    controls.append(ft.Text(text, size=TYPE_XL, weight=ft.FontWeight.BOLD, color=TEXT))
    return ft.Row(controls, spacing=8)


def make_empty_state(icon, message, sub_message=""):
    """Stato vuoto con icona e messaggi."""
    controls = [
        ft.Container(
            width=62,
            height=62,
            border_radius=18,
            bgcolor=ft.Colors.with_opacity(0.16, SURFACE_2),
            border=ft.Border.all(1, ft.Colors.with_opacity(0.28, OUTLINE_SOFT)),
            content=ft.Icon(icon, color=TEXT_DIM, size=34),
            alignment=ft.Alignment(0, 0),
        ),
        ft.Text(message, color=TEXT, size=TYPE_LG, text_align=ft.TextAlign.CENTER),
    ]
    if sub_message:
        controls.append(ft.Text(sub_message, color=TEXT_DIM, size=TYPE_SM, text_align=ft.TextAlign.CENTER))
    return ft.Container(
        content=ft.Column(controls, horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=8),
        bgcolor=ft.Colors.with_opacity(0.10, SURFACE_2),
        border=ft.Border.all(1, ft.Colors.with_opacity(0.35, OUTLINE_SOFT)),
        border_radius=16,
        alignment=ft.Alignment(0, 0),
        padding=30,
    )


def make_loading_indicator(message="Caricamento..."):
    """Spinner di caricamento."""
    return ft.Container(
        content=ft.Row([
            ft.ProgressRing(width=18, height=18, color=ACCENT, stroke_width=2),
            ft.Text(message, color=TEXT_DIM, size=TYPE_MD),
        ], spacing=10, alignment=ft.MainAxisAlignment.CENTER),
        bgcolor=ft.Colors.with_opacity(0.10, SURFACE_2),
        border=ft.Border.all(1, ft.Colors.with_opacity(0.28, OUTLINE_SOFT)),
        border_radius=14,
        padding=pad(h=12, v=10),
    )


def make_kv_row(key, value, value_color=TEXT):
    """Riga chiave-valore per pannelli info."""
    return ft.Row([
        ft.Text(key, size=TYPE_MD, color=TEXT_DIM, width=140),
        ft.Container(height=1, expand=True, bgcolor=ft.Colors.with_opacity(0.18, OUTLINE_SOFT)),
        ft.Text(str(value), size=TYPE_MD, color=value_color, weight=ft.FontWeight.BOLD),
    ], spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER)


def z_score_color(z):
    """Colore per z-score: verde normale, giallo moderato, rosso estremo."""
    from weather_trader.constants import YELLOW, ORANGE, RED
    az = abs(z)
    if az < 1.5:
        return TEXT_DIM
    if az < 2.0:
        return YELLOW
    if az < 3.0:
        return ORANGE
    return RED


def uv_color(uv):
    """Colore UV index: verde basso, giallo moderato, rosso/viola alto."""
    if uv < 3:
        return UV_COLORS[0]
    if uv < 6:
        return UV_COLORS[1]
    if uv < 8:
        return UV_COLORS[2]
    if uv < 11:
        return UV_COLORS[3]
    return UV_COLORS[4]
