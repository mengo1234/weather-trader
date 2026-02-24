"""Distribution and edge bar charts."""

import flet as ft

from weather_trader.constants import ACCENT, GREEN, OUTLINE_SOFT, RED, SURFACE_2, TEXT, TEXT_DIM
from weather_trader.widgets.factory import pad


def build_distribution_chart(stats, width=400, height=60):
    """Box plot orizzontale con p10/p25/median/p75/p90."""
    if not stats:
        return ft.Container(width=width, height=height)

    p10 = stats.get("p10", 0)
    p25 = stats.get("p25", 0)
    median = stats.get("median", 0)
    p75 = stats.get("p75", 0)
    p90 = stats.get("p90", 0)
    smin = stats.get("min", p10)
    smax = stats.get("max", p90)

    total_range = smax - smin if smax > smin else 1
    usable_w = width - 40

    def to_x(val):
        return 20 + max(0, min(usable_w, (val - smin) / total_range * usable_w))

    x_p10 = to_x(p10)
    x_p25 = to_x(p25)
    x_med = to_x(median)
    x_p75 = to_x(p75)
    x_p90 = to_x(p90)
    mid_y = height // 2

    chart = ft.Stack([
            # Whisker line (p10 to p90)
            ft.Container(
                width=x_p90 - x_p10, height=2,
                bgcolor=ft.Colors.with_opacity(0.7, TEXT_DIM),
                left=x_p10, top=mid_y,
            ),
            # Box (p25 to p75)
            ft.Container(
                width=max(4, x_p75 - x_p25), height=24,
                bgcolor=ft.Colors.with_opacity(0.2, ACCENT),
                border=ft.Border.all(1, ACCENT),
                border_radius=6,
                left=x_p25, top=mid_y - 12,
            ),
            # Median line
            ft.Container(
                width=3, height=28,
                bgcolor=ACCENT,
                border_radius=1,
                left=x_med, top=mid_y - 14,
            ),
            # Labels
            ft.Text(f"{p10:.1f}", size=8, color=TEXT_DIM, left=x_p10 - 10, top=mid_y + 16),
            ft.Text(f"{median:.1f}", size=9, color=ACCENT, weight=ft.FontWeight.BOLD,
                     left=x_med - 10, top=mid_y - 28),
            ft.Text(f"{p90:.1f}", size=8, color=TEXT_DIM, left=x_p90 - 10, top=mid_y + 16),
        ])
    return ft.Container(
        content=chart,
        width=width,
        height=height + 20,
        border_radius=14,
        bgcolor=ft.Colors.with_opacity(0.22, SURFACE_2),
        border=ft.Border.all(1, ft.Colors.with_opacity(0.55, OUTLINE_SOFT)),
        padding=pad(h=8, v=6),
    )


def build_edge_bar_chart(opportunities, width=400, height=30):
    """Barre edge orizzontali (verde positivo, rosso negativo)."""
    if not opportunities:
        return ft.Container(width=width, height=height)

    bars = []
    for opp in opportunities[:8]:
        rec = opp.get("recommendation", {})
        edge = rec.get("expected_value", 0)
        label = rec.get("best_bet", "?")
        if len(label) > 18:
            label = label[:18] + "…"
        color = GREEN if edge > 0 else RED
        bar_w = min(abs(edge) * 400, width * 0.6)

        bars.append(ft.Row([
            ft.Text(label, size=10, color=TEXT, width=120,
                     overflow=ft.TextOverflow.ELLIPSIS),
            ft.Container(
                content=ft.Stack([
                    ft.Container(
                        width=max(70, width * 0.4), height=14,
                        bgcolor=ft.Colors.with_opacity(0.08, TEXT),
                        border_radius=7,
                    ),
                    ft.Container(
                        width=max(4, bar_w), height=14,
                        bgcolor=ft.Colors.with_opacity(0.72, color),
                        border_radius=7,
                    ),
                ]),
                width=max(70, width * 0.4),
                height=14,
            ),
            ft.Text(f"{edge:+.1%}", size=10, color=color, weight=ft.FontWeight.BOLD),
        ], spacing=6))

    return ft.Container(
        content=ft.Column(bars, spacing=6),
        border_radius=12,
        bgcolor=ft.Colors.with_opacity(0.18, SURFACE_2),
        border=ft.Border.all(1, ft.Colors.with_opacity(0.45, OUTLINE_SOFT)),
        padding=pad(h=10, v=8),
    )


def build_consensus_indicator(spread, width=200, height=50):
    """Gauge indicatore spread ensemble: verde se stretto, rosso se largo."""
    from weather_trader.constants import YELLOW
    norm = min(spread / 10, 1.0) if spread else 0
    bar_color = GREEN if norm < 0.3 else (YELLOW if norm < 0.6 else RED)
    label = "Alto consenso" if norm < 0.3 else ("Moderato" if norm < 0.6 else "Basso consenso")

    return ft.Column([
        ft.Row([
            ft.Text("Consensus", size=11, color=TEXT_DIM),
            ft.Container(expand=True),
            ft.Text(label, size=11, color=bar_color, weight=ft.FontWeight.BOLD),
        ]),
        ft.Container(
            content=ft.Stack([
                ft.Container(width=width, height=9, bgcolor=ft.Colors.with_opacity(0.08, TEXT),
                              border_radius=5),
                ft.Container(width=max(4, norm * width), height=9, bgcolor=bar_color,
                              border_radius=5),
            ]),
            width=width, height=9,
        ),
        ft.Text(f"Spread: ±{spread * 5 / 9:.1f}°C", size=10, color=TEXT_DIM),
    ], spacing=5, width=width)
