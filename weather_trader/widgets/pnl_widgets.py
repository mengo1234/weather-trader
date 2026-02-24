"""P&L sparkline and risk gauge widgets."""

import flet as ft

from weather_trader.constants import ACCENT, BG, GREEN, OUTLINE_SOFT, RED, SURFACE_2, TEXT, TEXT_DIM, YELLOW
from weather_trader.widgets.charts import build_sparkline
from weather_trader.widgets.factory import pad


def build_pnl_sparkline(bets, width=200, height=40):
    """Sparkline del P&L cumulativo."""
    if not bets:
        return ft.Container(width=width, height=height)

    resolved = [b for b in bets if b["status"] != "pending"]
    if not resolved:
        return ft.Container(width=width, height=height,
                             content=ft.Text("Nessuna scommessa risolta", size=10, color=TEXT_DIM))

    cumulative = []
    total = 0
    for b in resolved:
        total += b["pnl"]
        cumulative.append(total)

    return build_sparkline(cumulative, width=width, height=height,
                            color=GREEN if total >= 0 else RED)


def build_risk_gauge(kelly_frac, suggested_pct, bankroll, width=200):
    """Gauge di rischio per una scommessa."""
    risk_level = suggested_pct * 100
    if risk_level < 2:
        risk_color = GREEN
        risk_label = "Basso"
    elif risk_level < 5:
        risk_color = YELLOW
        risk_label = "Moderato"
    else:
        risk_color = RED
        risk_label = "Alto"

    stake = bankroll * suggested_pct
    content = ft.Column([
        ft.Row([
            ft.Text("Rischio", size=11, color=TEXT_DIM),
            ft.Container(
                content=ft.Text(risk_label, size=9, color=BG, weight=ft.FontWeight.BOLD),
                bgcolor=risk_color, border_radius=999, padding=pad(h=6, v=2),
            ),
        ], spacing=6),
        ft.Container(
            content=ft.Stack([
                ft.Container(width=width, height=8,
                              bgcolor=ft.Colors.with_opacity(0.08, TEXT), border_radius=4),
                ft.Container(width=max(2, min(width, risk_level / 10 * width)), height=8,
                              bgcolor=risk_color, border_radius=4),
            ]),
            width=width, height=8,
        ),
        ft.Row([
            ft.Text(f"Kelly: {kelly_frac:.1%}", size=9, color=TEXT_DIM),
            ft.Text(f"Size: {suggested_pct:.1%}", size=9, color=TEXT_DIM),
            ft.Text(f"Stake: ${stake:.0f}", size=9, color=ACCENT, weight=ft.FontWeight.BOLD),
        ], spacing=8),
    ], spacing=4, width=width)
    return ft.Container(
        content=content,
        width=width + 20,
        padding=pad(h=10, v=8),
        border_radius=12,
        bgcolor=ft.Colors.with_opacity(0.22, SURFACE_2),
        border=ft.Border.all(1, ft.Colors.with_opacity(0.50, OUTLINE_SOFT)),
    )
