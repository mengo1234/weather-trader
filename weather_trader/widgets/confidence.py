"""Confidence meter, risk gauge, and betting signal widgets."""

import flet as ft

from weather_trader.constants import (
    ACCENT, BG, GREEN, ORANGE, OUTLINE_SOFT, RED, SURFACE_2, TEXT, TEXT_DIM, YELLOW,
)
from weather_trader.widgets.factory import pad


def calculate_confidence(ensemble_data: dict, edge: float,
                         accuracy_data: dict | None = None,
                         n_outcomes: int = 2,
                         days_ahead: int = 1) -> dict:
    """Calcola score di confidenza multi-fattore per una scommessa.

    CORRETTO: Edge NON entra nel calcolo confidenza (evita logica circolare).
    La confidenza misura QUANTO POSSIAMO FIDARCI della previsione.

    Fattori (5):
    1. Ensemble agreement (35%)
    2. Forecast accuracy (25%)
    3. Horizon decay (15%)
    4. Market complexity (10%)
    5. Sample size ensemble (15%)
    """
    scores = {}
    data_quality = True

    # 1. Ensemble spread (0-100, stretto = meglio)
    ens_std = ensemble_data.get("ensemble_std") if ensemble_data else None
    n_members = ensemble_data.get("n_members", 0) if ensemble_data else 0

    if ens_std is None or n_members == 0:
        spread_score = 10
        data_quality = False
    else:
        spread_score = max(0, min(100, 110 - ens_std * 12))
    scores["ensemble"] = spread_score

    # 2. Accuracy del modello (0-100)
    if accuracy_data and isinstance(accuracy_data, dict):
        mae = accuracy_data.get("mae")
        if mae is not None:
            acc_score = max(0, min(100, 105 - mae * 18))
        else:
            acc_score = 25
            data_quality = False
    else:
        acc_score = 20
        data_quality = False
    scores["accuracy"] = acc_score

    # 3. Horizon decay
    if days_ahead <= 1:
        horizon_score = 100
    elif days_ahead <= 3:
        horizon_score = 85
    elif days_ahead <= 5:
        horizon_score = 65
    elif days_ahead <= 7:
        horizon_score = 45
    elif days_ahead <= 10:
        horizon_score = 25
    else:
        horizon_score = 10
    scores["horizon"] = horizon_score

    # 4. Complessità mercato
    complexity_score = max(15, min(100, 115 - n_outcomes * 15))
    scores["complexity"] = complexity_score

    # 5. Sample size ensemble
    if n_members >= 40:
        sample_score = 100
    elif n_members >= 20:
        sample_score = 75
    elif n_members >= 10:
        sample_score = 50
    elif n_members > 0:
        sample_score = 30
    else:
        sample_score = 5
        data_quality = False
    scores["sample_size"] = sample_score

    weights = {
        "ensemble": 0.35,
        "accuracy": 0.25,
        "horizon": 0.15,
        "complexity": 0.10,
        "sample_size": 0.15,
    }

    total = sum(scores[k] * weights[k] for k in scores)

    if not data_quality:
        total = min(total, 35)

    if total >= 70:
        rating = "FORTE"
        rating_color = GREEN
    elif total >= 50:
        rating = "BUONO"
        rating_color = YELLOW
    elif total >= 30:
        rating = "DEBOLE"
        rating_color = ORANGE
    else:
        rating = "EVITA"
        rating_color = RED

    return {
        "total": total,
        "scores": scores,
        "rating": rating,
        "rating_color": rating_color,
        "data_quality": data_quality,
    }


def calculate_betting_signal(conf: dict, edge: float,
                             bias: float = 0, ens_std: float = 5,
                             days_ahead: int = 1) -> dict:
    """Segnale di scommessa chiaro: SCOMMETTI / CAUTELA / NON SCOMMETTERE."""
    reasons = []
    go = True

    conf_total = conf.get("total", 0)
    data_ok = conf.get("data_quality", True)

    # GATE 1: Dati sufficienti
    if not data_ok:
        reasons.append("Dati insufficienti per fidarsi della previsione")
        go = False

    # GATE 2: Confidenza minima
    if conf_total < 35:
        reasons.append(f"Confidenza troppo bassa ({conf_total:.0f}/100)")
        go = False
    elif conf_total < 50:
        reasons.append(f"Confidenza moderata ({conf_total:.0f}/100) — prudenza")

    # GATE 3: Edge minimo dopo correzione bias
    bias_penalty = abs(bias) * 0.01
    effective_edge = edge - bias_penalty
    if effective_edge < 0.02:
        reasons.append(f"Edge troppo piccolo dopo correzione bias ({effective_edge:.1%})")
        go = False
    elif effective_edge < 0.05:
        reasons.append(f"Edge modesto ({effective_edge:.1%}) — solo se sicuri")
    else:
        reasons.append(f"Edge buono ({effective_edge:.1%})")

    # GATE 4: Orizzonte temporale
    if days_ahead > 7:
        reasons.append(f"Previsione a {days_ahead} giorni — troppo lontana")
        go = False
    elif days_ahead > 4:
        reasons.append(f"Previsione a {days_ahead} giorni — incertezza alta")

    # GATE 5: Spread ensemble
    ens_std_c = ens_std * 5 / 9
    if ens_std_c > 5:
        reasons.append(f"Modelli molto discordi (±{ens_std_c:.1f}°C) — rischio alto")
        go = False
    elif ens_std_c > 3:
        reasons.append(f"Modelli moderatamente discordi (±{ens_std_c:.1f}°C)")

    # VERDETTO FINALE
    if go and conf_total >= 50 and effective_edge >= 0.03:
        signal = "SCOMMETTI"
        signal_color = GREEN
        signal_icon = ft.Icons.CHECK_CIRCLE
    elif go or (conf_total >= 35 and effective_edge >= 0.02):
        signal = "CAUTELA"
        signal_color = YELLOW
        signal_icon = ft.Icons.WARNING_AMBER
    else:
        signal = "NON SCOMMETTERE"
        signal_color = RED
        signal_icon = ft.Icons.BLOCK

    if signal == "SCOMMETTI":
        kelly_mult = 0.25
    elif signal == "CAUTELA":
        kelly_mult = 0.10
    else:
        kelly_mult = 0.0

    return {
        "signal": signal,
        "signal_color": signal_color,
        "signal_icon": signal_icon,
        "reasons": reasons,
        "effective_edge": effective_edge,
        "kelly_multiplier": kelly_mult,
    }


def build_confidence_meter(conf: dict, width=220):
    """Widget meter di confidenza con breakdown fattori."""
    total = conf["total"]
    rating = conf["rating"]
    rating_color = conf["rating_color"]
    scores = conf["scores"]

    factor_bars = []
    labels = {
        "ensemble": "Concordanza",
        "accuracy": "Precisione",
        "horizon": "Orizzonte",
        "complexity": "Semplicità",
        "sample_size": "N. Modelli",
    }
    for key, label in labels.items():
        val = scores.get(key, 0)
        bar_color = GREEN if val >= 60 else (YELLOW if val >= 35 else RED)
        factor_bars.append(ft.Row([
            ft.Text(label, size=9, color=TEXT_DIM, width=72),
            ft.Container(
                content=ft.Stack([
                    ft.Container(width=width - 110, height=7,
                                  bgcolor=ft.Colors.with_opacity(0.08, TEXT), border_radius=4),
                    ft.Container(width=max(2, val / 100 * (width - 110)), height=7,
                                  bgcolor=bar_color, border_radius=4),
                ]),
                width=width - 110, height=7,
            ),
            ft.Text(f"{val:.0f}", size=9, color=bar_color, width=25),
        ], spacing=4))

    return ft.Container(
        content=ft.Column([
            ft.Row([
                ft.Text(f"{total:.0f}/100", size=16, color=rating_color,
                         weight=ft.FontWeight.BOLD),
                ft.Container(
                    content=ft.Text(rating, size=10, color=BG, weight=ft.FontWeight.BOLD),
                    bgcolor=rating_color, border_radius=999, padding=pad(h=8, v=3),
                ),
            ], spacing=8),
            ft.Container(
                content=ft.Stack([
                    ft.Container(width=width - 20, height=9,
                                  bgcolor=ft.Colors.with_opacity(0.08, TEXT), border_radius=5),
                    ft.Container(width=max(4, total / 100 * (width - 20)), height=9,
                                  bgcolor=rating_color, border_radius=5),
                ]),
                width=width - 20, height=9,
            ),
            ft.Container(height=2),
            *factor_bars,
        ], spacing=4),
        width=width,
        padding=pad(h=10, v=8),
        bgcolor=ft.Colors.with_opacity(0.22, SURFACE_2),
        border=ft.Border.all(1, ft.Colors.with_opacity(0.50, OUTLINE_SOFT)),
        border_radius=12,
    )


def build_betting_signal_widget(signal: dict, width=250):
    """Widget segnale scommessa con motivi."""
    sig = signal["signal"]
    sig_color = signal["signal_color"]
    sig_icon = signal["signal_icon"]
    reasons = signal.get("reasons", [])
    eff_edge = signal.get("effective_edge", 0)
    kelly_m = signal.get("kelly_multiplier", 0)

    reason_widgets = []
    for r in reasons[:4]:
        reason_widgets.append(ft.Row([
            ft.Text("•", size=10, color=TEXT_DIM),
            ft.Text(r, size=9, color=TEXT_DIM, expand=True),
        ], spacing=4))

    return ft.Container(
        content=ft.Column([
            ft.Row([
                ft.Icon(sig_icon, color=sig_color, size=20),
                ft.Text(sig, size=14, color=sig_color, weight=ft.FontWeight.BOLD),
                ft.Container(expand=True),
                ft.Text(f"Edge eff: {eff_edge:+.1%}", size=10, color=TEXT_DIM),
            ], spacing=6),
            ft.Container(height=2),
            *reason_widgets,
            ft.Container(height=2),
            ft.Text(
                f"Kelly: {kelly_m:.0%}" if kelly_m > 0 else "Nessuna puntata consigliata",
                size=9, color=sig_color, italic=True,
            ),
        ], spacing=2),
        bgcolor=ft.Colors.with_opacity(0.10, sig_color),
        border=ft.Border.all(1, ft.Colors.with_opacity(0.22, sig_color)),
        border_radius=12,
        padding=pad(h=12, v=10),
        width=width,
    )
