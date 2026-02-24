"""Betting signal: clear SCOMMETTI / CAUTELA / NON SCOMMETTERE verdict.

Combines confidence, edge, bias, horizon, liquidity and convergence
into a single actionable signal.
"""

import logging

logger = logging.getLogger(__name__)


def calculate_betting_signal(
    confidence: dict,
    edge: float,
    bias: float = 0,
    ens_std: float = 5,
    days_ahead: int = 1,
    convergence_trend: str | None = None,
    spread_signal_boost: int = 0,
    liquidity: float = 50000.0,
) -> dict:
    """Generate a clear betting signal.

    Args:
        confidence: Result from calculate_confidence()
        edge: Raw edge (our_prob - market_price)
        bias: Model bias from verification
        ens_std: Ensemble standard deviation (Fahrenheit)
        days_ahead: Forecast horizon in days
        convergence_trend: Convergence trend from check_convergence()
        spread_signal_boost: Signal boost from spread_trajectory_signal()
        liquidity: Total market liquidity in USD

    Returns:
        Dict with signal, reasons, effective_edge, kelly_multiplier
    """
    reasons = []
    go = True

    conf_total = confidence.get("total", 0)
    data_ok = confidence.get("data_quality", True)

    # GATE 1: Sufficient data
    if not data_ok:
        reasons.append("Dati insufficienti per fidarsi della previsione")
        go = False

    # GATE 2: Minimum confidence
    if conf_total < 35:
        reasons.append(f"Confidenza troppo bassa ({conf_total:.0f}/100)")
        go = False
    elif conf_total < 50:
        reasons.append(f"Confidenza moderata ({conf_total:.0f}/100) — prudenza")

    # GATE 3: Minimum edge after bias correction
    bias_penalty = abs(bias) * 0.01
    effective_edge = edge - bias_penalty
    if effective_edge < 0.02:
        reasons.append(f"Edge troppo piccolo dopo correzione bias ({effective_edge:.1%})")
        go = False
    elif effective_edge < 0.05:
        reasons.append(f"Edge modesto ({effective_edge:.1%}) — solo se sicuri")
    else:
        reasons.append(f"Edge buono ({effective_edge:.1%})")

    # GATE 4: Time horizon
    if days_ahead > 7:
        reasons.append(f"Previsione a {days_ahead} giorni — troppo lontana")
        go = False
    elif days_ahead > 4:
        reasons.append(f"Previsione a {days_ahead} giorni — incertezza alta")

    # GATE 5: Ensemble spread
    ens_std_c = ens_std * 5 / 9
    if ens_std_c > 5:
        reasons.append(f"Modelli molto discordi (±{ens_std_c:.1f}°C) — rischio alto")
        go = False
    elif ens_std_c > 3:
        reasons.append(f"Modelli moderatamente discordi (±{ens_std_c:.1f}°C)")

    # GATE 6: Liquidity
    if liquidity < 2000:
        reasons.append(f"Liquidita' troppo bassa (${liquidity:.0f})")
        go = False
    elif liquidity < 5000:
        reasons.append(f"Liquidita' modesta (${liquidity:.0f}) — sizing ridotto")

    # GATE 7: Convergence (spread signal boost)
    if spread_signal_boost >= 5:
        # Converging models — boost effective edge
        effective_edge += 0.01
        reasons.append("Modelli in convergenza — edge potenziato +1%")
    elif spread_signal_boost <= -10:
        # Diverging models — penalize confidence
        conf_total = max(0, conf_total - 10)
        reasons.append("Modelli in divergenza — confidenza penalizzata")

    # FINAL VERDICT
    # Liquidity < 2000 is a hard block — no betting at all
    if liquidity < 2000:
        signal = "NON SCOMMETTERE"
    elif go and conf_total >= 50 and effective_edge >= 0.03:
        signal = "SCOMMETTI"
    elif go or (conf_total >= 35 and effective_edge >= 0.02):
        signal = "CAUTELA"
    else:
        signal = "NON SCOMMETTERE"

    if signal == "SCOMMETTI":
        kelly_mult = 0.25
    elif signal == "CAUTELA":
        kelly_mult = 0.10
    else:
        kelly_mult = 0.0

    return {
        "signal": signal,
        "reasons": reasons,
        "effective_edge": round(effective_edge, 4),
        "kelly_multiplier": kelly_mult,
    }
