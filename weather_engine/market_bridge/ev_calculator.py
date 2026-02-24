"""Expected Value and Kelly Criterion calculations for betting decisions."""
import math


def calculate_ev(our_probability: float, market_price: float) -> float:
    """Calculate expected value of a bet.

    EV = (our_prob * payout) - (1 - our_prob) * cost
    For a binary market where you buy at market_price:
      win: you get 1.0, net profit = 1.0 - market_price
      lose: you lose market_price
    EV = our_prob * (1 - market_price) - (1 - our_prob) * market_price
       = our_prob - market_price
    """
    return our_probability - market_price


def kelly_criterion(our_probability: float, market_price: float, max_fraction: float = 0.10) -> float:
    """Calculate Kelly criterion fraction of bankroll to bet.

    For binary outcome at price p with our probability q:
    Kelly = (q * (1/p - 1) - (1-q)) / (1/p - 1)
          = (q/p - 1) / (1/p - 1)
          = (q - p) / (1 - p)

    We cap at max_fraction to avoid over-betting.
    """
    if market_price <= 0 or market_price >= 1:
        return 0.0
    if our_probability <= market_price:
        return 0.0

    kelly = (our_probability - market_price) / (1 - market_price)
    return min(kelly, max_fraction)


def adaptive_kelly(
    our_prob: float,
    market_price: float,
    confidence_total: float = 50.0,
    signal: str = "SCOMMETTI",
    liquidity: float = 50000.0,
) -> float:
    """Adaptive Kelly sizing that scales by confidence, signal, and liquidity.

    Returns a kelly fraction (0.0 to 0.08).
    """
    if market_price <= 0 or market_price >= 1:
        return 0.0
    if our_prob <= market_price:
        return 0.0

    # 1. Base Kelly
    kelly = (our_prob - market_price) / (1 - market_price)

    # 2. Confidence scaling
    kelly *= confidence_total / 100.0

    # 3. Signal enforcement
    if signal == "NON SCOMMETTERE":
        return 0.0
    elif signal == "CAUTELA":
        kelly *= 0.20  # fifth-Kelly
    else:
        # SCOMMETTI (default)
        kelly *= 0.50  # half-Kelly

    # 4. Liquidity adjustment
    if liquidity < 2000:
        return 0.0
    elif liquidity < 5000:
        kelly *= 0.5

    # 5. Floor and ceiling
    if kelly < 0.005:
        return 0.0  # not worth it
    kelly = min(kelly, 0.08)

    return round(kelly, 6)


def should_bet(
    our_probability: float,
    market_price: float,
    min_edge: float = 0.05,
    min_confidence: float = 0.60,
    confidence: float = 1.0,
) -> bool:
    """Determine if we should place a bet based on edge and confidence thresholds."""
    edge = our_probability - market_price
    if edge < min_edge:
        return False
    if confidence < min_confidence:
        return False
    return True


def calculate_position_size(
    kelly_fraction: float,
    bankroll: float,
    max_bet_pct: float = 0.10,
    max_exposure_pct: float = 0.50,
    current_exposure: float = 0.0,
    adaptive: bool = False,
    our_prob: float = 0.0,
    market_price: float = 0.0,
    confidence_total: float = 50.0,
    signal: str = "SCOMMETTI",
    liquidity: float = 50000.0,
) -> float:
    """Calculate actual dollar amount to bet.

    If adaptive=True, uses adaptive_kelly() instead of fixed half-Kelly.
    Otherwise uses half-Kelly for safety, capped by max bet size and total exposure limits.
    """
    if adaptive:
        fraction = adaptive_kelly(our_prob, market_price, confidence_total, signal, liquidity)
    else:
        # Half-Kelly
        fraction = kelly_fraction * 0.5

    # Cap to max per bet
    capped = min(fraction, max_bet_pct)

    # Check total exposure
    remaining_capacity = max_exposure_pct - (current_exposure / bankroll) if bankroll > 0 else 0
    capped = min(capped, max(0, remaining_capacity))

    return capped * bankroll
