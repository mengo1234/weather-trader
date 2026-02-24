"""Portfolio optimization for correlated weather bets.

Computes bet correlations, CVaR via Monte Carlo, and greedy position sizing
that respects exposure limits and risk constraints.
"""
import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PortfolioPosition:
    bet_id: int | None
    city_slug: str
    variable: str
    target_date: str
    outcome: str
    probability: float
    edge: float
    kelly_fraction: float
    allocated_fraction: float  # final position size as fraction of bankroll
    allocated_amount: float


@dataclass
class PortfolioAnalysis:
    positions: list[PortfolioPosition]
    total_exposure: float
    portfolio_cvar: float
    sharpe_estimate: float
    n_positions: int
    correlation_summary: dict


def compute_bet_correlations(positions: list[dict], db=None) -> np.ndarray:
    """Compute correlation matrix between bet positions.

    Correlation heuristics:
    - Same city = high correlation (0.7)
    - Same target_date = moderate correlation (0.4)
    - Same variable, different cities = low correlation (0.2)
    - Different everything = near zero (0.05)
    """
    n = len(positions)
    if n == 0:
        return np.array([[]])

    corr = np.eye(n)

    for i in range(n):
        for j in range(i + 1, n):
            pi = positions[i]
            pj = positions[j]

            same_city = pi.get("city_slug") == pj.get("city_slug")
            same_date = pi.get("target_date") == pj.get("target_date")
            same_var = pi.get("variable") == pj.get("variable")

            if same_city and same_date and same_var:
                rho = 0.85  # nearly identical bets
            elif same_city and same_date:
                rho = 0.70  # same city/date, different variable
            elif same_city:
                rho = 0.40  # same city, different date
            elif same_date and same_var:
                rho = 0.30  # same date/variable, different city
            elif same_date:
                rho = 0.20  # same date only
            elif same_var:
                rho = 0.15  # same variable only
            else:
                rho = 0.05  # different everything

            corr[i, j] = rho
            corr[j, i] = rho

    return corr


def portfolio_cvar(
    positions: list[dict],
    corr_matrix: np.ndarray,
    confidence: float = 0.95,
    n_sim: int = 10000,
) -> float:
    """Compute Conditional Value at Risk (CVaR) via Monte Carlo.

    Simulates correlated Bernoulli outcomes (win/lose per position)
    and computes the expected loss in the worst (1-confidence)% of scenarios.
    """
    n = len(positions)
    if n == 0:
        return 0.0

    rng = np.random.default_rng(42)

    # Extract probabilities and stakes
    probs = np.array([p.get("probability", 0.5) for p in positions])
    stakes = np.array([p.get("stake", 0.0) for p in positions])
    odds = np.array([p.get("odds", 2.0) for p in positions])

    # Generate correlated uniform draws via Gaussian copula
    try:
        L = np.linalg.cholesky(corr_matrix)
    except np.linalg.LinAlgError:
        # Not positive definite — add small diagonal
        corr_fixed = corr_matrix + np.eye(n) * 0.01
        try:
            L = np.linalg.cholesky(corr_fixed)
        except np.linalg.LinAlgError:
            L = np.eye(n)

    z = rng.standard_normal((n_sim, n))
    corr_z = z @ L.T

    # Convert to uniform via Gaussian CDF, then to Bernoulli
    from scipy.stats import norm
    u = norm.cdf(corr_z)
    wins = u < probs  # True = win

    # Compute PnL per simulation
    pnl_per_sim = np.zeros(n_sim)
    for s in range(n_sim):
        for i in range(n):
            if wins[s, i]:
                pnl_per_sim[s] += stakes[i] * (odds[i] - 1)
            else:
                pnl_per_sim[s] -= stakes[i]

    # CVaR = expected loss in worst (1-confidence)% scenarios
    sorted_pnl = np.sort(pnl_per_sim)
    cutoff_idx = int(n_sim * (1 - confidence))
    if cutoff_idx < 1:
        cutoff_idx = 1
    worst_pnls = sorted_pnl[:cutoff_idx]
    cvar = -float(np.mean(worst_pnls))  # positive number = expected loss

    return round(max(0, cvar), 2)


def optimize_position_sizes(
    candidates: list[dict],
    corr_matrix: np.ndarray,
    max_exposure: float = 0.50,
    max_single: float = 0.10,
    max_cvar: float = 0.20,
    bankroll: float = 1000,
) -> list[PortfolioPosition]:
    """Greedy iterative position sizing with CVaR constraint.

    1. Rank candidates by Kelly-adjusted edge
    2. Add one at a time
    3. Check CVaR after each addition
    4. Stop when constraints are violated
    """
    if not candidates:
        return []

    # Rank by edge * probability (Kelly-like criterion)
    ranked = sorted(candidates, key=lambda c: c.get("edge", 0) * c.get("probability", 0.5), reverse=True)

    positions = []
    total_exposure = 0.0

    for candidate in ranked:
        edge = candidate.get("edge", 0)
        prob = candidate.get("probability", 0.5)

        if edge <= 0:
            continue

        # Kelly fraction (fractional)
        market_price = candidate.get("market_price", 0.5)
        if market_price <= 0 or market_price >= 1:
            continue
        odds = 1.0 / market_price
        kelly = (prob * odds - 1) / (odds - 1) if odds > 1 else 0
        kelly = max(0, min(kelly * 0.25, max_single))  # quarter Kelly, capped

        allocated_frac = kelly
        allocated_amt = round(bankroll * allocated_frac, 2)

        if allocated_amt < 1:
            continue

        # Check single position limit
        if allocated_frac > max_single:
            allocated_frac = max_single
            allocated_amt = round(bankroll * allocated_frac, 2)

        # Check total exposure
        if total_exposure + allocated_frac > max_exposure:
            remaining = max_exposure - total_exposure
            if remaining < 0.01:
                break
            allocated_frac = remaining
            allocated_amt = round(bankroll * allocated_frac, 2)

        # Build trial position list for CVaR check
        trial_pos = [
            {
                "probability": p.probability,
                "stake": p.allocated_amount,
                "odds": 1.0 / max(0.01, p.probability),
            }
            for p in positions
        ]
        trial_pos.append({
            "probability": prob,
            "stake": allocated_amt,
            "odds": odds,
        })

        # Compute CVaR for trial
        n_trial = len(trial_pos)
        if n_trial <= corr_matrix.shape[0]:
            trial_corr = corr_matrix[:n_trial, :n_trial]
        else:
            trial_corr = np.eye(n_trial)

        cvar = portfolio_cvar(trial_pos, trial_corr)
        if cvar / bankroll > max_cvar:
            # CVaR too high, try reducing size
            allocated_frac *= 0.5
            allocated_amt = round(bankroll * allocated_frac, 2)
            if allocated_amt < 1:
                continue

        positions.append(PortfolioPosition(
            bet_id=candidate.get("bet_id"),
            city_slug=candidate.get("city_slug", ""),
            variable=candidate.get("variable", ""),
            target_date=str(candidate.get("target_date", "")),
            outcome=candidate.get("outcome", ""),
            probability=prob,
            edge=round(edge, 4),
            kelly_fraction=round(kelly, 4),
            allocated_fraction=round(allocated_frac, 4),
            allocated_amount=allocated_amt,
        ))
        total_exposure += allocated_frac

    return positions


def portfolio_sharpe_estimate(positions: list[dict], corr_matrix: np.ndarray) -> float:
    """Estimate portfolio Sharpe ratio from position edges and correlations."""
    n = len(positions)
    if n == 0:
        return 0.0

    edges = np.array([p.get("edge", 0) for p in positions])
    weights = np.array([p.get("allocated_fraction", 0) for p in positions])

    if weights.sum() == 0:
        return 0.0

    # Portfolio expected return
    port_return = np.dot(weights, edges)

    # Portfolio variance (using correlations as proxy)
    # Var = w^T * Sigma * w where Sigma_ij = rho_ij * std_i * std_j
    # Use edge as proxy for std
    stds = np.maximum(np.abs(edges), 0.01)
    cov = np.outer(stds, stds) * corr_matrix[:n, :n]
    port_var = weights @ cov @ weights

    if port_var <= 0:
        return 0.0

    return round(float(port_return / np.sqrt(port_var)), 3)


def analyze_portfolio(db, candidates: list[dict] | None = None) -> PortfolioAnalysis:
    """Full portfolio analysis: correlations, CVaR, optimal sizing."""
    if candidates is None:
        # Fetch active bets from DB
        rows = db.execute(
            """SELECT id, city_slug, target_date, outcome, our_prob, edge, stake
            FROM bets WHERE status = 'pending'
            ORDER BY timestamp DESC LIMIT 50""",
        ).fetchall()
        candidates = [
            {
                "bet_id": r[0],
                "city_slug": r[1],
                "target_date": str(r[2]) if r[2] else "",
                "variable": "temperature_2m_max",
                "outcome": r[3],
                "probability": r[4] or 0.5,
                "edge": r[5] or 0,
                "market_price": max(0.01, 1 - (r[5] or 0) - (r[4] or 0.5)) if r[4] else 0.5,
                "stake": r[6] or 0,
            }
            for r in rows
        ]

    if not candidates:
        return PortfolioAnalysis(
            positions=[], total_exposure=0, portfolio_cvar=0,
            sharpe_estimate=0, n_positions=0, correlation_summary={},
        )

    corr = compute_bet_correlations(candidates)
    positions = optimize_position_sizes(candidates, corr)

    # Compute portfolio-level metrics
    pos_dicts = [
        {"probability": p.probability, "stake": p.allocated_amount,
         "odds": 1.0 / max(0.01, p.probability), "edge": p.edge,
         "allocated_fraction": p.allocated_fraction}
        for p in positions
    ]
    n = len(pos_dicts)
    final_corr = corr[:n, :n] if n <= corr.shape[0] else np.eye(n)

    total_exp = sum(p.allocated_fraction for p in positions)
    cvar = portfolio_cvar(pos_dicts, final_corr) if positions else 0
    sharpe = portfolio_sharpe_estimate(pos_dicts, final_corr) if positions else 0

    # Correlation summary
    if n > 1:
        upper = corr[np.triu_indices(n, k=1)]
        corr_summary = {
            "mean": round(float(np.mean(upper)), 3),
            "max": round(float(np.max(upper)), 3),
            "min": round(float(np.min(upper)), 3),
            "n_high": int(np.sum(upper > 0.5)),
        }
    else:
        corr_summary = {"mean": 0, "max": 0, "min": 0, "n_high": 0}

    return PortfolioAnalysis(
        positions=positions,
        total_exposure=round(total_exp, 4),
        portfolio_cvar=cvar,
        sharpe_estimate=sharpe,
        n_positions=len(positions),
        correlation_summary=corr_summary,
    )
