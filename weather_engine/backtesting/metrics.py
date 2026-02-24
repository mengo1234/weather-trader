"""Extended backtest metrics: Sharpe, Sortino, drawdown analysis."""

import numpy as np


def sharpe_ratio(returns: list[float], risk_free_rate: float = 0.0) -> float:
    if not returns or len(returns) < 2:
        return 0.0
    arr = np.array(returns)
    excess = arr - risk_free_rate / 252
    return float(np.mean(excess) / np.std(excess) * np.sqrt(252)) if np.std(excess) > 0 else 0.0


def sortino_ratio(returns: list[float], risk_free_rate: float = 0.0) -> float:
    if not returns or len(returns) < 2:
        return 0.0
    arr = np.array(returns) - risk_free_rate / 252
    downside = arr[arr < 0]
    if len(downside) == 0:
        return float('inf')
    return float(np.mean(arr) / np.std(downside) * np.sqrt(252)) if np.std(downside) > 0 else 0.0


def max_drawdown(equity_curve: list[float]) -> float:
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for val in equity_curve:
        peak = max(peak, val)
        dd = (peak - val) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)
    return max_dd


def profit_factor(trades_pnl: list[float]) -> float:
    wins = sum(p for p in trades_pnl if p > 0)
    losses = abs(sum(p for p in trades_pnl if p < 0))
    return wins / losses if losses > 0 else float('inf')


def calmar_ratio(total_return: float, max_dd: float) -> float:
    return total_return / max_dd if max_dd > 0 else 0.0
