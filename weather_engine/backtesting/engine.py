"""Backtesting engine: simulate trading strategy on historical data."""

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np

from weather_engine.db import get_db
from weather_engine.market_bridge.ev_calculator import calculate_ev, kelly_criterion

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    date: date
    city_slug: str
    outcome: str
    our_prob: float
    market_price: float
    edge: float
    stake: float
    odds: float
    won: bool
    pnl: float


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    total_pnl: float = 0
    win_rate: float = 0
    sharpe_ratio: float = 0
    max_drawdown: float = 0
    profit_factor: float = 0
    n_trades: int = 0
    avg_edge: float = 0
    equity_curve: list[float] = field(default_factory=list)
    drawdown_curve: list[float] = field(default_factory=list)


class BacktestEngine:
    def __init__(self, db=None):
        self.db = db or get_db()

    def run(
        self,
        city_slug: str,
        start_date: date,
        end_date: date,
        variable: str = "temperature_2m_max",
        min_edge: float = 0.05,
        min_confidence: float = 0.6,
        kelly_multiplier: float = 0.25,
        initial_bankroll: float = 1000.0,
        use_real_prices: bool = False,
    ) -> BacktestResult:
        """Run a backtest simulation over historical data.

        For each day in [start_date, end_date]:
        1. Get ensemble forecast values for that day
        2. Generate probability estimates from ensemble
        3. Get market prices (real from snapshots or simulated with noise)
        4. Apply strategy rules (min_edge, min_confidence)
        5. Determine win/loss from actual observations
        """
        result = BacktestResult()
        bankroll = initial_bankroll
        peak_bankroll = initial_bankroll
        result.equity_curve = [initial_bankroll]
        result.drawdown_curve = [0]

        current = start_date
        while current <= end_date:
            trade = self._simulate_day(
                city_slug, current, variable,
                min_edge, min_confidence, kelly_multiplier, bankroll,
                use_real_prices=use_real_prices,
            )
            if trade is not None:
                result.trades.append(trade)
                bankroll += trade.pnl
                peak_bankroll = max(peak_bankroll, bankroll)

            result.equity_curve.append(bankroll)
            dd = (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0
            result.drawdown_curve.append(dd)

            current += timedelta(days=1)

        # Compute metrics
        result.n_trades = len(result.trades)
        if result.n_trades > 0:
            result.total_pnl = bankroll - initial_bankroll
            wins = [t for t in result.trades if t.won]
            losses = [t for t in result.trades if not t.won]
            result.win_rate = len(wins) / result.n_trades

            pnls = [t.pnl for t in result.trades]
            result.avg_edge = np.mean([t.edge for t in result.trades])
            result.max_drawdown = max(result.drawdown_curve) if result.drawdown_curve else 0

            if len(pnls) > 1:
                daily_returns = np.array(pnls)
                mean_ret = np.mean(daily_returns)
                std_ret = np.std(daily_returns)
                result.sharpe_ratio = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0

            total_wins = sum(t.pnl for t in wins) if wins else 0
            total_losses = abs(sum(t.pnl for t in losses)) if losses else 0
            result.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        return result

    def _simulate_day(
        self, city_slug, target_date, variable,
        min_edge, min_confidence, kelly_multiplier, bankroll,
        use_real_prices: bool = False,
    ) -> Trade | None:
        """Simulate a single trading day."""
        # Get ensemble data for the target date
        ens_rows = self.db.execute(
            """SELECT MAX(temperature_2m) as val
            FROM ensemble_members
            WHERE city_slug = ? AND time::DATE = ? AND temperature_2m IS NOT NULL
            GROUP BY model, member_id""",
            [city_slug, target_date],
        ).fetchall()

        if not ens_rows or len(ens_rows) < 4:
            return None

        ensemble_values = np.array([r[0] for r in ens_rows])
        ens_mean = np.mean(ensemble_values)
        ens_std = np.std(ensemble_values)

        # Get actual observation
        obs = self.db.execute(
            f"SELECT {variable} FROM observations WHERE city_slug = ? AND date = ?",
            [city_slug, target_date],
        ).fetchone()

        if obs is None or obs[0] is None:
            return None

        actual = obs[0]

        # Create thresholds around ensemble mean
        low = round(ens_mean - ens_std)
        high = round(ens_mean + ens_std)
        thresholds = [
            (-200, float(low), f"Below {low}\u00b0F"),
            (float(low), float(high), f"{low}-{high}\u00b0F"),
            (float(high), 200, f"Above {high}\u00b0F"),
        ]

        # Estimate probabilities from ensemble
        probs = []
        for lo, hi, label in thresholds:
            count = np.sum((ensemble_values >= lo) & (ensemble_values < hi))
            probs.append(max(0.01, count / len(ensemble_values)))

        # Normalize
        total = sum(probs)
        probs = [p / total for p in probs]

        # Get market prices
        if use_real_prices:
            market_prices = self._get_real_prices(city_slug, target_date, len(probs))
            if market_prices is None:
                return None  # Skip days without real price data
        else:
            # Simulate market prices (slightly off from true prob)
            np.random.seed(hash(f"{city_slug}{target_date}") % 2**31)
            noise = np.random.uniform(-0.08, 0.08, size=len(probs))
            market_prices = [max(0.02, min(0.95, p + n)) for p, n in zip(probs, noise)]

        # Find best edge
        best_idx = None
        best_edge = 0
        for i, (p, mp) in enumerate(zip(probs, market_prices)):
            edge = p - mp
            if edge > best_edge:
                best_edge = edge
                best_idx = i

        if best_idx is None or best_edge < min_edge:
            return None

        # Confidence (simplified)
        confidence = 1.0 - min(1.0, ens_std / 10)
        if confidence < min_confidence:
            return None

        our_prob = probs[best_idx]
        mkt_price = market_prices[best_idx]
        odds = 1.0 / mkt_price
        kelly = kelly_criterion(our_prob, mkt_price)
        stake = bankroll * kelly * kelly_multiplier
        stake = max(0, min(stake, bankroll * 0.1))  # Cap at 10%

        # Determine win/loss
        lo, hi, label = thresholds[best_idx]
        won = lo <= actual < hi

        pnl = stake * (odds - 1) if won else -stake

        return Trade(
            date=target_date,
            city_slug=city_slug,
            outcome=label,
            our_prob=our_prob,
            market_price=mkt_price,
            edge=best_edge,
            stake=round(stake, 2),
            odds=round(odds, 2),
            won=won,
            pnl=round(pnl, 2),
        )

    def _get_real_prices(self, city_slug: str, target_date: date, n_outcomes: int) -> list[float] | None:
        """Get real market prices from snapshots for a given date.

        Finds the snapshot closest to the target date and returns prices.
        Returns None if no snapshots are available.
        """
        rows = self.db.execute(
            """SELECT outcome, price FROM market_price_snapshots
            WHERE collected_at::DATE = ?
            ORDER BY collected_at DESC
            LIMIT ?""",
            [target_date, n_outcomes * 5],
        ).fetchall()

        if not rows:
            return None

        # Get the most recent prices per outcome
        prices = {}
        for outcome, price in rows:
            if outcome not in prices:
                prices[outcome] = price

        if len(prices) < 2:
            return None

        # Return as list matching n_outcomes
        price_list = list(prices.values())[:n_outcomes]
        while len(price_list) < n_outcomes:
            price_list.append(0.5 / n_outcomes)

        return [max(0.02, min(0.95, p)) for p in price_list]

    def walk_forward(
        self,
        city_slug: str,
        start_date: date,
        end_date: date,
        variable: str = "temperature_2m_max",
        train_days: int = 30,
        test_days: int = 7,
        initial_bankroll: float = 1000.0,
        use_real_prices: bool = False,
    ) -> dict:
        """Walk-forward validation: optimize on training window, test on next window.

        Divides [start_date, end_date] into rolling windows:
        - Train: optimize min_edge and min_confidence
        - Test: evaluate out-of-sample performance
        """
        results = []
        all_oos_trades = []
        window_start = start_date

        while window_start + timedelta(days=train_days + test_days) <= end_date:
            train_end = window_start + timedelta(days=train_days - 1)
            test_start = train_end + timedelta(days=1)
            test_end = test_start + timedelta(days=test_days - 1)

            # Optimize on training window
            best_params = self._optimize_params(
                city_slug, window_start, train_end, variable,
                initial_bankroll, use_real_prices,
            )

            # Test on next window
            oos_result = self.run(
                city_slug, test_start, test_end, variable,
                min_edge=best_params["min_edge"],
                min_confidence=best_params["min_confidence"],
                kelly_multiplier=best_params["kelly_multiplier"],
                initial_bankroll=initial_bankroll,
                use_real_prices=use_real_prices,
            )

            results.append({
                "train_start": str(window_start),
                "train_end": str(train_end),
                "test_start": str(test_start),
                "test_end": str(test_end),
                "best_params": best_params,
                "oos_n_trades": oos_result.n_trades,
                "oos_pnl": round(oos_result.total_pnl, 2),
                "oos_win_rate": round(oos_result.win_rate, 3),
                "oos_sharpe": round(oos_result.sharpe_ratio, 3),
            })
            all_oos_trades.extend(oos_result.trades)

            window_start += timedelta(days=test_days)

        # Aggregate out-of-sample metrics
        n_total = len(all_oos_trades)
        total_pnl = sum(t.pnl for t in all_oos_trades)
        wins = [t for t in all_oos_trades if t.won]

        oos_pnls = [t.pnl for t in all_oos_trades]
        if len(oos_pnls) > 1:
            mean_ret = np.mean(oos_pnls)
            std_ret = np.std(oos_pnls)
            oos_sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0
        else:
            oos_sharpe = 0

        return {
            "windows": results,
            "n_windows": len(results),
            "oos_total_trades": n_total,
            "oos_total_pnl": round(total_pnl, 2),
            "oos_win_rate": round(len(wins) / n_total, 3) if n_total > 0 else 0,
            "oos_sharpe": round(oos_sharpe, 3),
        }

    def _optimize_params(
        self, city_slug, start_date, end_date, variable,
        initial_bankroll, use_real_prices,
    ) -> dict:
        """Grid search over min_edge and min_confidence to maximize Sharpe on training data."""
        best_sharpe = -999
        best_params = {"min_edge": 0.05, "min_confidence": 0.6, "kelly_multiplier": 0.25}

        for min_edge in [0.03, 0.05, 0.08, 0.10]:
            for min_conf in [0.4, 0.5, 0.6, 0.7]:
                for kelly_mult in [0.15, 0.25, 0.35]:
                    result = self.run(
                        city_slug, start_date, end_date, variable,
                        min_edge=min_edge, min_confidence=min_conf,
                        kelly_multiplier=kelly_mult,
                        initial_bankroll=initial_bankroll,
                        use_real_prices=use_real_prices,
                    )
                    if result.n_trades >= 3 and result.sharpe_ratio > best_sharpe:
                        best_sharpe = result.sharpe_ratio
                        best_params = {
                            "min_edge": min_edge,
                            "min_confidence": min_conf,
                            "kelly_multiplier": kelly_mult,
                        }

        return best_params
