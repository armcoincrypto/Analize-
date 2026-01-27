"""
Walk-Forward Optimization Engine
================================
Implements walk-forward testing for strategy parameter optimization.

Walk-forward testing:
1. Split data into train/test windows
2. Optimize parameters on train window
3. Test on out-of-sample test window
4. Roll forward and repeat

This prevents overfitting and provides realistic performance estimates.
"""

import logging
import itertools
import random
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Iterator
from dataclasses import dataclass, field
import numpy as np

from .metrics import (
    TradeResult,
    StrategyMetrics,
    calculate_metrics,
    calculate_pnl_after_costs,
    metrics_to_dict,
)

logger = logging.getLogger(__name__)


@dataclass
class ParameterSet:
    """A set of strategy parameters to test."""
    take_profit_pct: float = 0.15
    stop_loss_pct: float = 0.10
    time_stop_seconds: int = 90
    # Additional parameters can be added
    min_imbalance: float = 0.60

    def to_dict(self) -> Dict:
        return {
            "take_profit_pct": self.take_profit_pct,
            "stop_loss_pct": self.stop_loss_pct,
            "time_stop_seconds": self.time_stop_seconds,
            "min_imbalance": self.min_imbalance,
        }

    def __hash__(self):
        return hash((self.take_profit_pct, self.stop_loss_pct, self.time_stop_seconds, self.min_imbalance))


@dataclass
class ParameterGrid:
    """Define parameter ranges for grid search."""
    take_profit_pct: List[float] = field(default_factory=lambda: [0.15, 0.20, 0.25])
    stop_loss_pct: List[float] = field(default_factory=lambda: [0.10, 0.15])
    time_stop_seconds: List[int] = field(default_factory=lambda: [60, 90, 120])
    min_imbalance: List[float] = field(default_factory=lambda: [0.60])

    def iterate(self) -> Iterator[ParameterSet]:
        """Iterate over all parameter combinations."""
        for tp, sl, ts, mi in itertools.product(
            self.take_profit_pct,
            self.stop_loss_pct,
            self.time_stop_seconds,
            self.min_imbalance
        ):
            yield ParameterSet(
                take_profit_pct=tp,
                stop_loss_pct=sl,
                time_stop_seconds=ts,
                min_imbalance=mi
            )

    def total_combinations(self) -> int:
        """Get total number of parameter combinations."""
        return (
            len(self.take_profit_pct) *
            len(self.stop_loss_pct) *
            len(self.time_stop_seconds) *
            len(self.min_imbalance)
        )

    @classmethod
    def from_string(cls, grid_str: str) -> "ParameterGrid":
        """
        Parse grid string like "tp=0.2,0.3,0.4; sl=0.1,0.15; tstop=30,60".

        Args:
            grid_str: Parameter grid specification

        Returns:
            ParameterGrid instance
        """
        grid = cls(
            take_profit_pct=[],
            stop_loss_pct=[],
            time_stop_seconds=[],
            min_imbalance=[0.60]  # Default
        )

        for part in grid_str.split(";"):
            part = part.strip()
            if "=" not in part:
                continue

            key, values = part.split("=", 1)
            key = key.strip().lower()
            values = [v.strip() for v in values.split(",")]

            if key in ("tp", "take_profit", "take_profit_pct"):
                grid.take_profit_pct = [float(v) for v in values]
            elif key in ("sl", "stop_loss", "stop_loss_pct"):
                grid.stop_loss_pct = [float(v) for v in values]
            elif key in ("tstop", "time_stop", "time_stop_seconds"):
                grid.time_stop_seconds = [int(v) for v in values]
            elif key in ("imb", "imbalance", "min_imbalance"):
                grid.min_imbalance = [float(v) for v in values]

        return grid


@dataclass
class WindowResult:
    """Result from a single train/test window."""
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    best_params: ParameterSet
    train_metrics: StrategyMetrics
    test_metrics: StrategyMetrics


@dataclass
class WalkForwardResult:
    """Complete walk-forward optimization result."""
    symbol: str
    start_date: datetime
    end_date: datetime
    train_days: int
    test_days: int
    total_windows: int
    window_results: List[WindowResult]
    # Aggregate out-of-sample metrics
    oos_trade_count: int
    oos_win_rate: float
    oos_total_pnl_pct: float
    oos_avg_pnl_pct: float
    oos_max_drawdown_pct: float
    oos_sharpe_ratio: float
    # Best parameters (most frequently selected)
    best_params_frequency: Dict[str, int]


class SimpleBacktestEngine:
    """
    Simple backtest engine using candle data.

    This is a simplified version that simulates trades based on
    price momentum (candle patterns) rather than full orderbook data.
    """

    def __init__(self, candles: List[Dict], random_seed: int = 42):
        """
        Initialize with candle data.

        Args:
            candles: List of candle dicts with keys: timestamp, open, high, low, close, volume
            random_seed: Random seed for deterministic results
        """
        self.candles = candles
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)

    def check_entry_signal(self, idx: int, params: ParameterSet) -> Optional[str]:
        """
        Check for entry signal at candle index.

        Simplified momentum signal:
        - Look for price drop followed by buying pressure (lower wick)
        - Volume spike

        Returns:
            "long" or "short" if signal, None otherwise
        """
        if idx < 10:
            return None

        candle = self.candles[idx]
        prev_candles = self.candles[max(0, idx-5):idx]

        # Calculate price change over last 5 candles
        price_5_ago = prev_candles[0]["close"] if prev_candles else candle["close"]
        price_change_pct = (candle["close"] - price_5_ago) / price_5_ago * 100

        # Volume spike (compare to last 20 candles)
        recent_volumes = [c["volume"] for c in self.candles[max(0, idx-20):idx]]
        avg_volume = np.mean(recent_volumes) if recent_volumes else 0
        volume_spike = candle["volume"] / avg_volume if avg_volume > 0 else 1

        # Candle analysis
        body = abs(candle["close"] - candle["open"])
        lower_wick = min(candle["open"], candle["close"]) - candle["low"]
        upper_wick = candle["high"] - max(candle["open"], candle["close"])

        # Simple momentum signal
        if price_change_pct < -0.3 and volume_spike > 1.5 and lower_wick > body * 0.5:
            # Oversold bounce - long signal
            return "long"
        elif price_change_pct > 0.3 and volume_spike > 1.5 and upper_wick > body * 0.5:
            # Overbought rejection - short signal
            return "short"

        return None

    def simulate_trade(
        self,
        entry_idx: int,
        side: str,
        params: ParameterSet
    ) -> Optional[TradeResult]:
        """
        Simulate a trade from entry candle.

        Args:
            entry_idx: Candle index for entry
            side: "long" or "short"
            params: Strategy parameters

        Returns:
            TradeResult if trade closed, None if invalid
        """
        if entry_idx >= len(self.candles) - 1:
            return None

        entry_candle = self.candles[entry_idx]
        entry_price = entry_candle["close"]
        entry_time = entry_candle["timestamp"]

        # Simulate forward
        max_candles = min(100, len(self.candles) - entry_idx - 1)

        for i in range(1, max_candles):
            current_idx = entry_idx + i
            current = self.candles[current_idx]
            current_time = current["timestamp"]
            hold_time_sec = (current_time - entry_time) / 1000

            # Calculate PnL based on side
            if side == "long":
                pnl_at_high = (current["high"] - entry_price) / entry_price * 100
                pnl_at_low = (current["low"] - entry_price) / entry_price * 100
                pnl_at_close = (current["close"] - entry_price) / entry_price * 100
            else:  # short
                pnl_at_high = (entry_price - current["low"]) / entry_price * 100
                pnl_at_low = (entry_price - current["high"]) / entry_price * 100
                pnl_at_close = (entry_price - current["close"]) / entry_price * 100

            # Check stop loss (use worst case)
            if pnl_at_low <= -params.stop_loss_pct:
                pnl_pct = -params.stop_loss_pct
                exit_price = entry_price * (1 - params.stop_loss_pct / 100) if side == "long" else entry_price * (1 + params.stop_loss_pct / 100)
                return TradeResult(
                    entry_time=entry_time,
                    exit_time=current_time,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    pnl_pct=pnl_pct,
                    pnl_after_costs_pct=calculate_pnl_after_costs(pnl_pct),
                    side=side,
                    exit_reason="stop_loss"
                )

            # Check take profit (use best case)
            if pnl_at_high >= params.take_profit_pct:
                pnl_pct = params.take_profit_pct
                exit_price = entry_price * (1 + params.take_profit_pct / 100) if side == "long" else entry_price * (1 - params.take_profit_pct / 100)
                return TradeResult(
                    entry_time=entry_time,
                    exit_time=current_time,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    pnl_pct=pnl_pct,
                    pnl_after_costs_pct=calculate_pnl_after_costs(pnl_pct),
                    side=side,
                    exit_reason="take_profit"
                )

            # Check time stop
            if hold_time_sec >= params.time_stop_seconds:
                pnl_pct = pnl_at_close
                exit_price = current["close"]
                return TradeResult(
                    entry_time=entry_time,
                    exit_time=current_time,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    pnl_pct=pnl_pct,
                    pnl_after_costs_pct=calculate_pnl_after_costs(pnl_pct),
                    side=side,
                    exit_reason="time_stop"
                )

        # End of data
        last = self.candles[-1]
        if side == "long":
            pnl_pct = (last["close"] - entry_price) / entry_price * 100
        else:
            pnl_pct = (entry_price - last["close"]) / entry_price * 100

        return TradeResult(
            entry_time=entry_time,
            exit_time=last["timestamp"],
            entry_price=entry_price,
            exit_price=last["close"],
            pnl_pct=pnl_pct,
            pnl_after_costs_pct=calculate_pnl_after_costs(pnl_pct),
            side=side,
            exit_reason="end_of_data"
        )

    def run_backtest(
        self,
        params: ParameterSet,
        start_time: int = None,
        end_time: int = None
    ) -> List[TradeResult]:
        """
        Run backtest with given parameters over time range.

        Args:
            params: Strategy parameters
            start_time: Start timestamp (ms)
            end_time: End timestamp (ms)

        Returns:
            List of TradeResult
        """
        trades = []
        last_trade_idx = -30  # Min candles between trades

        for i, candle in enumerate(self.candles):
            # Filter by time range
            if start_time and candle["timestamp"] < start_time:
                continue
            if end_time and candle["timestamp"] > end_time:
                break

            # Min gap between trades
            if i - last_trade_idx < 30:
                continue

            # Check for signal
            signal = self.check_entry_signal(i, params)
            if signal:
                trade = self.simulate_trade(i, signal, params)
                if trade:
                    trades.append(trade)
                    last_trade_idx = i

        return trades


class WalkForwardEngine:
    """
    Walk-forward optimization engine.

    Performs rolling optimization:
    1. Train on N days, find best parameters
    2. Test on M days out-of-sample
    3. Roll forward and repeat
    """

    def __init__(
        self,
        candles: List[Dict],
        train_days: int = 14,
        test_days: int = 7,
        random_seed: int = 42
    ):
        """
        Initialize walk-forward engine.

        Args:
            candles: Historical candle data
            train_days: Days for training window
            test_days: Days for test window
            random_seed: Random seed for reproducibility
        """
        self.candles = sorted(candles, key=lambda x: x["timestamp"])
        self.train_days = train_days
        self.test_days = test_days
        self.random_seed = random_seed

        # Convert timestamps to datetime for windowing
        self.start_time = self.candles[0]["timestamp"]
        self.end_time = self.candles[-1]["timestamp"]
        self.start_date = datetime.fromtimestamp(self.start_time / 1000)
        self.end_date = datetime.fromtimestamp(self.end_time / 1000)

    def _get_windows(self) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        """
        Generate train/test windows.

        Returns:
            List of (train_start, train_end, test_start, test_end) tuples
        """
        windows = []
        train_start = self.start_date

        while True:
            train_end = train_start + timedelta(days=self.train_days)
            test_start = train_end
            test_end = test_start + timedelta(days=self.test_days)

            if test_end > self.end_date:
                break

            windows.append((train_start, train_end, test_start, test_end))

            # Roll forward by test_days
            train_start = train_start + timedelta(days=self.test_days)

        return windows

    def optimize_on_window(
        self,
        train_start: datetime,
        train_end: datetime,
        param_grid: ParameterGrid
    ) -> Tuple[ParameterSet, StrategyMetrics]:
        """
        Find best parameters on training window.

        Optimizes for Sharpe ratio.

        Args:
            train_start: Training period start
            train_end: Training period end
            param_grid: Parameter grid to search

        Returns:
            (best_params, train_metrics)
        """
        start_ms = int(train_start.timestamp() * 1000)
        end_ms = int(train_end.timestamp() * 1000)

        best_params = None
        best_metrics = None
        best_score = float('-inf')

        engine = SimpleBacktestEngine(self.candles, self.random_seed)

        for params in param_grid.iterate():
            trades = engine.run_backtest(params, start_ms, end_ms)
            metrics = calculate_metrics(trades)

            # Score: Sharpe ratio with trade count penalty for low activity
            score = metrics.sharpe_ratio
            if metrics.trade_count < 10:
                score -= (10 - metrics.trade_count) * 0.1  # Penalty for low trades

            if score > best_score:
                best_score = score
                best_params = params
                best_metrics = metrics

        return best_params, best_metrics

    def test_on_window(
        self,
        test_start: datetime,
        test_end: datetime,
        params: ParameterSet
    ) -> StrategyMetrics:
        """
        Test parameters on out-of-sample window.

        Args:
            test_start: Test period start
            test_end: Test period end
            params: Parameters to test

        Returns:
            Test metrics
        """
        start_ms = int(test_start.timestamp() * 1000)
        end_ms = int(test_end.timestamp() * 1000)

        engine = SimpleBacktestEngine(self.candles, self.random_seed)
        trades = engine.run_backtest(params, start_ms, end_ms)

        return calculate_metrics(trades)

    def run(self, param_grid: ParameterGrid) -> WalkForwardResult:
        """
        Run complete walk-forward optimization.

        Args:
            param_grid: Parameter grid to search

        Returns:
            WalkForwardResult with all window results and aggregate metrics
        """
        windows = self._get_windows()

        if not windows:
            raise ValueError(f"Not enough data for {self.train_days}+{self.test_days} day windows")

        logger.info(f"Walk-forward: {len(windows)} windows, {param_grid.total_combinations()} params each")

        window_results = []
        all_oos_trades = []
        param_counts = {}

        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            logger.info(f"Window {i+1}/{len(windows)}: Train {train_start.date()}-{train_end.date()}, Test {test_start.date()}-{test_end.date()}")

            # Optimize on train
            best_params, train_metrics = self.optimize_on_window(train_start, train_end, param_grid)

            if best_params is None:
                logger.warning(f"  No valid params found for window {i+1}")
                continue

            # Test out-of-sample
            test_metrics = self.test_on_window(test_start, test_end, best_params)

            window_results.append(WindowResult(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                best_params=best_params,
                train_metrics=train_metrics,
                test_metrics=test_metrics
            ))

            # Track params frequency
            params_key = str(best_params.to_dict())
            param_counts[params_key] = param_counts.get(params_key, 0) + 1

            logger.info(f"  Best params: TP={best_params.take_profit_pct}, SL={best_params.stop_loss_pct}, TS={best_params.time_stop_seconds}")
            logger.info(f"  Train: {train_metrics.trade_count} trades, {train_metrics.win_rate:.1f}% WR, {train_metrics.avg_pnl_after_costs_pct:.4f}% avg")
            logger.info(f"  Test:  {test_metrics.trade_count} trades, {test_metrics.win_rate:.1f}% WR, {test_metrics.avg_pnl_after_costs_pct:.4f}% avg")

        # Aggregate out-of-sample metrics
        oos_trade_counts = [w.test_metrics.trade_count for w in window_results]
        oos_win_rates = [w.test_metrics.win_rate for w in window_results if w.test_metrics.trade_count > 0]
        oos_pnls = [w.test_metrics.total_pnl_after_costs_pct for w in window_results]
        oos_avg_pnls = [w.test_metrics.avg_pnl_after_costs_pct for w in window_results if w.test_metrics.trade_count > 0]
        oos_drawdowns = [w.test_metrics.max_drawdown_pct for w in window_results]
        oos_sharpes = [w.test_metrics.sharpe_ratio for w in window_results if w.test_metrics.trade_count > 0]

        return WalkForwardResult(
            symbol="",  # Set by caller
            start_date=self.start_date,
            end_date=self.end_date,
            train_days=self.train_days,
            test_days=self.test_days,
            total_windows=len(window_results),
            window_results=window_results,
            oos_trade_count=sum(oos_trade_counts),
            oos_win_rate=np.mean(oos_win_rates) if oos_win_rates else 0,
            oos_total_pnl_pct=sum(oos_pnls),
            oos_avg_pnl_pct=np.mean(oos_avg_pnls) if oos_avg_pnls else 0,
            oos_max_drawdown_pct=max(oos_drawdowns) if oos_drawdowns else 0,
            oos_sharpe_ratio=np.mean(oos_sharpes) if oos_sharpes else 0,
            best_params_frequency=param_counts
        )


def walk_forward_run(
    symbol: str,
    candles: List[Dict],
    start_date: datetime,
    end_date: datetime,
    train_days: int = 14,
    test_days: int = 7,
    param_grid: ParameterGrid = None,
    random_seed: int = 42
) -> WalkForwardResult:
    """
    Convenience function to run walk-forward optimization.

    Args:
        symbol: Trading symbol
        candles: Historical candle data
        start_date: Start date for analysis
        end_date: End date for analysis
        train_days: Training window size
        test_days: Test window size
        param_grid: Parameter grid (uses default if None)
        random_seed: Random seed

    Returns:
        WalkForwardResult
    """
    if param_grid is None:
        param_grid = ParameterGrid()

    # Filter candles to date range
    start_ms = int(start_date.timestamp() * 1000)
    end_ms = int(end_date.timestamp() * 1000)
    filtered = [c for c in candles if start_ms <= c["timestamp"] <= end_ms]

    if len(filtered) < 100:
        raise ValueError(f"Not enough candles ({len(filtered)}) for walk-forward analysis")

    engine = WalkForwardEngine(filtered, train_days, test_days, random_seed)
    result = engine.run(param_grid)
    result.symbol = symbol

    return result
