"""
Walk-Forward Optimization Engine
================================
Implements walk-forward testing for strategy parameter optimization.

Walk-forward testing (3-way split):
1. Split data into TRAIN/VALID/FINAL_TEST windows
2. Optimize parameters on TRAIN window
3. Validate on VALID window (used for model selection)
4. Evaluate on FINAL_TEST window (held out, NO selection)
5. Roll forward and repeat

This prevents overfitting and provides realistic performance estimates.
The FINAL_TEST data is NEVER used for parameter selection - only for
final unbiased performance evaluation.

Multiple Comparisons Warning:
When testing many parameter combinations, there's a risk of finding
spuriously good results by chance. We track configurations tested
and report a Bonferroni-corrected significance warning.
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
    """Result from a single train/test window (legacy 2-way split)."""
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    best_params: ParameterSet
    train_metrics: StrategyMetrics
    test_metrics: StrategyMetrics


@dataclass
class WindowResult3Way:
    """Result from a single train/valid/final_test window (3-way split)."""
    train_start: datetime
    train_end: datetime
    valid_start: datetime
    valid_end: datetime
    final_test_start: datetime
    final_test_end: datetime
    best_params: ParameterSet
    train_metrics: StrategyMetrics
    valid_metrics: StrategyMetrics
    final_test_metrics: StrategyMetrics
    configs_tried: int  # Number of param configs tested in this window


@dataclass
class WalkForwardResult:
    """Complete walk-forward optimization result (legacy 2-way split)."""
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


@dataclass
class WalkForwardResult3Way:
    """Complete walk-forward optimization result with 3-way split."""
    symbol: str
    start_date: datetime
    end_date: datetime
    train_days: int
    valid_days: int
    final_test_days: int
    total_windows: int
    window_results: List[WindowResult3Way]

    # Aggregate VALIDATION metrics (used for selection)
    valid_trade_count: int
    valid_win_rate: float
    valid_total_pnl_pct: float
    valid_avg_pnl_pct: float
    valid_max_drawdown_pct: float
    valid_sharpe_ratio: float

    # Aggregate FINAL TEST metrics (NOT used for selection - unbiased)
    final_test_trade_count: int
    final_test_win_rate: float
    final_test_total_pnl_pct: float
    final_test_avg_pnl_pct: float
    final_test_max_drawdown_pct: float
    final_test_sharpe_ratio: float

    # Best parameters (most frequently selected)
    best_params_frequency: Dict[str, int]

    # Multiple comparisons tracking
    total_configs_tried: int  # Total parameter combinations tested
    unique_configs: int  # Unique parameter sets
    bonferroni_alpha: float  # Corrected significance level
    multiple_comparisons_warning: str  # Warning message if applicable


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


class WalkForwardEngine3Way:
    """
    Walk-forward optimization engine with 3-way split.

    Performs rolling optimization with proper holdout:
    1. TRAIN on N days - find best parameters
    2. VALIDATE on M days - verify params (used for selection)
    3. FINAL TEST on K days - unbiased evaluation (NOT used for selection)
    4. Roll forward and repeat

    The key difference from 2-way split: FINAL_TEST data is NEVER used
    for parameter selection. It provides unbiased performance estimate.
    """

    def __init__(
        self,
        candles: List[Dict],
        train_days: int = 14,
        valid_days: int = 7,
        final_test_days: int = 7,
        random_seed: int = 42
    ):
        """
        Initialize 3-way walk-forward engine.

        Args:
            candles: Historical candle data
            train_days: Days for training window
            valid_days: Days for validation window
            final_test_days: Days for final test window (held out)
            random_seed: Random seed for reproducibility
        """
        self.candles = sorted(candles, key=lambda x: x["timestamp"])
        self.train_days = train_days
        self.valid_days = valid_days
        self.final_test_days = final_test_days
        self.random_seed = random_seed

        # Convert timestamps to datetime for windowing
        self.start_time = self.candles[0]["timestamp"]
        self.end_time = self.candles[-1]["timestamp"]
        self.start_date = datetime.fromtimestamp(self.start_time / 1000)
        self.end_date = datetime.fromtimestamp(self.end_time / 1000)

        # Track configs for multiple comparisons warning
        self.total_configs_tried = 0

    def _get_windows(self) -> List[Tuple[datetime, datetime, datetime, datetime, datetime, datetime]]:
        """
        Generate train/valid/final_test windows.

        Returns:
            List of (train_start, train_end, valid_start, valid_end, final_test_start, final_test_end) tuples
        """
        windows = []
        train_start = self.start_date
        window_size = self.train_days + self.valid_days + self.final_test_days

        while True:
            train_end = train_start + timedelta(days=self.train_days)
            valid_start = train_end
            valid_end = valid_start + timedelta(days=self.valid_days)
            final_test_start = valid_end
            final_test_end = final_test_start + timedelta(days=self.final_test_days)

            if final_test_end > self.end_date:
                break

            windows.append((train_start, train_end, valid_start, valid_end, final_test_start, final_test_end))

            # Roll forward by (valid_days + final_test_days) to avoid overlap
            train_start = train_start + timedelta(days=self.valid_days + self.final_test_days)

        return windows

    def optimize_on_train(
        self,
        train_start: datetime,
        train_end: datetime,
        param_grid: ParameterGrid
    ) -> Tuple[List[Tuple[ParameterSet, StrategyMetrics, float]], int]:
        """
        Find candidate parameters on training window.

        Returns ALL param results (sorted by score) so validation can pick.

        Args:
            train_start: Training period start
            train_end: Training period end
            param_grid: Parameter grid to search

        Returns:
            (List of (params, metrics, score) sorted by score desc, configs_tried)
        """
        start_ms = int(train_start.timestamp() * 1000)
        end_ms = int(train_end.timestamp() * 1000)

        results = []
        configs_tried = 0

        engine = SimpleBacktestEngine(self.candles, self.random_seed)

        for params in param_grid.iterate():
            configs_tried += 1
            trades = engine.run_backtest(params, start_ms, end_ms)
            metrics = calculate_metrics(trades)

            # Score: Sharpe ratio with trade count penalty for low activity
            score = metrics.sharpe_ratio
            if metrics.trade_count < 10:
                score -= (10 - metrics.trade_count) * 0.1

            results.append((params, metrics, score))

        # Sort by score descending
        results.sort(key=lambda x: x[2], reverse=True)

        self.total_configs_tried += configs_tried
        return results, configs_tried

    def validate_params(
        self,
        valid_start: datetime,
        valid_end: datetime,
        candidates: List[Tuple[ParameterSet, StrategyMetrics, float]],
        top_n: int = 5
    ) -> Tuple[ParameterSet, StrategyMetrics, StrategyMetrics]:
        """
        Validate top candidates on validation window and pick best.

        This prevents selection bias: we pick params based on TRAIN,
        verify on VALID, and only then evaluate on FINAL_TEST.

        Args:
            valid_start: Validation period start
            valid_end: Validation period end
            candidates: List of (params, train_metrics, train_score) from training
            top_n: Number of top candidates to validate

        Returns:
            (best_params, train_metrics, valid_metrics)
        """
        start_ms = int(valid_start.timestamp() * 1000)
        end_ms = int(valid_end.timestamp() * 1000)

        engine = SimpleBacktestEngine(self.candles, self.random_seed)

        best_params = None
        best_train_metrics = None
        best_valid_metrics = None
        best_valid_score = float('-inf')

        # Test top N candidates on validation set
        for params, train_metrics, _ in candidates[:top_n]:
            trades = engine.run_backtest(params, start_ms, end_ms)
            valid_metrics = calculate_metrics(trades)

            # Score on validation (same formula as training)
            valid_score = valid_metrics.sharpe_ratio
            if valid_metrics.trade_count < 5:
                valid_score -= (5 - valid_metrics.trade_count) * 0.1

            if valid_score > best_valid_score:
                best_valid_score = valid_score
                best_params = params
                best_train_metrics = train_metrics
                best_valid_metrics = valid_metrics

        return best_params, best_train_metrics, best_valid_metrics

    def evaluate_final_test(
        self,
        final_test_start: datetime,
        final_test_end: datetime,
        params: ParameterSet
    ) -> StrategyMetrics:
        """
        Evaluate on final test window (NO selection - unbiased).

        IMPORTANT: This result is NEVER used for parameter selection.
        It is only for reporting unbiased out-of-sample performance.

        Args:
            final_test_start: Final test period start
            final_test_end: Final test period end
            params: Parameters selected from train+valid

        Returns:
            Final test metrics (unbiased)
        """
        start_ms = int(final_test_start.timestamp() * 1000)
        end_ms = int(final_test_end.timestamp() * 1000)

        engine = SimpleBacktestEngine(self.candles, self.random_seed)
        trades = engine.run_backtest(params, start_ms, end_ms)

        return calculate_metrics(trades)

    def _calculate_multiple_comparisons_warning(
        self,
        total_configs: int,
        windows: int,
        alpha: float = 0.05
    ) -> Tuple[float, str]:
        """
        Calculate multiple comparisons warning.

        Uses Bonferroni correction to adjust significance level.

        Args:
            total_configs: Total configurations tested
            windows: Number of windows
            alpha: Desired significance level (default 0.05)

        Returns:
            (bonferroni_alpha, warning_message)
        """
        # Effective tests = configs per window (since we pick one per window)
        # But total exposure is configs * windows
        effective_tests = total_configs

        bonferroni_alpha = alpha / effective_tests if effective_tests > 0 else alpha

        # Warning thresholds
        if effective_tests > 1000:
            warning = (
                f"HIGH RISK: {effective_tests} configs tested. "
                f"Bonferroni-corrected alpha = {bonferroni_alpha:.6f}. "
                f"Results may be spurious due to multiple comparisons. "
                f"Consider reducing parameter grid or using cross-validation."
            )
        elif effective_tests > 100:
            warning = (
                f"MODERATE RISK: {effective_tests} configs tested. "
                f"Bonferroni-corrected alpha = {bonferroni_alpha:.4f}. "
                f"Interpret results with caution."
            )
        elif effective_tests > 20:
            warning = (
                f"LOW RISK: {effective_tests} configs tested. "
                f"Bonferroni-corrected alpha = {bonferroni_alpha:.4f}. "
                f"Results reasonably reliable."
            )
        else:
            warning = f"MINIMAL RISK: Only {effective_tests} configs tested."

        return bonferroni_alpha, warning

    def run(self, param_grid: ParameterGrid, top_n_validate: int = 5) -> WalkForwardResult3Way:
        """
        Run complete 3-way walk-forward optimization.

        Args:
            param_grid: Parameter grid to search
            top_n_validate: Number of top train candidates to validate

        Returns:
            WalkForwardResult3Way with all window results and aggregate metrics
        """
        windows = self._get_windows()

        if not windows:
            raise ValueError(
                f"Not enough data for {self.train_days}+{self.valid_days}+{self.final_test_days} day windows"
            )

        logger.info(f"Walk-forward 3-way: {len(windows)} windows, {param_grid.total_combinations()} params each")
        logger.info(f"  Split: TRAIN={self.train_days}d / VALID={self.valid_days}d / FINAL_TEST={self.final_test_days}d")

        window_results = []
        param_counts = {}
        self.total_configs_tried = 0

        for i, (train_start, train_end, valid_start, valid_end, ft_start, ft_end) in enumerate(windows):
            logger.info(
                f"Window {i+1}/{len(windows)}: "
                f"Train {train_start.date()}-{train_end.date()}, "
                f"Valid {valid_start.date()}-{valid_end.date()}, "
                f"FinalTest {ft_start.date()}-{ft_end.date()}"
            )

            # Step 1: Optimize on TRAIN
            candidates, configs_tried = self.optimize_on_train(train_start, train_end, param_grid)

            if not candidates or candidates[0][0] is None:
                logger.warning(f"  No valid params found for window {i+1}")
                continue

            # Step 2: Validate on VALID (pick best from top N)
            best_params, train_metrics, valid_metrics = self.validate_params(
                valid_start, valid_end, candidates, top_n_validate
            )

            if best_params is None:
                logger.warning(f"  Validation failed for window {i+1}")
                continue

            # Step 3: Evaluate on FINAL_TEST (NO selection influence)
            final_test_metrics = self.evaluate_final_test(ft_start, ft_end, best_params)

            window_results.append(WindowResult3Way(
                train_start=train_start,
                train_end=train_end,
                valid_start=valid_start,
                valid_end=valid_end,
                final_test_start=ft_start,
                final_test_end=ft_end,
                best_params=best_params,
                train_metrics=train_metrics,
                valid_metrics=valid_metrics,
                final_test_metrics=final_test_metrics,
                configs_tried=configs_tried
            ))

            # Track params frequency
            params_key = str(best_params.to_dict())
            param_counts[params_key] = param_counts.get(params_key, 0) + 1

            logger.info(f"  Best params: TP={best_params.take_profit_pct}, SL={best_params.stop_loss_pct}, TS={best_params.time_stop_seconds}")
            logger.info(f"  Train:      {train_metrics.trade_count} trades, {train_metrics.win_rate:.1f}% WR, {train_metrics.avg_pnl_after_costs_pct:.4f}% avg")
            logger.info(f"  Valid:      {valid_metrics.trade_count} trades, {valid_metrics.win_rate:.1f}% WR, {valid_metrics.avg_pnl_after_costs_pct:.4f}% avg")
            logger.info(f"  FinalTest:  {final_test_metrics.trade_count} trades, {final_test_metrics.win_rate:.1f}% WR, {final_test_metrics.avg_pnl_after_costs_pct:.4f}% avg")

        # Aggregate VALIDATION metrics
        valid_trade_counts = [w.valid_metrics.trade_count for w in window_results]
        valid_win_rates = [w.valid_metrics.win_rate for w in window_results if w.valid_metrics.trade_count > 0]
        valid_pnls = [w.valid_metrics.total_pnl_after_costs_pct for w in window_results]
        valid_avg_pnls = [w.valid_metrics.avg_pnl_after_costs_pct for w in window_results if w.valid_metrics.trade_count > 0]
        valid_drawdowns = [w.valid_metrics.max_drawdown_pct for w in window_results]
        valid_sharpes = [w.valid_metrics.sharpe_ratio for w in window_results if w.valid_metrics.trade_count > 0]

        # Aggregate FINAL TEST metrics (unbiased)
        ft_trade_counts = [w.final_test_metrics.trade_count for w in window_results]
        ft_win_rates = [w.final_test_metrics.win_rate for w in window_results if w.final_test_metrics.trade_count > 0]
        ft_pnls = [w.final_test_metrics.total_pnl_after_costs_pct for w in window_results]
        ft_avg_pnls = [w.final_test_metrics.avg_pnl_after_costs_pct for w in window_results if w.final_test_metrics.trade_count > 0]
        ft_drawdowns = [w.final_test_metrics.max_drawdown_pct for w in window_results]
        ft_sharpes = [w.final_test_metrics.sharpe_ratio for w in window_results if w.final_test_metrics.trade_count > 0]

        # Multiple comparisons warning
        bonferroni_alpha, mc_warning = self._calculate_multiple_comparisons_warning(
            self.total_configs_tried, len(windows)
        )

        return WalkForwardResult3Way(
            symbol="",  # Set by caller
            start_date=self.start_date,
            end_date=self.end_date,
            train_days=self.train_days,
            valid_days=self.valid_days,
            final_test_days=self.final_test_days,
            total_windows=len(window_results),
            window_results=window_results,
            # Validation metrics
            valid_trade_count=sum(valid_trade_counts),
            valid_win_rate=np.mean(valid_win_rates) if valid_win_rates else 0,
            valid_total_pnl_pct=sum(valid_pnls),
            valid_avg_pnl_pct=np.mean(valid_avg_pnls) if valid_avg_pnls else 0,
            valid_max_drawdown_pct=max(valid_drawdowns) if valid_drawdowns else 0,
            valid_sharpe_ratio=np.mean(valid_sharpes) if valid_sharpes else 0,
            # Final test metrics (unbiased)
            final_test_trade_count=sum(ft_trade_counts),
            final_test_win_rate=np.mean(ft_win_rates) if ft_win_rates else 0,
            final_test_total_pnl_pct=sum(ft_pnls),
            final_test_avg_pnl_pct=np.mean(ft_avg_pnls) if ft_avg_pnls else 0,
            final_test_max_drawdown_pct=max(ft_drawdowns) if ft_drawdowns else 0,
            final_test_sharpe_ratio=np.mean(ft_sharpes) if ft_sharpes else 0,
            # Params
            best_params_frequency=param_counts,
            # Multiple comparisons
            total_configs_tried=self.total_configs_tried,
            unique_configs=param_grid.total_combinations(),
            bonferroni_alpha=bonferroni_alpha,
            multiple_comparisons_warning=mc_warning
        )


def walk_forward_run_3way(
    symbol: str,
    candles: List[Dict],
    start_date: datetime,
    end_date: datetime,
    train_days: int = 14,
    valid_days: int = 7,
    final_test_days: int = 7,
    param_grid: ParameterGrid = None,
    random_seed: int = 42,
    top_n_validate: int = 5
) -> WalkForwardResult3Way:
    """
    Convenience function to run 3-way walk-forward optimization.

    IMPORTANT: Uses 3-way split to prevent data leakage:
    - TRAIN: Find best parameters
    - VALID: Validate and select (early stopping)
    - FINAL_TEST: Unbiased evaluation (NOT used for selection)

    Args:
        symbol: Trading symbol
        candles: Historical candle data
        start_date: Start date for analysis
        end_date: End date for analysis
        train_days: Training window size
        valid_days: Validation window size
        final_test_days: Final test window size (held out)
        param_grid: Parameter grid (uses default if None)
        random_seed: Random seed
        top_n_validate: Number of top candidates to validate

    Returns:
        WalkForwardResult3Way with unbiased final test metrics
    """
    if param_grid is None:
        param_grid = ParameterGrid()

    # Filter candles to date range
    start_ms = int(start_date.timestamp() * 1000)
    end_ms = int(end_date.timestamp() * 1000)
    filtered = [c for c in candles if start_ms <= c["timestamp"] <= end_ms]

    if len(filtered) < 100:
        raise ValueError(f"Not enough candles ({len(filtered)}) for walk-forward analysis")

    engine = WalkForwardEngine3Way(filtered, train_days, valid_days, final_test_days, random_seed)
    result = engine.run(param_grid, top_n_validate)
    result.symbol = symbol

    return result


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
