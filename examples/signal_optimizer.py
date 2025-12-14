#!/usr/bin/env python3
"""
SIGNAL OPTIMIZER
================
Finds high-frequency, high-edge trading configurations.

The goal: More trades + maintained edge = Real profitability

Tests:
- Multiple timeframes (1h, 15m, 5m)
- Multiple signal combinations
- Multiple symbols
- Different risk parameters

Ranks by: Edge Score = Profit Factor × √(Trade Frequency)

Usage:
    python3 signal_optimizer.py --symbol ATOMUSDT
    python3 signal_optimizer.py --symbol ATOMUSDT --deep
    python3 signal_optimizer.py --all-symbols
"""

import os
import sys
import json
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import time
import math

# Disable SSL warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
os.environ['PYTHONWARNINGS'] = 'ignore:Unverified HTTPS request'

try:
    import requests
except ImportError:
    print("[ERROR] requests library required: pip install requests")
    sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Symbols to test
TEST_SYMBOLS = [
    "ATOMUSDT",   # Current best performer
    "SOLUSDT",    # Previously tested
    "AVAXUSDT",   # High volatility alt
    "DOTUSDT",    # Polkadot
    "MATICUSDT",  # Polygon (if available)
    "LINKUSDT",   # Chainlink
    "ADAUSDT",    # Cardano
]

# Timeframes to test
TEST_TIMEFRAMES = [
    ("1h", 90),    # 1 hour, 90 days
    ("15m", 30),   # 15 min, 30 days (more data points)
    ("5m", 14),    # 5 min, 14 days (high frequency)
]

# Signal combinations to test
SIGNAL_COMBINATIONS = [
    ["BB", "MACD"],           # Current strategy
    ["BB", "RSI"],            # BB + oversold/overbought
    ["MACD", "RSI"],          # Momentum combo
    ["BB", "MACD", "RSI"],    # Triple confirmation
    ["BB"],                   # BB only (high frequency)
    ["MACD"],                 # MACD only
    ["RSI"],                  # RSI only
    ["BB", "MACD", "VOL"],    # With volume confirmation
]

# Risk parameters to test
RISK_PARAMS = [
    (1.5, 2.0),  # Tight: 1.5% stop, 2% target
    (2.0, 3.0),  # Standard: 2% stop, 3% target (current)
    (2.0, 4.0),  # Wide target: 2% stop, 4% target
    (1.5, 3.0),  # Tight stop, standard target
    (2.5, 5.0),  # Swing: wider parameters
]


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Candle:
    """OHLCV candle."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class OptimizationResult:
    """Result of testing one configuration."""
    symbol: str
    timeframe: str
    signals: List[str]
    stop_pct: float
    target_pct: float

    # Performance
    trades: int
    wins: int
    losses: int
    win_rate: float
    profit_factor: float
    total_pnl_pct: float
    max_drawdown: float
    max_consecutive_losses: int

    # Quality metrics
    avg_mfe: float
    avg_mae: float

    # Computed scores
    edge_score: float = 0.0
    frequency_score: float = 0.0
    combined_score: float = 0.0

    def compute_scores(self, max_trades: int = 100):
        """Compute optimization scores."""
        # Edge score: profit factor (capped at 5)
        self.edge_score = min(self.profit_factor, 5.0)

        # Frequency score: normalized trade count
        self.frequency_score = min(self.trades / max_trades, 1.0)

        # Combined score: Edge × √Frequency
        # This rewards both edge AND frequency, but edge matters more
        if self.win_rate >= 50 and self.profit_factor >= 1.0:
            self.combined_score = self.edge_score * math.sqrt(self.frequency_score + 0.1)
        else:
            self.combined_score = 0  # No edge = no score


# =============================================================================
# DATA FETCHER
# =============================================================================

class DataFetcher:
    """Fetch historical candles from Binance US."""

    BASE_URL = "https://api.binance.us/api/v3"

    def __init__(self):
        self.session = requests.Session()
        self.session.verify = False
        self.cache = {}  # Cache fetched data

    def fetch(self, symbol: str, interval: str, days: int) -> List[Candle]:
        """Fetch candles with caching."""
        cache_key = f"{symbol}_{interval}_{days}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        candles = []
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        current_start = start_time

        while current_start < end_time:
            try:
                response = self.session.get(
                    f"{self.BASE_URL}/klines",
                    params={
                        "symbol": symbol,
                        "interval": interval,
                        "startTime": current_start,
                        "endTime": end_time,
                        "limit": 1000
                    },
                    timeout=30
                )

                if response.status_code == 451:
                    return []  # Geo-restricted

                if response.status_code != 200:
                    break

                data = response.json()
                if not data:
                    break

                for kline in data:
                    candles.append(Candle(
                        timestamp=datetime.fromtimestamp(kline[0] / 1000),
                        open=float(kline[1]),
                        high=float(kline[2]),
                        low=float(kline[3]),
                        close=float(kline[4]),
                        volume=float(kline[5])
                    ))

                current_start = int(data[-1][0]) + 1
                time.sleep(0.1)  # Rate limit

            except Exception as e:
                break

        self.cache[cache_key] = candles
        return candles


# =============================================================================
# SIGNAL GENERATOR
# =============================================================================

class SignalGenerator:
    """Generate signals with configurable indicators."""

    def __init__(self, required_signals: List[str]):
        self.required_signals = required_signals

    def calculate_bb(self, candles: List[Candle], period: int = 20, std: float = 2.0):
        """Bollinger Bands."""
        if len(candles) < period:
            return 0, 0, 0
        closes = [c.close for c in candles[-period:]]
        sma = sum(closes) / period
        variance = sum((x - sma) ** 2 for x in closes) / period
        std_dev = variance ** 0.5
        return sma - (std * std_dev), sma, sma + (std * std_dev)

    def calculate_macd(self, candles: List[Candle]):
        """MACD indicator."""
        if len(candles) < 35:
            return 0, 0, 0

        closes = [c.close for c in candles]

        def ema(data, period):
            result = [data[0]]
            mult = 2 / (period + 1)
            for i in range(1, len(data)):
                result.append((data[i] - result[-1]) * mult + result[-1])
            return result

        ema12 = ema(closes, 12)
        ema26 = ema(closes, 26)
        macd_line = [f - s for f, s in zip(ema12, ema26)]
        signal_line = ema(macd_line[25:], 9)

        if not signal_line:
            return 0, 0, 0

        return macd_line[-1], signal_line[-1], macd_line[-1] - signal_line[-1]

    def calculate_rsi(self, candles: List[Candle], period: int = 14):
        """RSI indicator."""
        if len(candles) < period + 1:
            return 50

        closes = [c.close for c in candles[-(period+1):]]
        gains, losses = [], []

        for i in range(1, len(closes)):
            change = closes[i] - closes[i-1]
            gains.append(max(change, 0))
            losses.append(abs(min(change, 0)))

        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period

        if avg_loss == 0:
            return 100

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def calculate_volume_signal(self, candles: List[Candle], period: int = 20):
        """Volume above average."""
        if len(candles) < period:
            return False
        volumes = [c.volume for c in candles[-period:]]
        avg_vol = sum(volumes) / period
        return candles[-1].volume > avg_vol * 1.5

    def generate(self, candles: List[Candle], index: int) -> Optional[Dict]:
        """Generate signal at index if conditions met."""
        if index < 35:
            return None

        history = candles[:index+1]
        current = candles[index]

        signals = []
        direction = None
        confidence = 0.5

        # Check BB
        if "BB" in self.required_signals:
            lower, mid, upper = self.calculate_bb(history)
            if current.close < lower:
                signals.append("BB")
                direction = "BUY"
                confidence += 0.15
            elif current.close > upper:
                signals.append("BB")
                direction = "SELL"
                confidence += 0.15

        # Check MACD
        if "MACD" in self.required_signals:
            macd, signal, hist = self.calculate_macd(history)
            prev_hist = self.calculate_macd(history[:-1])[2] if len(history) > 1 else 0

            if hist > 0 and prev_hist <= 0:
                signals.append("MACD")
                if direction is None:
                    direction = "BUY"
                confidence += 0.15
            elif hist < 0 and prev_hist >= 0:
                signals.append("MACD")
                if direction is None:
                    direction = "SELL"
                confidence += 0.15

        # Check RSI
        if "RSI" in self.required_signals:
            rsi = self.calculate_rsi(history)
            if rsi < 30:
                signals.append("RSI")
                if direction is None:
                    direction = "BUY"
                confidence += 0.10
            elif rsi > 70:
                signals.append("RSI")
                if direction is None:
                    direction = "SELL"
                confidence += 0.10

        # Check Volume
        if "VOL" in self.required_signals:
            if self.calculate_volume_signal(history):
                signals.append("VOL")
                confidence += 0.05

        # Check if all required signals are present
        required_count = len([s for s in self.required_signals if s != "VOL"])
        actual_count = len([s for s in signals if s != "VOL"])

        if actual_count < required_count:
            return None

        if direction is None:
            return None

        return {
            "timestamp": current.timestamp,
            "direction": direction,
            "signals": signals,
            "price": current.close,
            "confidence": min(confidence, 0.95)
        }


# =============================================================================
# BACKTESTER
# =============================================================================

class QuickBacktest:
    """Fast backtester for optimization."""

    def __init__(self, stop_pct: float, target_pct: float):
        self.stop_pct = stop_pct
        self.target_pct = target_pct

    def run(self, candles: List[Candle], generator: SignalGenerator) -> OptimizationResult:
        """Run backtest and return results."""
        trades = []
        wins = 0
        losses = 0
        total_pnl = 0.0
        max_dd = 0.0
        peak = 10000.0
        capital = 10000.0
        consecutive_losses = 0
        max_consecutive = 0
        mfe_sum = 0.0
        mae_sum = 0.0

        for i in range(35, len(candles) - 50):
            signal = generator.generate(candles, i)

            if signal:
                # Simulate trade
                entry = signal["price"]
                direction = signal["direction"]

                if direction == "BUY":
                    stop = entry * (1 - self.stop_pct / 100)
                    target = entry * (1 + self.target_pct / 100)
                else:
                    stop = entry * (1 + self.stop_pct / 100)
                    target = entry * (1 - self.target_pct / 100)

                # Track outcome
                exit_price = None
                pnl_pct = 0
                mfe = 0
                mae = 0

                for future in candles[i+1:i+51]:
                    if direction == "BUY":
                        # Check stop/target
                        if future.low <= stop:
                            exit_price = stop
                            pnl_pct = -self.stop_pct
                            break
                        if future.high >= target:
                            exit_price = target
                            pnl_pct = self.target_pct
                            break
                        # Track MFE/MAE
                        mfe = max(mfe, (future.high - entry) / entry * 100)
                        mae = max(mae, (entry - future.low) / entry * 100)
                    else:
                        if future.high >= stop:
                            exit_price = stop
                            pnl_pct = -self.stop_pct
                            break
                        if future.low <= target:
                            exit_price = target
                            pnl_pct = self.target_pct
                            break
                        mfe = max(mfe, (entry - future.low) / entry * 100)
                        mae = max(mae, (future.high - entry) / entry * 100)

                # Timeout exit
                if exit_price is None:
                    last = candles[min(i+50, len(candles)-1)].close
                    if direction == "BUY":
                        pnl_pct = (last - entry) / entry * 100
                    else:
                        pnl_pct = (entry - last) / entry * 100

                # Record trade
                trades.append(pnl_pct)
                mfe_sum += mfe
                mae_sum += mae

                if pnl_pct > 0:
                    wins += 1
                    consecutive_losses = 0
                else:
                    losses += 1
                    consecutive_losses += 1
                    max_consecutive = max(max_consecutive, consecutive_losses)

                # Update capital
                position_size = capital * 0.02
                pnl_usd = position_size * (pnl_pct / 100)
                capital += pnl_usd
                total_pnl += pnl_usd

                # Update drawdown
                if capital > peak:
                    peak = capital
                dd = (peak - capital) / peak * 100
                max_dd = max(max_dd, dd)

        # Calculate metrics
        total_trades = len(trades)
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

        gross_profit = sum(t for t in trades if t > 0)
        gross_loss = abs(sum(t for t in trades if t < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        avg_mfe = mfe_sum / total_trades if total_trades > 0 else 0
        avg_mae = mae_sum / total_trades if total_trades > 0 else 0

        return OptimizationResult(
            symbol="",  # Set by caller
            timeframe="",  # Set by caller
            signals=generator.required_signals,
            stop_pct=self.stop_pct,
            target_pct=self.target_pct,
            trades=total_trades,
            wins=wins,
            losses=losses,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_pnl_pct=(total_pnl / 10000) * 100,
            max_drawdown=max_dd,
            max_consecutive_losses=max_consecutive,
            avg_mfe=avg_mfe,
            avg_mae=avg_mae
        )


# =============================================================================
# OPTIMIZER
# =============================================================================

class SignalOptimizer:
    """Find optimal trading configurations."""

    def __init__(self):
        self.fetcher = DataFetcher()
        self.results: List[OptimizationResult] = []

    def optimize_symbol(
        self,
        symbol: str,
        deep: bool = False,
        progress_callback=None
    ) -> List[OptimizationResult]:
        """
        Test all configurations for a symbol.

        Args:
            symbol: Trading pair
            deep: If True, test all combinations. If False, test common ones.
            progress_callback: Optional progress update function
        """
        results = []

        # Select configurations to test
        if deep:
            timeframes = TEST_TIMEFRAMES
            signal_combos = SIGNAL_COMBINATIONS
            risk_params = RISK_PARAMS
        else:
            # Quick test: fewer combinations
            timeframes = [("1h", 90), ("15m", 30)]
            signal_combos = [["BB", "MACD"], ["BB", "RSI"], ["BB"]]
            risk_params = [(2.0, 3.0), (1.5, 2.0)]

        total_tests = len(timeframes) * len(signal_combos) * len(risk_params)
        current_test = 0

        print(f"\n[OPTIMIZE] Testing {symbol} ({total_tests} configurations)...")

        for interval, days in timeframes:
            # Fetch data once per timeframe
            candles = self.fetcher.fetch(symbol, interval, days)

            if len(candles) < 100:
                print(f"  [SKIP] {interval}: Not enough data")
                continue

            for signals in signal_combos:
                for stop, target in risk_params:
                    current_test += 1

                    # Run backtest
                    generator = SignalGenerator(signals)
                    backtester = QuickBacktest(stop, target)
                    result = backtester.run(candles, generator)

                    # Set metadata
                    result.symbol = symbol
                    result.timeframe = interval
                    result.compute_scores()

                    results.append(result)

                    if progress_callback:
                        progress_callback(current_test, total_tests)
                    elif current_test % 10 == 0:
                        print(f"  Progress: {current_test}/{total_tests}", end="\r")

        print(f"  Completed {len(results)} configurations")
        return results

    def find_best(
        self,
        results: List[OptimizationResult],
        min_trades: int = 10,
        min_win_rate: float = 55.0,
        min_profit_factor: float = 1.2
    ) -> List[OptimizationResult]:
        """Filter and rank results by combined score."""

        # Filter by minimum criteria
        filtered = [
            r for r in results
            if r.trades >= min_trades
            and r.win_rate >= min_win_rate
            and r.profit_factor >= min_profit_factor
            and r.max_drawdown <= 20.0
        ]

        # Sort by combined score
        filtered.sort(key=lambda x: x.combined_score, reverse=True)

        return filtered


# =============================================================================
# REPORTING
# =============================================================================

def print_optimization_report(results: List[OptimizationResult], symbol: str):
    """Print comprehensive optimization report."""

    print("\n" + "=" * 80)
    print(f"SIGNAL OPTIMIZATION REPORT - {symbol}")
    print("=" * 80)

    if not results:
        print("\n[NO RESULTS] No configurations met minimum criteria")
        print("  Try lowering min_trades or min_win_rate thresholds")
        return

    # Top configurations
    print(f"\n[TOP 10 CONFIGURATIONS]")
    print("-" * 80)
    print(f"{'Rank':<5} {'TF':<5} {'Signals':<20} {'Stop/Tgt':<10} "
          f"{'Trades':<7} {'Win%':<7} {'PF':<6} {'Score':<8}")
    print("-" * 80)

    for i, r in enumerate(results[:10], 1):
        signals_str = "+".join(r.signals)[:18]
        risk_str = f"{r.stop_pct}/{r.target_pct}"
        print(f"{i:<5} {r.timeframe:<5} {signals_str:<20} {risk_str:<10} "
              f"{r.trades:<7} {r.win_rate:<7.1f} {r.profit_factor:<6.2f} {r.combined_score:<8.2f}")

    # Best by category
    print(f"\n[BEST BY CATEGORY]")
    print("-" * 80)

    # Best edge (highest profit factor)
    best_edge = max(results, key=lambda x: x.profit_factor)
    print(f"  Best Edge (PF):     {'+'.join(best_edge.signals)} on {best_edge.timeframe} "
          f"(PF={best_edge.profit_factor:.2f}, {best_edge.trades} trades)")

    # Best frequency
    best_freq = max(results, key=lambda x: x.trades if x.profit_factor >= 1.0 else 0)
    print(f"  Best Frequency:     {'+'.join(best_freq.signals)} on {best_freq.timeframe} "
          f"({best_freq.trades} trades, PF={best_freq.profit_factor:.2f})")

    # Best balanced
    best_balanced = results[0] if results else None
    if best_balanced:
        print(f"  Best Balanced:      {'+'.join(best_balanced.signals)} on {best_balanced.timeframe} "
              f"(Score={best_balanced.combined_score:.2f})")

    # Timeframe analysis
    print(f"\n[TIMEFRAME ANALYSIS]")
    print("-" * 80)

    tf_stats = {}
    for r in results:
        if r.timeframe not in tf_stats:
            tf_stats[r.timeframe] = {"count": 0, "avg_pf": 0, "avg_trades": 0, "profitable": 0}
        tf_stats[r.timeframe]["count"] += 1
        tf_stats[r.timeframe]["avg_pf"] += r.profit_factor
        tf_stats[r.timeframe]["avg_trades"] += r.trades
        if r.profit_factor > 1.0:
            tf_stats[r.timeframe]["profitable"] += 1

    for tf, stats in tf_stats.items():
        avg_pf = stats["avg_pf"] / stats["count"]
        avg_trades = stats["avg_trades"] / stats["count"]
        profit_pct = stats["profitable"] / stats["count"] * 100
        print(f"  {tf}: Avg PF={avg_pf:.2f}, Avg Trades={avg_trades:.0f}, "
              f"Profitable={profit_pct:.0f}%")

    # Signal combination analysis
    print(f"\n[SIGNAL COMBINATION ANALYSIS]")
    print("-" * 80)

    signal_stats = {}
    for r in results:
        key = "+".join(sorted(r.signals))
        if key not in signal_stats:
            signal_stats[key] = {"count": 0, "avg_pf": 0, "avg_trades": 0, "best_pf": 0}
        signal_stats[key]["count"] += 1
        signal_stats[key]["avg_pf"] += r.profit_factor
        signal_stats[key]["avg_trades"] += r.trades
        signal_stats[key]["best_pf"] = max(signal_stats[key]["best_pf"], r.profit_factor)

    # Sort by average profit factor
    sorted_signals = sorted(
        signal_stats.items(),
        key=lambda x: x[1]["avg_pf"] / x[1]["count"],
        reverse=True
    )

    for signals, stats in sorted_signals[:5]:
        avg_pf = stats["avg_pf"] / stats["count"]
        avg_trades = stats["avg_trades"] / stats["count"]
        print(f"  {signals}: Avg PF={avg_pf:.2f}, Best PF={stats['best_pf']:.2f}, "
              f"Avg Trades={avg_trades:.0f}")

    # Recommendation
    print(f"\n[RECOMMENDATION]")
    print("=" * 80)

    if results:
        top = results[0]

        # Estimate annual trades
        if top.timeframe == "1h":
            annual_multiplier = 365 / 90
        elif top.timeframe == "15m":
            annual_multiplier = 365 / 30
        else:
            annual_multiplier = 365 / 14

        est_annual_trades = int(top.trades * annual_multiplier)
        est_annual_pnl = top.total_pnl_pct * annual_multiplier

        print(f"\n  OPTIMAL CONFIGURATION:")
        print(f"  ─────────────────────────────────────────")
        print(f"  Timeframe:     {top.timeframe}")
        print(f"  Signals:       {' + '.join(top.signals)}")
        print(f"  Stop Loss:     {top.stop_pct}%")
        print(f"  Take Profit:   {top.target_pct}%")
        print(f"  ─────────────────────────────────────────")
        print(f"  Win Rate:      {top.win_rate:.1f}%")
        print(f"  Profit Factor: {top.profit_factor:.2f}")
        print(f"  Max Drawdown:  {top.max_drawdown:.2f}%")
        print(f"  ─────────────────────────────────────────")
        print(f"  Est. Annual Trades:  {est_annual_trades}")
        print(f"  Est. Annual P&L:     {est_annual_pnl:+.1f}%")
        print(f"  ─────────────────────────────────────────")

        if top.win_rate >= 60 and top.profit_factor >= 1.5 and est_annual_trades >= 100:
            print(f"\n  ✓ HIGH CONFIDENCE - Ready for paper trading")
        elif top.win_rate >= 55 and top.profit_factor >= 1.2:
            print(f"\n  ~ MODERATE CONFIDENCE - Proceed with caution")
        else:
            print(f"\n  ✗ LOW CONFIDENCE - Needs more refinement")

    print("\n" + "=" * 80)


def print_comparison_report(all_results: Dict[str, List[OptimizationResult]]):
    """Compare results across multiple symbols."""

    print("\n" + "=" * 80)
    print("MULTI-SYMBOL COMPARISON")
    print("=" * 80)

    # Find best per symbol
    best_per_symbol = {}
    for symbol, results in all_results.items():
        if results:
            best_per_symbol[symbol] = results[0]

    if not best_per_symbol:
        print("\n[NO RESULTS] No symbols met minimum criteria")
        return

    # Rank symbols
    print(f"\n[SYMBOL RANKING]")
    print("-" * 80)
    print(f"{'Rank':<5} {'Symbol':<12} {'Best Config':<25} {'Trades':<8} "
          f"{'Win%':<7} {'PF':<6} {'Score':<8}")
    print("-" * 80)

    ranked = sorted(best_per_symbol.items(), key=lambda x: x[1].combined_score, reverse=True)

    for i, (symbol, r) in enumerate(ranked, 1):
        config = f"{r.timeframe} {'+'.join(r.signals)}"[:23]
        print(f"{i:<5} {symbol:<12} {config:<25} {r.trades:<8} "
              f"{r.win_rate:<7.1f} {r.profit_factor:<6.2f} {r.combined_score:<8.2f}")

    # Top overall
    print(f"\n[TOP SYMBOL]")
    print("-" * 80)
    top_symbol, top_result = ranked[0]
    print(f"  {top_symbol} with {'+'.join(top_result.signals)} on {top_result.timeframe}")
    print(f"  Win Rate: {top_result.win_rate:.1f}%, PF: {top_result.profit_factor:.2f}, "
          f"Trades: {top_result.trades}")

    print("\n" + "=" * 80)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Signal Optimizer - Find profitable configurations")
    parser.add_argument("--symbol", default="ATOMUSDT", help="Symbol to optimize")
    parser.add_argument("--deep", action="store_true", help="Deep optimization (more combinations)")
    parser.add_argument("--all-symbols", action="store_true", help="Test all symbols")
    parser.add_argument("--min-trades", type=int, default=10, help="Minimum trades required")
    parser.add_argument("--min-winrate", type=float, default=55.0, help="Minimum win rate")
    parser.add_argument("--min-pf", type=float, default=1.2, help="Minimum profit factor")

    args = parser.parse_args()

    print("[SSL] SSL verification disabled for optimization")
    print("\n" + "=" * 80)
    print("SIGNAL OPTIMIZER")
    print("=" * 80)
    print(f"  Goal: Find high-frequency, high-edge configurations")
    print(f"  Scoring: Edge × √Frequency (rewards both)")
    print("=" * 80)

    optimizer = SignalOptimizer()

    if args.all_symbols:
        # Test all symbols
        all_results = {}
        for symbol in TEST_SYMBOLS:
            print(f"\n{'─' * 40}")
            results = optimizer.optimize_symbol(symbol, deep=args.deep)
            filtered = optimizer.find_best(
                results,
                min_trades=args.min_trades,
                min_win_rate=args.min_winrate,
                min_profit_factor=args.min_pf
            )
            all_results[symbol] = filtered

            if filtered:
                print(f"  {symbol}: {len(filtered)} profitable configs found")
            else:
                print(f"  {symbol}: No profitable configs")

        print_comparison_report(all_results)

    else:
        # Single symbol
        results = optimizer.optimize_symbol(args.symbol, deep=args.deep)
        filtered = optimizer.find_best(
            results,
            min_trades=args.min_trades,
            min_win_rate=args.min_winrate,
            min_profit_factor=args.min_pf
        )
        optimizer.results = filtered
        print_optimization_report(filtered, args.symbol)

    print("\n[NEXT STEPS]")
    print("  1. Test recommended configuration with historical_replay.py")
    print("  2. Run paper trading with auto_executor.py")
    print("  3. Update production_config.py with optimal parameters")


if __name__ == "__main__":
    main()
