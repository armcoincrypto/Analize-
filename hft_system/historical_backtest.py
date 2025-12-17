"""
Historical Data Collector & Backtester
======================================
Fetches historical data from Binance and tests the HFT strategy.

This allows validating the strategy on past data before forward testing.
"""

import asyncio
import logging
import time
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass
import sqlite3
import numpy as np

from .config import SYSTEM_CONFIG, ASSETS, get_asset_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class HistoricalCandle:
    """Historical candlestick data."""
    symbol: str
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


class HistoricalCollector:
    """
    Fetches historical kline data from Binance.
    """

    BASE_URL = "https://api.binance.com/api/v3/klines"

    def __init__(self):
        self.data: Dict[str, List[HistoricalCandle]] = {}

    def fetch_klines(
        self,
        symbol: str,
        interval: str = "1m",
        days: int = 7,
        limit: int = 1000
    ) -> List[HistoricalCandle]:
        """
        Fetch historical klines for a symbol.

        Args:
            symbol: Trading pair (e.g., "ATOMUSDT")
            interval: Candle interval (1m, 5m, 15m, 1h, etc.)
            days: Number of days of history
            limit: Max candles per request (1000 max)
        """
        candles = []
        end_time = int(time.time() * 1000)
        start_time = end_time - (days * 24 * 60 * 60 * 1000)

        logger.info(f"Fetching {days} days of {interval} data for {symbol}...")

        while start_time < end_time:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": start_time,
                "limit": limit
            }

            try:
                response = requests.get(self.BASE_URL, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()

                if not data:
                    break

                for k in data:
                    candle = HistoricalCandle(
                        symbol=symbol,
                        timestamp=k[0],
                        open=float(k[1]),
                        high=float(k[2]),
                        low=float(k[3]),
                        close=float(k[4]),
                        volume=float(k[5])
                    )
                    candles.append(candle)

                # Move to next batch
                start_time = data[-1][0] + 1

                # Rate limit
                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
                break

        logger.info(f"  Fetched {len(candles)} candles for {symbol}")
        self.data[symbol] = candles
        return candles

    def fetch_all_assets(self, interval: str = "1m", days: int = 7):
        """Fetch data for all enabled assets."""
        for symbol in SYSTEM_CONFIG.enabled_assets:
            config = get_asset_config(symbol)
            self.fetch_klines(config.exchange_symbol, interval, days)

        return self.data


class HFTBacktester:
    """
    Backtests the HFT strategy on historical data.

    Simulates the 5-condition entry system and TP/SL/Time exits.
    """

    def __init__(self, data: Dict[str, List[HistoricalCandle]]):
        self.data = data
        self.trades = []
        self.capital = SYSTEM_CONFIG.initial_capital
        self.peak_capital = self.capital

    def calculate_rsi(self, prices: List[float], period: int = 14) -> Optional[float]:
        """Calculate RSI from price list."""
        if len(prices) < period + 1:
            return None

        deltas = np.diff(prices[-period - 1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def calculate_volume_spike(
        self,
        volumes: List[float],
        short_window: int = 5,
        long_window: int = 20
    ) -> Optional[float]:
        """Calculate volume spike ratio."""
        if len(volumes) < long_window:
            return None

        short_avg = np.mean(volumes[-short_window:])
        long_avg = np.mean(volumes[-long_window:])

        if long_avg == 0:
            return None

        return short_avg / long_avg

    def check_entry_conditions(
        self,
        candles: List[HistoricalCandle],
        idx: int,
        config
    ) -> Dict:
        """
        Check all 5 entry conditions at a point in time.

        Returns dict with condition results.
        """
        if idx < 20:  # Need history
            return {"conditions_met": 0, "details": {}}

        prices = [c.close for c in candles[max(0, idx-60):idx+1]]
        volumes = [c.volume for c in candles[max(0, idx-20):idx+1]]

        current = candles[idx]
        conditions = {}

        # 1. Price movement (drop in last few candles)
        if len(prices) >= 5:
            price_change = (prices[-1] - prices[-5]) / prices[-5] * 100
            conditions["price_drop"] = (
                price_change <= -config.min_price_drop_pct and
                price_change >= -config.max_price_drop_pct
            )
        else:
            conditions["price_drop"] = False

        # 2. Volume spike
        volume_spike = self.calculate_volume_spike(volumes)
        conditions["volume_spike"] = (
            volume_spike is not None and
            volume_spike >= config.volume_spike_multiplier
        )

        # 3. RSI oversold
        rsi = self.calculate_rsi(prices)
        conditions["rsi_oversold"] = rsi is not None and rsi <= config.rsi_oversold

        # 4. Candle pattern (proxy for orderbook - lower wick)
        candle_body = abs(current.close - current.open)
        lower_wick = min(current.open, current.close) - current.low
        conditions["buying_pressure"] = lower_wick > candle_body * 0.5

        # 5. Trend filter (price above 50-period MA)
        if len(prices) >= 50:
            ma50 = np.mean(prices[-50:])
            conditions["trend_filter"] = current.close < ma50 * 1.02  # Below MA (mean reversion)
        else:
            conditions["trend_filter"] = False

        conditions_met = sum(1 for v in conditions.values() if v)

        return {
            "conditions_met": conditions_met,
            "details": conditions,
            "rsi": rsi,
            "volume_spike": volume_spike
        }

    def simulate_trade(
        self,
        candles: List[HistoricalCandle],
        entry_idx: int,
        config
    ) -> Optional[Dict]:
        """
        Simulate a trade from entry point.

        Checks TP, SL, and time stop.
        """
        entry = candles[entry_idx]
        entry_price = entry.close

        # Position size (1% of capital)
        position_value = self.capital * 0.01
        quantity = position_value / entry_price

        # Simulate forward
        for i in range(entry_idx + 1, min(entry_idx + 100, len(candles))):
            current = candles[i]

            # Calculate P&L
            pnl_pct = (current.close - entry_price) / entry_price * 100

            # Check low for stop loss (intracandle)
            low_pnl = (current.low - entry_price) / entry_price * 100
            if low_pnl <= -config.stop_loss_pct:
                exit_price = entry_price * (1 - config.stop_loss_pct / 100)
                return {
                    "entry_time": entry.timestamp,
                    "exit_time": current.timestamp,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl_pct": -config.stop_loss_pct,
                    "pnl": quantity * (exit_price - entry_price),
                    "reason": "stop_loss",
                    "hold_candles": i - entry_idx
                }

            # Check high for take profit (intracandle)
            high_pnl = (current.high - entry_price) / entry_price * 100
            if high_pnl >= config.take_profit_pct:
                exit_price = entry_price * (1 + config.take_profit_pct / 100)
                return {
                    "entry_time": entry.timestamp,
                    "exit_time": current.timestamp,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl_pct": config.take_profit_pct,
                    "pnl": quantity * (exit_price - entry_price),
                    "reason": "take_profit",
                    "hold_candles": i - entry_idx
                }

            # Time stop (90 seconds = ~1.5 candles at 1m)
            hold_time = (current.timestamp - entry.timestamp) / 1000
            if hold_time >= config.time_stop_seconds:
                if pnl_pct < config.min_profit_for_time_check:
                    return {
                        "entry_time": entry.timestamp,
                        "exit_time": current.timestamp,
                        "entry_price": entry_price,
                        "exit_price": current.close,
                        "pnl_pct": pnl_pct,
                        "pnl": quantity * (current.close - entry_price),
                        "reason": "time_stop",
                        "hold_candles": i - entry_idx
                    }

        # If we get here, position still open (use last candle)
        last = candles[-1]
        pnl_pct = (last.close - entry_price) / entry_price * 100
        return {
            "entry_time": entry.timestamp,
            "exit_time": last.timestamp,
            "entry_price": entry_price,
            "exit_price": last.close,
            "pnl_pct": pnl_pct,
            "pnl": quantity * (last.close - entry_price),
            "reason": "end_of_data",
            "hold_candles": len(candles) - entry_idx - 1
        }

    def run_backtest(self, min_conditions: int = 3) -> Dict:
        """
        Run full backtest across all assets.
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"RUNNING BACKTEST (min {min_conditions}/5 conditions)")
        logger.info(f"{'='*60}")

        all_trades = []

        for symbol in SYSTEM_CONFIG.enabled_assets:
            config = get_asset_config(symbol)
            candles = self.data.get(config.exchange_symbol, [])

            if not candles:
                continue

            logger.info(f"\nProcessing {symbol} ({len(candles)} candles)...")

            last_trade_idx = -100  # Prevent immediate re-entry

            for i in range(20, len(candles) - 10):
                # Skip if too close to last trade
                if i - last_trade_idx < 30:
                    continue

                # Check entry conditions
                result = self.check_entry_conditions(candles, i, config)

                if result["conditions_met"] >= min_conditions:
                    # Simulate trade
                    trade = self.simulate_trade(candles, i, config)
                    if trade:
                        trade["symbol"] = symbol
                        trade["conditions_met"] = result["conditions_met"]
                        all_trades.append(trade)
                        last_trade_idx = i

                        # Update capital
                        self.capital += trade["pnl"]
                        if self.capital > self.peak_capital:
                            self.peak_capital = self.capital

        self.trades = all_trades
        return self.analyze_results()

    def analyze_results(self) -> Dict:
        """Analyze backtest results."""
        if not self.trades:
            return {"message": "No trades"}

        wins = [t for t in self.trades if t["pnl"] > 0]
        losses = [t for t in self.trades if t["pnl"] <= 0]

        total_pnl = sum(t["pnl"] for t in self.trades)

        # By exit reason
        by_reason = {}
        for t in self.trades:
            reason = t["reason"]
            if reason not in by_reason:
                by_reason[reason] = {"count": 0, "pnl": 0, "wins": 0}
            by_reason[reason]["count"] += 1
            by_reason[reason]["pnl"] += t["pnl"]
            if t["pnl"] > 0:
                by_reason[reason]["wins"] += 1

        # By symbol
        by_symbol = {}
        for t in self.trades:
            sym = t["symbol"]
            if sym not in by_symbol:
                by_symbol[sym] = {"count": 0, "pnl": 0, "wins": 0}
            by_symbol[sym]["count"] += 1
            by_symbol[sym]["pnl"] += t["pnl"]
            if t["pnl"] > 0:
                by_symbol[sym]["wins"] += 1

        results = {
            "total_trades": len(self.trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(self.trades) * 100 if self.trades else 0,
            "total_pnl": total_pnl,
            "avg_pnl": total_pnl / len(self.trades),
            "avg_win": sum(t["pnl"] for t in wins) / len(wins) if wins else 0,
            "avg_loss": sum(t["pnl"] for t in losses) / len(losses) if losses else 0,
            "profit_factor": abs(sum(t["pnl"] for t in wins)) / abs(sum(t["pnl"] for t in losses)) if losses and sum(t["pnl"] for t in losses) != 0 else float('inf'),
            "final_capital": self.capital,
            "max_drawdown": (self.peak_capital - min(self.capital, self.peak_capital)) / self.peak_capital * 100,
            "by_reason": by_reason,
            "by_symbol": by_symbol
        }

        # Print results
        logger.info(f"\n{'='*60}")
        logger.info("BACKTEST RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Total Trades: {results['total_trades']}")
        logger.info(f"Win Rate: {results['win_rate']:.1f}%")
        logger.info(f"Total PnL: ${results['total_pnl']:.2f}")
        logger.info(f"Profit Factor: {results['profit_factor']:.2f}")
        logger.info(f"Final Capital: ${results['final_capital']:.2f}")

        logger.info(f"\nBy Exit Reason:")
        for reason, stats in by_reason.items():
            wr = stats["wins"] / stats["count"] * 100 if stats["count"] else 0
            logger.info(f"  {reason}: {stats['count']} trades, ${stats['pnl']:.2f}, {wr:.0f}% WR")

        logger.info(f"\nBy Symbol:")
        for sym, stats in by_symbol.items():
            wr = stats["wins"] / stats["count"] * 100 if stats["count"] else 0
            logger.info(f"  {sym}: {stats['count']} trades, ${stats['pnl']:.2f}, {wr:.0f}% WR")

        logger.info(f"{'='*60}\n")

        return results


def run_historical_backtest(days: int = 7):
    """
    Main function to run historical backtest.

    Args:
        days: Number of days of history to fetch
    """
    # Collect data
    collector = HistoricalCollector()
    data = collector.fetch_all_assets(interval="1m", days=days)

    # Run backtest
    backtester = HFTBacktester(data)
    results = backtester.run_backtest(min_conditions=3)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HFT Historical Backtest")
    parser.add_argument("--days", type=int, default=7, help="Days of history")
    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════════════════════════╗
║              HFT HISTORICAL BACKTESTER                        ║
╚══════════════════════════════════════════════════════════════╝
    """)

    results = run_historical_backtest(days=args.days)
