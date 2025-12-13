#!/usr/bin/env python3
"""
HISTORICAL REPLAY ENGINE
========================
Phase 0 of the trading roadmap.

Replays historical market data through the full execution hierarchy
to collect statistically significant evidence before risking real capital.

Target: 1000-5000 historical trades for meaningful statistics.

Usage:
    python3 historical_replay.py --symbol ATOMUSDT --days 90
    python3 historical_replay.py --symbol ATOMUSDT --days 180 --analyze
"""

import os
import sys
import json
import sqlite3
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import time

# Disable SSL warnings for Binance
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
os.environ['PYTHONWARNINGS'] = 'ignore:Unverified HTTPS request'

print("[SSL] SSL verification disabled for historical data fetch")

# Try to import requests
try:
    import requests
except ImportError:
    print("[ERROR] requests library required: pip install requests")
    sys.exit(1)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Candle:
    """Single OHLCV candle."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def range_pct(self) -> float:
        """Candle range as percentage."""
        return ((self.high - self.low) / self.open) * 100 if self.open > 0 else 0

    @property
    def body_pct(self) -> float:
        """Candle body as percentage (direction agnostic)."""
        return abs(self.close - self.open) / self.open * 100 if self.open > 0 else 0

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open


@dataclass
class ReplaySignal:
    """Signal generated from historical data."""
    timestamp: datetime
    symbol: str
    direction: str  # BUY or SELL
    signals: List[str]
    price: float
    confidence: float
    candle: Candle

    # For outcome tracking
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    exit_price: float = 0.0
    exit_reason: str = ""
    pnl_pct: float = 0.0
    mfe_pct: float = 0.0  # Max favorable excursion
    mae_pct: float = 0.0  # Max adverse excursion


class ReplayResult(Enum):
    """Outcome of a replayed trade."""
    WIN = "win"
    LOSS = "loss"
    BREAKEVEN = "breakeven"
    BLOCKED = "blocked"


# =============================================================================
# HISTORICAL DATA FETCHER
# =============================================================================

class HistoricalDataFetcher:
    """
    Fetches historical candle data from Binance.
    Uses Binance US API for reliability.
    """

    BASE_URL = "https://api.binance.us/api/v3"

    def __init__(self):
        self.session = requests.Session()
        self.session.verify = False  # Disable SSL verification

    def fetch_candles(
        self,
        symbol: str,
        interval: str = "1h",
        days: int = 90,
        progress_callback=None
    ) -> List[Candle]:
        """
        Fetch historical candles.

        Args:
            symbol: Trading pair (e.g., ATOMUSDT)
            interval: Candle interval (1m, 5m, 15m, 1h, 4h, 1d)
            days: Number of days of history
            progress_callback: Optional callback for progress updates

        Returns:
            List of Candle objects
        """
        candles = []
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

        # Binance limit is 1000 candles per request
        limit = 1000
        current_start = start_time

        print(f"\n[FETCH] Downloading {days} days of {interval} candles for {symbol}...")

        request_count = 0
        while current_start < end_time:
            try:
                response = self.session.get(
                    f"{self.BASE_URL}/klines",
                    params={
                        "symbol": symbol,
                        "interval": interval,
                        "startTime": current_start,
                        "endTime": end_time,
                        "limit": limit
                    },
                    timeout=30
                )

                if response.status_code == 451:
                    print("[ERROR] Binance US geo-restricted. Try VPN or different region.")
                    break

                response.raise_for_status()
                data = response.json()

                if not data:
                    break

                for kline in data:
                    candle = Candle(
                        timestamp=datetime.fromtimestamp(kline[0] / 1000),
                        open=float(kline[1]),
                        high=float(kline[2]),
                        low=float(kline[3]),
                        close=float(kline[4]),
                        volume=float(kline[5])
                    )
                    candles.append(candle)

                # Move to next batch
                current_start = int(data[-1][0]) + 1
                request_count += 1

                if progress_callback:
                    progress_callback(len(candles))
                else:
                    print(f"  Fetched {len(candles)} candles...", end="\r")

                # Rate limiting
                time.sleep(0.1)

            except requests.exceptions.RequestException as e:
                print(f"\n[ERROR] Failed to fetch candles: {e}")
                break

        print(f"\n[FETCH] Downloaded {len(candles)} candles total")
        return candles

    def fetch_btc_reference(self, days: int = 90) -> Dict[datetime, float]:
        """Fetch BTC prices for regime/correlation analysis."""
        candles = self.fetch_candles("BTCUSDT", "1h", days)
        return {c.timestamp: c.close for c in candles}


# =============================================================================
# SIGNAL GENERATOR (from candles)
# =============================================================================

class SignalGenerator:
    """
    Generates trading signals from historical candles.
    Uses the same logic as the live analyzer.
    """

    def __init__(self, symbol: str):
        self.symbol = symbol

    def calculate_bollinger_bands(
        self,
        candles: List[Candle],
        period: int = 20,
        std_dev: float = 2.0
    ) -> Tuple[float, float, float]:
        """Calculate BB for the last candle."""
        if len(candles) < period:
            return 0, 0, 0

        closes = [c.close for c in candles[-period:]]
        sma = sum(closes) / period
        variance = sum((x - sma) ** 2 for x in closes) / period
        std = variance ** 0.5

        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)

        return lower, sma, upper

    def calculate_macd(
        self,
        candles: List[Candle],
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[float, float, float]:
        """Calculate MACD for the last candle."""
        if len(candles) < slow + signal:
            return 0, 0, 0

        closes = [c.close for c in candles]

        # EMA calculation
        def ema(data: List[float], period: int) -> List[float]:
            result = [data[0]]
            multiplier = 2 / (period + 1)
            for i in range(1, len(data)):
                result.append((data[i] - result[-1]) * multiplier + result[-1])
            return result

        ema_fast = ema(closes, fast)
        ema_slow = ema(closes, slow)

        macd_line = [f - s for f, s in zip(ema_fast, ema_slow)]
        signal_line = ema(macd_line[slow-1:], signal)

        if not signal_line:
            return 0, 0, 0

        macd_val = macd_line[-1]
        signal_val = signal_line[-1]
        histogram = macd_val - signal_val

        return macd_val, signal_val, histogram

    def calculate_rsi(self, candles: List[Candle], period: int = 14) -> float:
        """Calculate RSI for the last candle."""
        if len(candles) < period + 1:
            return 50

        closes = [c.close for c in candles[-(period+1):]]
        gains = []
        losses = []

        for i in range(1, len(closes)):
            change = closes[i] - closes[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))

        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period

        if avg_loss == 0:
            return 100

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def generate_signal(
        self,
        candles: List[Candle],
        index: int
    ) -> Optional[ReplaySignal]:
        """
        Generate a signal at a specific candle index.

        Returns signal if conditions met, None otherwise.
        """
        if index < 30:  # Need enough history for indicators
            return None

        history = candles[:index+1]
        current = candles[index]

        # Calculate indicators
        bb_lower, bb_mid, bb_upper = self.calculate_bollinger_bands(history)
        macd_val, signal_val, histogram = self.calculate_macd(history)
        rsi = self.calculate_rsi(history)

        signals = []
        direction = None
        confidence = 0.5

        # BB signal: price below lower band = oversold (BUY)
        if current.close < bb_lower:
            signals.append("BB")
            direction = "BUY"
            confidence += 0.15
        elif current.close > bb_upper:
            signals.append("BB")
            direction = "SELL"
            confidence += 0.15

        # MACD signal: histogram crossing
        if len(history) >= 2:
            prev_hist = self.calculate_macd(history[:-1])[2]
            if histogram > 0 and prev_hist <= 0:
                signals.append("MACD")
                if direction is None:
                    direction = "BUY"
                confidence += 0.15
            elif histogram < 0 and prev_hist >= 0:
                signals.append("MACD")
                if direction is None:
                    direction = "SELL"
                confidence += 0.15

        # RSI confirmation
        if rsi < 30 and direction == "BUY":
            signals.append("RSI")
            confidence += 0.10
        elif rsi > 70 and direction == "SELL":
            signals.append("RSI")
            confidence += 0.10

        # Need at least BB + MACD (as per production config)
        if "BB" not in signals or "MACD" not in signals:
            return None

        if direction is None:
            return None

        return ReplaySignal(
            timestamp=current.timestamp,
            symbol=self.symbol,
            direction=direction,
            signals=signals,
            price=current.close,
            confidence=min(confidence, 0.95),
            candle=current
        )


# =============================================================================
# REPLAY ENGINE
# =============================================================================

class ReplayEngine:
    """
    Replays historical signals through the execution hierarchy.
    Tracks outcomes for statistical analysis.
    """

    def __init__(
        self,
        symbol: str,
        initial_capital: float = 10000.0,
        stop_loss_pct: float = 2.0,
        take_profit_pct: float = 3.0
    ):
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        # Tracking
        self.trades: List[ReplaySignal] = []
        self.blocked_signals: List[Tuple[ReplaySignal, str]] = []

        # Statistics
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_capital = initial_capital
        self.consecutive_losses = 0
        self.max_consecutive_losses = 0

        # Import production config for validation
        try:
            from production_config import (
                PRODUCTION_CONFIG, validate_symbol, validate_signal, validate_time
            )
            self.config = PRODUCTION_CONFIG
            self.validate_symbol = validate_symbol
            self.validate_signal = validate_signal
            self.validate_time = validate_time
            print("[CONFIG] Production config loaded for replay validation")
        except ImportError:
            self.config = None
            print("[CONFIG] No production config - running without filters")

    def check_hierarchy(self, signal: ReplaySignal) -> Tuple[bool, str]:
        """
        Check if signal passes the full hierarchy.

        Returns:
            (allowed, reason)
        """
        if not self.config:
            return True, "No config - all signals allowed"

        # Check symbol
        sym_check = self.validate_symbol(signal.symbol, self.config)
        if not sym_check["valid"]:
            return False, f"SYMBOL: {sym_check['reason']}"

        # Check signals (need BB + MACD)
        sig_check = self.validate_signal(signal.signals, self.config)
        if not sig_check["valid"]:
            return False, f"SIGNAL: {sig_check['reason']}"

        # Check time
        time_check = self.validate_time(signal.timestamp, self.config)
        if not time_check["valid"]:
            return False, f"TIME: {time_check['reason']}"

        return True, "Passed all checks"

    def simulate_trade(
        self,
        signal: ReplaySignal,
        future_candles: List[Candle]
    ) -> ReplaySignal:
        """
        Simulate trade outcome using future candles.

        Args:
            signal: The entry signal
            future_candles: Candles after entry for outcome simulation

        Returns:
            Signal with outcome fields populated
        """
        entry = signal.price

        # Calculate stop and target
        if signal.direction == "BUY":
            stop_loss = entry * (1 - self.stop_loss_pct / 100)
            take_profit = entry * (1 + self.take_profit_pct / 100)
        else:
            stop_loss = entry * (1 + self.stop_loss_pct / 100)
            take_profit = entry * (1 - self.take_profit_pct / 100)

        signal.entry_price = entry
        signal.stop_loss = stop_loss
        signal.take_profit = take_profit

        # Track MFE/MAE
        max_favorable = 0.0
        max_adverse = 0.0

        # Simulate through future candles
        for candle in future_candles[:50]:  # Max 50 candles holding period
            if signal.direction == "BUY":
                # Check if stopped out (hit low)
                if candle.low <= stop_loss:
                    signal.exit_price = stop_loss
                    signal.exit_reason = "STOP_LOSS"
                    signal.pnl_pct = -self.stop_loss_pct
                    break

                # Check if target hit (hit high)
                if candle.high >= take_profit:
                    signal.exit_price = take_profit
                    signal.exit_reason = "TAKE_PROFIT"
                    signal.pnl_pct = self.take_profit_pct
                    break

                # Track excursions
                favorable = ((candle.high - entry) / entry) * 100
                adverse = ((entry - candle.low) / entry) * 100
                max_favorable = max(max_favorable, favorable)
                max_adverse = max(max_adverse, adverse)

            else:  # SELL
                # Check if stopped out (hit high)
                if candle.high >= stop_loss:
                    signal.exit_price = stop_loss
                    signal.exit_reason = "STOP_LOSS"
                    signal.pnl_pct = -self.stop_loss_pct
                    break

                # Check if target hit (hit low)
                if candle.low <= take_profit:
                    signal.exit_price = take_profit
                    signal.exit_reason = "TAKE_PROFIT"
                    signal.pnl_pct = self.take_profit_pct
                    break

                # Track excursions
                favorable = ((entry - candle.low) / entry) * 100
                adverse = ((candle.high - entry) / entry) * 100
                max_favorable = max(max_favorable, favorable)
                max_adverse = max(max_adverse, adverse)

        # If no exit after 50 candles, close at last price
        if signal.exit_price == 0:
            last_price = future_candles[-1].close if future_candles else entry
            signal.exit_price = last_price
            signal.exit_reason = "TIMEOUT"
            if signal.direction == "BUY":
                signal.pnl_pct = ((last_price - entry) / entry) * 100
            else:
                signal.pnl_pct = ((entry - last_price) / entry) * 100

        signal.mfe_pct = max_favorable
        signal.mae_pct = max_adverse

        return signal

    def process_signal(
        self,
        signal: ReplaySignal,
        future_candles: List[Candle]
    ) -> Tuple[bool, str, Optional[ReplaySignal]]:
        """
        Process a signal through hierarchy and simulate outcome.

        Returns:
            (executed, reason, completed_signal)
        """
        # Check hierarchy
        allowed, reason = self.check_hierarchy(signal)

        if not allowed:
            self.blocked_signals.append((signal, reason))
            return False, reason, None

        # Simulate trade
        completed = self.simulate_trade(signal, future_candles)
        self.trades.append(completed)

        # Update statistics
        if completed.pnl_pct > 0:
            self.wins += 1
            self.consecutive_losses = 0
        elif completed.pnl_pct < 0:
            self.losses += 1
            self.consecutive_losses += 1
            self.max_consecutive_losses = max(
                self.max_consecutive_losses,
                self.consecutive_losses
            )

        # Update capital
        position_size = self.current_capital * 0.02  # 2% position
        pnl_usd = position_size * (completed.pnl_pct / 100)
        self.current_capital += pnl_usd
        self.total_pnl += pnl_usd

        # Update drawdown
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital

        current_dd = ((self.peak_capital - self.current_capital) / self.peak_capital) * 100
        self.max_drawdown = max(self.max_drawdown, current_dd)

        return True, completed.exit_reason, completed

    def run_replay(self, candles: List[Candle]) -> Dict[str, Any]:
        """
        Run full replay on historical candles.

        Returns:
            Dictionary of statistics
        """
        print(f"\n[REPLAY] Starting replay on {len(candles)} candles...")

        generator = SignalGenerator(self.symbol)
        signals_generated = 0

        for i in range(30, len(candles) - 50):  # Need history and future
            signal = generator.generate_signal(candles, i)

            if signal:
                signals_generated += 1
                future = candles[i+1:i+51]
                executed, reason, result = self.process_signal(signal, future)

                # Progress update
                if signals_generated % 50 == 0:
                    print(f"  Processed {signals_generated} signals, "
                          f"{len(self.trades)} trades executed...")

        # Calculate final statistics
        total_trades = len(self.trades)
        win_rate = (self.wins / total_trades * 100) if total_trades > 0 else 0

        # Profit factor
        gross_profit = sum(t.pnl_pct for t in self.trades if t.pnl_pct > 0)
        gross_loss = abs(sum(t.pnl_pct for t in self.trades if t.pnl_pct < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Average MFE/MAE
        avg_mfe = sum(t.mfe_pct for t in self.trades) / total_trades if total_trades > 0 else 0
        avg_mae = sum(t.mae_pct for t in self.trades) / total_trades if total_trades > 0 else 0

        stats = {
            "symbol": self.symbol,
            "candles_processed": len(candles),
            "signals_generated": signals_generated,
            "signals_blocked": len(self.blocked_signals),
            "trades_executed": total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_pnl_pct": (self.total_pnl / self.initial_capital) * 100,
            "max_drawdown": self.max_drawdown,
            "max_consecutive_losses": self.max_consecutive_losses,
            "avg_mfe": avg_mfe,
            "avg_mae": avg_mae,
            "final_capital": self.current_capital,
            "initial_capital": self.initial_capital
        }

        return stats


# =============================================================================
# DATABASE STORAGE
# =============================================================================

class ReplayDatabase:
    """Store replay results for analysis."""

    def __init__(self, db_path: str = "replay_evidence.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Replay sessions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS replay_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                candles INTEGER,
                trades INTEGER,
                win_rate REAL,
                profit_factor REAL,
                max_drawdown REAL,
                total_pnl_pct REAL,
                config_hash TEXT
            )
        """)

        # Individual trades
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS replay_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                timestamp TEXT,
                symbol TEXT,
                direction TEXT,
                signals TEXT,
                entry_price REAL,
                exit_price REAL,
                stop_loss REAL,
                take_profit REAL,
                exit_reason TEXT,
                pnl_pct REAL,
                mfe_pct REAL,
                mae_pct REAL,
                FOREIGN KEY (session_id) REFERENCES replay_sessions(id)
            )
        """)

        # Blocked signals
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS replay_blocked (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                timestamp TEXT,
                symbol TEXT,
                direction TEXT,
                signals TEXT,
                block_reason TEXT,
                FOREIGN KEY (session_id) REFERENCES replay_sessions(id)
            )
        """)

        conn.commit()
        conn.close()

    def save_session(
        self,
        stats: Dict[str, Any],
        trades: List[ReplaySignal],
        blocked: List[Tuple[ReplaySignal, str]]
    ) -> int:
        """Save replay session to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Save session
        cursor.execute("""
            INSERT INTO replay_sessions
            (timestamp, symbol, candles, trades, win_rate, profit_factor,
             max_drawdown, total_pnl_pct, config_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            stats["symbol"],
            stats["candles_processed"],
            stats["trades_executed"],
            stats["win_rate"],
            stats["profit_factor"],
            stats["max_drawdown"],
            stats["total_pnl_pct"],
            "production_config_v1"
        ))

        session_id = cursor.lastrowid

        # Save trades
        for trade in trades:
            cursor.execute("""
                INSERT INTO replay_trades
                (session_id, timestamp, symbol, direction, signals, entry_price,
                 exit_price, stop_loss, take_profit, exit_reason, pnl_pct, mfe_pct, mae_pct)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                trade.timestamp.isoformat(),
                trade.symbol,
                trade.direction,
                json.dumps(trade.signals),
                trade.entry_price,
                trade.exit_price,
                trade.stop_loss,
                trade.take_profit,
                trade.exit_reason,
                trade.pnl_pct,
                trade.mfe_pct,
                trade.mae_pct
            ))

        # Save blocked signals
        for signal, reason in blocked:
            cursor.execute("""
                INSERT INTO replay_blocked
                (session_id, timestamp, symbol, direction, signals, block_reason)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                signal.timestamp.isoformat(),
                signal.symbol,
                signal.direction,
                json.dumps(signal.signals),
                reason
            ))

        conn.commit()
        conn.close()

        return session_id


# =============================================================================
# ANALYSIS & REPORTING
# =============================================================================

def print_replay_report(stats: Dict[str, Any], trades: List[ReplaySignal]):
    """Print comprehensive replay analysis."""

    print("\n" + "=" * 70)
    print("HISTORICAL REPLAY REPORT")
    print("=" * 70)

    print(f"\n[OVERVIEW]")
    print(f"  Symbol: {stats['symbol']}")
    print(f"  Candles Processed: {stats['candles_processed']:,}")
    print(f"  Signals Generated: {stats['signals_generated']:,}")
    print(f"  Signals Blocked: {stats['signals_blocked']:,} ({stats['signals_blocked']/max(1,stats['signals_generated'])*100:.1f}%)")
    print(f"  Trades Executed: {stats['trades_executed']:,}")

    print(f"\n[PERFORMANCE]")
    print(f"  Win Rate: {stats['win_rate']:.1f}%")
    print(f"  Profit Factor: {stats['profit_factor']:.2f}")
    print(f"  Total P&L: {stats['total_pnl_pct']:+.2f}%")
    print(f"  Max Drawdown: {stats['max_drawdown']:.2f}%")
    print(f"  Max Consecutive Losses: {stats['max_consecutive_losses']}")

    print(f"\n[CAPITAL]")
    print(f"  Initial: ${stats['initial_capital']:,.2f}")
    print(f"  Final: ${stats['final_capital']:,.2f}")
    print(f"  Growth: {((stats['final_capital']/stats['initial_capital'])-1)*100:+.2f}%")

    print(f"\n[EXIT QUALITY]")
    print(f"  Avg MFE (Max Favorable): {stats['avg_mfe']:.2f}%")
    print(f"  Avg MAE (Max Adverse): {stats['avg_mae']:.2f}%")
    print(f"  MFE/MAE Ratio: {stats['avg_mfe']/max(0.01, stats['avg_mae']):.2f}")

    # Exit reason breakdown
    if trades:
        exit_reasons = {}
        for t in trades:
            exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

        print(f"\n[EXIT REASONS]")
        for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
            pct = count / len(trades) * 100
            print(f"  {reason}: {count} ({pct:.1f}%)")

    # Time analysis
    if trades:
        hourly_performance = {}
        for t in trades:
            hour = t.timestamp.hour
            if hour not in hourly_performance:
                hourly_performance[hour] = {"wins": 0, "losses": 0, "pnl": 0}
            if t.pnl_pct > 0:
                hourly_performance[hour]["wins"] += 1
            else:
                hourly_performance[hour]["losses"] += 1
            hourly_performance[hour]["pnl"] += t.pnl_pct

        print(f"\n[HOURLY BREAKDOWN]")
        print(f"  {'Hour':<6} {'Trades':<8} {'Win%':<8} {'P&L':<10}")
        print(f"  {'-'*32}")
        for hour in sorted(hourly_performance.keys()):
            data = hourly_performance[hour]
            total = data["wins"] + data["losses"]
            wr = data["wins"] / total * 100 if total > 0 else 0
            print(f"  {hour:02d}:00  {total:<8} {wr:<8.1f} {data['pnl']:+.2f}%")

    # Day of week analysis
    if trades:
        daily_performance = {}
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        for t in trades:
            day = days[t.timestamp.weekday()]
            if day not in daily_performance:
                daily_performance[day] = {"wins": 0, "losses": 0, "pnl": 0}
            if t.pnl_pct > 0:
                daily_performance[day]["wins"] += 1
            else:
                daily_performance[day]["losses"] += 1
            daily_performance[day]["pnl"] += t.pnl_pct

        print(f"\n[DAILY BREAKDOWN]")
        print(f"  {'Day':<6} {'Trades':<8} {'Win%':<8} {'P&L':<10}")
        print(f"  {'-'*32}")
        for day in days:
            if day in daily_performance:
                data = daily_performance[day]
                total = data["wins"] + data["losses"]
                wr = data["wins"] / total * 100 if total > 0 else 0
                print(f"  {day:<6} {total:<8} {wr:<8.1f} {data['pnl']:+.2f}%")

    # Verdict
    print(f"\n[VERDICT]")
    if stats['win_rate'] >= 60 and stats['profit_factor'] >= 1.2 and stats['max_drawdown'] <= 15:
        print("  ✓ READY FOR PAPER TRADING")
        print("  Strategy shows statistical edge with controlled risk.")
    elif stats['win_rate'] >= 55 and stats['profit_factor'] >= 1.0:
        print("  ~ MARGINAL - Needs refinement")
        print("  Edge exists but risk parameters need adjustment.")
    else:
        print("  ✗ NOT READY - Needs significant work")
        print("  Insufficient edge or excessive risk.")

    print("\n" + "=" * 70)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Historical Replay Engine - Phase 0 Validation"
    )
    parser.add_argument(
        "--symbol",
        default="ATOMUSDT",
        help="Symbol to replay (default: ATOMUSDT)"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Days of history to fetch (default: 90)"
    )
    parser.add_argument(
        "--interval",
        default="1h",
        help="Candle interval (default: 1h)"
    )
    parser.add_argument(
        "--stop",
        type=float,
        default=2.0,
        help="Stop loss percentage (default: 2.0)"
    )
    parser.add_argument(
        "--target",
        type=float,
        default=3.0,
        help="Take profit percentage (default: 3.0)"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Show detailed analysis"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results to database"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("HISTORICAL REPLAY ENGINE")
    print("=" * 70)
    print(f"  Symbol: {args.symbol}")
    print(f"  Days: {args.days}")
    print(f"  Interval: {args.interval}")
    print(f"  Stop Loss: {args.stop}%")
    print(f"  Take Profit: {args.target}%")
    print("=" * 70)

    # Fetch historical data
    fetcher = HistoricalDataFetcher()
    candles = fetcher.fetch_candles(
        symbol=args.symbol,
        interval=args.interval,
        days=args.days
    )

    if len(candles) < 100:
        print("[ERROR] Not enough candles fetched. Check API access.")
        return

    # Run replay
    engine = ReplayEngine(
        symbol=args.symbol,
        stop_loss_pct=args.stop,
        take_profit_pct=args.target
    )

    stats = engine.run_replay(candles)

    # Print report
    print_replay_report(stats, engine.trades)

    # Save to database
    if args.save:
        db = ReplayDatabase()
        session_id = db.save_session(stats, engine.trades, engine.blocked_signals)
        print(f"\n[SAVED] Results stored in replay_evidence.db (session {session_id})")

    print("\n[NEXT STEPS]")
    print("  1. Run with different symbols: --symbol SOLUSDT")
    print("  2. Try longer history: --days 180")
    print("  3. Save for comparison: --save")
    print("  4. After validation, proceed to Phase 1 (Paper Trading)")


if __name__ == "__main__":
    main()
