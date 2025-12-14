#!/usr/bin/env python3
"""
LIVE PAPER TRADING BOT
======================
Phase A: Paper trading with REAL market signals.

This bot:
1. Monitors real 15-minute candles from Binance
2. Generates BB signals when price crosses lower band
3. Validates through full hierarchy (Gate, Config, Session)
4. Executes paper trades with stop/target tracking
5. Logs everything for analysis

Strategy (from optimizer):
- Timeframe: 15m
- Signal: Bollinger Bands (mean reversion)
- Entry: Price closes below lower BB
- Stop Loss: 1.5%
- Take Profit: 2.0%
- Target: ATOMUSDT only

Usage:
    python3 live_paper_trader.py                    # Run once (check for signals now)
    python3 live_paper_trader.py --watch            # Watch mode (continuous)
    python3 live_paper_trader.py --watch --interval 60   # Check every 60 seconds
    python3 live_paper_trader.py --status           # Show current positions

IMPORTANT: This is PAPER trading. No real money is used.
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

# Strategy parameters (from optimizer)
SYMBOL = "ATOMUSDT"
TIMEFRAME = "15m"
BB_PERIOD = 20
BB_STD_DEV = 2.0
STOP_LOSS_PCT = 1.5
TAKE_PROFIT_PCT = 2.0
POSITION_SIZE_PCT = 2.0  # 2% of capital per trade

# Paper trading capital
INITIAL_CAPITAL = 10000.0

# API
BINANCE_API = "https://api.binance.us/api/v3"


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
class Position:
    """Open paper position."""
    id: int
    symbol: str
    direction: str  # BUY or SELL
    entry_time: datetime
    entry_price: float
    size_usd: float
    size_qty: float
    stop_loss: float
    take_profit: float
    status: str  # OPEN, CLOSED
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl_usd: Optional[float] = None
    pnl_pct: Optional[float] = None


# =============================================================================
# DATABASE
# =============================================================================

class PaperTradingDB:
    """SQLite database for paper trading."""

    def __init__(self, db_path: str = "paper_trading.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Positions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                direction TEXT,
                entry_time TEXT,
                entry_price REAL,
                size_usd REAL,
                size_qty REAL,
                stop_loss REAL,
                take_profit REAL,
                status TEXT,
                exit_time TEXT,
                exit_price REAL,
                exit_reason TEXT,
                pnl_usd REAL,
                pnl_pct REAL
            )
        """)

        # Capital tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS capital (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                capital REAL,
                pnl_total REAL,
                trades_total INTEGER,
                wins INTEGER,
                losses INTEGER
            )
        """)

        # Signals log
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                direction TEXT,
                price REAL,
                bb_lower REAL,
                bb_upper REAL,
                action TEXT,
                reason TEXT
            )
        """)

        conn.commit()
        conn.close()

    def get_open_positions(self) -> List[Position]:
        """Get all open positions."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM positions WHERE status = 'OPEN'")
        rows = cursor.fetchall()
        conn.close()

        positions = []
        for row in rows:
            positions.append(Position(
                id=row[0],
                symbol=row[1],
                direction=row[2],
                entry_time=datetime.fromisoformat(row[3]),
                entry_price=row[4],
                size_usd=row[5],
                size_qty=row[6],
                stop_loss=row[7],
                take_profit=row[8],
                status=row[9]
            ))
        return positions

    def open_position(self, pos: Position) -> int:
        """Open a new position."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO positions
            (symbol, direction, entry_time, entry_price, size_usd, size_qty,
             stop_loss, take_profit, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            pos.symbol, pos.direction, pos.entry_time.isoformat(),
            pos.entry_price, pos.size_usd, pos.size_qty,
            pos.stop_loss, pos.take_profit, "OPEN"
        ))
        pos_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return pos_id

    def close_position(self, pos_id: int, exit_price: float, exit_reason: str):
        """Close a position."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get position
        cursor.execute("SELECT * FROM positions WHERE id = ?", (pos_id,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            return

        entry_price = row[4]
        size_usd = row[5]
        direction = row[2]

        # Calculate P&L
        if direction == "BUY":
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100
        else:
            pnl_pct = ((entry_price - exit_price) / entry_price) * 100

        pnl_usd = size_usd * (pnl_pct / 100)

        # Update position
        cursor.execute("""
            UPDATE positions SET
                status = 'CLOSED',
                exit_time = ?,
                exit_price = ?,
                exit_reason = ?,
                pnl_usd = ?,
                pnl_pct = ?
            WHERE id = ?
        """, (
            datetime.utcnow().isoformat(),
            exit_price,
            exit_reason,
            pnl_usd,
            pnl_pct,
            pos_id
        ))

        conn.commit()
        conn.close()

        return pnl_usd, pnl_pct

    def log_signal(self, symbol: str, direction: str, price: float,
                   bb_lower: float, bb_upper: float, action: str, reason: str):
        """Log a signal event."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO signals (timestamp, symbol, direction, price,
                                bb_lower, bb_upper, action, reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.utcnow().isoformat(),
            symbol, direction, price, bb_lower, bb_upper, action, reason
        ))
        conn.commit()
        conn.close()

    def get_stats(self) -> Dict:
        """Get trading statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Total trades
        cursor.execute("SELECT COUNT(*) FROM positions WHERE status = 'CLOSED'")
        total_trades = cursor.fetchone()[0]

        # Wins/losses
        cursor.execute("SELECT COUNT(*) FROM positions WHERE status = 'CLOSED' AND pnl_usd > 0")
        wins = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM positions WHERE status = 'CLOSED' AND pnl_usd <= 0")
        losses = cursor.fetchone()[0]

        # Total P&L
        cursor.execute("SELECT SUM(pnl_usd) FROM positions WHERE status = 'CLOSED'")
        total_pnl = cursor.fetchone()[0] or 0

        # Open positions
        cursor.execute("SELECT COUNT(*) FROM positions WHERE status = 'OPEN'")
        open_count = cursor.fetchone()[0]

        conn.close()

        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

        return {
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "open_positions": open_count
        }


# =============================================================================
# MARKET DATA
# =============================================================================

class MarketData:
    """Fetch real-time market data from Binance."""

    def __init__(self):
        self.session = requests.Session()
        self.session.verify = False

    def get_candles(self, symbol: str, interval: str, limit: int = 50) -> List[Candle]:
        """Fetch recent candles."""
        try:
            response = self.session.get(
                f"{BINANCE_API}/klines",
                params={
                    "symbol": symbol,
                    "interval": interval,
                    "limit": limit
                },
                timeout=10
            )

            if response.status_code != 200:
                print(f"[ERROR] API returned {response.status_code}")
                return []

            data = response.json()
            candles = []

            for kline in data:
                candles.append(Candle(
                    timestamp=datetime.fromtimestamp(kline[0] / 1000),
                    open=float(kline[1]),
                    high=float(kline[2]),
                    low=float(kline[3]),
                    close=float(kline[4]),
                    volume=float(kline[5])
                ))

            return candles

        except Exception as e:
            print(f"[ERROR] Failed to fetch candles: {e}")
            return []

    def get_price(self, symbol: str) -> Optional[float]:
        """Get current price."""
        try:
            response = self.session.get(
                f"{BINANCE_API}/ticker/price",
                params={"symbol": symbol},
                timeout=10
            )

            if response.status_code == 200:
                return float(response.json()["price"])
            return None

        except Exception as e:
            print(f"[ERROR] Failed to get price: {e}")
            return None


# =============================================================================
# SIGNAL GENERATOR
# =============================================================================

class BBSignalGenerator:
    """Generate Bollinger Band signals."""

    def __init__(self, period: int = 20, std_dev: float = 2.0):
        self.period = period
        self.std_dev = std_dev

    def calculate_bb(self, candles: List[Candle]) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands."""
        if len(candles) < self.period:
            return 0, 0, 0

        closes = [c.close for c in candles[-self.period:]]
        sma = sum(closes) / self.period
        variance = sum((x - sma) ** 2 for x in closes) / self.period
        std = variance ** 0.5

        lower = sma - (self.std_dev * std)
        upper = sma + (self.std_dev * std)

        return lower, sma, upper

    def check_signal(self, candles: List[Candle]) -> Tuple[Optional[str], Dict]:
        """
        Check for BB signal on the most recent CLOSED candle.

        Returns:
            (direction, details) or (None, details) if no signal
        """
        if len(candles) < self.period + 1:
            return None, {"reason": "Not enough candles"}

        # Use the second-to-last candle (most recent CLOSED)
        # The last candle is still forming
        signal_candle = candles[-2]
        lower, mid, upper = self.calculate_bb(candles[:-1])

        details = {
            "price": signal_candle.close,
            "bb_lower": lower,
            "bb_mid": mid,
            "bb_upper": upper,
            "candle_time": signal_candle.timestamp
        }

        # BUY signal: price closed below lower band
        if signal_candle.close < lower:
            return "BUY", details

        # SELL signal: price closed above upper band
        if signal_candle.close > upper:
            return "SELL", details

        return None, details


# =============================================================================
# TRADING LOGIC
# =============================================================================

class LivePaperTrader:
    """Live paper trading execution."""

    def __init__(self, capital: float = INITIAL_CAPITAL):
        self.capital = capital
        self.db = PaperTradingDB()
        self.market = MarketData()
        self.signal_gen = BBSignalGenerator(BB_PERIOD, BB_STD_DEV)

        # Try to load production config for hierarchy validation
        try:
            from production_config import ProductionConfig, is_trading_allowed
            self.config = ProductionConfig()
            self.check_trading_allowed = is_trading_allowed
            print("[CONFIG] Production config loaded")
        except ImportError:
            self.config = None
            self.check_trading_allowed = None
            print("[CONFIG] No production config - running without filters")

        # Try to load trading gate
        try:
            from trading_gate import TradingGate, GateDecision
            self.gate_class = TradingGate
            self.gate_decision = GateDecision
            print("[GATE] Trading gate loaded")
        except ImportError:
            self.gate_class = None
            print("[GATE] No trading gate - running without gate checks")

    def check_hierarchy(self) -> Tuple[bool, str]:
        """Check if trading is allowed by hierarchy."""

        # Check trading gate
        if self.gate_class:
            gate = self.gate_class(SYMBOL)
            status = gate.evaluate()
            if status.decision == self.gate_decision.CLOSED:
                return False, f"GATE CLOSED: {status.summary}"

        # Check production config time rules
        if self.check_trading_allowed:
            time_check = self.check_trading_allowed(self.config)
            if not time_check["allowed"]:
                return False, f"CONFIG: {time_check['reason']}"

        return True, "Hierarchy OK"

    def check_and_close_positions(self, current_price: float) -> List[str]:
        """Check open positions for stop/target hits."""
        messages = []
        positions = self.db.get_open_positions()

        for pos in positions:
            should_close = False
            reason = ""
            exit_price = current_price

            if pos.direction == "BUY":
                if current_price <= pos.stop_loss:
                    should_close = True
                    reason = "STOP_LOSS"
                    exit_price = pos.stop_loss
                elif current_price >= pos.take_profit:
                    should_close = True
                    reason = "TAKE_PROFIT"
                    exit_price = pos.take_profit
            else:  # SELL
                if current_price >= pos.stop_loss:
                    should_close = True
                    reason = "STOP_LOSS"
                    exit_price = pos.stop_loss
                elif current_price <= pos.take_profit:
                    should_close = True
                    reason = "TAKE_PROFIT"
                    exit_price = pos.take_profit

            if should_close:
                pnl_usd, pnl_pct = self.db.close_position(pos.id, exit_price, reason)
                self.capital += pnl_usd

                emoji = "✓" if pnl_usd > 0 else "✗"
                messages.append(
                    f"  [{emoji}] CLOSED {pos.direction} @ ${exit_price:.4f} "
                    f"({reason}) P&L: ${pnl_usd:+.2f} ({pnl_pct:+.2f}%)"
                )

        return messages

    def execute_signal(self, direction: str, price: float, details: Dict) -> Tuple[bool, str]:
        """Execute a paper trade."""

        # Check if we already have an open position
        open_positions = self.db.get_open_positions()
        if len(open_positions) > 0:
            return False, "Already have open position"

        # Calculate position size
        size_usd = self.capital * (POSITION_SIZE_PCT / 100)
        size_qty = size_usd / price

        # Calculate stop and target
        if direction == "BUY":
            stop_loss = price * (1 - STOP_LOSS_PCT / 100)
            take_profit = price * (1 + TAKE_PROFIT_PCT / 100)
        else:
            stop_loss = price * (1 + STOP_LOSS_PCT / 100)
            take_profit = price * (1 - TAKE_PROFIT_PCT / 100)

        # Create position
        pos = Position(
            id=0,
            symbol=SYMBOL,
            direction=direction,
            entry_time=datetime.utcnow(),
            entry_price=price,
            size_usd=size_usd,
            size_qty=size_qty,
            stop_loss=stop_loss,
            take_profit=take_profit,
            status="OPEN"
        )

        pos_id = self.db.open_position(pos)

        # Log signal
        self.db.log_signal(
            SYMBOL, direction, price,
            details["bb_lower"], details["bb_upper"],
            "EXECUTED", "Signal met all criteria"
        )

        return True, (
            f"OPENED {direction} @ ${price:.4f}\n"
            f"    Size: ${size_usd:.2f} ({size_qty:.4f} {SYMBOL[:-4]})\n"
            f"    Stop: ${stop_loss:.4f} (-{STOP_LOSS_PCT}%)\n"
            f"    Target: ${take_profit:.4f} (+{TAKE_PROFIT_PCT}%)"
        )

    def run_once(self) -> Dict:
        """Run one check cycle."""
        result = {
            "timestamp": datetime.utcnow().isoformat(),
            "signal": None,
            "action": None,
            "messages": []
        }

        # Get current price
        price = self.market.get_price(SYMBOL)
        if not price:
            result["messages"].append("[ERROR] Could not get price")
            return result

        result["price"] = price

        # Check and close positions
        close_msgs = self.check_and_close_positions(price)
        result["messages"].extend(close_msgs)

        # Get candles
        candles = self.market.get_candles(SYMBOL, TIMEFRAME, 50)
        if len(candles) < BB_PERIOD + 1:
            result["messages"].append("[ERROR] Not enough candles")
            return result

        # Check for signal
        direction, details = self.signal_gen.check_signal(candles)
        result["bb_lower"] = details.get("bb_lower", 0)
        result["bb_upper"] = details.get("bb_upper", 0)

        if direction:
            result["signal"] = direction

            # Check hierarchy
            allowed, reason = self.check_hierarchy()

            if not allowed:
                result["action"] = "BLOCKED"
                result["messages"].append(f"  [BLOCKED] {reason}")
                self.db.log_signal(
                    SYMBOL, direction, price,
                    details["bb_lower"], details["bb_upper"],
                    "BLOCKED", reason
                )
            else:
                # Execute trade
                success, msg = self.execute_signal(direction, price, details)
                if success:
                    result["action"] = "EXECUTED"
                    result["messages"].append(f"  [TRADE] {msg}")
                else:
                    result["action"] = "SKIPPED"
                    result["messages"].append(f"  [SKIP] {msg}")

        return result

    def print_status(self):
        """Print current status."""
        print("\n" + "=" * 60)
        print("LIVE PAPER TRADING STATUS")
        print("=" * 60)

        # Get stats
        stats = self.db.get_stats()

        print(f"\n[CAPITAL]")
        print(f"  Initial: ${INITIAL_CAPITAL:,.2f}")
        print(f"  Current: ${self.capital:,.2f}")
        print(f"  P&L: ${stats['total_pnl']:+,.2f} ({(stats['total_pnl']/INITIAL_CAPITAL)*100:+.2f}%)")

        print(f"\n[STATISTICS]")
        print(f"  Total Trades: {stats['total_trades']}")
        print(f"  Wins: {stats['wins']}")
        print(f"  Losses: {stats['losses']}")
        print(f"  Win Rate: {stats['win_rate']:.1f}%")

        print(f"\n[OPEN POSITIONS]")
        positions = self.db.get_open_positions()
        if positions:
            price = self.market.get_price(SYMBOL)
            for pos in positions:
                if pos.direction == "BUY":
                    unrealized = ((price - pos.entry_price) / pos.entry_price) * 100
                else:
                    unrealized = ((pos.entry_price - price) / pos.entry_price) * 100
                print(f"  {pos.symbol} {pos.direction} @ ${pos.entry_price:.4f}")
                print(f"    Size: ${pos.size_usd:.2f}")
                print(f"    Stop: ${pos.stop_loss:.4f} | Target: ${pos.take_profit:.4f}")
                print(f"    Current: ${price:.4f} ({unrealized:+.2f}%)")
        else:
            print("  No open positions")

        print(f"\n[STRATEGY]")
        print(f"  Symbol: {SYMBOL}")
        print(f"  Timeframe: {TIMEFRAME}")
        print(f"  Signal: Bollinger Bands ({BB_PERIOD}, {BB_STD_DEV}σ)")
        print(f"  Stop: {STOP_LOSS_PCT}% | Target: {TAKE_PROFIT_PCT}%")

        print("\n" + "=" * 60)

    def watch(self, interval_seconds: int = 60):
        """Watch mode - continuously monitor for signals."""
        print("\n" + "=" * 60)
        print("LIVE PAPER TRADING - WATCH MODE")
        print("=" * 60)
        print(f"  Symbol: {SYMBOL}")
        print(f"  Timeframe: {TIMEFRAME}")
        print(f"  Check Interval: {interval_seconds}s")
        print(f"  Strategy: BB Mean Reversion")
        print(f"  Stop: {STOP_LOSS_PCT}% | Target: {TAKE_PROFIT_PCT}%")
        print("=" * 60)
        print("\nPress Ctrl+C to stop\n")

        try:
            while True:
                now = datetime.utcnow()
                result = self.run_once()

                # Status line
                price = result.get("price", 0)
                bb_lower = result.get("bb_lower", 0)
                bb_upper = result.get("bb_upper", 0)

                status_char = "─"
                if price and bb_lower and bb_upper:
                    if price < bb_lower:
                        status_char = "▼"  # Below lower band
                    elif price > bb_upper:
                        status_char = "▲"  # Above upper band
                    else:
                        status_char = "●"  # Inside bands

                print(f"[{now.strftime('%H:%M:%S')}] {SYMBOL} ${price:.4f} "
                      f"[{bb_lower:.4f} {status_char} {bb_upper:.4f}]", end="")

                if result["signal"]:
                    print(f" → {result['signal']} signal!", end="")

                if result["action"]:
                    print(f" [{result['action']}]", end="")

                print()

                # Print any messages
                for msg in result["messages"]:
                    print(msg)

                # Wait for next check
                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            print("\n\n[STOPPED] Watch mode ended")
            self.print_status()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Live Paper Trading Bot")
    parser.add_argument("--watch", action="store_true", help="Continuous watch mode")
    parser.add_argument("--interval", type=int, default=60, help="Check interval in seconds")
    parser.add_argument("--status", action="store_true", help="Show current status")
    parser.add_argument("--capital", type=float, default=INITIAL_CAPITAL, help="Starting capital")

    args = parser.parse_args()

    print("[SSL] SSL verification disabled for API access")

    trader = LivePaperTrader(capital=args.capital)

    if args.status:
        trader.print_status()
    elif args.watch:
        trader.watch(interval_seconds=args.interval)
    else:
        # Run once
        print("\n" + "=" * 60)
        print("LIVE PAPER TRADING - SINGLE CHECK")
        print("=" * 60)

        result = trader.run_once()

        print(f"\n[{result['timestamp']}]")
        print(f"  Price: ${result.get('price', 0):.4f}")
        print(f"  BB Lower: ${result.get('bb_lower', 0):.4f}")
        print(f"  BB Upper: ${result.get('bb_upper', 0):.4f}")

        if result["signal"]:
            print(f"\n  SIGNAL: {result['signal']}")
            print(f"  ACTION: {result['action']}")
        else:
            print(f"\n  No signal (price inside bands)")

        for msg in result["messages"]:
            print(msg)

        print("\n" + "-" * 60)
        trader.print_status()

        print("\n[TIP] Run with --watch for continuous monitoring")
        print("      Run with --status to see positions only")


if __name__ == "__main__":
    main()
