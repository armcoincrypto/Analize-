#!/usr/bin/env python3
"""
MULTI-STRATEGY PAPER TRADER
============================
Runs multiple strategies simultaneously for more signals.

Strategies:
1. BB Mean Reversion (ATOMUSDT 15m) - Original
2. BB Mean Reversion (DOTUSDT 15m) - Secondary
3. RSI Oversold (ATOMUSDT 15m) - High frequency
4. BB + RSI Combo (ATOMUSDT 15m) - Higher confidence

Usage:
    python3 multi_strategy_trader.py --watch
    python3 multi_strategy_trader.py --status
"""

import os
import sys
import json
import sqlite3
import argparse
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import time

# Disable SSL warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
os.environ['PYTHONWARNINGS'] = 'ignore:Unverified HTTPS request'

try:
    import requests
except ImportError:
    print("[ERROR] pip install requests")
    sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================

STRATEGIES = [
    {
        "name": "BB_ATOM_15m",
        "symbol": "ATOMUSDT",
        "timeframe": "15m",
        "signal_type": "BB",
        "stop_pct": 1.5,
        "target_pct": 2.0,
        "enabled": True
    },
    {
        "name": "BB_DOT_15m",
        "symbol": "DOTUSDT",
        "timeframe": "15m",
        "signal_type": "BB",
        "stop_pct": 1.5,
        "target_pct": 2.0,
        "enabled": True
    },
    {
        "name": "RSI_ATOM_15m",
        "symbol": "ATOMUSDT",
        "timeframe": "15m",
        "signal_type": "RSI",
        "stop_pct": 2.0,
        "target_pct": 3.0,
        "enabled": True
    },
    {
        "name": "BB_RSI_ATOM_15m",
        "symbol": "ATOMUSDT",
        "timeframe": "15m",
        "signal_type": "BB_RSI",
        "stop_pct": 1.5,
        "target_pct": 2.5,
        "enabled": True
    },
]

INITIAL_CAPITAL = 10000.0
POSITION_SIZE_PCT = 2.0
BINANCE_API = "https://api.binance.us/api/v3"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Signal:
    strategy: str
    symbol: str
    direction: str
    price: float
    indicator_value: float
    threshold: float
    timestamp: datetime


# =============================================================================
# DATABASE
# =============================================================================

class MultiStrategyDB:
    def __init__(self, db_path: str = "multi_strategy.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy TEXT,
                symbol TEXT,
                direction TEXT,
                entry_time TEXT,
                entry_price REAL,
                size_usd REAL,
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

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                strategy TEXT,
                symbol TEXT,
                direction TEXT,
                price REAL,
                indicator_value REAL,
                threshold REAL,
                action TEXT,
                reason TEXT
            )
        """)

        conn.commit()
        conn.close()

    def get_open_positions(self, strategy: str = None) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if strategy:
            cursor.execute("SELECT * FROM positions WHERE status='OPEN' AND strategy=?", (strategy,))
        else:
            cursor.execute("SELECT * FROM positions WHERE status='OPEN'")

        rows = cursor.fetchall()
        conn.close()

        positions = []
        for row in rows:
            positions.append({
                "id": row[0], "strategy": row[1], "symbol": row[2],
                "direction": row[3], "entry_time": row[4], "entry_price": row[5],
                "size_usd": row[6], "stop_loss": row[7], "take_profit": row[8],
                "status": row[9]
            })
        return positions

    def open_position(self, strategy: str, symbol: str, direction: str,
                      entry_price: float, size_usd: float, stop_loss: float, take_profit: float) -> int:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO positions (strategy, symbol, direction, entry_time, entry_price,
                                   size_usd, stop_loss, take_profit, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')
        """, (strategy, symbol, direction, datetime.now(timezone.utc).isoformat(),
              entry_price, size_usd, stop_loss, take_profit))
        pos_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return pos_id

    def close_position(self, pos_id: int, exit_price: float, exit_reason: str) -> Tuple[float, float]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT entry_price, size_usd, direction FROM positions WHERE id=?", (pos_id,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            return 0, 0

        entry_price, size_usd, direction = row

        if direction == "BUY":
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100
        else:
            pnl_pct = ((entry_price - exit_price) / entry_price) * 100

        pnl_usd = size_usd * (pnl_pct / 100)

        cursor.execute("""
            UPDATE positions SET status='CLOSED', exit_time=?, exit_price=?,
                                 exit_reason=?, pnl_usd=?, pnl_pct=?
            WHERE id=?
        """, (datetime.now(timezone.utc).isoformat(), exit_price, exit_reason, pnl_usd, pnl_pct, pos_id))

        conn.commit()
        conn.close()
        return pnl_usd, pnl_pct

    def log_signal(self, strategy: str, symbol: str, direction: str, price: float,
                   indicator_value: float, threshold: float, action: str, reason: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO signals (timestamp, strategy, symbol, direction, price,
                                indicator_value, threshold, action, reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (datetime.now(timezone.utc).isoformat(), strategy, symbol, direction,
              price, indicator_value, threshold, action, reason))
        conn.commit()
        conn.close()

    def get_stats(self) -> Dict:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM positions WHERE status='CLOSED'")
        total = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM positions WHERE status='CLOSED' AND pnl_usd > 0")
        wins = cursor.fetchone()[0]

        cursor.execute("SELECT SUM(pnl_usd) FROM positions WHERE status='CLOSED'")
        total_pnl = cursor.fetchone()[0] or 0

        cursor.execute("SELECT COUNT(*) FROM positions WHERE status='OPEN'")
        open_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM signals")
        signal_count = cursor.fetchone()[0]

        # Per strategy stats
        cursor.execute("""
            SELECT strategy, COUNT(*), SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END), SUM(pnl_usd)
            FROM positions WHERE status='CLOSED' GROUP BY strategy
        """)
        strategy_stats = {}
        for row in cursor.fetchall():
            strategy_stats[row[0]] = {"trades": row[1], "wins": row[2], "pnl": row[3] or 0}

        conn.close()

        return {
            "total_trades": total,
            "wins": wins,
            "losses": total - wins,
            "win_rate": (wins / total * 100) if total > 0 else 0,
            "total_pnl": total_pnl,
            "open_positions": open_count,
            "signals_detected": signal_count,
            "strategy_stats": strategy_stats
        }


# =============================================================================
# MARKET DATA
# =============================================================================

class MarketData:
    def __init__(self):
        self.session = requests.Session()
        self.session.verify = False
        self.cache = {}

    def get_candles(self, symbol: str, interval: str, limit: int = 50) -> List[Candle]:
        try:
            response = self.session.get(
                f"{BINANCE_API}/klines",
                params={"symbol": symbol, "interval": interval, "limit": limit},
                timeout=10
            )
            if response.status_code != 200:
                return []

            candles = []
            for k in response.json():
                candles.append(Candle(
                    timestamp=datetime.fromtimestamp(k[0]/1000, tz=timezone.utc),
                    open=float(k[1]), high=float(k[2]), low=float(k[3]),
                    close=float(k[4]), volume=float(k[5])
                ))
            return candles
        except:
            return []

    def get_price(self, symbol: str) -> Optional[float]:
        try:
            response = self.session.get(
                f"{BINANCE_API}/ticker/price",
                params={"symbol": symbol},
                timeout=10
            )
            if response.status_code == 200:
                return float(response.json()["price"])
        except:
            pass
        return None


# =============================================================================
# SIGNAL GENERATORS
# =============================================================================

class SignalGenerator:
    @staticmethod
    def calculate_bb(candles: List[Candle], period: int = 20, std: float = 2.0) -> Tuple[float, float, float]:
        if len(candles) < period:
            return 0, 0, 0
        closes = [c.close for c in candles[-period:]]
        sma = sum(closes) / period
        variance = sum((x - sma) ** 2 for x in closes) / period
        std_dev = variance ** 0.5
        return sma - (std * std_dev), sma, sma + (std * std_dev)

    @staticmethod
    def calculate_rsi(candles: List[Candle], period: int = 14) -> float:
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

    @staticmethod
    def check_bb_signal(candles: List[Candle]) -> Tuple[Optional[str], float, float]:
        """Returns (direction, price, bb_threshold)"""
        if len(candles) < 22:
            return None, 0, 0

        signal_candle = candles[-2]  # Last closed candle
        lower, mid, upper = SignalGenerator.calculate_bb(candles[:-1])

        if signal_candle.close < lower:
            return "BUY", signal_candle.close, lower
        elif signal_candle.close > upper:
            return "SELL", signal_candle.close, upper
        return None, signal_candle.close, lower

    @staticmethod
    def check_rsi_signal(candles: List[Candle]) -> Tuple[Optional[str], float, float]:
        """Returns (direction, rsi_value, threshold)"""
        if len(candles) < 16:
            return None, 0, 0

        rsi = SignalGenerator.calculate_rsi(candles[:-1])

        if rsi < 30:
            return "BUY", rsi, 30
        elif rsi > 70:
            return "SELL", rsi, 70
        return None, rsi, 30

    @staticmethod
    def check_bb_rsi_signal(candles: List[Candle]) -> Tuple[Optional[str], float, float]:
        """BB + RSI combo for higher confidence"""
        bb_dir, price, bb_thresh = SignalGenerator.check_bb_signal(candles)
        rsi_dir, rsi_val, rsi_thresh = SignalGenerator.check_rsi_signal(candles)

        if bb_dir == "BUY" and rsi_val < 35:
            return "BUY", rsi_val, 35
        elif bb_dir == "SELL" and rsi_val > 65:
            return "SELL", rsi_val, 65
        return None, rsi_val, 35


# =============================================================================
# MULTI-STRATEGY TRADER
# =============================================================================

class MultiStrategyTrader:
    def __init__(self, capital: float = INITIAL_CAPITAL):
        self.capital = capital
        self.db = MultiStrategyDB()
        self.market = MarketData()
        self.strategies = [s for s in STRATEGIES if s["enabled"]]
        print(f"[INIT] {len(self.strategies)} strategies enabled")

    def check_positions(self):
        """Check all open positions for stop/target"""
        messages = []
        positions = self.db.get_open_positions()

        for pos in positions:
            price = self.market.get_price(pos["symbol"])
            if not price:
                continue

            should_close = False
            reason = ""
            exit_price = price

            if pos["direction"] == "BUY":
                if price <= pos["stop_loss"]:
                    should_close, reason, exit_price = True, "STOP_LOSS", pos["stop_loss"]
                elif price >= pos["take_profit"]:
                    should_close, reason, exit_price = True, "TAKE_PROFIT", pos["take_profit"]
            else:
                if price >= pos["stop_loss"]:
                    should_close, reason, exit_price = True, "STOP_LOSS", pos["stop_loss"]
                elif price <= pos["take_profit"]:
                    should_close, reason, exit_price = True, "TAKE_PROFIT", pos["take_profit"]

            if should_close:
                pnl_usd, pnl_pct = self.db.close_position(pos["id"], exit_price, reason)
                self.capital += pnl_usd
                emoji = "✓" if pnl_usd > 0 else "✗"
                messages.append(f"  [{emoji}] {pos['strategy']}: CLOSED @ ${exit_price:.4f} ({reason}) P&L: ${pnl_usd:+.2f}")

        return messages

    def check_strategy(self, strategy: Dict) -> Optional[Dict]:
        """Check one strategy for signals"""
        candles = self.market.get_candles(strategy["symbol"], strategy["timeframe"], 50)
        if len(candles) < 25:
            return None

        # Check if already have position in this strategy
        open_pos = self.db.get_open_positions(strategy["name"])
        if open_pos:
            return None

        # Get signal based on type
        signal_type = strategy["signal_type"]

        if signal_type == "BB":
            direction, value, threshold = SignalGenerator.check_bb_signal(candles)
        elif signal_type == "RSI":
            direction, value, threshold = SignalGenerator.check_rsi_signal(candles)
        elif signal_type == "BB_RSI":
            direction, value, threshold = SignalGenerator.check_bb_rsi_signal(candles)
        else:
            return None

        if direction:
            return {
                "strategy": strategy["name"],
                "symbol": strategy["symbol"],
                "direction": direction,
                "price": self.market.get_price(strategy["symbol"]),
                "indicator_value": value,
                "threshold": threshold,
                "stop_pct": strategy["stop_pct"],
                "target_pct": strategy["target_pct"]
            }
        return None

    def execute_signal(self, signal: Dict) -> Tuple[bool, str]:
        """Execute a paper trade"""
        price = signal["price"]
        if not price:
            return False, "No price"

        size_usd = self.capital * (POSITION_SIZE_PCT / 100)

        if signal["direction"] == "BUY":
            stop = price * (1 - signal["stop_pct"] / 100)
            target = price * (1 + signal["target_pct"] / 100)
        else:
            stop = price * (1 + signal["stop_pct"] / 100)
            target = price * (1 - signal["target_pct"] / 100)

        self.db.open_position(
            signal["strategy"], signal["symbol"], signal["direction"],
            price, size_usd, stop, target
        )

        self.db.log_signal(
            signal["strategy"], signal["symbol"], signal["direction"],
            price, signal["indicator_value"], signal["threshold"],
            "EXECUTED", "Signal met criteria"
        )

        return True, f"{signal['direction']} @ ${price:.4f} | Stop: ${stop:.4f} | Target: ${target:.4f}"

    def run_once(self) -> List[str]:
        """Run one check cycle"""
        messages = []

        # Check positions
        close_msgs = self.check_positions()
        messages.extend(close_msgs)

        # Check each strategy
        for strategy in self.strategies:
            signal = self.check_strategy(strategy)
            if signal:
                success, msg = self.execute_signal(signal)
                if success:
                    messages.append(f"  [TRADE] {signal['strategy']}: {msg}")
                    self.db.log_signal(
                        signal["strategy"], signal["symbol"], signal["direction"],
                        signal["price"], signal["indicator_value"], signal["threshold"],
                        "EXECUTED", msg
                    )
                else:
                    self.db.log_signal(
                        signal["strategy"], signal["symbol"], signal["direction"],
                        signal["price"] or 0, signal["indicator_value"], signal["threshold"],
                        "SKIPPED", msg
                    )

        return messages

    def print_status(self):
        """Print current status"""
        stats = self.db.get_stats()

        print("\n" + "=" * 70)
        print("MULTI-STRATEGY PAPER TRADING STATUS")
        print("=" * 70)

        print(f"\n[CAPITAL]")
        print(f"  Initial: ${INITIAL_CAPITAL:,.2f}")
        print(f"  Current: ${self.capital:,.2f}")
        print(f"  P&L: ${stats['total_pnl']:+,.2f} ({(stats['total_pnl']/INITIAL_CAPITAL)*100:+.2f}%)")

        print(f"\n[OVERALL STATISTICS]")
        print(f"  Total Trades: {stats['total_trades']}")
        print(f"  Wins: {stats['wins']} | Losses: {stats['losses']}")
        print(f"  Win Rate: {stats['win_rate']:.1f}%")
        print(f"  Signals Detected: {stats['signals_detected']}")
        print(f"  Open Positions: {stats['open_positions']}")

        print(f"\n[PER-STRATEGY PERFORMANCE]")
        print(f"  {'Strategy':<20} {'Trades':<8} {'Wins':<6} {'P&L':<12}")
        print(f"  {'-'*46}")
        for name, data in stats.get('strategy_stats', {}).items():
            wr = (data['wins']/data['trades']*100) if data['trades'] > 0 else 0
            print(f"  {name:<20} {data['trades']:<8} {data['wins']:<6} ${data['pnl']:+.2f}")

        print(f"\n[ACTIVE STRATEGIES]")
        for s in self.strategies:
            print(f"  • {s['name']}: {s['symbol']} {s['timeframe']} ({s['signal_type']})")

        print(f"\n[OPEN POSITIONS]")
        positions = self.db.get_open_positions()
        if positions:
            for pos in positions:
                price = self.market.get_price(pos["symbol"]) or 0
                if pos["direction"] == "BUY":
                    unrealized = ((price - pos["entry_price"]) / pos["entry_price"]) * 100
                else:
                    unrealized = ((pos["entry_price"] - price) / pos["entry_price"]) * 100
                print(f"  {pos['strategy']}: {pos['direction']} @ ${pos['entry_price']:.4f} → ${price:.4f} ({unrealized:+.2f}%)")
        else:
            print("  No open positions")

        print("\n" + "=" * 70)

    def watch(self, interval: int = 60):
        """Watch mode - continuous monitoring"""
        print("\n" + "=" * 70)
        print("MULTI-STRATEGY PAPER TRADING - WATCH MODE")
        print("=" * 70)
        print(f"  Strategies: {len(self.strategies)}")
        for s in self.strategies:
            print(f"    • {s['name']}: {s['symbol']} ({s['signal_type']})")
        print(f"  Check Interval: {interval}s")
        print("=" * 70)
        print("\nPress Ctrl+C to stop\n")

        try:
            while True:
                now = datetime.now(timezone.utc)
                messages = self.run_once()

                # Status line
                prices = {}
                for s in self.strategies:
                    if s["symbol"] not in prices:
                        prices[s["symbol"]] = self.market.get_price(s["symbol"]) or 0

                price_str = " | ".join([f"{sym}: ${p:.4f}" for sym, p in prices.items()])
                print(f"[{now.strftime('%H:%M:%S')}] {price_str}")

                for msg in messages:
                    print(msg)

                time.sleep(interval)

        except KeyboardInterrupt:
            print("\n\n[STOPPED]")
            self.print_status()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Multi-Strategy Paper Trader")
    parser.add_argument("--watch", action="store_true", help="Watch mode")
    parser.add_argument("--interval", type=int, default=60, help="Check interval")
    parser.add_argument("--status", action="store_true", help="Show status")

    args = parser.parse_args()

    print("[SSL] SSL verification disabled")

    trader = MultiStrategyTrader()

    if args.status:
        trader.print_status()
    elif args.watch:
        trader.watch(args.interval)
    else:
        print("\n[SINGLE CHECK]")
        messages = trader.run_once()
        for msg in messages:
            print(msg)
        if not messages:
            print("  No signals detected")
        trader.print_status()


if __name__ == "__main__":
    main()
