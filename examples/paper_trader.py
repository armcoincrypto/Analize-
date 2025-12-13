"""
Paper Trading Simulator - Test strategies in real-time without real money.

Connects to live price feeds and simulates trades with the optimized strategy.
Tracks performance, logs all trades, and provides real-time statistics.
"""

import sqlite3
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path
import time
import json
from typing import Optional
from analize.features.indicators import TechnicalIndicators

# Import quant database for trade persistence
try:
    from quant_database import QuantDatabase, Trade as DBTrade
    HAS_QUANT_DB = True
except ImportError:
    HAS_QUANT_DB = False


# Configuration
DB_PATH = Path(__file__).parent.parent / "data" / "market_data.db"
TRADES_LOG = Path(__file__).parent.parent / "data" / "paper_trades.json"
QUANT_DB_PATH = Path(__file__).parent / "quant_signals.db"

# Trading costs
TRADING_FEE_PCT = 0.1
SLIPPAGE_PCT = 0.05
SPREAD_PCT = 0.02


@dataclass
class Position:
    """Current trading position."""
    is_open: bool = False
    entry_price: float = 0.0
    entry_time: datetime = None
    quantity: float = 0.0
    side: str = ""  # "long" or "short"


@dataclass
class Trade:
    """Completed trade record."""
    trade_id: int
    symbol: str
    side: str
    entry_price: float
    entry_time: datetime
    exit_price: float
    exit_time: datetime
    quantity: float
    gross_pnl: float
    fees: float
    net_pnl: float
    net_pnl_pct: float
    strategy: str
    params: dict


@dataclass
class PaperTrader:
    """Paper trading simulator."""
    symbol: str = "XRPUSDT"
    initial_capital: float = 1000.0
    capital: float = 1000.0
    position: Position = field(default_factory=Position)
    trades: list = field(default_factory=list)
    trade_counter: int = 0

    # Strategy parameters (best from optimization)
    strategy_name: str = "Stochastic"
    strategy_params: dict = field(default_factory=lambda: {
        "k_period": 21,
        "d_period": 3,
        "buy_threshold": 20,
        "sell_threshold": 75,
    })

    # Data buffer for indicators
    price_buffer: list = field(default_factory=list)
    buffer_size: int = 100

    # Quant database for trade persistence
    quant_db: Optional[any] = None

    def __post_init__(self):
        """Initialize quant database if available."""
        if HAS_QUANT_DB:
            try:
                self.quant_db = QuantDatabase(str(QUANT_DB_PATH))
                print(f"  [DB] Connected to quant database: {QUANT_DB_PATH}")
            except Exception as e:
                print(f"  [WARN] Could not connect to quant DB: {e}")

    def fetch_current_price(self) -> Optional[dict]:
        """Fetch current price from Binance."""
        urls = [
            f"https://api.binance.com/api/v3/ticker/price?symbol={self.symbol}",
            f"https://api.binance.us/api/v3/ticker/price?symbol={self.symbol}",
        ]

        for url in urls:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    return {"price": float(data["price"]), "time": datetime.now()}
            except:
                continue
        return None

    def fetch_recent_candles(self, limit: int = 100) -> pd.DataFrame:
        """Fetch recent candles for indicator calculation."""
        urls = [
            "https://api.binance.us/api/v3/klines",  # US first (works in geo-restricted regions)
            "https://api.binance.com/api/v3/klines",
        ]

        params = {
            "symbol": self.symbol,
            "interval": "1h",
            "limit": limit
        }

        for url in urls:
            try:
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data:
                        df = pd.DataFrame(data, columns=[
                            "timestamp", "open", "high", "low", "close", "volume",
                            "close_time", "quote_volume", "trades", "taker_buy_base",
                            "taker_buy_quote", "ignore"
                        ])
                        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                        for col in ["open", "high", "low", "close", "volume"]:
                            df[col] = df[col].astype(float)
                        return df[["timestamp", "open", "high", "low", "close", "volume"]]
            except:
                continue
        return pd.DataFrame()

    def calculate_signal(self, df: pd.DataFrame) -> str:
        """Calculate trading signal based on strategy."""
        if len(df) < 30:
            return "HOLD"

        k, d = TechnicalIndicators.stochastic(
            df["high"], df["low"], df["close"],
            k_period=self.strategy_params["k_period"],
            d_period=self.strategy_params["d_period"]
        )

        current_k = k.iloc[-1]
        if pd.isna(current_k):
            return "HOLD"

        buy_threshold = self.strategy_params["buy_threshold"]
        sell_threshold = self.strategy_params["sell_threshold"]

        if current_k < buy_threshold:
            return "BUY"
        elif current_k > sell_threshold:
            return "SELL"
        return "HOLD"

    def calculate_fees(self, price: float, quantity: float) -> float:
        """Calculate trading fees for a trade."""
        value = price * quantity
        fee = value * (TRADING_FEE_PCT / 100)
        slippage = value * (SLIPPAGE_PCT / 100)
        spread = value * (SPREAD_PCT / 100)
        return fee + slippage + spread

    def open_position(self, price: float, signal: str):
        """Open a new position."""
        # Use 95% of capital (keep some for fees)
        available = self.capital * 0.95
        quantity = available / price

        entry_fees = self.calculate_fees(price, quantity)

        self.position = Position(
            is_open=True,
            entry_price=price,
            entry_time=datetime.now(),
            quantity=quantity,
            side="long" if signal == "BUY" else "short"
        )

        self.capital -= entry_fees

        print(f"\n{'='*50}")
        print(f"📈 OPENED POSITION")
        print(f"{'='*50}")
        print(f"  Time:     {self.position.entry_time}")
        print(f"  Side:     {self.position.side.upper()}")
        print(f"  Price:    ${price:.4f}")
        print(f"  Quantity: {quantity:.4f}")
        print(f"  Value:    ${price * quantity:.2f}")
        print(f"  Fees:     ${entry_fees:.4f}")

    def close_position(self, price: float):
        """Close current position."""
        if not self.position.is_open:
            return

        self.trade_counter += 1

        # Calculate P&L
        if self.position.side == "long":
            gross_pnl = (price - self.position.entry_price) * self.position.quantity
        else:
            gross_pnl = (self.position.entry_price - price) * self.position.quantity

        exit_fees = self.calculate_fees(price, self.position.quantity)
        net_pnl = gross_pnl - exit_fees

        entry_value = self.position.entry_price * self.position.quantity
        net_pnl_pct = (net_pnl / entry_value) * 100

        # Update capital
        self.capital += (entry_value + net_pnl)

        # Record trade
        trade = Trade(
            trade_id=self.trade_counter,
            symbol=self.symbol,
            side=self.position.side,
            entry_price=self.position.entry_price,
            entry_time=self.position.entry_time,
            exit_price=price,
            exit_time=datetime.now(),
            quantity=self.position.quantity,
            gross_pnl=gross_pnl,
            fees=exit_fees,
            net_pnl=net_pnl,
            net_pnl_pct=net_pnl_pct,
            strategy=self.strategy_name,
            params=self.strategy_params,
        )
        self.trades.append(trade)

        # Save to quant database for metrics analysis
        if self.quant_db and HAS_QUANT_DB:
            try:
                # Determine outcome
                if net_pnl_pct > 0.5:
                    outcome = "WIN"
                elif net_pnl_pct < -0.5:
                    outcome = "LOSS"
                else:
                    outcome = "BREAKEVEN"

                db_trade = DBTrade(
                    symbol=self.symbol,
                    entry_timestamp=self.position.entry_time.isoformat() + "Z",
                    entry_price=self.position.entry_price,
                    position_size=entry_value,
                    direction="LONG" if self.position.side == "long" else "SHORT",
                    exit_timestamp=datetime.now().isoformat() + "Z",
                    exit_price=price,
                    pnl_percent=net_pnl_pct,
                    pnl_usd=net_pnl,
                    outcome=outcome,
                    signals_used=json.dumps([self.strategy_name]),
                    notes=f"Paper trade via {self.strategy_name}"
                )
                self.quant_db.insert_trade(db_trade)
                print(f"  [DB] Trade saved to database")
            except Exception as e:
                print(f"  [WARN] Could not save to DB: {e}")

        result = "WIN ✓" if net_pnl > 0 else "LOSS ✗"

        print(f"\n{'='*50}")
        print(f"📉 CLOSED POSITION - {result}")
        print(f"{'='*50}")
        print(f"  Entry:     ${self.position.entry_price:.4f} @ {self.position.entry_time}")
        print(f"  Exit:      ${price:.4f} @ {datetime.now()}")
        print(f"  Gross P&L: ${gross_pnl:+.4f}")
        print(f"  Fees:      ${exit_fees:.4f}")
        print(f"  NET P&L:   ${net_pnl:+.4f} ({net_pnl_pct:+.2f}%)")
        print(f"  Capital:   ${self.capital:.2f}")

        # Reset position
        self.position = Position()

    def get_statistics(self) -> dict:
        """Calculate trading statistics."""
        if not self.trades:
            return {}

        wins = [t for t in self.trades if t.net_pnl > 0]
        losses = [t for t in self.trades if t.net_pnl <= 0]

        total_profit = sum(t.net_pnl for t in self.trades)
        total_fees = sum(t.fees for t in self.trades)

        return {
            "total_trades": len(self.trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(self.trades) * 100 if self.trades else 0,
            "total_profit": total_profit,
            "total_fees": total_fees,
            "avg_win": sum(t.net_pnl for t in wins) / len(wins) if wins else 0,
            "avg_loss": sum(t.net_pnl for t in losses) / len(losses) if losses else 0,
            "capital_return": (self.capital - self.initial_capital) / self.initial_capital * 100,
        }

    def save_trades(self):
        """Save trades to JSON file."""
        TRADES_LOG.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "symbol": self.symbol,
            "strategy": self.strategy_name,
            "params": self.strategy_params,
            "initial_capital": self.initial_capital,
            "current_capital": self.capital,
            "trades": [
                {
                    "trade_id": t.trade_id,
                    "symbol": t.symbol,
                    "side": t.side,
                    "entry_price": t.entry_price,
                    "entry_time": t.entry_time.isoformat(),
                    "exit_price": t.exit_price,
                    "exit_time": t.exit_time.isoformat(),
                    "quantity": t.quantity,
                    "gross_pnl": t.gross_pnl,
                    "fees": t.fees,
                    "net_pnl": t.net_pnl,
                    "net_pnl_pct": t.net_pnl_pct,
                }
                for t in self.trades
            ],
            "statistics": self.get_statistics(),
            "last_updated": datetime.now().isoformat(),
        }

        with open(TRADES_LOG, "w") as f:
            json.dump(data, f, indent=2)

    def print_status(self, current_price: float, signal: str, k_value: float):
        """Print current status."""
        stats = self.get_statistics()

        print(f"\n{'─'*60}")
        print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  {self.symbol}: ${current_price:.4f}")
        print(f"  Stochastic K: {k_value:.1f}")
        print(f"  Signal: {signal}")
        print(f"{'─'*60}")
        print(f"  Capital: ${self.capital:.2f} ({(self.capital/self.initial_capital-1)*100:+.2f}%)")
        print(f"  Position: {'OPEN' if self.position.is_open else 'NONE'}")
        if self.position.is_open:
            unrealized = (current_price - self.position.entry_price) * self.position.quantity
            print(f"  Unrealized P&L: ${unrealized:+.2f}")
        print(f"  Trades: {stats.get('total_trades', 0)} | "
              f"Win Rate: {stats.get('win_rate', 0):.1f}%")


def run_paper_trading(duration_minutes: int = 60, check_interval: int = 60,
                      buy_threshold: int = 20, sell_threshold: int = 75,
                      symbol: str = "XRPUSDT"):
    """Run paper trading simulation."""
    print("=" * 70)
    print("PAPER TRADING SIMULATOR")
    if buy_threshold != 20 or sell_threshold != 75:
        print("⚠️  DEMO MODE - Using tight thresholds for testing")
    print("=" * 70)

    trader = PaperTrader(
        symbol=symbol,
        initial_capital=1000.0,
        strategy_name="Stochastic",
        strategy_params={
            "k_period": 21,
            "d_period": 3,
            "buy_threshold": buy_threshold,
            "sell_threshold": sell_threshold,
        }
    )

    print(f"\n  Symbol:          {trader.symbol}")
    print(f"  Strategy:        {trader.strategy_name}")
    print(f"  Parameters:      {trader.strategy_params}")
    print(f"  Initial Capital: ${trader.initial_capital:.2f}")
    print(f"  Duration:        {duration_minutes} minutes")
    print(f"  Check Interval:  {check_interval} seconds")

    print("\n" + "=" * 70)
    print("Starting paper trading... (Press Ctrl+C to stop)")
    print("=" * 70)

    start_time = datetime.now()
    end_time = start_time + timedelta(minutes=duration_minutes)

    try:
        while datetime.now() < end_time:
            # Fetch data
            df = trader.fetch_recent_candles(100)

            if df.empty:
                print("  ⚠️  Could not fetch data, retrying...")
                time.sleep(check_interval)
                continue

            current_price = df["close"].iloc[-1]

            # Calculate indicator
            k, d = TechnicalIndicators.stochastic(
                df["high"], df["low"], df["close"],
                k_period=trader.strategy_params["k_period"],
                d_period=trader.strategy_params["d_period"]
            )
            k_value = k.iloc[-1] if not pd.isna(k.iloc[-1]) else 50

            # Get signal
            signal = trader.calculate_signal(df)

            # Execute trades - support both LONG and SHORT positions
            if signal == "BUY":
                if not trader.position.is_open:
                    # Open new LONG position
                    trader.open_position(current_price, signal)
                elif trader.position.side == "short":
                    # Close SHORT and open LONG (reversal)
                    trader.close_position(current_price)
                    trader.open_position(current_price, signal)
            elif signal == "SELL":
                if not trader.position.is_open:
                    # Open new SHORT position
                    trader.open_position(current_price, signal)
                elif trader.position.side == "long":
                    # Close LONG and open SHORT (reversal)
                    trader.close_position(current_price)
                    trader.open_position(current_price, signal)

            # Print status
            trader.print_status(current_price, signal, k_value)

            # Save trades
            trader.save_trades()

            # Wait for next check
            time.sleep(check_interval)

    except KeyboardInterrupt:
        print("\n\n⚠️  Paper trading stopped by user")

    # Final summary
    print("\n" + "=" * 70)
    print("PAPER TRADING SESSION COMPLETE")
    print("=" * 70)

    # Close any open position
    if trader.position.is_open:
        df = trader.fetch_recent_candles(10)
        if not df.empty:
            trader.close_position(df["close"].iloc[-1])

    stats = trader.get_statistics()

    print(f"""
  Session Duration: {(datetime.now() - start_time).total_seconds() / 60:.1f} minutes

  RESULTS:
  ─────────────────────────────────────
  Initial Capital:  ${trader.initial_capital:.2f}
  Final Capital:    ${trader.capital:.2f}
  Total Return:     {stats.get('capital_return', 0):+.2f}%

  Total Trades:     {stats.get('total_trades', 0)}
  Wins:             {stats.get('wins', 0)}
  Losses:           {stats.get('losses', 0)}
  Win Rate:         {stats.get('win_rate', 0):.1f}%

  Total Profit:     ${stats.get('total_profit', 0):+.2f}
  Total Fees:       ${stats.get('total_fees', 0):.2f}
  Avg Win:          ${stats.get('avg_win', 0):+.2f}
  Avg Loss:         ${stats.get('avg_loss', 0):.2f}

  Trade log saved to: {TRADES_LOG}
""")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Paper Trading Simulator")
    parser.add_argument("--symbol", default="XRPUSDT", help="Trading symbol (e.g., BTCUSDT, ETHUSDT, ATOMUSDT)")
    parser.add_argument("--demo", action="store_true", help="Use demo mode with tight thresholds")
    args = parser.parse_args()

    symbol = args.symbol.upper()
    if not symbol.endswith("USDT"):
        symbol = symbol + "USDT"

    print(f"\nSelected symbol: {symbol}")
    print("\nPAPER TRADING OPTIONS:")
    print("─" * 40)
    print("1. Quick test (5 minutes, 30s intervals)")
    print("2. Short session (30 minutes, 1m intervals)")
    print("3. Full session (4 hours, 5m intervals)")
    print("4. Custom")
    print("5. DEMO MODE (tight thresholds for testing)")

    # Auto-select demo mode if --demo flag passed
    if args.demo:
        choice = "5"
        print("\nAuto-selecting DEMO MODE (--demo flag)")
    else:
        try:
            choice = input("\nSelect option (1-5): ").strip()
        except:
            choice = "1"

    if choice == "1":
        run_paper_trading(duration_minutes=5, check_interval=30, symbol=symbol)
    elif choice == "2":
        run_paper_trading(duration_minutes=30, check_interval=60, symbol=symbol)
    elif choice == "3":
        run_paper_trading(duration_minutes=240, check_interval=300, symbol=symbol)
    elif choice == "5":
        # Demo mode with tight thresholds to generate trades
        run_paper_trading(
            duration_minutes=5,
            check_interval=30,
            buy_threshold=40,    # Normal: 20
            sell_threshold=60,   # Normal: 75
            symbol=symbol
        )
    else:
        try:
            mins = int(input("Duration (minutes): "))
            interval = int(input("Check interval (seconds): "))
            run_paper_trading(duration_minutes=mins, check_interval=interval, symbol=symbol)
        except:
            run_paper_trading(duration_minutes=5, check_interval=30, symbol=symbol)


if __name__ == "__main__":
    main()
