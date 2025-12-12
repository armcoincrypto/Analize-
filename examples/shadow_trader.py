#!/usr/bin/env python3
"""
Shadow Live Mode - Validate Strategies Without Real Execution

Shadow mode sends signals through the entire live pipeline but only
simulates trade execution. This allows 2-4 weeks of validation before
any real money deployment.

Features:
1. Full signal generation from live data
2. Risk manager validation
3. Simulated order execution with realistic fills
4. Performance tracking vs theoretical
5. Drift detection between shadow and backtest
6. Automatic alerting on anomalies

Author: Cloud AI Analyzer
"""

import json
import requests
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from enum import Enum
import sqlite3


# =============================================================================
# CONFIGURATION
# =============================================================================

SHADOW_DB_PATH = Path(__file__).parent / "shadow_trades.db"


class ShadowMode(Enum):
    """Shadow execution modes."""
    FULL_SHADOW = "full_shadow"        # No real orders at all
    PARTIAL_SHADOW = "partial_shadow"  # Small real orders for testing
    LIVE = "live"                       # Real execution


@dataclass
class ShadowOrder:
    """Shadow order record."""
    order_id: str
    timestamp: datetime
    symbol: str
    direction: str          # LONG or SHORT
    size_usd: float
    entry_price: float
    stop_loss: float
    take_profit: Optional[float]
    signals_used: List[str]
    risk_assessment: Dict
    status: str = "OPEN"    # OPEN, FILLED, STOPPED, TARGET, CANCELLED
    exit_price: Optional[float] = None
    exit_timestamp: Optional[datetime] = None
    pnl_usd: Optional[float] = None
    pnl_pct: Optional[float] = None
    slippage_pct: float = 0.0
    fees_usd: float = 0.0


@dataclass
class ShadowPerformance:
    """Shadow trading performance metrics."""
    total_orders: int = 0
    filled_orders: int = 0
    stopped_orders: int = 0
    target_orders: int = 0
    cancelled_orders: int = 0
    total_pnl_usd: float = 0.0
    avg_pnl_pct: float = 0.0
    win_rate: float = 0.0
    avg_slippage: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    backtest_pnl: float = 0.0
    shadow_vs_backtest_drift: float = 0.0


# =============================================================================
# SHADOW TRADER CLASS
# =============================================================================

class ShadowTrader:
    """
    Shadow trading system for strategy validation.

    Runs the full trading pipeline without executing real orders.
    Tracks performance and compares against backtest expectations.
    """

    def __init__(
        self,
        mode: ShadowMode = ShadowMode.FULL_SHADOW,
        db_path: str = None,
        max_drift_pct: float = 25.0
    ):
        """
        Initialize shadow trader.

        Args:
            mode: Shadow execution mode
            db_path: Database path for shadow trades
            max_drift_pct: Max allowed drift from backtest before alerting
        """
        self.mode = mode
        self.db_path = db_path or str(SHADOW_DB_PATH)
        self.max_drift_pct = max_drift_pct

        self.orders: Dict[str, ShadowOrder] = {}
        self.closed_orders: List[ShadowOrder] = []
        self.order_counter = 0

        # Performance tracking
        self.daily_pnl: List[float] = []
        self.backtest_expectations: Dict[str, float] = {}

        self._init_database()

        print("=" * 60)
        print(f"SHADOW TRADER INITIALIZED")
        print("=" * 60)
        print(f"  Mode:           {mode.value}")
        print(f"  Database:       {self.db_path}")
        print(f"  Max Drift:      {max_drift_pct}%")
        print("=" * 60)

    def _init_database(self):
        """Initialize shadow trades database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS shadow_orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id TEXT UNIQUE NOT NULL,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                size_usd REAL NOT NULL,
                entry_price REAL NOT NULL,
                stop_loss REAL NOT NULL,
                take_profit REAL,
                signals_used TEXT,
                risk_assessment TEXT,
                status TEXT DEFAULT 'OPEN',
                exit_price REAL,
                exit_timestamp TEXT,
                pnl_usd REAL,
                pnl_pct REAL,
                slippage_pct REAL DEFAULT 0,
                fees_usd REAL DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS shadow_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT UNIQUE NOT NULL,
                total_orders INTEGER DEFAULT 0,
                total_pnl_usd REAL DEFAULT 0,
                win_rate REAL DEFAULT 0,
                backtest_pnl REAL DEFAULT 0,
                drift_pct REAL DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_shadow_symbol_status
            ON shadow_orders(symbol, status)
        """)

        conn.commit()
        conn.close()

    # =========================================================================
    # ORDER MANAGEMENT
    # =========================================================================

    def create_shadow_order(
        self,
        symbol: str,
        direction: str,
        size_usd: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float = None,
        signals_used: List[str] = None,
        risk_assessment: Dict = None
    ) -> ShadowOrder:
        """
        Create a shadow order (simulated, not executed).

        Args:
            symbol: Trading symbol
            direction: LONG or SHORT
            size_usd: Position size in USD
            entry_price: Entry price
            stop_loss: Stop-loss price
            take_profit: Take-profit price (optional)
            signals_used: List of signal types that generated this order
            risk_assessment: Risk assessment data

        Returns:
            Created shadow order
        """
        self.order_counter += 1
        order_id = f"SHADOW-{datetime.now().strftime('%Y%m%d%H%M%S')}-{self.order_counter:04d}"

        # Simulate slippage (0.05% - 0.15%)
        import random
        slippage_pct = random.uniform(0.05, 0.15)

        if direction == "LONG":
            actual_entry = entry_price * (1 + slippage_pct / 100)
        else:
            actual_entry = entry_price * (1 - slippage_pct / 100)

        # Simulate fees (0.1% trading fee)
        fees_usd = size_usd * 0.001

        order = ShadowOrder(
            order_id=order_id,
            timestamp=datetime.now(),
            symbol=symbol,
            direction=direction,
            size_usd=size_usd,
            entry_price=actual_entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            signals_used=signals_used or [],
            risk_assessment=risk_assessment or {},
            slippage_pct=slippage_pct,
            fees_usd=fees_usd,
        )

        self.orders[order_id] = order
        self._save_order(order)

        print(f"\n{'='*50}")
        print(f"📊 SHADOW ORDER CREATED")
        print(f"{'='*50}")
        print(f"  Order ID:   {order_id}")
        print(f"  Symbol:     {symbol}")
        print(f"  Direction:  {direction}")
        print(f"  Size:       ${size_usd:,.2f}")
        print(f"  Entry:      ${actual_entry:.4f} (slippage: {slippage_pct:.2f}%)")
        print(f"  Stop-Loss:  ${stop_loss:.4f}")
        print(f"  Signals:    {', '.join(signals_used or [])}")

        return order

    def check_orders(self) -> List[ShadowOrder]:
        """
        Check all open orders against current prices.

        Returns:
            List of orders that were closed
        """
        closed = []

        for order_id, order in list(self.orders.items()):
            if order.status != "OPEN":
                continue

            current_price = self._fetch_price(order.symbol)
            if current_price is None:
                continue

            closed_order = self._check_order_exit(order, current_price)
            if closed_order:
                closed.append(closed_order)

        return closed

    def _check_order_exit(self, order: ShadowOrder, current_price: float) -> Optional[ShadowOrder]:
        """Check if order should be closed."""

        exit_reason = None
        exit_price = None

        if order.direction == "LONG":
            # Check stop-loss
            if current_price <= order.stop_loss:
                exit_reason = "STOPPED"
                exit_price = order.stop_loss * 0.998  # Slippage on stop

            # Check take-profit
            elif order.take_profit and current_price >= order.take_profit:
                exit_reason = "TARGET"
                exit_price = order.take_profit

        else:  # SHORT
            # Check stop-loss
            if current_price >= order.stop_loss:
                exit_reason = "STOPPED"
                exit_price = order.stop_loss * 1.002  # Slippage on stop

            # Check take-profit
            elif order.take_profit and current_price <= order.take_profit:
                exit_reason = "TARGET"
                exit_price = order.take_profit

        if exit_reason:
            return self._close_order(order, exit_price, exit_reason)

        return None

    def _close_order(
        self,
        order: ShadowOrder,
        exit_price: float,
        reason: str
    ) -> ShadowOrder:
        """Close an order and calculate P&L."""

        # Calculate P&L
        if order.direction == "LONG":
            pnl_pct = ((exit_price - order.entry_price) / order.entry_price) * 100
        else:
            pnl_pct = ((order.entry_price - exit_price) / order.entry_price) * 100

        pnl_usd = order.size_usd * (pnl_pct / 100) - order.fees_usd

        # Update order
        order.status = reason
        order.exit_price = exit_price
        order.exit_timestamp = datetime.now()
        order.pnl_usd = pnl_usd
        order.pnl_pct = pnl_pct

        # Move to closed orders
        self.closed_orders.append(order)
        del self.orders[order.order_id]

        # Save to database
        self._update_order(order)

        # Track daily P&L
        self.daily_pnl.append(pnl_usd)

        result = "WIN ✓" if pnl_usd > 0 else "LOSS ✗"

        print(f"\n{'='*50}")
        print(f"📊 SHADOW ORDER CLOSED - {result}")
        print(f"{'='*50}")
        print(f"  Order ID:   {order.order_id}")
        print(f"  Reason:     {reason}")
        print(f"  Exit:       ${exit_price:.4f}")
        print(f"  P&L:        ${pnl_usd:+,.2f} ({pnl_pct:+.2f}%)")

        return order

    def close_order_manual(self, order_id: str, exit_price: float = None) -> Optional[ShadowOrder]:
        """Manually close an order."""
        if order_id not in self.orders:
            return None

        order = self.orders[order_id]

        if exit_price is None:
            exit_price = self._fetch_price(order.symbol)
            if exit_price is None:
                return None

        return self._close_order(order, exit_price, "MANUAL")

    # =========================================================================
    # PERFORMANCE TRACKING
    # =========================================================================

    def get_performance(self) -> ShadowPerformance:
        """Calculate current shadow trading performance."""

        all_orders = self.closed_orders.copy()

        if not all_orders:
            return ShadowPerformance()

        wins = [o for o in all_orders if o.pnl_usd and o.pnl_usd > 0]
        losses = [o for o in all_orders if o.pnl_usd and o.pnl_usd <= 0]

        total_pnl = sum(o.pnl_usd or 0 for o in all_orders)
        avg_pnl = sum(o.pnl_pct or 0 for o in all_orders) / len(all_orders) if all_orders else 0
        win_rate = len(wins) / len(all_orders) * 100 if all_orders else 0
        avg_slippage = sum(o.slippage_pct for o in all_orders) / len(all_orders) if all_orders else 0

        # Calculate max drawdown
        cumulative = 0
        peak = 0
        max_dd = 0
        for order in all_orders:
            cumulative += order.pnl_usd or 0
            peak = max(peak, cumulative)
            dd = (peak - cumulative) / peak * 100 if peak > 0 else 0
            max_dd = max(max_dd, dd)

        # Calculate drift from backtest
        backtest_pnl = self.backtest_expectations.get('total_pnl', total_pnl)
        drift = abs(total_pnl - backtest_pnl) / abs(backtest_pnl) * 100 if backtest_pnl != 0 else 0

        return ShadowPerformance(
            total_orders=len(all_orders),
            filled_orders=len([o for o in all_orders if o.status != "CANCELLED"]),
            stopped_orders=len([o for o in all_orders if o.status == "STOPPED"]),
            target_orders=len([o for o in all_orders if o.status == "TARGET"]),
            cancelled_orders=len([o for o in all_orders if o.status == "CANCELLED"]),
            total_pnl_usd=total_pnl,
            avg_pnl_pct=avg_pnl,
            win_rate=win_rate,
            avg_slippage=avg_slippage,
            max_drawdown=max_dd,
            backtest_pnl=backtest_pnl,
            shadow_vs_backtest_drift=drift,
        )

    def set_backtest_expectations(self, total_pnl: float, win_rate: float, sharpe: float):
        """Set backtest expectations for drift comparison."""
        self.backtest_expectations = {
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'sharpe': sharpe,
        }

    def check_drift_alert(self) -> Tuple[bool, str]:
        """
        Check if shadow performance drifts too far from backtest.

        Returns:
            (should_alert, message)
        """
        perf = self.get_performance()

        if perf.shadow_vs_backtest_drift > self.max_drift_pct:
            return True, (
                f"Shadow vs backtest drift {perf.shadow_vs_backtest_drift:.1f}% "
                f"exceeds threshold {self.max_drift_pct}%"
            )

        return False, "OK"

    def print_performance_report(self):
        """Print shadow trading performance report."""

        perf = self.get_performance()

        print("\n" + "=" * 60)
        print("SHADOW TRADING PERFORMANCE REPORT")
        print("=" * 60)

        print(f"""
  MODE: {self.mode.value.upper()}

  ORDERS:
  ─────────────────────────────────────
  Total Orders:     {perf.total_orders}
  Filled:           {perf.filled_orders}
  Stopped Out:      {perf.stopped_orders}
  Hit Target:       {perf.target_orders}
  Cancelled:        {perf.cancelled_orders}

  PERFORMANCE:
  ─────────────────────────────────────
  Total P&L:        ${perf.total_pnl_usd:+,.2f}
  Avg P&L:          {perf.avg_pnl_pct:+.2f}%
  Win Rate:         {perf.win_rate:.1f}%
  Max Drawdown:     {perf.max_drawdown:.1f}%
  Avg Slippage:     {perf.avg_slippage:.2f}%

  VALIDATION:
  ─────────────────────────────────────
  Backtest P&L:     ${perf.backtest_pnl:+,.2f}
  Shadow P&L:       ${perf.total_pnl_usd:+,.2f}
  Drift:            {perf.shadow_vs_backtest_drift:.1f}%
  Status:           {"✓ OK" if perf.shadow_vs_backtest_drift <= self.max_drift_pct else "⚠ DRIFT ALERT"}
""")

        # Open positions
        if self.orders:
            print("  OPEN POSITIONS:")
            print("  ─────────────────────────────────────")
            for order in self.orders.values():
                current = self._fetch_price(order.symbol) or order.entry_price
                if order.direction == "LONG":
                    unrealized = ((current - order.entry_price) / order.entry_price) * 100
                else:
                    unrealized = ((order.entry_price - current) / order.entry_price) * 100
                print(f"    {order.symbol} {order.direction}: ${order.size_usd:,.0f} @ {order.entry_price:.4f} "
                      f"(unrealized: {unrealized:+.2f}%)")

        print("=" * 60)

    # =========================================================================
    # DATABASE OPERATIONS
    # =========================================================================

    def _save_order(self, order: ShadowOrder):
        """Save order to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO shadow_orders
            (order_id, timestamp, symbol, direction, size_usd, entry_price,
             stop_loss, take_profit, signals_used, risk_assessment, status,
             slippage_pct, fees_usd)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            order.order_id,
            order.timestamp.isoformat(),
            order.symbol,
            order.direction,
            order.size_usd,
            order.entry_price,
            order.stop_loss,
            order.take_profit,
            json.dumps(order.signals_used),
            json.dumps(order.risk_assessment),
            order.status,
            order.slippage_pct,
            order.fees_usd,
        ))

        conn.commit()
        conn.close()

    def _update_order(self, order: ShadowOrder):
        """Update order in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE shadow_orders
            SET status = ?, exit_price = ?, exit_timestamp = ?,
                pnl_usd = ?, pnl_pct = ?
            WHERE order_id = ?
        """, (
            order.status,
            order.exit_price,
            order.exit_timestamp.isoformat() if order.exit_timestamp else None,
            order.pnl_usd,
            order.pnl_pct,
            order.order_id,
        ))

        conn.commit()
        conn.close()

    def _fetch_price(self, symbol: str) -> Optional[float]:
        """Fetch current price for a symbol."""
        try:
            url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return float(response.json()['price'])
        except:
            pass
        return None

    def load_orders_from_db(self):
        """Load open orders from database."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM shadow_orders WHERE status = 'OPEN'")
        for row in cursor.fetchall():
            order = ShadowOrder(
                order_id=row['order_id'],
                timestamp=datetime.fromisoformat(row['timestamp']),
                symbol=row['symbol'],
                direction=row['direction'],
                size_usd=row['size_usd'],
                entry_price=row['entry_price'],
                stop_loss=row['stop_loss'],
                take_profit=row['take_profit'],
                signals_used=json.loads(row['signals_used']) if row['signals_used'] else [],
                risk_assessment=json.loads(row['risk_assessment']) if row['risk_assessment'] else {},
                status=row['status'],
                slippage_pct=row['slippage_pct'] or 0,
                fees_usd=row['fees_usd'] or 0,
            )
            self.orders[order.order_id] = order

        conn.close()
        print(f"  [SHADOW] Loaded {len(self.orders)} open orders from database")


# =============================================================================
# DEMO / TEST
# =============================================================================

def demo_shadow_trader():
    """Demonstrate shadow trading functionality."""
    print("\n" + "=" * 70)
    print("SHADOW TRADER DEMO")
    print("=" * 70)

    shadow = ShadowTrader(mode=ShadowMode.FULL_SHADOW)

    # Set backtest expectations
    shadow.set_backtest_expectations(
        total_pnl=500.0,
        win_rate=55.0,
        sharpe=1.2
    )

    # Create some shadow orders
    print("\n[1] Creating shadow orders...")

    shadow.create_shadow_order(
        symbol="XRPUSDT",
        direction="LONG",
        size_usd=200,
        entry_price=2.03,
        stop_loss=1.95,
        take_profit=2.20,
        signals_used=["RSI", "MACD", "Whale"],
        risk_assessment={'confidence': 0.75, 'regime': 'SIDEWAYS'},
    )

    shadow.create_shadow_order(
        symbol="BTCUSDT",
        direction="SHORT",
        size_usd=300,
        entry_price=97500,
        stop_loss=100000,
        take_profit=92000,
        signals_used=["Funding", "ML"],
        risk_assessment={'confidence': 0.68, 'regime': 'VOLATILE'},
    )

    # Check orders (simulate price check)
    print("\n[2] Checking orders against current prices...")
    closed = shadow.check_orders()
    print(f"    Closed orders: {len(closed)}")

    # Print performance
    shadow.print_performance_report()

    # Check drift
    should_alert, msg = shadow.check_drift_alert()
    if should_alert:
        print(f"\n⚠️  DRIFT ALERT: {msg}")


if __name__ == "__main__":
    demo_shadow_trader()
