#!/usr/bin/env python3
"""
Maker Fill Experiment Harness
=============================
Test maker order fill rates in paper mode.

Posts maker orders near touch, waits for fill or timeout, cancels if needed.
ALWAYS writes maker_order_telemetry row for analysis.
If filled, creates/closes a trade with execution_mode='maker'.

Usage:
    python tools/maker_smoke_test.py --symbol SUIUSDT --n 5 --timeout-sec 30
    python tools/maker_smoke_test.py --symbol XRPUSDT --n 10 --timeout-sec 60 --cooldown-ms 5000

Safety:
    - Paper mode only (no real funds)
    - Uses simulated orderbook if no live data
"""

import argparse
import asyncio
import logging
import os
import random
import sqlite3
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Add parent dir for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class MakerOrderResult:
    """Result of a maker order attempt."""
    order_id: str
    symbol: str
    side: str
    limit_price: float
    quantity: float
    posted_ts: int
    filled_ts: Optional[int] = None
    cancelled_ts: Optional[int] = None
    fill_ratio: float = 0.0
    time_to_fill_ms: Optional[int] = None
    slippage_from_limit_pct: float = 0.0
    cancel_reason: Optional[str] = None
    fill_price: Optional[float] = None


def open_sqlite(db_path: str) -> sqlite3.Connection:
    """Open SQLite with WAL mode and busy_timeout."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def ensure_tables(conn: sqlite3.Connection):
    """Ensure required tables exist."""
    cursor = conn.cursor()

    # Check if maker_order_telemetry exists
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='maker_order_telemetry'
    """)
    if not cursor.fetchone():
        logger.info("Creating maker_order_telemetry table...")
        cursor.execute("""
            CREATE TABLE maker_order_telemetry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT,
                symbol TEXT,
                side TEXT,
                order_id TEXT,
                limit_price REAL,
                quantity REAL,
                posted_ts INTEGER,
                filled_ts INTEGER,
                cancelled_ts INTEGER,
                fill_ratio REAL,
                time_to_fill_ms INTEGER,
                slippage_from_limit_pct REAL,
                cancel_reason TEXT
            )
        """)

    # Check if trades exists
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='trades'
    """)
    if not cursor.fetchone():
        logger.info("Creating trades table...")
        cursor.execute("""
            CREATE TABLE trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT UNIQUE,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                quantity REAL NOT NULL,
                entry_time INTEGER NOT NULL,
                exit_price REAL,
                exit_time INTEGER,
                pnl REAL DEFAULT 0,
                pnl_pct REAL DEFAULT 0,
                hold_time_sec REAL DEFAULT 0,
                exit_reason TEXT,
                mfe REAL DEFAULT 0,
                mae REAL DEFAULT 0,
                conditions_met INTEGER DEFAULT 0,
                signal_confidence REAL DEFAULT 0,
                status TEXT DEFAULT 'open',
                is_probe_cause INTEGER DEFAULT 0,
                cause_class TEXT,
                execution_mode TEXT DEFAULT 'taker',
                entry_fee_pct REAL DEFAULT 0,
                exit_fee_pct REAL DEFAULT 0,
                fees_paid_pct REAL DEFAULT 0,
                spread_cost_pct REAL DEFAULT 0,
                slippage_pct REAL DEFAULT 0,
                total_costs_pct REAL DEFAULT 0,
                pnl_after_costs_pct REAL DEFAULT 0,
                pocket_id TEXT
            )
        """)

    conn.commit()


def log_maker_order(conn: sqlite3.Connection, result: MakerOrderResult, trade_id: str = None):
    """Log maker order to telemetry table."""
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO maker_order_telemetry (
            trade_id, symbol, side, order_id, limit_price, quantity,
            posted_ts, filled_ts, cancelled_ts, fill_ratio,
            time_to_fill_ms, slippage_from_limit_pct, cancel_reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        trade_id,
        result.symbol,
        result.side,
        result.order_id,
        result.limit_price,
        result.quantity,
        result.posted_ts,
        result.filled_ts,
        result.cancelled_ts,
        result.fill_ratio,
        result.time_to_fill_ms,
        result.slippage_from_limit_pct,
        result.cancel_reason
    ))
    conn.commit()
    logger.debug(f"Logged maker order: {result.order_id}")


def create_maker_trade(conn: sqlite3.Connection, result: MakerOrderResult) -> str:
    """Create a trade record for a filled maker order."""
    trade_id = f"MKR_{result.symbol}_{result.filled_ts}_{result.order_id[-4:]}"
    entry_time = result.filled_ts
    exit_time = entry_time + 1000  # 1 second later

    cursor = conn.cursor()

    # Insert entry
    cursor.execute("""
        INSERT INTO trades (
            trade_id, symbol, side, entry_price, quantity,
            entry_time, conditions_met, signal_confidence, status,
            is_probe_cause, cause_class, execution_mode
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'open', ?, ?, ?)
    """, (
        trade_id,
        result.symbol,
        result.side.upper(),
        result.fill_price or result.limit_price,
        result.quantity,
        entry_time,
        3,   # conditions_met
        0.8, # signal_confidence
        0,   # is_probe_cause
        "MAKER_SMOKE_TEST",
        "maker"
    ))

    # Immediately close with maker fees (lower than taker)
    maker_fee = 0.02  # 0.02% maker fee
    total_fees = maker_fee * 2  # entry + exit

    cursor.execute("""
        UPDATE trades SET
            exit_price = ?,
            exit_time = ?,
            pnl = 0,
            pnl_pct = 0,
            hold_time_sec = 1,
            exit_reason = 'maker_smoke_test',
            mfe = 0,
            mae = 0,
            entry_fee_pct = ?,
            exit_fee_pct = ?,
            fees_paid_pct = ?,
            spread_cost_pct = 0,
            slippage_pct = ?,
            total_costs_pct = ?,
            pnl_after_costs_pct = ?,
            status = 'closed'
        WHERE trade_id = ?
    """, (
        result.fill_price or result.limit_price,
        exit_time,
        maker_fee,
        maker_fee,
        total_fees,
        result.slippage_from_limit_pct,
        total_fees + result.slippage_from_limit_pct,
        -(total_fees + result.slippage_from_limit_pct),
        trade_id
    ))

    conn.commit()
    logger.info(f"Created maker trade: {trade_id}")
    return trade_id


def get_simulated_orderbook(symbol: str) -> Tuple[float, float, float]:
    """
    Get simulated orderbook prices.
    Returns (best_bid, best_ask, spread_pct).

    In real mode, this would connect to exchange websocket.
    For smoke testing, we simulate realistic prices.
    """
    # Simulated base prices
    base_prices = {
        "SUIUSDT": 3.50,
        "XRPUSDT": 2.20,
        "BTCUSDT": 95000.0,
        "ETHUSDT": 3200.0,
    }

    base = base_prices.get(symbol, 1.0)

    # Add some random noise (-0.5% to +0.5%)
    noise = random.uniform(-0.005, 0.005)
    mid_price = base * (1 + noise)

    # Typical spread for these assets (0.01% to 0.05%)
    spread_pct = random.uniform(0.0001, 0.0005)
    half_spread = mid_price * spread_pct / 2

    best_bid = mid_price - half_spread
    best_ask = mid_price + half_spread

    return best_bid, best_ask, spread_pct * 100


async def simulate_maker_order(
    symbol: str,
    side: str,
    timeout_sec: float
) -> MakerOrderResult:
    """
    Simulate posting a maker order and waiting for fill.

    In paper mode, simulates realistic fill behavior:
    - ~60-70% fill rate for orders at touch
    - Fill time varies from instant to near-timeout
    - Small slippage on fills
    """
    order_id = f"MKR_SMOKE_{uuid.uuid4().hex[:8]}"
    posted_ts = int(time.time() * 1000)

    # Get orderbook
    best_bid, best_ask, spread_pct = get_simulated_orderbook(symbol)

    # Place order at touch (best bid for buy, best ask for sell)
    if side.upper() == "LONG":
        limit_price = best_bid
    else:
        limit_price = best_ask

    quantity = 10.0  # Fixed quantity for smoke test

    logger.info(f"Posting maker order: {order_id}")
    logger.info(f"  Symbol: {symbol}, Side: {side}")
    logger.info(f"  Limit: {limit_price:.6f}, Qty: {quantity}")
    logger.info(f"  Spread: {spread_pct:.4f}%")

    result = MakerOrderResult(
        order_id=order_id,
        symbol=symbol,
        side=side.upper(),
        limit_price=limit_price,
        quantity=quantity,
        posted_ts=posted_ts
    )

    # Simulate fill probability and timing
    # Higher chance of fill with longer timeout
    base_fill_prob = 0.65
    timeout_bonus = min(timeout_sec / 60, 0.2)  # Up to 20% bonus for longer waits
    fill_probability = base_fill_prob + timeout_bonus

    will_fill = random.random() < fill_probability

    if will_fill:
        # Random fill time (1s to 80% of timeout)
        max_fill_time = timeout_sec * 0.8
        fill_time = random.uniform(1.0, max_fill_time)

        logger.info(f"  Waiting for fill... (simulated)")
        await asyncio.sleep(min(fill_time, timeout_sec))

        filled_ts = int(time.time() * 1000)
        time_to_fill_ms = filled_ts - posted_ts

        # Small slippage on fill (usually favorable for maker)
        slippage_pct = random.uniform(-0.005, 0.01)  # -0.5bp to +1bp
        fill_price = limit_price * (1 + slippage_pct / 100)

        result.filled_ts = filled_ts
        result.fill_ratio = 1.0
        result.time_to_fill_ms = time_to_fill_ms
        result.slippage_from_limit_pct = slippage_pct
        result.fill_price = fill_price

        logger.info(f"  FILLED in {time_to_fill_ms}ms at {fill_price:.6f}")

    else:
        # Wait full timeout then cancel
        logger.info(f"  Waiting {timeout_sec}s for fill...")
        await asyncio.sleep(timeout_sec)

        cancelled_ts = int(time.time() * 1000)
        result.cancelled_ts = cancelled_ts
        result.fill_ratio = 0.0
        result.cancel_reason = "timeout"

        logger.info(f"  CANCELLED (timeout after {timeout_sec}s)")

    return result


async def run_maker_smoke_test(
    symbol: str,
    n_attempts: int,
    timeout_sec: float,
    cooldown_ms: int,
    db_path: str
) -> dict:
    """
    Run maker fill experiment.

    Returns summary statistics.
    """
    logger.info("=" * 60)
    logger.info("MAKER SMOKE TEST")
    logger.info("=" * 60)
    logger.info(f"  Symbol: {symbol}")
    logger.info(f"  Attempts: {n_attempts}")
    logger.info(f"  Timeout: {timeout_sec}s")
    logger.info(f"  Cooldown: {cooldown_ms}ms")
    logger.info(f"  Database: {db_path}")
    logger.info("=" * 60)

    conn = open_sqlite(db_path)
    ensure_tables(conn)

    results = {
        "total": n_attempts,
        "filled": 0,
        "cancelled": 0,
        "fill_times_ms": [],
        "slippages_pct": [],
        "trade_ids": []
    }

    for i in range(n_attempts):
        logger.info(f"\n--- Attempt {i+1}/{n_attempts} ---")

        # Alternate sides
        side = "LONG" if i % 2 == 0 else "SHORT"

        # Execute order
        result = await simulate_maker_order(symbol, side, timeout_sec)

        # Log to telemetry (ALWAYS)
        trade_id = None
        if result.filled_ts:
            # Create trade for filled order
            trade_id = create_maker_trade(conn, result)
            results["trade_ids"].append(trade_id)
            results["filled"] += 1
            results["fill_times_ms"].append(result.time_to_fill_ms)
            results["slippages_pct"].append(result.slippage_from_limit_pct)
        else:
            results["cancelled"] += 1

        log_maker_order(conn, result, trade_id)

        # Cooldown between attempts
        if i < n_attempts - 1 and cooldown_ms > 0:
            logger.info(f"  Cooldown: {cooldown_ms}ms")
            await asyncio.sleep(cooldown_ms / 1000)

    conn.close()

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Total attempts:  {results['total']}")
    logger.info(f"  Filled:          {results['filled']}")
    logger.info(f"  Cancelled:       {results['cancelled']}")

    fill_rate = results['filled'] / results['total'] * 100 if results['total'] > 0 else 0
    logger.info(f"  Fill rate:       {fill_rate:.1f}%")

    if results['fill_times_ms']:
        avg_fill_time = sum(results['fill_times_ms']) / len(results['fill_times_ms'])
        logger.info(f"  Avg fill time:   {avg_fill_time:.0f}ms")

    if results['slippages_pct']:
        avg_slippage = sum(results['slippages_pct']) / len(results['slippages_pct'])
        logger.info(f"  Avg slippage:    {avg_slippage:.4f}%")

    if results['trade_ids']:
        logger.info(f"  Trade IDs created: {len(results['trade_ids'])}")
        for tid in results['trade_ids'][:5]:
            logger.info(f"    {tid}")
        if len(results['trade_ids']) > 5:
            logger.info(f"    ... and {len(results['trade_ids']) - 5} more")

    logger.info("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Maker Fill Experiment Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test: 5 attempts, 30s timeout
  python tools/maker_smoke_test.py --symbol SUIUSDT --n 5 --timeout-sec 30

  # Longer test with cooldown
  python tools/maker_smoke_test.py --symbol XRPUSDT --n 10 --timeout-sec 60 --cooldown-ms 5000

  # Fast test for validation
  python tools/maker_smoke_test.py --symbol SUIUSDT --n 3 --timeout-sec 5
        """
    )
    parser.add_argument("--symbol", type=str, required=True,
                        help="Symbol to test (e.g., SUIUSDT)")
    parser.add_argument("--n", type=int, default=5,
                        help="Number of attempts (default: 5)")
    parser.add_argument("--timeout-sec", type=float, default=30,
                        help="Timeout per order in seconds (default: 30)")
    parser.add_argument("--cooldown-ms", type=int, default=2000,
                        help="Cooldown between attempts in ms (default: 2000)")
    parser.add_argument("--db", type=str, default="hft_trades.db",
                        help="Database path (default: hft_trades.db)")

    args = parser.parse_args()

    # Normalize symbol
    symbol = args.symbol.upper()
    if not symbol.endswith("USDT"):
        symbol = f"{symbol}USDT"

    # Find database
    db_path = Path(args.db)
    if not db_path.exists():
        for try_path in [
            Path.cwd() / "hft_trades.db",
            Path.home() / "Analize-" / "hft_trades.db",
            Path("/root/Analize-/hft_trades.db")
        ]:
            if try_path.exists():
                db_path = try_path
                break

    # Run async
    asyncio.run(run_maker_smoke_test(
        symbol=symbol,
        n_attempts=args.n,
        timeout_sec=args.timeout_sec,
        cooldown_ms=args.cooldown_ms,
        db_path=str(db_path)
    ))


if __name__ == "__main__":
    main()
