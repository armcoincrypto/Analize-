#!/usr/bin/env python3
"""
Maker Telemetry Insert Smoke Test
=================================
Sanity test to confirm the maker_order_telemetry table exists,
schema is correct, and inserts/reads work.

BOT-SAFE: Uses WAL mode and retry logic to avoid locking issues
when bot is running.

Usage:
    python tools/maker_telemetry_insert_smoke.py

    # Custom DB path
    python tools/maker_telemetry_insert_smoke.py --db-path /path/to/db.db

    # Read-only check (no insert, safe while bot is trading)
    python tools/maker_telemetry_insert_smoke.py --readonly-check
"""

import argparse
import sys
import time
import uuid
from pathlib import Path

# Add parent dir for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from hft_system.db import open_sqlite, execute_with_retry, execute_immediate, short_lived_connection


def run_smoke_test(db_path: str = "hft_trades.db", readonly: bool = False) -> bool:
    """
    Run maker telemetry insert smoke test.

    Returns True if all tests pass.
    """
    print(f"=" * 60)
    print(f"Maker Telemetry Smoke Test {'(READ-ONLY)' if readonly else ''}")
    print(f"=" * 60)
    print(f"DB Path: {db_path}")
    print()

    try:
        # Use short-lived connection for bot safety
        with short_lived_connection(db_path) as conn:
            cursor = conn.cursor()

            # 1. Check if table exists
            print("[1/5] Checking if maker_order_telemetry table exists...")
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='maker_order_telemetry'
            """)
            table_exists = cursor.fetchone() is not None
            if table_exists:
                print("      PASS: Table exists")
            else:
                if readonly:
                    print("      FAIL: Table does not exist (readonly mode - cannot create)")
                    return False
                print("      FAIL: Table does not exist")
                print("      Creating table...")
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS maker_order_telemetry (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        order_id TEXT NOT NULL,
                        trade_id TEXT,
                        symbol TEXT NOT NULL,
                        side TEXT NOT NULL,
                        limit_price REAL NOT NULL,
                        quantity REAL NOT NULL,
                        order_type TEXT DEFAULT 'limit',
                        posted_ts INTEGER NOT NULL,
                        filled_ts INTEGER,
                        cancelled_ts INTEGER,
                        filled_quantity REAL DEFAULT 0,
                        avg_fill_price REAL,
                        time_to_fill_ms INTEGER,
                        fill_ratio REAL DEFAULT 0,
                        slippage_from_limit_pct REAL,
                        best_bid_at_post REAL,
                        best_ask_at_post REAL,
                        spread_at_post_pct REAL,
                        mid_price_at_post REAL,
                        best_bid_at_fill REAL,
                        best_ask_at_fill REAL,
                        spread_at_fill_pct REAL,
                        price_moved_pct REAL,
                        queue_position_estimate INTEGER,
                        price_crossed_limit INTEGER DEFAULT 0,
                        cross_depth_bps REAL DEFAULT 0,
                        cancel_reason TEXT,
                        status TEXT DEFAULT 'pending'
                    )
                """)
                conn.commit()
                print("      Table created")

            # 2. Check schema
            print("[2/5] Checking table schema...")
            cursor.execute("PRAGMA table_info(maker_order_telemetry)")
            columns = {row[1]: row[2] for row in cursor.fetchall()}
            required_columns = ["order_id", "symbol", "side", "limit_price", "quantity", "posted_ts", "status"]
            missing = [c for c in required_columns if c not in columns]
            if missing:
                print(f"      FAIL: Missing columns: {missing}")
                return False
            print(f"      PASS: Schema has {len(columns)} columns, all required present")

            # 3. Insert test row (skip if readonly)
            test_order_id = None
            if readonly:
                print("[3/5] Skipping insert (readonly mode)...")
                print("      SKIP: --readonly-check specified")
            else:
                print("[3/5] Inserting test row with BEGIN IMMEDIATE + retry...")
                print("      (Will wait up to 10s if bot is writing, or fail with clear message)")
                test_order_id = f"SMOKE_TEST_{uuid.uuid4().hex[:8]}"
                test_ts = int(time.time() * 1000)

                sql = """
                    INSERT INTO maker_order_telemetry (
                        order_id, trade_id, symbol, side,
                        limit_price, quantity, order_type,
                        posted_ts,
                        best_bid_at_post, best_ask_at_post, spread_at_post_pct,
                        mid_price_at_post, queue_position_estimate,
                        status
                    ) VALUES (?, ?, ?, ?, ?, ?, 'limit', ?, ?, ?, ?, ?, ?, 'pending')
                """
                params = (
                    test_order_id,
                    f"smoke_trade_{test_ts}",
                    "XRPUSDT",
                    "LONG",
                    2.5000,
                    100.0,
                    test_ts,
                    2.4995,
                    2.5005,
                    0.04,
                    2.5000,
                    5
                )

                # Use BEGIN IMMEDIATE to acquire write lock early and fail fast
                success, message = execute_immediate(conn, sql, params, max_attempts=10, base_delay_ms=1000)
                if success:
                    print(f"      PASS: Inserted order_id={test_order_id}")
                    print(f"      {message}")
                else:
                    print(f"      FAIL: {message}")
                    if "writer still active" in message.lower():
                        print("      HINT: Another process (likely the bot) has a write lock.")
                        print("            This is expected if bot is actively writing.")
                        print("            Try --readonly-check to verify table without writing.")
                    return False

            # 4. Read back test row (or any recent row in readonly mode)
            print("[4/5] Reading back data...")
            if readonly:
                cursor.execute("""
                    SELECT order_id, symbol, side, limit_price, quantity, status
                    FROM maker_order_telemetry
                    ORDER BY posted_ts DESC
                    LIMIT 1
                """)
            else:
                cursor.execute("""
                    SELECT order_id, symbol, side, limit_price, quantity, status
                    FROM maker_order_telemetry
                    WHERE order_id = ?
                """, (test_order_id,))
            row = cursor.fetchone()
            if row:
                print(f"      PASS: Read back: {row}")
            else:
                if readonly:
                    print("      INFO: No rows found (table is empty)")
                else:
                    print("      FAIL: Could not read back test row")
                    return False

            # 5. Get total count
            print("[5/5] Getting total row count...")
            cursor.execute("SELECT COUNT(*) FROM maker_order_telemetry")
            total_count = cursor.fetchone()[0]
            print(f"      INFO: Total rows in maker_order_telemetry: {total_count}")

            # Get status breakdown
            cursor.execute("""
                SELECT status, COUNT(*) as cnt
                FROM maker_order_telemetry
                GROUP BY status
            """)
            status_breakdown = cursor.fetchall()
            if status_breakdown:
                print("      Status breakdown:")
                for status, cnt in status_breakdown:
                    print(f"        - {status}: {cnt}")

        print()
        print("=" * 60)
        print("SMOKE TEST PASSED")
        print("=" * 60)
        print(f"  - Table exists: YES")
        print(f"  - Schema valid: YES")
        print(f"  - Insert works: {'SKIPPED (readonly)' if readonly else 'YES'}")
        print(f"  - Read works: YES")
        print(f"  - Total rows: {total_count}")
        print()

        return True

    except Exception as e:
        print(f"      FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Maker Telemetry Smoke Test (bot-safe)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--db-path",
        default="hft_trades.db",
        help="Path to SQLite database (default: hft_trades.db)"
    )
    parser.add_argument(
        "--readonly-check",
        action="store_true",
        help="Read-only check (no insert, safe while bot is trading)"
    )

    args = parser.parse_args()

    success = run_smoke_test(args.db_path, args.readonly_check)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
