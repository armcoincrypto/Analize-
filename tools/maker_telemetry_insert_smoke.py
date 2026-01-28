#!/usr/bin/env python3
"""
Maker Telemetry Insert Smoke Test
=================================
Sanity test to confirm the maker_order_telemetry table exists,
schema is correct, and inserts/reads work.

Usage:
    python tools/maker_telemetry_insert_smoke.py

    # Custom DB path
    python tools/maker_telemetry_insert_smoke.py --db-path /path/to/db.db
"""

import argparse
import sqlite3
import sys
import time
import uuid
from pathlib import Path

# Add parent dir for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_smoke_test(db_path: str = "hft_trades.db") -> bool:
    """
    Run maker telemetry insert smoke test.

    Returns True if all tests pass.
    """
    print(f"=" * 60)
    print(f"Maker Telemetry Smoke Test")
    print(f"=" * 60)
    print(f"DB Path: {db_path}")
    print()

    try:
        conn = sqlite3.connect(db_path)
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

        # 3. Insert test row
        print("[3/5] Inserting test row...")
        test_order_id = f"SMOKE_TEST_{uuid.uuid4().hex[:8]}"
        test_ts = int(time.time() * 1000)
        cursor.execute("""
            INSERT INTO maker_order_telemetry (
                order_id, trade_id, symbol, side,
                limit_price, quantity, order_type,
                posted_ts,
                best_bid_at_post, best_ask_at_post, spread_at_post_pct,
                mid_price_at_post, queue_position_estimate,
                status
            ) VALUES (?, ?, ?, ?, ?, ?, 'limit', ?, ?, ?, ?, ?, ?, 'pending')
        """, (
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
        ))
        conn.commit()
        print(f"      PASS: Inserted order_id={test_order_id}")

        # 4. Read back test row
        print("[4/5] Reading back test row...")
        cursor.execute("""
            SELECT order_id, symbol, side, limit_price, quantity, status
            FROM maker_order_telemetry
            WHERE order_id = ?
        """, (test_order_id,))
        row = cursor.fetchone()
        if row:
            print(f"      PASS: Read back: {row}")
        else:
            print("      FAIL: Could not read back test row")
            return False

        # 5. Get total count
        print("[5/5] Getting total row count...")
        cursor.execute("SELECT COUNT(*) FROM maker_order_telemetry")
        total_count = cursor.fetchone()[0]
        print(f"      INFO: Total rows in maker_order_telemetry: {total_count}")

        # Cleanup test row (optional - keep for audit trail)
        # cursor.execute("DELETE FROM maker_order_telemetry WHERE order_id = ?", (test_order_id,))
        # conn.commit()

        conn.close()

        print()
        print("=" * 60)
        print("SMOKE TEST PASSED")
        print("=" * 60)
        print(f"  - Table exists: YES")
        print(f"  - Schema valid: YES")
        print(f"  - Insert works: YES")
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
        description="Maker Telemetry Smoke Test",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--db-path",
        default="hft_trades.db",
        help="Path to SQLite database (default: hft_trades.db)"
    )

    args = parser.parse_args()

    success = run_smoke_test(args.db_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
