#!/usr/bin/env python3
"""
Maker Telemetry Smoke Test
==========================
Diagnostic tool to verify maker order telemetry is being logged.

Usage:
    python tools/maker_telemetry_smoke.py --db hft_trades.db
    python tools/maker_telemetry_smoke.py --db hft_trades.db --days 30
    python tools/maker_telemetry_smoke.py --db hft_trades.db --symbol XRP

Output:
    - Count of maker orders in telemetry table
    - Last 10 orders with status
    - Fill rate statistics
    - Symbol breakdown
"""

import argparse
import sqlite3
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

# Add parent dir for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from hft_system.symbol_utils import normalize_db_symbol


def check_table_exists(conn: sqlite3.Connection, table: str) -> bool:
    """Check if a table exists."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name=?
    """, (table,))
    return cursor.fetchone() is not None


def get_table_columns(conn: sqlite3.Connection, table: str) -> list:
    """Get column names for a table."""
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table})")
    return [row[1] for row in cursor.fetchall()]


def run_smoke_test(conn: sqlite3.Connection, days: int, symbol: Optional[str] = None):
    """Run maker telemetry smoke test."""
    cursor = conn.cursor()
    cutoff_ms = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

    # Check if table exists
    if not check_table_exists(conn, "maker_order_telemetry"):
        print("\n[ERROR] maker_order_telemetry table does not exist!")
        print("        The bot needs to create this table on first run.")
        print("        Run the bot once to initialize the database schema.")
        return False

    # Get table columns
    columns = get_table_columns(conn, "maker_order_telemetry")
    print(f"\n  Table columns: {', '.join(columns[:10])}...")

    # Build WHERE clause
    where_parts = ["posted_ts > ?"]
    params = [cutoff_ms]
    if symbol:
        where_parts.append("symbol LIKE ?")
        params.append(f"%{symbol.upper()}%")
    where_clause = " AND ".join(where_parts)

    # Total count
    cursor.execute(f"""
        SELECT COUNT(*) FROM maker_order_telemetry
        WHERE {where_clause}
    """, params)
    total_count = cursor.fetchone()[0]

    symbol_filter = f" for {symbol.upper()}" if symbol else ""
    print(f"""
+================================================================================+
|                     MAKER TELEMETRY SMOKE TEST                                  |
+================================================================================+
| Period:        Last {days} days{symbol_filter}
| Total Orders:  {total_count}
+================================================================================+
""")

    if total_count == 0:
        print("  [WARNING] No maker orders found in telemetry!")
        print()
        print("  Possible reasons:")
        print("  1. Bot is not placing maker orders (execution_mode='taker')")
        print("  2. Maker orders are placed but not logged (missing log_maker_order_posted call)")
        print("  3. Time filter too restrictive (try --days 90)")
        print("  4. Symbol filter not matching (check symbol format in DB)")
        print()

        # Check if there are ANY orders in the table
        cursor.execute("SELECT COUNT(*) FROM maker_order_telemetry")
        all_time_count = cursor.fetchone()[0]
        print(f"  Total orders (all time): {all_time_count}")

        if all_time_count > 0:
            cursor.execute("""
                SELECT MIN(posted_ts), MAX(posted_ts) FROM maker_order_telemetry
            """)
            min_ts, max_ts = cursor.fetchone()
            if min_ts and max_ts:
                min_date = datetime.fromtimestamp(min_ts / 1000)
                max_date = datetime.fromtimestamp(max_ts / 1000)
                print(f"  Date range: {min_date.date()} to {max_date.date()}")
        return False

    # Status breakdown
    cursor.execute(f"""
        SELECT
            status,
            COUNT(*) as count,
            AVG(time_to_fill_ms) as avg_time
        FROM maker_order_telemetry
        WHERE {where_clause}
        GROUP BY status
        ORDER BY count DESC
    """, params)

    print("  --- STATUS BREAKDOWN ---")
    status_rows = cursor.fetchall()
    for status, count, avg_time in status_rows:
        pct = count / total_count * 100
        time_str = f"{avg_time:.0f}ms" if avg_time else "N/A"
        print(f"  {status or 'pending':15s}: {count:5d} ({pct:5.1f}%)  avg_time: {time_str}")
    print()

    # Fill rate
    filled = sum(c for s, c, _ in status_rows if s in ('filled', 'partial'))
    fill_rate = filled / total_count * 100 if total_count > 0 else 0
    print(f"  Fill Rate: {fill_rate:.1f}%")
    print()

    # Symbol breakdown
    cursor.execute(f"""
        SELECT
            symbol,
            COUNT(*) as count,
            SUM(CASE WHEN status = 'filled' THEN 1 ELSE 0 END) as filled
        FROM maker_order_telemetry
        WHERE {where_clause}
        GROUP BY symbol
        ORDER BY count DESC
        LIMIT 10
    """, params)

    print("  --- SYMBOL BREAKDOWN ---")
    for sym, count, filled in cursor.fetchall():
        fill_rate = filled / count * 100 if count > 0 else 0
        print(f"  {sym:15s}: {count:5d} orders, {fill_rate:5.1f}% fill rate")
    print()

    # Last 10 orders
    cursor.execute(f"""
        SELECT
            order_id,
            symbol,
            side,
            status,
            posted_ts,
            filled_ts,
            cancelled_ts,
            cancel_reason,
            time_to_fill_ms
        FROM maker_order_telemetry
        WHERE {where_clause}
        ORDER BY posted_ts DESC
        LIMIT 10
    """, params)

    print("  --- LAST 10 ORDERS ---")
    rows = cursor.fetchall()
    for row in rows:
        order_id, sym, side, status, posted_ts, filled_ts, cancelled_ts, cancel_reason, ttf = row
        posted_dt = datetime.fromtimestamp(posted_ts / 1000) if posted_ts else None

        status_detail = status or "pending"
        if cancel_reason:
            status_detail += f" ({cancel_reason})"
        if ttf:
            status_detail += f" [{ttf:.0f}ms]"

        print(f"  {order_id[:12] if order_id else 'N/A':12s} | {sym:10s} | {side:5s} | {status_detail:30s} | {posted_dt}")
    print()

    # Cancel reasons
    cursor.execute(f"""
        SELECT
            cancel_reason,
            COUNT(*) as count
        FROM maker_order_telemetry
        WHERE {where_clause} AND status = 'cancelled'
        GROUP BY cancel_reason
        ORDER BY count DESC
        LIMIT 10
    """, params)

    cancel_rows = cursor.fetchall()
    if cancel_rows:
        print("  --- CANCEL REASONS ---")
        for reason, count in cancel_rows:
            print(f"  {reason or 'unknown':25s}: {count:5d}")
        print()

    return total_count > 0


def main():
    parser = argparse.ArgumentParser(
        description="Maker Telemetry Smoke Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/maker_telemetry_smoke.py --db hft_trades.db
  python tools/maker_telemetry_smoke.py --db hft_trades.db --days 30
  python tools/maker_telemetry_smoke.py --db hft_trades.db --symbol XRPUSDT
        """
    )
    parser.add_argument("--db", default="hft_trades.db", help="Database path")
    parser.add_argument("--days", type=int, default=7, help="Days to analyze (default: 7)")
    parser.add_argument("--symbol", type=str, default=None, help="Symbol filter")
    args = parser.parse_args()

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

    if not db_path.exists():
        print(f"ERROR: Database not found at {args.db}")
        return 1

    print(f"Using database: {db_path}")

    # Use bot-safe connection with WAL mode
    from hft_system.db import open_sqlite
    conn = open_sqlite(str(db_path))

    # Normalize symbol for consistent DB queries
    symbol = normalize_db_symbol(args.symbol) if args.symbol else None
    success = run_smoke_test(conn, args.days, symbol)
    conn.close()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
