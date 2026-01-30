#!/usr/bin/env python3
"""
Report Sanity Check Tool
========================
Quick verification that database tables are consistent and reports are accurate.

Usage:
    python tools/report_sanity_check.py --db hft_trades.db --days 1

Output:
    - count(maker_order_telemetry rows in range)
    - count(trades rows in range)
    - maker filled/unfilled/pending breakdown
    - maker rows with NULL symbol/order_id (should be 0)
    - Data quality issues flagged
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent dir for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from hft_system.db import open_sqlite


def get_time_cutoff_ms(days: int) -> int:
    """Get timestamp cutoff in milliseconds."""
    return int((datetime.now() - timedelta(days=days)).timestamp() * 1000)


def run_sanity_check(db_path: str, days: int = 1) -> bool:
    """
    Run sanity checks on database.

    Returns True if all checks pass.
    """
    print("=" * 70)
    print(" REPORT SANITY CHECK")
    print("=" * 70)
    print(f"  Database: {db_path}")
    print(f"  Period: Last {days} days")
    print()

    cutoff_ms = get_time_cutoff_ms(days)
    print(f"  Cutoff timestamp: {cutoff_ms} ({datetime.fromtimestamp(cutoff_ms/1000).isoformat()})")
    print()

    conn = open_sqlite(db_path)
    cursor = conn.cursor()

    issues = []

    # 1. Count maker_order_telemetry rows
    print("-" * 70)
    print(" 1. MAKER ORDER TELEMETRY")
    print("-" * 70)

    cursor.execute("""
        SELECT COUNT(*) FROM maker_order_telemetry
        WHERE posted_ts >= ?
    """, (cutoff_ms,))
    maker_telemetry_count = cursor.fetchone()[0]
    print(f"  Rows in range:    {maker_telemetry_count}")

    cursor.execute("SELECT COUNT(*) FROM maker_order_telemetry")
    maker_telemetry_total = cursor.fetchone()[0]
    print(f"  Total rows:       {maker_telemetry_total}")

    # Maker status breakdown
    cursor.execute("""
        SELECT
            SUM(CASE WHEN filled_ts IS NOT NULL THEN 1 ELSE 0 END) as filled,
            SUM(CASE WHEN filled_ts IS NULL AND cancelled_ts IS NOT NULL THEN 1 ELSE 0 END) as cancelled,
            SUM(CASE WHEN filled_ts IS NULL AND cancelled_ts IS NULL THEN 1 ELSE 0 END) as pending
        FROM maker_order_telemetry
        WHERE posted_ts >= ?
    """, (cutoff_ms,))
    row = cursor.fetchone()
    filled = row[0] or 0
    cancelled = row[1] or 0
    pending = row[2] or 0
    print(f"\n  Status breakdown (in range):")
    print(f"    Filled:     {filled}")
    print(f"    Cancelled:  {cancelled}")
    print(f"    Pending:    {pending}")

    # Check for NULL fields
    cursor.execute("""
        SELECT COUNT(*) FROM maker_order_telemetry
        WHERE symbol IS NULL OR order_id IS NULL
    """)
    null_fields = cursor.fetchone()[0]
    print(f"\n  Rows with NULL symbol/order_id: {null_fields}")
    if null_fields > 0:
        issues.append(f"maker_order_telemetry has {null_fields} rows with NULL symbol or order_id")

    # 2. Count trades rows
    print()
    print("-" * 70)
    print(" 2. TRADES TABLE")
    print("-" * 70)

    cursor.execute("""
        SELECT COUNT(*) FROM trades
        WHERE entry_time >= ?
    """, (cutoff_ms,))
    trades_in_range = cursor.fetchone()[0]
    print(f"  Rows in range:    {trades_in_range}")

    cursor.execute("SELECT COUNT(*) FROM trades WHERE status = 'closed'")
    trades_closed = cursor.fetchone()[0]
    print(f"  Total closed:     {trades_closed}")

    # Trades by execution_mode
    cursor.execute("""
        SELECT
            COALESCE(execution_mode, 'NULL/taker') as mode,
            COUNT(*) as count
        FROM trades
        WHERE entry_time >= ?
        GROUP BY execution_mode
        ORDER BY count DESC
    """, (cutoff_ms,))
    rows = cursor.fetchall()
    print(f"\n  By execution_mode (in range):")
    for row in rows:
        print(f"    {row[0]}: {row[1]}")

    # 3. Cross-check
    print()
    print("-" * 70)
    print(" 3. CROSS-CHECK")
    print("-" * 70)

    # Count maker trades in trades table
    cursor.execute("""
        SELECT COUNT(*) FROM trades
        WHERE execution_mode = 'maker' AND entry_time >= ?
    """, (cutoff_ms,))
    maker_trades = cursor.fetchone()[0]
    print(f"  Maker trades in trades table:     {maker_trades}")
    print(f"  Maker orders in telemetry:        {maker_telemetry_count}")

    if maker_trades > filled:
        issues.append(f"More maker trades ({maker_trades}) than filled telemetry orders ({filled})")

    # 4. Sample data
    print()
    print("-" * 70)
    print(" 4. SAMPLE DATA (last 5 maker orders)")
    print("-" * 70)

    cursor.execute("""
        SELECT order_id, symbol, side, status, posted_ts, filled_ts, cancelled_ts
        FROM maker_order_telemetry
        ORDER BY posted_ts DESC
        LIMIT 5
    """)
    rows = cursor.fetchall()

    if rows:
        print(f"  {'Order ID':<25} {'Symbol':<10} {'Side':<6} {'Status':<10}")
        print(f"  {'-'*25} {'-'*10} {'-'*6} {'-'*10}")
        for row in rows:
            order_id = (row[0] or "NULL")[:24]
            symbol = row[1] or "NULL"
            side = row[2] or "NULL"
            # Determine status from timestamps
            if row[5]:  # filled_ts
                status = "filled"
            elif row[6]:  # cancelled_ts
                status = "cancelled"
            else:
                status = "pending"
            print(f"  {order_id:<25} {symbol:<10} {side:<6} {status:<10}")
    else:
        print("  (no data)")

    # 5. Summary
    print()
    print("=" * 70)
    print(" SUMMARY")
    print("=" * 70)

    if issues:
        print("\n  ISSUES FOUND:")
        for issue in issues:
            print(f"    - {issue}")
        print()
        print("  Result: FAIL")
        conn.close()
        return False
    else:
        print("\n  No issues found")
        print()
        print(f"  maker_order_telemetry: {maker_telemetry_count} rows in range")
        print(f"  trades: {trades_in_range} rows in range")
        print(f"  Maker fill rate: {filled}/{filled+cancelled} = {filled/(filled+cancelled)*100 if (filled+cancelled) > 0 else 0:.1f}%")
        print()
        print("  Result: PASS")
        conn.close()
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Report Sanity Check - verify database consistency"
    )
    parser.add_argument("--db", default="hft_trades.db", help="Database path")
    parser.add_argument("--days", type=int, default=1, help="Days to check (default: 1)")
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

    success = run_sanity_check(str(db_path), args.days)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
