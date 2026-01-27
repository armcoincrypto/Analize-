#!/usr/bin/env python3
"""
Database Schema Check Tool
==========================
Verifies that the database schema matches the required structure.
Uses the same REQUIRED_SCHEMA as the bot's startup check.

Usage:
    python tools/db_schema_check.py --db hft_trades.db
    python tools/db_schema_check.py --db hft_trades.db --migrate
"""

import argparse
import sqlite3
import sys
from pathlib import Path

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from hft_system.db_migrations import (
    REQUIRED_SCHEMA,
    check_schema,
    apply_migrations,
    print_schema_report,
    table_exists,
    get_table_columns
)


def check_data_integrity(conn):
    """Check data integrity and join quality."""
    cursor = conn.cursor()
    print(f"\n{'='*70}")
    print(" DATA INTEGRITY CHECKS")
    print('='*70)

    # Check 1: Total trades
    try:
        cursor.execute("SELECT COUNT(*) FROM trades WHERE status='closed'")
        total_trades = cursor.fetchone()[0]
        print(f"\n  Total closed trades: {total_trades}")
    except Exception as e:
        print(f"  ERROR checking trades: {e}")
        total_trades = 0

    # Check 2: Trades with MFE/MAE data
    if total_trades > 0:
        try:
            cursor.execute("SELECT COUNT(*) FROM trades WHERE status='closed' AND (mfe IS NOT NULL AND mfe != 0)")
            mfe_trades = cursor.fetchone()[0]
            pct = (mfe_trades / total_trades * 100)
            status = "OK" if pct > 50 else "LOW"
            print(f"  Trades with MFE/MAE: {mfe_trades}/{total_trades} ({pct:.1f}%) [{status}]")
        except Exception as e:
            print(f"  ERROR checking MFE/MAE: {e}")

    # Check 3: Trade causality join rate (using trade_id, not id)
    if total_trades > 0:
        try:
            cursor.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN tc.trade_id IS NOT NULL THEN 1 ELSE 0 END) as joined
                FROM trades t
                LEFT JOIN trade_causality tc ON t.trade_id = tc.trade_id
                WHERE t.status = 'closed'
            """)
            row = cursor.fetchone()
            join_rate = (row[1] / row[0] * 100) if row[0] > 0 else 0
            status = "OK" if join_rate > 80 else "LOW"
            print(f"  Trade->Causality join rate: {row[1]}/{row[0]} ({join_rate:.1f}%) [{status}]")
        except Exception as e:
            print(f"  ERROR checking causality join: {e}")

    # Check 4: Cost model coverage
    if total_trades > 0:
        try:
            cursor.execute("SELECT COUNT(*) FROM trades WHERE total_costs_pct IS NOT NULL AND total_costs_pct > 0 AND status='closed'")
            with_costs = cursor.fetchone()[0]
            pct = (with_costs / total_trades * 100)
            status = "OK" if pct > 50 else "LOW"
            print(f"  Trades with cost data: {with_costs}/{total_trades} ({pct:.1f}%) [{status}]")
        except Exception as e:
            print(f"  ERROR checking costs: {e}")

    # Check 5: Probe trades (new feature)
    try:
        cursor.execute("SELECT COUNT(*) FROM trades WHERE is_probe_cause = 1 AND status='closed'")
        probe_count = cursor.fetchone()[0]
        print(f"  PROBE trades: {probe_count}")
    except Exception as e:
        print(f"  (is_probe_cause column not found - needs migration)")


def show_recent_activity(conn):
    """Show recent trades and blocks."""
    cursor = conn.cursor()
    print(f"\n{'='*70}")
    print(" RECENT ACTIVITY")
    print('='*70)

    # Last 5 trades
    try:
        cursor.execute("""
            SELECT
                datetime(entry_time/1000, 'unixepoch') as entry,
                symbol, side, exit_reason,
                ROUND(pnl_after_costs_pct, 4) as net_pnl,
                COALESCE(cause_class, 'FULL') as class
            FROM trades
            WHERE status = 'closed'
            ORDER BY entry_time DESC
            LIMIT 5
        """)
        rows = cursor.fetchall()
        if rows:
            print("\n  Last 5 trades:")
            print(f"  {'Entry':<20} {'Symbol':<6} {'Side':<6} {'Exit':<15} {'Net%':<10} {'Class':<6}")
            print("  " + "-" * 65)
            for row in rows:
                entry, symbol, side, exit_reason, net_pnl, cls = row
                print(f"  {entry:<20} {symbol:<6} {side:<6} {exit_reason or 'N/A':<15} {net_pnl or 0:>+8.4f}% {cls:<6}")
        else:
            print("\n  No recent trades")
    except Exception as e:
        print(f"  ERROR fetching recent trades: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Database Schema Check - Verify required columns exist"
    )
    parser.add_argument("--db", default="hft_trades.db", help="Database path")
    parser.add_argument("--migrate", action="store_true", help="Auto-migrate missing columns")
    parser.add_argument("--quiet", action="store_true", help="Only show problems")
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

    # Run schema check
    if not args.quiet:
        print_schema_report(str(db_path))

    problems = check_schema(str(db_path))

    if problems and args.migrate:
        print("\nApplying migrations...")
        applied = apply_migrations(str(db_path))
        if applied:
            print(f"Applied {len(applied)} migrations:")
            for m in applied:
                print(f"  + {m}")

        # Re-check
        problems = check_schema(str(db_path))

    # Data integrity checks
    if not args.quiet:
        try:
            conn = sqlite3.connect(str(db_path))
            check_data_integrity(conn)
            show_recent_activity(conn)
            conn.close()
        except Exception as e:
            print(f"ERROR during integrity check: {e}")

    # Final summary
    print(f"\n{'='*70}")
    if problems:
        print(f" RESULT: {len(problems)} schema issues found")
        for p in problems:
            print(f"   - {p}")
        print("\n Run with --migrate to fix missing columns")
        return 1
    else:
        print(" RESULT: Schema OK - all required columns present")
    print('='*70)

    return 0


if __name__ == "__main__":
    exit(main())
