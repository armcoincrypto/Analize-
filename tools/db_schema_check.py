#!/usr/bin/env python3
"""
Database Schema Check Tool
==========================
Verifies that the database schema matches the expected structure.

Usage:
    python tools/db_schema_check.py --db hft_trades.db
"""

import argparse
import sqlite3
from pathlib import Path


def check_table_schema(conn, table_name: str):
    """Print table schema using PRAGMA table_info."""
    cursor = conn.cursor()
    print(f"\n{'='*60}")
    print(f"TABLE: {table_name}")
    print('='*60)

    try:
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()

        if not columns:
            print(f"  Table '{table_name}' does not exist!")
            return False

        print(f"  {'#':<3} {'Name':<25} {'Type':<15} {'NotNull':<8} {'Default':<15}")
        print("-" * 70)

        for col in columns:
            cid, name, dtype, notnull, default, pk = col
            default_str = str(default) if default else ""
            print(f"  {cid:<3} {name:<25} {dtype:<15} {notnull:<8} {default_str:<15}")

        print(f"\n  Total columns: {len(columns)}")
        return True

    except Exception as e:
        print(f"  Error checking table: {e}")
        return False


def show_last_rows(conn, table_name: str, n: int = 5):
    """Show last N rows from a table."""
    cursor = conn.cursor()
    print(f"\n  Last {n} rows:")

    try:
        # Get column names
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [col[1] for col in cursor.fetchall()]

        if not columns:
            return

        # Get last N rows
        cursor.execute(f"SELECT * FROM {table_name} ORDER BY rowid DESC LIMIT {n}")
        rows = cursor.fetchall()

        if not rows:
            print("  (no data)")
            return

        # Print header (first 6 columns max for readability)
        show_cols = columns[:6]
        header = " | ".join(f"{c[:15]:<15}" for c in show_cols)
        print(f"  {header}")
        print("  " + "-" * len(header))

        # Print rows
        for row in rows:
            row_data = row[:6]
            row_str = " | ".join(f"{str(v)[:15]:<15}" for v in row_data)
            print(f"  {row_str}")

    except Exception as e:
        print(f"  Error fetching rows: {e}")


def check_all_tables(conn):
    """Check all expected tables."""
    expected_tables = [
        "trades",
        "trade_causality",
        "trade_edge",
        "blocked_signals",
        "winner_gate_blocks",
        "market_regime",
        "position_sizing",
        "no_trade_zone_blocks"
    ]

    found = []
    missing = []

    for table in expected_tables:
        if check_table_schema(conn, table):
            found.append(table)
            show_last_rows(conn, table, 5)
        else:
            missing.append(table)

    return found, missing


def check_data_integrity(conn):
    """Check data integrity and join quality."""
    cursor = conn.cursor()
    print(f"\n{'='*60}")
    print("DATA INTEGRITY CHECKS")
    print('='*60)

    checks = []

    # Check 1: Total trades
    try:
        cursor.execute("SELECT COUNT(*) FROM trades WHERE status='closed'")
        total_trades = cursor.fetchone()[0]
        print(f"\n  Total closed trades: {total_trades}")
        checks.append(("total_trades", total_trades))
    except:
        print("  ERROR: Cannot count trades")

    # Check 2: Trades with MFE/MAE data
    try:
        cursor.execute("SELECT COUNT(*) FROM trades WHERE status='closed' AND (mfe != 0 OR mae != 0)")
        mfe_trades = cursor.fetchone()[0]
        pct = (mfe_trades / total_trades * 100) if total_trades > 0 else 0
        print(f"  Trades with MFE/MAE: {mfe_trades} ({pct:.1f}%)")
        checks.append(("mfe_coverage", pct))
    except:
        print("  ERROR: Cannot check MFE/MAE")

    # Check 3: Trade causality join rate
    try:
        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN tc.trade_id IS NOT NULL THEN 1 ELSE 0 END) as joined
            FROM trades t
            LEFT JOIN trade_causality tc ON t.id = tc.trade_id
            WHERE t.status = 'closed'
        """)
        row = cursor.fetchone()
        join_rate = (row[1] / row[0] * 100) if row[0] > 0 else 0
        print(f"  Trade->Causality join rate: {row[1]}/{row[0]} ({join_rate:.1f}%)")
        checks.append(("causality_join", join_rate))
    except:
        print("  ERROR: Cannot check causality join")

    # Check 4: Null execution_mode
    try:
        cursor.execute("SELECT COUNT(*) FROM trades WHERE execution_mode IS NULL AND status='closed'")
        null_mode = cursor.fetchone()[0]
        print(f"  Trades with NULL execution_mode: {null_mode}")
        checks.append(("null_exec_mode", null_mode))
    except:
        print("  ERROR: Cannot check execution_mode")

    # Check 5: Trades with 0 costs
    try:
        cursor.execute("SELECT COUNT(*) FROM trades WHERE total_costs_pct = 0 AND status='closed'")
        zero_cost = cursor.fetchone()[0]
        print(f"  Trades with zero costs: {zero_cost}")
        checks.append(("zero_cost", zero_cost))
    except:
        print("  ERROR: Cannot check zero costs")

    return checks


def main():
    parser = argparse.ArgumentParser(description="Database Schema Check")
    parser.add_argument("--db", default="hft_trades.db", help="Database path")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        # Try home directory
        home_path = Path.home() / "Analize-" / "hft_trades.db"
        if home_path.exists():
            db_path = home_path
        else:
            print(f"Error: Database not found at {db_path}")
            return 1

    print(f"Database Schema Check: {db_path}")
    print(f"{'='*60}")

    conn = sqlite3.connect(str(db_path))

    try:
        found, missing = check_all_tables(conn)
        check_data_integrity(conn)

        print(f"\n{'='*60}")
        print("SUMMARY")
        print('='*60)
        print(f"  Tables found: {len(found)}")
        print(f"  Tables missing: {len(missing)}")
        if missing:
            print(f"  Missing: {', '.join(missing)}")

    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    exit(main())
