#!/usr/bin/env python3
"""
Symbol Migration Script
=======================
One-time migration to update existing database rows from short symbols (XRP)
to full trading pair symbols (XRPUSDT).

Usage:
    python tools/migrate_symbols.py --db hft_trades.db
    python tools/migrate_symbols.py --db hft_trades.db --dry-run
    python tools/migrate_symbols.py --db hft_trades.db --quote USDT

This script:
1. Finds all rows with symbols that don't have a quote currency suffix
2. Updates them to add the default quote currency (USDT)
3. Reports the number of rows updated per table
"""

import argparse
import sqlite3
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

# Add parent dir for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from hft_system.symbol_utils import QUOTE_CURRENCIES, normalize_symbol
from hft_system.db import open_sqlite


# Tables that have a symbol column that should be migrated
TABLES_WITH_SYMBOL = [
    "trades",
    "signals",
    "market_snapshots",
    "tick_data",
    "orderbook_snapshots",
    "indicator_snapshots",
    "strategy_signals",
    "trade_flow",
    "cvd_snapshots",
    "trade_entry_flow",
    "trade_quality_metrics",
    "trade_causality",
    "trade_edge",
    "market_regime",
    "blocked_signals",
    "winner_gate_blocks",
    "position_sizing",
    "maker_order_telemetry",
]


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    """Check if a table exists."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name=?
    """, (table,))
    return cursor.fetchone() is not None


def has_symbol_column(conn: sqlite3.Connection, table: str) -> bool:
    """Check if table has a symbol column."""
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table})")
    columns = [row[1] for row in cursor.fetchall()]
    return "symbol" in columns


def get_unmigrated_symbols(conn: sqlite3.Connection, table: str) -> List[Tuple[str, int]]:
    """
    Get symbols that need migration (don't have quote currency suffix).

    Returns list of (symbol, count) tuples.
    """
    cursor = conn.cursor()

    # Build condition to find symbols without quote currencies
    # We look for symbols that don't end with known quote currencies
    conditions = []
    for quote in QUOTE_CURRENCIES:
        conditions.append(f"symbol NOT LIKE '%{quote}'")

    where_clause = " AND ".join(conditions)

    query = f"""
        SELECT symbol, COUNT(*) as count
        FROM {table}
        WHERE symbol IS NOT NULL
          AND symbol != ''
          AND ({where_clause})
        GROUP BY symbol
        ORDER BY count DESC
    """

    try:
        cursor.execute(query)
        return [(row[0], row[1]) for row in cursor.fetchall()]
    except sqlite3.OperationalError as e:
        print(f"  WARNING: Error querying {table}: {e}")
        return []


def migrate_table(conn: sqlite3.Connection, table: str, quote: str, dry_run: bool = False) -> int:
    """
    Migrate symbols in a table to full format.

    Returns number of rows updated.
    """
    cursor = conn.cursor()

    # Get unmigrated symbols
    unmigrated = get_unmigrated_symbols(conn, table)

    if not unmigrated:
        return 0

    total_updated = 0

    for old_symbol, count in unmigrated:
        new_symbol = normalize_symbol(old_symbol, quote)

        if old_symbol == new_symbol:
            continue  # Already normalized

        if dry_run:
            print(f"    [DRY-RUN] Would update {count} rows: '{old_symbol}' -> '{new_symbol}'")
            total_updated += count
        else:
            try:
                cursor.execute(f"""
                    UPDATE {table}
                    SET symbol = ?
                    WHERE symbol = ?
                """, (new_symbol, old_symbol))
                rows_affected = cursor.rowcount
                total_updated += rows_affected
                print(f"    Updated {rows_affected} rows: '{old_symbol}' -> '{new_symbol}'")
            except sqlite3.OperationalError as e:
                print(f"    ERROR updating '{old_symbol}': {e}")

    if not dry_run and total_updated > 0:
        conn.commit()

    return total_updated


def main():
    parser = argparse.ArgumentParser(
        description="Migrate database symbols from short format (XRP) to full format (XRPUSDT)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/migrate_symbols.py --db hft_trades.db --dry-run  # Preview changes
  python tools/migrate_symbols.py --db hft_trades.db            # Apply migration
  python tools/migrate_symbols.py --db hft_trades.db --quote USDT
        """
    )
    parser.add_argument("--db", required=True, help="Database path")
    parser.add_argument("--quote", default="USDT", help="Quote currency to append (default: USDT)")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without applying")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")
    args = parser.parse_args()

    # Find database
    db_path = Path(args.db)
    if not db_path.exists():
        # Try common locations
        for try_path in [
            Path.cwd() / args.db,
            Path.home() / "Analize-" / args.db,
            Path("/root/Analize-") / args.db
        ]:
            if try_path.exists():
                db_path = try_path
                break

    if not db_path.exists():
        print(f"ERROR: Database not found at {args.db}")
        return 1

    # Connect with backup warning
    if not args.dry_run:
        print(f"""
================================================================================
                    SYMBOL MIGRATION SCRIPT
================================================================================
Database: {db_path}
Quote:    {args.quote}
Mode:     {'DRY-RUN (preview only)' if args.dry_run else 'LIVE (will modify database)'}

This script will update all symbol columns from short format (e.g., 'XRP')
to full trading pair format (e.g., 'XRPUSDT').

""")
        if not args.quiet:
            response = input("IMPORTANT: Have you backed up your database? (y/n): ")
            if response.lower() != 'y':
                print("Please backup your database first:")
                print(f"  cp {db_path} {db_path}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                return 1
    else:
        print(f"""
================================================================================
                    SYMBOL MIGRATION SCRIPT (DRY-RUN)
================================================================================
Database: {db_path}
Quote:    {args.quote}
Mode:     DRY-RUN (preview only - no changes will be made)

""")

    # Connect (bot-safe with WAL + busy_timeout)
    conn = open_sqlite(str(db_path))

    # Process each table
    total_tables = 0
    total_rows = 0

    print("Scanning tables...")
    print()

    for table in TABLES_WITH_SYMBOL:
        if not table_exists(conn, table):
            continue

        if not has_symbol_column(conn, table):
            continue

        # Check for unmigrated symbols
        unmigrated = get_unmigrated_symbols(conn, table)

        if not unmigrated:
            if not args.quiet:
                print(f"  {table}: No migration needed")
            continue

        total_count = sum(count for _, count in unmigrated)
        print(f"  {table}: Found {total_count} rows to migrate")

        rows_updated = migrate_table(conn, table, args.quote, args.dry_run)

        if rows_updated > 0:
            total_tables += 1
            total_rows += rows_updated

    conn.close()

    # Summary
    print()
    print("=" * 60)
    if args.dry_run:
        print(f"DRY-RUN COMPLETE: Would update {total_rows} rows in {total_tables} tables")
        print()
        print("To apply changes, run without --dry-run flag:")
        print(f"  python tools/migrate_symbols.py --db {args.db}")
    else:
        print(f"MIGRATION COMPLETE: Updated {total_rows} rows in {total_tables} tables")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
