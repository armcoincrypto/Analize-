#!/usr/bin/env python3
"""
Maker vs Taker Performance Report
=================================
Compares profitability between maker and taker execution modes.

Single source of truth:
- MAKER orders: maker_order_telemetry table
- TAKER trades: trades table with execution_mode='taker' or NULL

Usage:
    python tools/maker_taker_report.py --db hft_trades.db
    python tools/maker_taker_report.py --db hft_trades.db --days 7
    python tools/maker_taker_report.py --db hft_trades.db --debug-sql
    python tools/maker_taker_report.py --db hft_trades.db --csv report.csv

Output:
    - Trade counts and win rates by execution mode
    - Maker order fill analysis from maker_order_telemetry
    - Average PnL after costs by mode
    - Average fees, spread costs, and slippage by mode
"""

import argparse
import sqlite3
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# Add parent dir for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from hft_system.db import open_sqlite

# Global debug flag
DEBUG_SQL = False


def debug_print(msg: str):
    """Print debug message if debug mode enabled."""
    if DEBUG_SQL:
        print(f"  [DEBUG] {msg}")


def print_header(title: str, char: str = "="):
    width = 70
    print(f"\n{char * width}")
    print(f" {title}")
    print(char * width)


def get_time_cutoff_ms(days: int) -> int:
    """Get timestamp cutoff in milliseconds."""
    return int((datetime.now() - timedelta(days=days)).timestamp() * 1000)


def get_maker_order_stats(conn: sqlite3.Connection, days: int = None) -> Dict:
    """
    Get maker order statistics from maker_order_telemetry table.

    This is the SINGLE SOURCE OF TRUTH for maker orders.

    Status definitions:
    - filled: filled_ts IS NOT NULL
    - cancelled: filled_ts IS NULL AND cancelled_ts IS NOT NULL
    - pending: filled_ts IS NULL AND cancelled_ts IS NULL
    """
    cursor = conn.cursor()

    # Build time filter using posted_ts (ms timestamp)
    time_filter = ""
    params = []
    if days:
        cutoff_ms = get_time_cutoff_ms(days)
        time_filter = "WHERE posted_ts >= ?"
        params.append(cutoff_ms)
        debug_print(f"Time filter: posted_ts >= {cutoff_ms} ({days} days ago)")

    # Count by status
    query = f"""
        SELECT
            CASE
                WHEN filled_ts IS NOT NULL THEN 'filled'
                WHEN cancelled_ts IS NOT NULL THEN 'cancelled'
                ELSE 'pending'
            END as status,
            COUNT(*) as count,
            AVG(fill_ratio) as avg_fill_ratio,
            AVG(time_to_fill_ms) as avg_time_to_fill_ms,
            AVG(slippage_from_limit_pct) as avg_slippage_pct
        FROM maker_order_telemetry
        {time_filter}
        GROUP BY status
    """

    debug_print(f"Maker order stats query:\n{query}")
    debug_print(f"Params: {params}")

    cursor.execute(query, params)
    rows = cursor.fetchall()

    stats = {
        "filled": {"count": 0, "avg_fill_ratio": 0, "avg_time_to_fill_ms": 0, "avg_slippage_pct": 0},
        "cancelled": {"count": 0, "avg_fill_ratio": 0, "avg_time_to_fill_ms": 0, "avg_slippage_pct": 0},
        "pending": {"count": 0, "avg_fill_ratio": 0, "avg_time_to_fill_ms": 0, "avg_slippage_pct": 0},
    }

    for row in rows:
        status = row[0]
        stats[status] = {
            "count": row[1],
            "avg_fill_ratio": row[2] or 0,
            "avg_time_to_fill_ms": row[3] or 0,
            "avg_slippage_pct": row[4] or 0,
        }
        debug_print(f"  {status}: {row[1]} orders")

    # Calculate totals
    total = sum(s["count"] for s in stats.values())
    filled = stats["filled"]["count"]
    cancelled = stats["cancelled"]["count"]
    pending = stats["pending"]["count"]

    stats["total"] = total
    stats["fill_rate_pct"] = (filled / (filled + cancelled) * 100) if (filled + cancelled) > 0 else 0

    debug_print(f"Total maker orders: {total}, Fill rate: {stats['fill_rate_pct']:.1f}%")

    return stats


def get_maker_order_by_symbol(conn: sqlite3.Connection, days: int = None) -> Dict:
    """Get maker order breakdown by symbol."""
    cursor = conn.cursor()

    time_filter = ""
    params = []
    if days:
        cutoff_ms = get_time_cutoff_ms(days)
        time_filter = "WHERE posted_ts >= ?"
        params.append(cutoff_ms)

    query = f"""
        SELECT
            symbol,
            COUNT(*) as total,
            SUM(CASE WHEN filled_ts IS NOT NULL THEN 1 ELSE 0 END) as filled,
            SUM(CASE WHEN cancelled_ts IS NOT NULL AND filled_ts IS NULL THEN 1 ELSE 0 END) as cancelled,
            AVG(CASE WHEN filled_ts IS NOT NULL THEN time_to_fill_ms END) as avg_fill_time_ms
        FROM maker_order_telemetry
        {time_filter}
        GROUP BY symbol
        ORDER BY total DESC
    """

    debug_print(f"Maker by symbol query:\n{query}")

    cursor.execute(query, params)
    rows = cursor.fetchall()

    result = {}
    for row in rows:
        symbol = row[0] or "UNKNOWN"
        result[symbol] = {
            "total": row[1],
            "filled": row[2],
            "cancelled": row[3],
            "fill_rate_pct": (row[2] / row[1] * 100) if row[1] > 0 else 0,
            "avg_fill_time_ms": row[4] or 0
        }
        debug_print(f"  {symbol}: {row[1]} orders, {row[2]} filled")

    return result


def get_taker_trade_stats(conn: sqlite3.Connection, days: int = None) -> Dict:
    """
    Get taker trade statistics from trades table.

    Taker = execution_mode IS NULL OR execution_mode = 'taker'
    """
    cursor = conn.cursor()

    time_filter = ""
    params = []
    if days:
        cutoff_ms = get_time_cutoff_ms(days)
        time_filter = "AND entry_time >= ?"
        params.append(cutoff_ms)
        debug_print(f"Taker time filter: entry_time >= {cutoff_ms}")

    query = f"""
        SELECT
            COUNT(*) as trade_count,
            SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN pnl_pct <= 0 THEN 1 ELSE 0 END) as losses,
            AVG(pnl_pct) as avg_pnl_pct,
            AVG(pnl_after_costs_pct) as avg_pnl_after_costs_pct,
            SUM(pnl) as total_pnl,
            AVG(entry_fee_pct) as avg_entry_fee,
            AVG(exit_fee_pct) as avg_exit_fee,
            AVG(spread_cost_pct) as avg_spread_cost,
            AVG(slippage_pct) as avg_slippage,
            AVG(total_costs_pct) as avg_total_costs,
            AVG(hold_time_sec) as avg_hold_time
        FROM trades
        WHERE status = 'closed'
        AND (execution_mode IS NULL OR execution_mode = 'taker')
        {time_filter}
    """

    debug_print(f"Taker stats query:\n{query}")

    cursor.execute(query, params)
    row = cursor.fetchone()

    if not row or row[0] == 0:
        debug_print("No taker trades found")
        return {}

    trade_count = row[0]
    wins = row[1] or 0

    stats = {
        "trade_count": trade_count,
        "wins": wins,
        "losses": row[2] or 0,
        "win_rate": (wins / trade_count * 100) if trade_count > 0 else 0,
        "avg_pnl_pct": row[3] or 0,
        "avg_pnl_after_costs_pct": row[4] or 0,
        "total_pnl": row[5] or 0,
        "avg_entry_fee_pct": row[6] or 0,
        "avg_exit_fee_pct": row[7] or 0,
        "avg_spread_cost_pct": row[8] or 0,
        "avg_slippage_pct": row[9] or 0,
        "avg_total_costs_pct": row[10] or 0,
        "avg_hold_time_sec": row[11] or 0,
    }

    debug_print(f"Taker trades: {trade_count}, Win rate: {stats['win_rate']:.1f}%")

    return stats


def get_maker_trade_stats(conn: sqlite3.Connection, days: int = None) -> Dict:
    """
    Get maker trade statistics from trades table.

    Maker = execution_mode = 'maker'
    """
    cursor = conn.cursor()

    time_filter = ""
    params = []
    if days:
        cutoff_ms = get_time_cutoff_ms(days)
        time_filter = "AND entry_time >= ?"
        params.append(cutoff_ms)

    query = f"""
        SELECT
            COUNT(*) as trade_count,
            SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN pnl_pct <= 0 THEN 1 ELSE 0 END) as losses,
            AVG(pnl_pct) as avg_pnl_pct,
            AVG(pnl_after_costs_pct) as avg_pnl_after_costs_pct,
            SUM(pnl) as total_pnl,
            AVG(entry_fee_pct) as avg_entry_fee,
            AVG(exit_fee_pct) as avg_exit_fee,
            AVG(spread_cost_pct) as avg_spread_cost,
            AVG(slippage_pct) as avg_slippage,
            AVG(total_costs_pct) as avg_total_costs,
            AVG(hold_time_sec) as avg_hold_time
        FROM trades
        WHERE status = 'closed'
        AND execution_mode = 'maker'
        {time_filter}
    """

    debug_print(f"Maker trade stats query:\n{query}")

    cursor.execute(query, params)
    row = cursor.fetchone()

    if not row or row[0] == 0:
        debug_print("No maker trades found in trades table")
        return {}

    trade_count = row[0]
    wins = row[1] or 0

    stats = {
        "trade_count": trade_count,
        "wins": wins,
        "losses": row[2] or 0,
        "win_rate": (wins / trade_count * 100) if trade_count > 0 else 0,
        "avg_pnl_pct": row[3] or 0,
        "avg_pnl_after_costs_pct": row[4] or 0,
        "total_pnl": row[5] or 0,
        "avg_entry_fee_pct": row[6] or 0,
        "avg_exit_fee_pct": row[7] or 0,
        "avg_spread_cost_pct": row[8] or 0,
        "avg_slippage_pct": row[9] or 0,
        "avg_total_costs_pct": row[10] or 0,
        "avg_hold_time_sec": row[11] or 0,
    }

    debug_print(f"Maker trades: {trade_count}, Win rate: {stats['win_rate']:.1f}%")

    return stats


def print_maker_order_stats(stats: Dict):
    """Print maker order statistics from telemetry."""
    print_header("MAKER ORDER TELEMETRY", "-")

    total = stats.get("total", 0)
    if total == 0:
        print("  No maker orders in telemetry")
        return

    filled = stats["filled"]["count"]
    cancelled = stats["cancelled"]["count"]
    pending = stats["pending"]["count"]
    fill_rate = stats["fill_rate_pct"]

    print(f"\n  Total orders:     {total}")
    print(f"  Filled:           {filled} ({filled/total*100:.1f}%)")
    print(f"  Cancelled:        {cancelled} ({cancelled/total*100:.1f}%)")
    print(f"  Pending:          {pending} ({pending/total*100:.1f}%)")
    print(f"\n  Fill rate (filled / (filled + cancelled)): {fill_rate:.1f}%")

    if stats["filled"]["avg_time_to_fill_ms"] > 0:
        print(f"  Avg time to fill: {stats['filled']['avg_time_to_fill_ms']:.0f}ms")
    if stats["filled"]["avg_slippage_pct"] != 0:
        print(f"  Avg slippage:     {stats['filled']['avg_slippage_pct']:.4f}%")


def print_maker_by_symbol(by_symbol: Dict):
    """Print maker order breakdown by symbol."""
    if not by_symbol:
        return

    print("\n  --- BY SYMBOL ---")
    print(f"  {'Symbol':<12} {'Total':>8} {'Filled':>8} {'Fill %':>8} {'Avg Fill ms':>12}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*12}")

    for symbol, data in by_symbol.items():
        print(f"  {symbol:<12} {data['total']:>8} {data['filled']:>8} "
              f"{data['fill_rate_pct']:>7.1f}% {data['avg_fill_time_ms']:>11.0f}")


def print_trade_stats(stats: Dict, mode: str):
    """Print trade stats for a mode."""
    if not stats:
        print(f"\n  No {mode} trades found")
        return

    print(f"\n  --- {mode.upper()} TRADES ---")
    print(f"  Trades:       {stats['trade_count']:>6}")
    print(f"  Wins:         {stats['wins']:>6} ({stats['win_rate']:.1f}% WR)")
    print(f"  Losses:       {stats['losses']:>6}")
    print()
    print(f"  Avg PnL (gross):       {stats['avg_pnl_pct']:>+8.4f}%")
    print(f"  Avg PnL (after costs): {stats['avg_pnl_after_costs_pct']:>+8.4f}%")
    print(f"  Total PnL:             ${stats['total_pnl']:>+10.2f}")
    print()
    print(f"  --- COSTS ---")
    print(f"  Entry fee:    {stats['avg_entry_fee_pct']:>8.4f}%")
    print(f"  Exit fee:     {stats['avg_exit_fee_pct']:>8.4f}%")
    print(f"  Spread cost:  {stats['avg_spread_cost_pct']:>8.4f}%")
    print(f"  Slippage:     {stats['avg_slippage_pct']:>8.4f}%")
    print(f"  TOTAL COSTS:  {stats['avg_total_costs_pct']:>8.4f}%")
    print()
    print(f"  Avg hold time: {stats['avg_hold_time_sec']:.1f}s")


def print_comparison_table(taker_stats: Dict, maker_stats: Dict):
    """Print side-by-side comparison table."""
    print_header("MAKER vs TAKER COMPARISON")

    if not taker_stats and not maker_stats:
        print("  No data available for either mode")
        return

    # Table header
    print(f"\n  {'Metric':<25} {'TAKER':>12} {'MAKER':>12} {'DIFF':>12}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12}")

    # Rows
    metrics = [
        ("Trade Count", "trade_count", "{:>12}", False),
        ("Win Rate %", "win_rate", "{:>11.1f}%", False),
        ("Avg PnL (gross) %", "avg_pnl_pct", "{:>+11.4f}%", True),
        ("Avg PnL (net) %", "avg_pnl_after_costs_pct", "{:>+11.4f}%", True),
        ("Total PnL $", "total_pnl", "{:>+11.2f}$", True),
        ("Total Costs %", "avg_total_costs_pct", "{:>11.4f}%", True),
        ("Avg Hold Time (s)", "avg_hold_time_sec", "{:>11.1f}s", False),
    ]

    for label, key, fmt, show_diff in metrics:
        t_val = taker_stats.get(key, 0) if taker_stats else 0
        m_val = maker_stats.get(key, 0) if maker_stats else 0

        t_str = fmt.format(t_val) if taker_stats else "N/A"
        m_str = fmt.format(m_val) if maker_stats else "N/A"

        if show_diff and taker_stats and maker_stats:
            diff = m_val - t_val
            diff_str = f"{diff:>+11.4f}%" if "%" in fmt else f"{diff:>+11.2f}"
        else:
            diff_str = ""

        print(f"  {label:<25} {t_str:>12} {m_str:>12} {diff_str:>12}")


def export_to_csv(taker_stats: Dict, maker_stats: Dict, maker_order_stats: Dict, filepath: str):
    """Export report to CSV."""
    with open(filepath, 'w') as f:
        # Summary section
        f.write("Maker vs Taker Performance Report\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")

        # Trade stats by mode
        f.write("Trade Stats by Mode\n")
        f.write("Mode,Trades,Wins,Losses,Win Rate %,Avg PnL %,Avg PnL After Costs %,Total Costs %\n")

        for mode, stats in [("taker", taker_stats), ("maker", maker_stats)]:
            if stats:
                f.write(f"{mode},{stats['trade_count']},{stats['wins']},{stats['losses']},")
                f.write(f"{stats['win_rate']:.2f},{stats['avg_pnl_pct']:.4f},")
                f.write(f"{stats['avg_pnl_after_costs_pct']:.4f},{stats['avg_total_costs_pct']:.4f}\n")

        f.write("\n")

        # Maker order telemetry
        f.write("Maker Order Telemetry\n")
        f.write("Status,Count\n")
        for status in ["filled", "cancelled", "pending"]:
            f.write(f"{status},{maker_order_stats[status]['count']}\n")
        f.write(f"fill_rate_pct,{maker_order_stats['fill_rate_pct']:.2f}\n")

    print(f"\n  Report exported to: {filepath}")


def main():
    global DEBUG_SQL

    parser = argparse.ArgumentParser(
        description="Maker vs Taker Performance Report"
    )
    parser.add_argument("--db", default="hft_trades.db", help="Database path")
    parser.add_argument("--days", type=int, help="Filter to last N days")
    parser.add_argument("--csv", help="Export to CSV file")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")
    parser.add_argument("--debug-sql", action="store_true", help="Print SQL queries and row counts")
    args = parser.parse_args()

    DEBUG_SQL = args.debug_sql

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

    # Connect with bot-safe WAL mode and busy_timeout
    conn = open_sqlite(str(db_path))

    # Get data from single sources of truth
    maker_order_stats = get_maker_order_stats(conn, args.days)
    maker_by_symbol = get_maker_order_by_symbol(conn, args.days)
    taker_stats = get_taker_trade_stats(conn, args.days)
    maker_trade_stats = get_maker_trade_stats(conn, args.days)

    if not args.quiet:
        # Print report
        print("""
+======================================================================+
|              MAKER vs TAKER PERFORMANCE REPORT                       |
+======================================================================+
        """)

        if args.days:
            print(f"  Period: Last {args.days} days")
        else:
            print("  Period: All time")

        print(f"  Database: {db_path}")

        # Maker order telemetry (single source of truth for maker orders)
        print_maker_order_stats(maker_order_stats)
        print_maker_by_symbol(maker_by_symbol)

        # Trade stats from trades table
        print_header("TRADE PERFORMANCE (from trades table)")
        print_trade_stats(taker_stats, "taker")
        print_trade_stats(maker_trade_stats, "maker")

        # Comparison table
        print_comparison_table(taker_stats, maker_trade_stats)

        # Summary
        print_header("SUMMARY")

        total_maker_orders = maker_order_stats.get("total", 0)
        maker_trades = maker_trade_stats.get("trade_count", 0) if maker_trade_stats else 0
        taker_trades = taker_stats.get("trade_count", 0) if taker_stats else 0

        print(f"\n  Maker orders in telemetry: {total_maker_orders}")
        print(f"  Maker trades in trades table: {maker_trades}")
        print(f"  Taker trades in trades table: {taker_trades}")

        if maker_trade_stats and taker_stats:
            maker_net = maker_trade_stats.get("avg_pnl_after_costs_pct", 0)
            taker_net = taker_stats.get("avg_pnl_after_costs_pct", 0)

            if maker_net > taker_net:
                print(f"\n  MAKER outperforms TAKER by {maker_net - taker_net:.4f}% per trade")
            elif taker_net > maker_net:
                print(f"\n  TAKER outperforms MAKER by {taker_net - maker_net:.4f}% per trade")
            else:
                print("\n  MAKER and TAKER perform equally")
        elif maker_trade_stats:
            print("\n  Only MAKER data available in trades table")
        elif taker_stats:
            print("\n  Only TAKER data available in trades table")
        else:
            print("\n  No trade data available in trades table")

        print()
        print("=" * 70)

    # Export to CSV if requested
    if args.csv:
        export_to_csv(taker_stats, maker_trade_stats, maker_order_stats, args.csv)

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
