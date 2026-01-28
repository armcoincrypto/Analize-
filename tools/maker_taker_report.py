#!/usr/bin/env python3
"""
Maker vs Taker Performance Report
=================================
Compares profitability between maker and taker execution modes.

Usage:
    python tools/maker_taker_report.py --db hft_trades.db
    python tools/maker_taker_report.py --db hft_trades.db --days 7
    python tools/maker_taker_report.py --db hft_trades.db --csv report.csv

Output:
    - Trade counts and win rates by execution mode
    - Average PnL after costs by mode
    - Average fees, spread costs, and slippage by mode
    - Fill rate analysis (if maker order cancellation data available)
"""

import argparse
import sqlite3
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Add parent dir for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from hft_system.db import open_sqlite


def print_header(title: str, char: str = "="):
    width = 70
    print(f"\n{char * width}")
    print(f" {title}")
    print(char * width)


def get_execution_mode_stats(conn: sqlite3.Connection, days: int = None) -> Dict:
    """Get performance statistics grouped by execution mode."""
    cursor = conn.cursor()

    # Build time filter
    time_filter = ""
    params = []
    if days:
        cutoff_ms = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        time_filter = "AND entry_time >= ?"
        params.append(cutoff_ms)

    # Main query for stats by execution mode
    query = f"""
        SELECT
            COALESCE(execution_mode, 'taker') as mode,
            COUNT(*) as trade_count,
            SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN pnl_pct <= 0 THEN 1 ELSE 0 END) as losses,
            AVG(pnl_pct) as avg_pnl_pct,
            AVG(pnl_after_costs_pct) as avg_pnl_after_costs_pct,
            SUM(pnl) as total_pnl,
            SUM(pnl_after_costs_pct * quantity * entry_price / 100) as total_pnl_after_costs,
            AVG(entry_fee_pct) as avg_entry_fee,
            AVG(exit_fee_pct) as avg_exit_fee,
            AVG(COALESCE(fees_paid_pct, entry_fee_pct + exit_fee_pct)) as avg_fees_paid,
            AVG(spread_cost_pct) as avg_spread_cost,
            AVG(slippage_pct) as avg_slippage,
            AVG(total_costs_pct) as avg_total_costs,
            AVG(hold_time_sec) as avg_hold_time
        FROM trades
        WHERE status = 'closed'
        {time_filter}
        GROUP BY COALESCE(execution_mode, 'taker')
        ORDER BY trade_count DESC
    """

    cursor.execute(query, params)
    rows = cursor.fetchall()

    stats = {}
    for row in rows:
        mode = row[0]
        trade_count = row[1]
        wins = row[2]
        losses = row[3]

        stats[mode] = {
            "trade_count": trade_count,
            "wins": wins,
            "losses": losses,
            "win_rate": (wins / trade_count * 100) if trade_count > 0 else 0,
            "avg_pnl_pct": row[4] or 0,
            "avg_pnl_after_costs_pct": row[5] or 0,
            "total_pnl": row[6] or 0,
            "total_pnl_after_costs": row[7] or 0,
            "avg_entry_fee_pct": row[8] or 0,
            "avg_exit_fee_pct": row[9] or 0,
            "avg_fees_paid_pct": row[10] or 0,
            "avg_spread_cost_pct": row[11] or 0,
            "avg_slippage_pct": row[12] or 0,
            "avg_total_costs_pct": row[13] or 0,
            "avg_hold_time_sec": row[14] or 0,
        }

    return stats


def get_exit_reason_breakdown(conn: sqlite3.Connection, days: int = None) -> Dict:
    """Get exit reason breakdown by execution mode."""
    cursor = conn.cursor()

    time_filter = ""
    params = []
    if days:
        cutoff_ms = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        time_filter = "AND entry_time >= ?"
        params.append(cutoff_ms)

    query = f"""
        SELECT
            COALESCE(execution_mode, 'taker') as mode,
            exit_reason,
            COUNT(*) as count,
            AVG(pnl_after_costs_pct) as avg_pnl
        FROM trades
        WHERE status = 'closed'
        {time_filter}
        GROUP BY COALESCE(execution_mode, 'taker'), exit_reason
        ORDER BY mode, count DESC
    """

    cursor.execute(query, params)
    rows = cursor.fetchall()

    breakdown = {}
    for row in rows:
        mode = row[0]
        reason = row[1] or "unknown"
        if mode not in breakdown:
            breakdown[mode] = {}
        breakdown[mode][reason] = {
            "count": row[2],
            "avg_pnl_pct": row[3] or 0
        }

    return breakdown


def get_cost_comparison(conn: sqlite3.Connection) -> Dict:
    """Get theoretical vs actual cost comparison."""
    cursor = conn.cursor()

    # Theoretical costs from config
    theoretical = {
        "taker": {
            "entry_fee": 0.10,
            "exit_fee": 0.10,
            "spread_cost": 0.02,
            "slippage": 0.02,
            "total": 0.26
        },
        "maker": {
            "entry_fee": 0.01,
            "exit_fee": 0.01,
            "spread_cost": 0.005,
            "slippage": 0.0,
            "total": 0.03
        }
    }

    # Actual costs from trades
    query = """
        SELECT
            COALESCE(execution_mode, 'taker') as mode,
            AVG(entry_fee_pct) as entry_fee,
            AVG(exit_fee_pct) as exit_fee,
            AVG(spread_cost_pct) as spread_cost,
            AVG(slippage_pct) as slippage,
            AVG(total_costs_pct) as total
        FROM trades
        WHERE status = 'closed'
        GROUP BY COALESCE(execution_mode, 'taker')
    """

    cursor.execute(query)
    rows = cursor.fetchall()

    actual = {}
    for row in rows:
        actual[row[0]] = {
            "entry_fee": row[1] or 0,
            "exit_fee": row[2] or 0,
            "spread_cost": row[3] or 0,
            "slippage": row[4] or 0,
            "total": row[5] or 0
        }

    return {"theoretical": theoretical, "actual": actual}


def get_maker_fill_analysis(conn: sqlite3.Connection) -> Optional[Dict]:
    """
    Analyze maker order fill rates if data available.

    Note: This requires tracking of unfilled/cancelled maker orders,
    which may not be available in the current schema.
    """
    cursor = conn.cursor()

    # Check if we have maker cancellation data
    try:
        cursor.execute("""
            SELECT COUNT(*) FROM blocked_signals
            WHERE block_reason = 'maker_no_fill'
        """)
        maker_no_fills = cursor.fetchone()[0]

        cursor.execute("""
            SELECT COUNT(*) FROM trades
            WHERE execution_mode = 'maker' AND status = 'closed'
        """)
        maker_fills = cursor.fetchone()[0]

        if maker_fills + maker_no_fills > 0:
            fill_rate = maker_fills / (maker_fills + maker_no_fills) * 100
            return {
                "maker_fills": maker_fills,
                "maker_no_fills": maker_no_fills,
                "fill_rate_pct": fill_rate
            }
    except Exception:
        pass

    return None


def print_mode_stats(stats: Dict, mode: str):
    """Print stats for a single execution mode."""
    s = stats.get(mode, {})
    if not s:
        print(f"  No data for {mode} mode")
        return

    print(f"\n  --- {mode.upper()} MODE ---")
    print(f"  Trades:       {s['trade_count']:>6}")
    print(f"  Wins:         {s['wins']:>6} ({s['win_rate']:.1f}% WR)")
    print(f"  Losses:       {s['losses']:>6}")
    print()
    print(f"  Avg PnL (gross):     {s['avg_pnl_pct']:>+8.4f}%")
    print(f"  Avg PnL (after costs): {s['avg_pnl_after_costs_pct']:>+8.4f}%")
    print(f"  Total PnL:           ${s['total_pnl']:>+10.2f}")
    print()
    print(f"  --- COSTS BREAKDOWN ---")
    print(f"  Entry fee:    {s['avg_entry_fee_pct']:>8.4f}%")
    print(f"  Exit fee:     {s['avg_exit_fee_pct']:>8.4f}%")
    print(f"  Fees paid:    {s['avg_fees_paid_pct']:>8.4f}%")
    print(f"  Spread cost:  {s['avg_spread_cost_pct']:>8.4f}%")
    print(f"  Slippage:     {s['avg_slippage_pct']:>8.4f}%")
    print(f"  TOTAL COSTS:  {s['avg_total_costs_pct']:>8.4f}%")
    print()
    print(f"  Avg hold time: {s['avg_hold_time_sec']:.1f}s")


def print_comparison_table(stats: Dict):
    """Print side-by-side comparison table."""
    print_header("MAKER vs TAKER COMPARISON")

    taker = stats.get("taker", {})
    maker = stats.get("maker", {})

    if not taker and not maker:
        print("  No data available")
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
        ("Avg Fees %", "avg_fees_paid_pct", "{:>11.4f}%", True),
        ("Avg Spread Cost %", "avg_spread_cost_pct", "{:>11.4f}%", True),
        ("Avg Slippage %", "avg_slippage_pct", "{:>11.4f}%", True),
        ("Total Costs %", "avg_total_costs_pct", "{:>11.4f}%", True),
        ("Avg Hold Time (s)", "avg_hold_time_sec", "{:>11.1f}s", False),
    ]

    for label, key, fmt, show_diff in metrics:
        t_val = taker.get(key, 0) or 0
        m_val = maker.get(key, 0) or 0

        t_str = fmt.format(t_val)
        m_str = fmt.format(m_val) if maker else "N/A"

        if show_diff and maker and taker:
            diff = m_val - t_val
            # For costs, negative diff is good (maker saves money)
            if "cost" in key.lower() or "fee" in key.lower() or "slippage" in key.lower():
                diff_str = f"{diff:>+11.4f}%" if "%" in fmt else f"{diff:>+11.2f}"
            else:
                diff_str = f"{diff:>+11.4f}%" if "%" in fmt else f"{diff:>+11.2f}"
        else:
            diff_str = ""

        print(f"  {label:<25} {t_str:>12} {m_str:>12} {diff_str:>12}")


def print_cost_comparison(comparison: Dict):
    """Print theoretical vs actual cost comparison."""
    print_header("THEORETICAL vs ACTUAL COSTS", "-")

    theoretical = comparison["theoretical"]
    actual = comparison["actual"]

    for mode in ["taker", "maker"]:
        theo = theoretical.get(mode, {})
        act = actual.get(mode, {})

        if not act:
            continue

        print(f"\n  --- {mode.upper()} ---")
        print(f"  {'Cost Type':<15} {'Theoretical':>12} {'Actual':>12} {'Diff':>12}")
        print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*12}")

        for cost_type in ["entry_fee", "exit_fee", "spread_cost", "slippage", "total"]:
            t_val = theo.get(cost_type, 0)
            a_val = act.get(cost_type, 0)
            diff = a_val - t_val

            print(f"  {cost_type:<15} {t_val:>11.4f}% {a_val:>11.4f}% {diff:>+11.4f}%")


def export_to_csv(stats: Dict, exit_breakdown: Dict, filepath: str):
    """Export report to CSV."""
    with open(filepath, 'w') as f:
        # Summary section
        f.write("Maker vs Taker Performance Report\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")

        # Stats by mode
        f.write("Mode,Trades,Wins,Losses,Win Rate %,Avg PnL %,Avg PnL After Costs %,")
        f.write("Avg Fees %,Avg Spread %,Avg Slippage %,Total Costs %\n")

        for mode in ["taker", "maker"]:
            s = stats.get(mode, {})
            if s:
                f.write(f"{mode},{s['trade_count']},{s['wins']},{s['losses']},")
                f.write(f"{s['win_rate']:.2f},{s['avg_pnl_pct']:.4f},{s['avg_pnl_after_costs_pct']:.4f},")
                f.write(f"{s['avg_fees_paid_pct']:.4f},{s['avg_spread_cost_pct']:.4f},")
                f.write(f"{s['avg_slippage_pct']:.4f},{s['avg_total_costs_pct']:.4f}\n")

        f.write("\n")

        # Exit reason breakdown
        f.write("Exit Reason Breakdown by Mode\n")
        f.write("Mode,Exit Reason,Count,Avg PnL %\n")

        for mode, reasons in exit_breakdown.items():
            for reason, data in reasons.items():
                f.write(f"{mode},{reason},{data['count']},{data['avg_pnl_pct']:.4f}\n")

    print(f"\n  Report exported to: {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="Maker vs Taker Performance Report"
    )
    parser.add_argument("--db", default="hft_trades.db", help="Database path")
    parser.add_argument("--days", type=int, help="Filter to last N days")
    parser.add_argument("--csv", help="Export to CSV file")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")
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

    # Connect with bot-safe WAL mode and busy_timeout
    conn = open_sqlite(str(db_path))

    # Get data
    stats = get_execution_mode_stats(conn, args.days)
    exit_breakdown = get_exit_reason_breakdown(conn, args.days)
    cost_comparison = get_cost_comparison(conn)
    fill_analysis = get_maker_fill_analysis(conn)

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

        # Comparison table
        print_comparison_table(stats)

        # Individual mode details
        print_header("DETAILED STATS BY MODE")
        print_mode_stats(stats, "taker")
        print_mode_stats(stats, "maker")

        # Cost comparison
        print_cost_comparison(cost_comparison)

        # Fill analysis
        if fill_analysis:
            print_header("MAKER FILL ANALYSIS", "-")
            print(f"\n  Maker orders filled:    {fill_analysis['maker_fills']}")
            print(f"  Maker orders unfilled:  {fill_analysis['maker_no_fills']}")
            print(f"  Fill rate:              {fill_analysis['fill_rate_pct']:.1f}%")

        # Summary
        print_header("SUMMARY")

        taker = stats.get("taker", {})
        maker = stats.get("maker", {})

        if taker and maker:
            taker_net = taker.get("avg_pnl_after_costs_pct", 0)
            maker_net = maker.get("avg_pnl_after_costs_pct", 0)

            if maker_net > taker_net:
                savings = maker_net - taker_net
                print(f"\n  MAKER outperforms TAKER by {savings:.4f}% per trade")
                print(f"  Cost savings: {taker.get('avg_total_costs_pct', 0) - maker.get('avg_total_costs_pct', 0):.4f}% per trade")
            elif taker_net > maker_net:
                print(f"\n  TAKER outperforms MAKER by {taker_net - maker_net:.4f}% per trade")
                print("  (May indicate maker fills are selective or insufficient data)")
            else:
                print("\n  MAKER and TAKER perform equally")
        elif taker:
            print("\n  Only TAKER data available - switch to maker mode to compare")
        elif maker:
            print("\n  Only MAKER data available - switch to taker mode to compare")
        else:
            print("\n  No data available")

        print()
        print("=" * 70)

    # Export to CSV if requested
    if args.csv:
        export_to_csv(stats, exit_breakdown, args.csv)

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
