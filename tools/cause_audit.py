#!/usr/bin/env python3
"""
Cause Policy Audit Tool - Evidence Collection for Cause Classification
======================================================================
Analyzes trading performance by cause to decide FULL/PROBE/BLOCK status.

Usage:
    python tools/cause_audit.py --db ~/Analize-/hft_trades.db

Output:
    1. Performance by cause (net expectancy, win%, N, total_net)
    2. Winners vs losers MFE/MAE by cause
    3. Exit reason distribution by cause
    4. Probe vs full performance comparison
    5. Decision recommendations

Targets:
    - ≥30 trades per cause for statistical significance
    - Positive net expectancy -> promote to FULL
    - Negative net expectancy -> keep BLOCKED
"""

import argparse
import sqlite3
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add parent dir for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from hft_system.db import open_sqlite


def print_header(title: str):
    """Print a formatted header."""
    print()
    print("=" * 80)
    print(f" {title}")
    print("=" * 80)


def print_table(cursor, title: str = None):
    """Print query results as a formatted table."""
    if title:
        print(f"\n{title}")
        print("-" * 70)

    if cursor.description is None:
        print("  No results")
        return []

    headers = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()

    if not rows:
        print("  No data found")
        return []

    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(str(val) if val is not None else "NULL"))

    # Print header row
    header_row = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(f"  {header_row}")
    print(f"  {'-' * len(header_row)}")

    # Print data rows
    for row in rows:
        data_row = " | ".join(
            str(val if val is not None else "NULL").ljust(widths[i])
            for i, val in enumerate(row)
        )
        print(f"  {data_row}")

    return rows


def analyze_performance_by_cause(conn):
    """Analyze net performance by cause."""
    print_header("1. PERFORMANCE BY CAUSE (Net After Costs)")

    cursor = conn.cursor()
    cursor.execute("""
        SELECT
            COALESCE(tc.primary_cause, 'unknown') as cause,
            COUNT(*) as N,
            ROUND(100.0 * SUM(CASE WHEN t.pnl_after_costs_pct > 0 THEN 1 ELSE 0 END) / COUNT(*), 1) as win_pct,
            ROUND(AVG(t.pnl_after_costs_pct), 4) as avg_net_pct,
            ROUND(SUM(t.pnl_after_costs_pct), 4) as total_net_pct,
            ROUND(AVG(t.total_costs_pct), 4) as avg_cost_pct,
            COALESCE(t.cause_class, 'FULL') as class
        FROM trades t
        LEFT JOIN trade_causality tc ON t.trade_id = tc.trade_id
        WHERE t.status = 'closed'
        GROUP BY COALESCE(tc.primary_cause, 'unknown')
        ORDER BY total_net_pct DESC
    """)

    rows = print_table(cursor, "Performance by Primary Cause:")

    # Summary
    print("\n  DECISION GUIDE:")
    print("  ---------------")
    for row in rows:
        cause, n, win_pct, avg_net, total_net, avg_cost, cause_class = row
        if n >= 30:
            if avg_net and avg_net > 0:
                decision = "✅ PROMOTE to FULL (positive expectancy)"
            else:
                decision = "❌ KEEP BLOCKED (negative expectancy)"
        else:
            decision = f"⏳ NEED MORE DATA ({30-n} more trades)"
        print(f"    {cause}: {decision}")


def analyze_mfe_mae_by_cause(conn):
    """Analyze MFE/MAE patterns by cause."""
    print_header("2. EDGE QUALITY BY CAUSE (MFE/MAE Analysis)")

    cursor = conn.cursor()

    # Winners MFE/MAE
    cursor.execute("""
        SELECT
            COALESCE(tc.primary_cause, 'unknown') as cause,
            COUNT(*) as winners,
            ROUND(AVG(t.mfe), 4) as avg_mfe,
            ROUND(AVG(t.mae), 4) as avg_mae,
            ROUND(AVG(t.mfe) - AVG(ABS(t.mae)), 4) as edge_quality
        FROM trades t
        LEFT JOIN trade_causality tc ON t.trade_id = tc.trade_id
        WHERE t.status = 'closed' AND t.pnl_after_costs_pct > 0
        AND t.mfe IS NOT NULL AND t.mfe > 0
        GROUP BY COALESCE(tc.primary_cause, 'unknown')
        ORDER BY edge_quality DESC
    """)
    print_table(cursor, "WINNERS - MFE/MAE by Cause:")

    # Losers MFE/MAE
    cursor.execute("""
        SELECT
            COALESCE(tc.primary_cause, 'unknown') as cause,
            COUNT(*) as losers,
            ROUND(AVG(t.mfe), 4) as avg_mfe,
            ROUND(AVG(t.mae), 4) as avg_mae,
            ROUND(AVG(t.mfe) - AVG(ABS(t.mae)), 4) as edge_quality
        FROM trades t
        LEFT JOIN trade_causality tc ON t.trade_id = tc.trade_id
        WHERE t.status = 'closed' AND t.pnl_after_costs_pct <= 0
        AND t.mfe IS NOT NULL
        GROUP BY COALESCE(tc.primary_cause, 'unknown')
        ORDER BY edge_quality DESC
    """)
    print_table(cursor, "LOSERS - MFE/MAE by Cause:")


def analyze_exit_distribution(conn):
    """Analyze exit reason distribution by cause."""
    print_header("3. EXIT REASON DISTRIBUTION BY CAUSE")

    cursor = conn.cursor()
    cursor.execute("""
        SELECT
            COALESCE(tc.primary_cause, 'unknown') as cause,
            t.exit_reason,
            COUNT(*) as N,
            ROUND(AVG(t.pnl_after_costs_pct), 4) as avg_net_pct
        FROM trades t
        LEFT JOIN trade_causality tc ON t.trade_id = tc.trade_id
        WHERE t.status = 'closed' AND t.exit_reason IS NOT NULL
        GROUP BY COALESCE(tc.primary_cause, 'unknown'), t.exit_reason
        ORDER BY cause, N DESC
    """)
    print_table(cursor)


def analyze_probe_vs_full(conn):
    """Compare PROBE vs FULL trade performance."""
    print_header("4. PROBE vs FULL PERFORMANCE COMPARISON")

    cursor = conn.cursor()

    # Check if cause_class column exists
    cursor.execute("PRAGMA table_info(trades)")
    columns = [col[1] for col in cursor.fetchall()]

    if 'cause_class' not in columns:
        print("  cause_class column not found - no PROBE trades yet")
        print("  (This column will be added after the code update)")
        return

    cursor.execute("""
        SELECT
            COALESCE(t.cause_class, 'FULL') as trade_class,
            COUNT(*) as N,
            ROUND(100.0 * SUM(CASE WHEN t.pnl_after_costs_pct > 0 THEN 1 ELSE 0 END) / COUNT(*), 1) as win_pct,
            ROUND(AVG(t.pnl_after_costs_pct), 4) as avg_net_pct,
            ROUND(SUM(t.pnl_after_costs_pct), 4) as total_net_pct,
            ROUND(AVG(t.total_costs_pct), 4) as avg_cost_pct
        FROM trades t
        WHERE t.status = 'closed'
        GROUP BY COALESCE(t.cause_class, 'FULL')
        ORDER BY trade_class
    """)
    print_table(cursor, "Performance by Trade Class:")

    # PROBE causes breakdown
    cursor.execute("""
        SELECT
            COALESCE(tc.primary_cause, 'unknown') as cause,
            COALESCE(t.cause_class, 'FULL') as class,
            COUNT(*) as N,
            ROUND(AVG(t.pnl_after_costs_pct), 4) as avg_net_pct
        FROM trades t
        LEFT JOIN trade_causality tc ON t.trade_id = tc.trade_id
        WHERE t.status = 'closed' AND t.cause_class = 'PROBE'
        GROUP BY COALESCE(tc.primary_cause, 'unknown')
        ORDER BY N DESC
    """)
    rows = print_table(cursor, "PROBE Trades by Cause (collecting evidence):")

    if rows:
        print("\n  PROBE CAUSE DECISIONS:")
        print("  ----------------------")
        for row in rows:
            cause, cls, n, avg_net = row
            if n >= 30:
                if avg_net and avg_net > 0:
                    print(f"    {cause}: ✅ READY TO PROMOTE (N={n}, avg={avg_net}%)")
                else:
                    print(f"    {cause}: ❌ KEEP BLOCKED (N={n}, avg={avg_net}%)")
            else:
                print(f"    {cause}: ⏳ {n}/30 trades - need {30-n} more for decision")


def analyze_execution_mode(conn):
    """Analyze performance by execution mode (maker vs taker)."""
    print_header("5. EXECUTION MODE ANALYSIS (Maker vs Taker)")

    cursor = conn.cursor()
    cursor.execute("""
        SELECT
            COALESCE(t.execution_mode, 'taker') as mode,
            COUNT(*) as N,
            ROUND(100.0 * SUM(CASE WHEN t.pnl_after_costs_pct > 0 THEN 1 ELSE 0 END) / COUNT(*), 1) as win_pct,
            ROUND(AVG(t.pnl_after_costs_pct), 4) as avg_net_pct,
            ROUND(AVG(t.total_costs_pct), 4) as avg_cost_pct
        FROM trades t
        WHERE t.status = 'closed'
        GROUP BY COALESCE(t.execution_mode, 'taker')
        ORDER BY avg_net_pct DESC
    """)
    print_table(cursor)


def show_recent_trades(conn, limit: int = 10):
    """Show recent trades with cause info."""
    print_header("6. RECENT TRADES (with Cause Info)")

    cursor = conn.cursor()
    cursor.execute(f"""
        SELECT
            datetime(t.entry_time/1000, 'unixepoch') as entry,
            t.symbol,
            COALESCE(tc.primary_cause, 'unknown') as cause,
            COALESCE(t.cause_class, 'FULL') as class,
            t.exit_reason,
            ROUND(t.pnl_after_costs_pct, 4) as net_pct
        FROM trades t
        LEFT JOIN trade_causality tc ON t.trade_id = tc.trade_id
        WHERE t.status = 'closed'
        ORDER BY t.entry_time DESC
        LIMIT {limit}
    """)
    print_table(cursor)


def show_summary(conn):
    """Show overall summary and next steps."""
    print_header("SUMMARY & RECOMMENDATIONS")

    cursor = conn.cursor()

    # Total trades
    cursor.execute("SELECT COUNT(*) FROM trades WHERE status = 'closed'")
    total = cursor.fetchone()[0]

    # Trades by cause with sample size assessment
    cursor.execute("""
        SELECT
            COALESCE(tc.primary_cause, 'unknown') as cause,
            COUNT(*) as N,
            ROUND(AVG(t.pnl_after_costs_pct), 4) as avg_net
        FROM trades t
        LEFT JOIN trade_causality tc ON t.trade_id = tc.trade_id
        WHERE t.status = 'closed'
        GROUP BY COALESCE(tc.primary_cause, 'unknown')
        ORDER BY N DESC
    """)
    causes = cursor.fetchall()

    print(f"\n  TOTAL CLOSED TRADES: {total}")
    print(f"\n  CAUSE SAMPLE SIZES:")

    sufficient_data = []
    need_more_data = []

    for cause, n, avg_net in causes:
        status = "✅" if n >= 30 else "⏳"
        print(f"    {status} {cause}: {n} trades (need 30)")
        if n >= 30:
            sufficient_data.append((cause, n, avg_net))
        else:
            need_more_data.append((cause, n, avg_net))

    print("\n  NEXT STEPS:")
    print("  -----------")

    if not sufficient_data and not need_more_data:
        print("    1. Deploy the cause policy update")
        print("    2. Wait 24-48h for PROBE trades to accumulate")
        print("    3. Re-run this audit to check sample sizes")
    elif need_more_data:
        trades_needed = sum(max(0, 30 - n) for _, n, _ in need_more_data)
        print(f"    1. Continue collecting data ({trades_needed} more trades needed)")
        print(f"    2. Re-run audit after 24h: python tools/cause_audit.py")

    if sufficient_data:
        print("\n  READY FOR DECISIONS:")
        for cause, n, avg_net in sufficient_data:
            if avg_net and avg_net > 0:
                print(f"    ✅ {cause}: PROMOTE to FULL (positive expectancy)")
            else:
                print(f"    ❌ {cause}: KEEP BLOCKED (negative expectancy)")


def main():
    parser = argparse.ArgumentParser(
        description="Cause Policy Audit - Evidence Collection for Trading Decisions"
    )
    parser.add_argument(
        "--db",
        type=str,
        default="hft_trades.db",
        help="Path to SQLite database"
    )
    parser.add_argument(
        "--recent",
        type=int,
        default=10,
        help="Number of recent trades to show"
    )

    args = parser.parse_args()

    # Find database
    db_path = Path(args.db)
    if not db_path.exists():
        # Try common locations
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
        print("Try: python tools/cause_audit.py --db /path/to/hft_trades.db")
        return 1

    print(f"\n📊 CAUSE POLICY AUDIT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Database: {db_path}")

    try:
        # Connect with bot-safe WAL mode and busy_timeout
        conn = open_sqlite(str(db_path))

        analyze_performance_by_cause(conn)
        analyze_mfe_mae_by_cause(conn)
        analyze_exit_distribution(conn)
        analyze_probe_vs_full(conn)
        analyze_execution_mode(conn)
        show_recent_trades(conn, args.recent)
        show_summary(conn)

        conn.close()

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
