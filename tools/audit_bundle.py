#!/usr/bin/env python3
"""
HFT Audit Bundle - Complete Evidence Collection
================================================
Single command to collect all evidence needed for trading decisions.

Usage:
    python tools/audit_bundle.py --db ~/Analize-/hft_trades.db --log ~/Analize-/hft.log

Output:
    - Git version/commit
    - Current config values
    - Last 200 log lines (ENTRY/EXIT/BLOCK/REGIME)
    - SQL query results with evidence tables
    - Data quality checks
    - Summary tables with STOP/GO decision
"""

import argparse
import sqlite3
import subprocess
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple


def print_header(title: str):
    """Print a formatted header."""
    print()
    print("=" * 80)
    print(f" {title}")
    print("=" * 80)


def get_git_info():
    """Get current git commit and branch."""
    print_header("GIT VERSION INFO")
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=os.path.dirname(__file__) or "."
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, cwd=os.path.dirname(__file__) or "."
        ).stdout.strip()
        print(f"  Commit: {commit}")
        print(f"  Branch: {branch}")
        print(f"  Timestamp: {datetime.now().isoformat()}")
    except Exception as e:
        print(f"  Error getting git info: {e}")


def get_config_values():
    """Extract current config values from config.py."""
    print_header("CURRENT CONFIG VALUES")
    config_path = Path(__file__).parent.parent / "hft_system" / "config.py"

    if not config_path.exists():
        print(f"  Config file not found: {config_path}")
        return

    try:
        with open(config_path, 'r') as f:
            content = f.read()

        # Extract key config values using simple parsing
        configs = {
            "execution_mode": None,
            "pocket_a_regimes": None,
            "pocket_a_min_imbalance": None,
            "pocket_b_enabled": None,
            "pocket_b_regimes": None,
            "pocket_b_min_imbalance": None,
            "max_trades_per_symbol_per_hour": None,
            "min_seconds_between_trades": None,
            "causality_filter_enabled": None,
            "blocked_regimes": None,
        }

        for line in content.split('\n'):
            for key in configs:
                if key in line and '=' in line and not line.strip().startswith('#'):
                    configs[key] = line.split('=', 1)[1].strip().rstrip(',')

        for key, value in configs.items():
            if value:
                print(f"  {key}: {value}")

    except Exception as e:
        print(f"  Error reading config: {e}")


def get_recent_logs(log_path: str, n: int = 200):
    """Get last N relevant log lines."""
    print_header(f"LAST {n} RELEVANT LOG LINES")

    if not os.path.exists(log_path):
        # Try alternative paths
        alternatives = [
            Path.home() / "Analize-" / "bot.log",
            Path.home() / "Analize-" / "hft.log",
            Path.home() / "bot.log"
        ]
        for alt in alternatives:
            if alt.exists():
                log_path = str(alt)
                break
        else:
            print(f"  Log file not found: {log_path}")
            return

    print(f"  Source: {log_path}")
    print()

    keywords = ["SIGNAL", "ENTRY", "EXIT", "BLOCK", "PASS", "REGIME", "TRADE", "WINNER_GATE", "CAUSALITY"]

    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()

        # Filter relevant lines
        relevant = []
        for line in lines:
            for kw in keywords:
                if kw in line:
                    relevant.append(line.rstrip())
                    break

        # Show last N
        for line in relevant[-n:]:
            print(f"  {line}")

    except Exception as e:
        print(f"  Error reading log: {e}")


def run_sql_query(conn, title: str, query: str, show_header: bool = True):
    """Run a SQL query and print results."""
    if show_header:
        print_header(title)

    cursor = conn.cursor()
    try:
        cursor.execute(query)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = cursor.fetchall()

        if not rows:
            print("  (no data)")
            return []

        # Calculate column widths
        widths = [len(str(c)) for c in columns]
        for row in rows:
            for i, val in enumerate(row):
                widths[i] = max(widths[i], len(str(val)[:20]))

        # Print header
        header = " | ".join(f"{c:<{widths[i]}}" for i, c in enumerate(columns))
        print(f"  {header}")
        print("  " + "-" * len(header))

        # Print rows
        for row in rows:
            row_str = " | ".join(f"{str(v)[:20]:<{widths[i]}}" for i, v in enumerate(row))
            print(f"  {row_str}")

        return rows

    except Exception as e:
        print(f"  Error: {e}")
        return []


def audit_sql_queries(conn):
    """Run all required audit SQL queries."""

    # Query 1: Performance by CAUSE
    run_sql_query(conn, "1) PERFORMANCE BY CAUSE (After Costs)", """
        SELECT
            COALESCE(tc.primary_cause, 'unknown') as cause,
            COUNT(*) as n,
            ROUND(AVG(t.total_costs_pct), 4) as avg_cost,
            ROUND(AVG(t.pnl_after_costs_pct), 4) as avg_net,
            ROUND(SUM(t.pnl_after_costs_pct), 4) as total_net,
            ROUND(100.0 * SUM(CASE WHEN t.pnl_after_costs_pct > 0 THEN 1 ELSE 0 END) / COUNT(*), 1) as win_pct
        FROM trades t
        LEFT JOIN trade_causality tc ON tc.trade_id = t.id
        WHERE t.status = 'closed'
        GROUP BY COALESCE(tc.primary_cause, 'unknown')
        ORDER BY total_net DESC
    """)

    # Query 2: Winners vs Losers MFE/MAE
    run_sql_query(conn, "2) WINNERS vs LOSERS - MFE/MAE ANALYSIS", """
        SELECT
            CASE WHEN pnl_after_costs_pct > 0 THEN 'WIN' ELSE 'LOSS' END as grp,
            COUNT(*) as n,
            ROUND(AVG(mfe), 4) as avg_mfe,
            ROUND(AVG(mae), 4) as avg_mae,
            ROUND(AVG(mfe) + AVG(mae), 4) as edge,
            ROUND(AVG(hold_time_sec), 1) as avg_hold_s
        FROM trades
        WHERE status = 'closed' AND (mfe != 0 OR mae != 0)
        GROUP BY CASE WHEN pnl_after_costs_pct > 0 THEN 'WIN' ELSE 'LOSS' END
    """)

    # Query 3: Performance by REGIME
    run_sql_query(conn, "3) PERFORMANCE BY REGIME", """
        SELECT
            COALESCE(regime, 'unknown') as regime,
            COUNT(*) as n,
            ROUND(AVG(pnl_after_costs_pct), 4) as avg_net,
            ROUND(SUM(pnl_after_costs_pct), 4) as total_net,
            ROUND(AVG(total_costs_pct), 4) as avg_cost,
            ROUND(100.0 * SUM(CASE WHEN pnl_after_costs_pct > 0 THEN 1 ELSE 0 END) / COUNT(*), 1) as win_pct
        FROM trades
        WHERE status = 'closed'
        GROUP BY regime
        ORDER BY total_net DESC
    """)

    # Query 4: Performance by POCKET
    run_sql_query(conn, "4) PERFORMANCE BY POCKET", """
        SELECT
            COALESCE(pocket_id, 'unknown') as pocket,
            COUNT(*) as n,
            ROUND(AVG(pnl_after_costs_pct), 4) as avg_net,
            ROUND(SUM(pnl_after_costs_pct), 4) as total_net,
            ROUND(AVG(total_costs_pct), 4) as avg_cost,
            ROUND(100.0 * SUM(CASE WHEN pnl_after_costs_pct > 0 THEN 1 ELSE 0 END) / COUNT(*), 1) as win_pct
        FROM trades
        WHERE status = 'closed'
        GROUP BY pocket_id
        ORDER BY total_net DESC
    """)

    # Query 5: Performance by EXIT reason
    run_sql_query(conn, "5) PERFORMANCE BY EXIT REASON", """
        SELECT
            exit_reason,
            COUNT(*) as n,
            ROUND(AVG(pnl_after_costs_pct), 4) as avg_net,
            ROUND(SUM(pnl_after_costs_pct), 4) as total_net,
            ROUND(100.0 * SUM(CASE WHEN pnl_after_costs_pct > 0 THEN 1 ELSE 0 END) / COUNT(*), 1) as win_pct
        FROM trades
        WHERE status = 'closed'
        GROUP BY exit_reason
        ORDER BY total_net DESC
    """)

    # Query 6: Maker vs Taker
    run_sql_query(conn, "6) MAKER vs TAKER COMPARISON", """
        SELECT
            COALESCE(execution_mode, 'taker') as mode,
            COUNT(*) as n,
            ROUND(AVG(total_costs_pct), 4) as avg_cost,
            ROUND(AVG(pnl_after_costs_pct), 4) as avg_net,
            ROUND(SUM(pnl_after_costs_pct), 4) as total_net,
            ROUND(100.0 * SUM(CASE WHEN pnl_after_costs_pct > 0 THEN 1 ELSE 0 END) / COUNT(*), 1) as win_pct
        FROM trades
        WHERE status = 'closed'
        GROUP BY execution_mode
        ORDER BY total_net DESC
    """)


def check_data_integrity(conn) -> List[Tuple[str, str, bool]]:
    """Run data integrity checks and return pass/fail status."""
    print_header("7) DATA INTEGRITY CHECKS")

    cursor = conn.cursor()
    checks = []

    # Check 1: MFE/MAE coverage
    try:
        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN mfe = 0 AND mae = 0 THEN 1 ELSE 0 END) as zero_mfe
            FROM trades WHERE status = 'closed'
        """)
        row = cursor.fetchone()
        total, zero = row[0], row[1]
        pct = (zero / total * 100) if total > 0 else 0
        status = "PASS" if pct < 50 else "WARN" if pct < 80 else "FAIL"
        print(f"  MFE/MAE zero rate: {zero}/{total} ({pct:.1f}%) - {status}")
        checks.append(("mfe_mae_coverage", f"{pct:.1f}%", pct < 80))
    except Exception as e:
        print(f"  MFE/MAE check error: {e}")

    # Check 2: Causality join rate
    try:
        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN tc.trade_id IS NULL THEN 1 ELSE 0 END) as missing
            FROM trades t
            LEFT JOIN trade_causality tc ON t.id = tc.trade_id
            WHERE t.status = 'closed'
        """)
        row = cursor.fetchone()
        total, missing = row[0], row[1]
        pct = (missing / total * 100) if total > 0 else 0
        status = "PASS" if pct < 20 else "WARN" if pct < 50 else "FAIL"
        print(f"  Missing causality: {missing}/{total} ({pct:.1f}%) - {status}")
        checks.append(("causality_coverage", f"{pct:.1f}%", pct < 50))
    except Exception as e:
        print(f"  Causality check error: {e}")

    # Check 3: Null execution_mode
    try:
        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN execution_mode IS NULL THEN 1 ELSE 0 END) as null_mode
            FROM trades WHERE status = 'closed'
        """)
        row = cursor.fetchone()
        total, null_mode = row[0], row[1]
        pct = (null_mode / total * 100) if total > 0 else 0
        status = "PASS" if pct < 10 else "WARN"
        print(f"  Null execution_mode: {null_mode}/{total} ({pct:.1f}%) - {status}")
        checks.append(("exec_mode_coverage", f"{pct:.1f}%", pct < 50))
    except Exception as e:
        print(f"  Execution mode check error: {e}")

    # Check 4: Total sample size
    try:
        cursor.execute("SELECT COUNT(*) FROM trades WHERE status = 'closed'")
        total = cursor.fetchone()[0]
        status = "SUFFICIENT" if total >= 50 else "INSUFFICIENT"
        print(f"  Total closed trades: {total} - {status}")
        checks.append(("sample_size", str(total), total >= 50))
    except Exception as e:
        print(f"  Sample size check error: {e}")

    return checks


def generate_verdict(conn) -> str:
    """Generate final STOP/CONTINUE verdict based on evidence."""
    print_header("AUDIT VERDICT")

    cursor = conn.cursor()

    # Get overall stats
    try:
        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(pnl_after_costs_pct) as total_net,
                AVG(pnl_after_costs_pct) as avg_net,
                SUM(CASE WHEN pnl_after_costs_pct > 0 THEN 1 ELSE 0 END) as wins
            FROM trades WHERE status = 'closed'
        """)
        row = cursor.fetchone()
        total, total_net, avg_net, wins = row

        if total < 30:
            verdict = "INSUFFICIENT_DATA"
            reason = f"Only {total} trades - need at least 30 for conclusions"
        elif total_net > 0 and avg_net > 0:
            verdict = "EDGE_DETECTED"
            reason = f"Positive net: {total_net:.4f}% total, {avg_net:.4f}% avg"
        elif total_net < -0.5 and total >= 50:
            verdict = "EDGE_DISPROVEN"
            reason = f"Significant loss: {total_net:.4f}% with {total} trades"
        else:
            verdict = "CONTINUE_RESEARCH"
            reason = f"Inconclusive: {total_net:.4f}% with {total} trades"

        print(f"\n  VERDICT: {verdict}")
        print(f"  REASON:  {reason}")
        print(f"\n  Stats: {total} trades, {wins} wins ({wins/total*100:.1f}% win rate)")
        print(f"         Total net: {total_net:.4f}%, Avg net: {avg_net:.4f}%")

        # Recommendations
        print("\n  RECOMMENDATIONS:")
        if verdict == "EDGE_DISPROVEN":
            print("    ⛔ STOP TRADING this configuration")
            print("    → Analyze which pockets/causes are losing")
            print("    → Consider tightening filters or reducing position size")
        elif verdict == "EDGE_DETECTED":
            print("    ✅ Edge appears positive - continue with caution")
            print("    → Monitor for regime changes")
            print("    → Consider increasing position size gradually")
        elif verdict == "INSUFFICIENT_DATA":
            print("    ⏳ Continue collecting data")
            print(f"    → Need {50 - total} more trades for basic conclusions")
            print("    → Keep current filters active")
        else:
            print("    ⚠️  Results inconclusive")
            print("    → Continue with current configuration")
            print("    → Re-evaluate after 20 more trades")

        return verdict

    except Exception as e:
        print(f"  Error generating verdict: {e}")
        return "ERROR"


def main():
    parser = argparse.ArgumentParser(description="HFT Audit Bundle")
    parser.add_argument("--db", default="hft_trades.db", help="Database path")
    parser.add_argument("--log", default="bot.log", help="Log file path")
    parser.add_argument("--no-logs", action="store_true", help="Skip log file analysis")
    args = parser.parse_args()

    print("=" * 80)
    print(" HFT AUDIT BUNDLE - Evidence Collection")
    print(" Generated: " + datetime.now().isoformat())
    print("=" * 80)

    # Find database
    db_path = Path(args.db)
    if not db_path.exists():
        for path in [
            Path.home() / "Analize-" / "hft_trades.db",
            Path("hft_trades.db"),
            Path.home() / "hft_trades.db"
        ]:
            if path.exists():
                db_path = path
                break

    if not db_path.exists():
        print(f"\nERROR: Database not found at {args.db}")
        print("Try: python tools/audit_bundle.py --db /path/to/hft_trades.db")
        return 1

    print(f"\nDatabase: {db_path}")

    # Part 1: Git info
    get_git_info()

    # Part 2: Config values
    get_config_values()

    # Part 3: Recent logs (optional)
    if not args.no_logs:
        get_recent_logs(args.log, 100)

    # Part 4: Connect to database
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    try:
        # Run all SQL queries
        audit_sql_queries(conn)

        # Data integrity checks
        checks = check_data_integrity(conn)

        # Generate verdict
        verdict = generate_verdict(conn)

        # Summary footer
        print_header("END OF AUDIT BUNDLE")
        print(f"  Verdict: {verdict}")
        print(f"  Time: {datetime.now().isoformat()}")

    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    exit(main())
