#!/usr/bin/env python3
"""
HFT Bot Healthcheck Tool
========================
Quick status check for the running bot.

Usage:
    python tools/healthcheck.py [--db PATH]

Displays:
- WebSocket status
- Current regimes (raw + stable)
- Gate thresholds
- Last 20 blocks (grouped)
- Last 20 trades (net-after-costs)
"""

import argparse
import sqlite3
from pathlib import Path
from datetime import datetime


def get_db_connection(db_path: str):
    """Connect to the database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def print_header(title: str):
    """Print a formatted header."""
    print()
    print("=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_section(title: str):
    """Print a section header."""
    print()
    print(f"--- {title} ---")


def check_trades(conn):
    """Show last 20 trades with net-after-costs."""
    print_header("LAST 20 TRADES (Net-After-Costs)")

    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT
                id, symbol, side,
                pnl_pct, total_costs_pct, pnl_after_costs_pct,
                exit_reason, pocket_id,
                datetime(entry_time/1000, 'unixepoch') as entry_dt,
                hold_time_sec
            FROM trades
            WHERE status = 'closed'
            ORDER BY entry_time DESC
            LIMIT 20
        """)
        rows = cursor.fetchall()

        if not rows:
            print("  No trades found.")
            return

        print(f"  {'ID':<5} {'Symbol':<6} {'Side':<5} {'PnL%':>8} {'Costs%':>8} {'Net%':>10} {'Exit':<12} {'Pocket':<6}")
        print("-" * 70)

        total_pnl = 0
        total_costs = 0
        total_net = 0

        for row in rows:
            pnl = row['pnl_pct'] or 0
            costs = row['total_costs_pct'] or 0
            net = row['pnl_after_costs_pct'] or pnl

            total_pnl += pnl
            total_costs += costs
            total_net += net

            net_str = f"{net:+.4f}%" if net else "N/A"
            pocket = row['pocket_id'] or "?"

            print(f"  {row['id']:<5} {row['symbol']:<6} {row['side']:<5} "
                  f"{pnl:+.4f}% {costs:.4f}% {net_str:>10} {row['exit_reason']:<12} {pocket:<6}")

        print("-" * 70)
        print(f"  {'TOTAL':<18} {total_pnl:+.4f}% {total_costs:.4f}% {total_net:+.4f}%")

    except Exception as e:
        print(f"  Error: {e}")


def check_gate_blocks(conn):
    """Show last 20 gate blocks grouped by reason."""
    print_header("WINNER GATE BLOCKS (Last 24h)")

    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT
                CASE
                    WHEN block_reason LIKE 'regime=%' THEN SUBSTR(block_reason, 1, INSTR(block_reason, ' ')-1)
                    WHEN block_reason LIKE 'pocket_a:%' THEN 'pocket_a_fail'
                    WHEN block_reason LIKE 'pocket_b:%' THEN 'pocket_b_fail'
                    WHEN block_reason LIKE 'tier=%' THEN 'tier_blocked'
                    WHEN block_reason LIKE 'imbalance=%' THEN 'imbalance_low'
                    WHEN block_reason LIKE 'rate_limit%' THEN 'rate_limit'
                    WHEN block_reason LIKE 'cooldown%' THEN 'cooldown'
                    ELSE block_reason
                END as reason,
                COUNT(*) as cnt
            FROM winner_gate_blocks
            WHERE timestamp > strftime('%s','now','-24 hours')*1000
            GROUP BY 1
            ORDER BY cnt DESC
            LIMIT 20
        """)
        rows = cursor.fetchall()

        if not rows:
            print("  No gate blocks in last 24h.")
            return

        total = sum(row['cnt'] for row in rows)
        print(f"  Total blocks: {total}")
        print()
        print(f"  {'Reason':<30} {'Count':>8} {'%':>8}")
        print("-" * 50)

        for row in rows:
            pct = (row['cnt'] / total) * 100 if total > 0 else 0
            print(f"  {row['reason']:<30} {row['cnt']:>8} {pct:>7.1f}%")

    except Exception as e:
        print(f"  Error: {e}")


def check_regime_distribution(conn):
    """Show regime distribution in last 24h."""
    print_header("REGIME DISTRIBUTION (Last 24h)")

    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT
                regime,
                COUNT(*) as cnt
            FROM market_regime
            WHERE timestamp > strftime('%s','now','-24 hours')*1000
            GROUP BY regime
            ORDER BY cnt DESC
        """)
        rows = cursor.fetchall()

        if not rows:
            print("  No regime data.")
            return

        total = sum(row['cnt'] for row in rows)
        print(f"  {'Regime':<20} {'Samples':>10} {'%':>8}")
        print("-" * 40)

        for row in rows:
            pct = (row['cnt'] / total) * 100 if total > 0 else 0
            print(f"  {row['regime']:<20} {row['cnt']:>10} {pct:>7.1f}%")

    except Exception as e:
        print(f"  Error: {e}")


def check_performance_summary(conn):
    """Show performance summary by pocket."""
    print_header("PERFORMANCE BY POCKET (Net-After-Costs)")

    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT
                COALESCE(pocket_id, 'unknown') as pocket,
                COUNT(*) as n,
                SUM(CASE WHEN pnl_after_costs_pct > 0 THEN 1 ELSE 0 END) as wins,
                ROUND(AVG(pnl_pct), 4) as avg_raw_pnl,
                ROUND(AVG(total_costs_pct), 4) as avg_costs,
                ROUND(AVG(pnl_after_costs_pct), 4) as avg_net_pnl
            FROM trades
            WHERE status = 'closed'
            GROUP BY pocket_id
            ORDER BY avg_net_pnl DESC
        """)
        rows = cursor.fetchall()

        if not rows:
            print("  No closed trades.")
            return

        print(f"  {'Pocket':<10} {'N':>6} {'Win%':>8} {'Raw PnL':>10} {'Costs':>10} {'Net PnL':>10}")
        print("-" * 60)

        for row in rows:
            win_pct = (row['wins'] / row['n']) * 100 if row['n'] > 0 else 0
            avg_raw = row['avg_raw_pnl'] or 0
            avg_costs = row['avg_costs'] or 0
            avg_net = row['avg_net_pnl'] or 0

            print(f"  {row['pocket']:<10} {row['n']:>6} {win_pct:>7.1f}% "
                  f"{avg_raw:>+9.4f}% {avg_costs:>9.4f}% {avg_net:>+9.4f}%")

    except Exception as e:
        print(f"  Error: {e}")


def check_cost_model_impact(conn):
    """Show cost model impact on profitability."""
    print_header("COST MODEL IMPACT")

    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN pnl_pct > 0 THEN 1 ELSE 0 END) as raw_wins,
                SUM(CASE WHEN pnl_after_costs_pct > 0 THEN 1 ELSE 0 END) as net_wins,
                ROUND(SUM(pnl_pct), 4) as total_raw_pnl,
                ROUND(SUM(total_costs_pct), 4) as total_costs,
                ROUND(SUM(pnl_after_costs_pct), 4) as total_net_pnl,
                ROUND(AVG(entry_fee_pct), 4) as avg_entry_fee,
                ROUND(AVG(exit_fee_pct), 4) as avg_exit_fee,
                ROUND(AVG(spread_cost_pct), 4) as avg_spread,
                ROUND(AVG(slippage_pct), 4) as avg_slippage
            FROM trades
            WHERE status = 'closed'
        """)
        row = cursor.fetchone()

        if not row or row['total'] == 0:
            print("  No closed trades.")
            return

        raw_win_pct = (row['raw_wins'] / row['total']) * 100
        net_win_pct = (row['net_wins'] / row['total']) * 100

        print(f"  Total trades:     {row['total']}")
        print()
        print(f"  Before costs:")
        print(f"    Win rate:       {raw_win_pct:.1f}%")
        print(f"    Total PnL:      {row['total_raw_pnl']:+.4f}%")
        print()
        print(f"  Cost Breakdown (avg per trade):")
        print(f"    Entry fee:      {row['avg_entry_fee']:.4f}%")
        print(f"    Exit fee:       {row['avg_exit_fee']:.4f}%")
        print(f"    Spread cost:    {row['avg_spread']:.4f}%")
        print(f"    Slippage:       {row['avg_slippage']:.4f}%")
        print(f"    TOTAL:          {row['total_costs']/row['total']:.4f}%")
        print()
        print(f"  After costs:")
        print(f"    Win rate:       {net_win_pct:.1f}%")
        print(f"    Total PnL:      {row['total_net_pnl']:+.4f}%")
        print()

        if row['total_net_pnl'] < 0:
            print("  ⚠️  NEGATIVE NET PnL - Strategy not profitable after costs!")
        elif row['total_net_pnl'] > 0:
            print("  ✓  POSITIVE NET PnL - Strategy profitable after costs!")

    except Exception as e:
        print(f"  Error: {e}")


def check_mfe_mae_analysis(conn):
    """Show MFE/MAE analysis by pocket - reveals edge quality."""
    print_header("MFE/MAE ANALYSIS (Edge Quality)")

    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT
                COALESCE(pocket_id, '?') as pocket,
                COUNT(*) as n,
                ROUND(AVG(mfe), 4) as avg_mfe,
                ROUND(MAX(mfe), 4) as max_mfe,
                ROUND(AVG(mae), 4) as avg_mae,
                ROUND(MIN(mae), 4) as worst_mae,
                ROUND(AVG(mfe) - ABS(AVG(mae)), 4) as edge_ratio,
                ROUND(AVG(CASE WHEN pnl_after_costs_pct > 0 THEN mfe ELSE NULL END), 4) as winner_mfe,
                ROUND(AVG(CASE WHEN pnl_after_costs_pct <= 0 THEN mfe ELSE NULL END), 4) as loser_mfe
            FROM trades
            WHERE status = 'closed' AND (mfe != 0 OR mae != 0)
            GROUP BY pocket_id
            ORDER BY edge_ratio DESC
        """)
        rows = cursor.fetchall()

        if not rows:
            print("  No MFE/MAE data available yet.")
            print("  (Fix was deployed - waiting for new trades)")
            return

        print(f"  {'Pocket':<8} {'N':>5} {'Avg MFE':>10} {'Max MFE':>10} {'Avg MAE':>10} {'Edge':>10} {'W-MFE':>10} {'L-MFE':>10}")
        print("-" * 85)

        for row in rows:
            edge_ratio = row['edge_ratio'] or 0
            edge_symbol = "+" if edge_ratio > 0 else ""

            winner_mfe = row['winner_mfe'] or 0
            loser_mfe = row['loser_mfe'] or 0

            print(f"  {row['pocket']:<8} {row['n']:>5} "
                  f"{row['avg_mfe']:>+9.4f}% {row['max_mfe']:>+9.4f}% "
                  f"{row['avg_mae']:>9.4f}% {edge_symbol}{edge_ratio:>8.4f}% "
                  f"{winner_mfe:>+9.4f}% {loser_mfe:>+9.4f}%")

        print()
        print("  MFE = Max Favorable Excursion (best profit during trade)")
        print("  MAE = Max Adverse Excursion (worst drawdown during trade)")
        print("  Edge = MFE - |MAE| (positive = good entry timing)")
        print("  W-MFE = Avg MFE for winners, L-MFE = Avg MFE for losers")

    except Exception as e:
        print(f"  Error: {e}")


def check_causality_performance(conn):
    """Show performance by primary cause - which WHY is profitable?"""
    print_header("CAUSALITY PERFORMANCE (Winners vs Losers by WHY)")

    cursor = conn.cursor()
    try:
        # Join trades with trade_causality to get cause performance
        # NOTE: Must join on trade_id TEXT = TEXT (not id INTEGER)
        cursor.execute("""
            SELECT
                COALESCE(c.primary_cause, 'unknown') as cause,
                COUNT(*) as n,
                SUM(CASE WHEN t.pnl_after_costs_pct > 0 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN t.pnl_after_costs_pct <= 0 THEN 1 ELSE 0 END) as losses,
                ROUND(AVG(t.pnl_after_costs_pct), 4) as avg_net_pnl,
                ROUND(SUM(t.pnl_after_costs_pct), 4) as total_net_pnl
            FROM trades t
            LEFT JOIN trade_causality c ON t.trade_id = c.trade_id
            WHERE t.status = 'closed'
            GROUP BY c.primary_cause
            ORDER BY total_net_pnl DESC
        """)
        rows = cursor.fetchall()

        if not rows:
            print("  No causality data available.")
            return

        print(f"  {'Cause':<25} {'N':>5} {'Wins':>5} {'Loss':>5} {'Win%':>7} {'Avg PnL':>10} {'Total PnL':>12}")
        print("-" * 80)

        for row in rows:
            win_pct = (row['wins'] / row['n']) * 100 if row['n'] > 0 else 0
            avg_pnl = row['avg_net_pnl'] or 0
            total_pnl = row['total_net_pnl'] or 0

            # Highlight profitable causes
            pnl_marker = "✓" if total_pnl > 0 else "✗" if total_pnl < -0.1 else " "

            print(f"{pnl_marker} {row['cause']:<24} {row['n']:>5} {row['wins']:>5} {row['losses']:>5} "
                  f"{win_pct:>6.1f}% {avg_pnl:>+9.4f}% {total_pnl:>+11.4f}%")

        print()
        print("  ✓ = Profitable cause, ✗ = Losing cause")
        print("  Focus on causes with positive Total PnL!")

    except Exception as e:
        print(f"  Error: {e}")


def check_blocked_causes(conn):
    """Show causality filter blocks in last 24h."""
    print_header("CAUSALITY FILTER BLOCKS (Last 24h)")

    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT
                REPLACE(block_reason, 'cause_filtered:', '') as cause,
                COUNT(*) as cnt
            FROM blocked_signals
            WHERE block_reason LIKE 'cause_filtered:%'
              AND timestamp > strftime('%s','now','-24 hours')*1000
            GROUP BY block_reason
            ORDER BY cnt DESC
            LIMIT 10
        """)
        rows = cursor.fetchall()

        if not rows:
            print("  No causality filter blocks in last 24h.")
            return

        total = sum(row['cnt'] for row in rows)
        print(f"  Total blocked by causality filter: {total}")
        print()
        print(f"  {'Cause Blocked':<30} {'Count':>8}")
        print("-" * 45)

        for row in rows:
            print(f"  {row['cause']:<30} {row['cnt']:>8}")

    except Exception as e:
        print(f"  Error: {e}")


def check_performance_by_execution_mode(conn):
    """Show performance by execution mode (taker vs maker)."""
    print_header("PERFORMANCE BY EXECUTION MODE")

    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT
                COALESCE(execution_mode, 'taker') as mode,
                COUNT(*) as n,
                SUM(CASE WHEN pnl_after_costs_pct > 0 THEN 1 ELSE 0 END) as wins,
                ROUND(AVG(pnl_pct), 4) as avg_raw_pnl,
                ROUND(AVG(total_costs_pct), 4) as avg_costs,
                ROUND(AVG(pnl_after_costs_pct), 4) as avg_net_pnl,
                ROUND(SUM(pnl_after_costs_pct), 4) as total_net_pnl
            FROM trades
            WHERE status = 'closed'
            GROUP BY execution_mode
            ORDER BY avg_net_pnl DESC
        """)
        rows = cursor.fetchall()

        if not rows:
            print("  No closed trades.")
            return

        print(f"  {'Mode':<10} {'N':>6} {'Win%':>8} {'Raw PnL':>10} {'Costs':>10} {'Net PnL':>10} {'Total':>10}")
        print("-" * 70)

        for row in rows:
            win_pct = (row['wins'] / row['n']) * 100 if row['n'] > 0 else 0
            avg_raw = row['avg_raw_pnl'] or 0
            avg_costs = row['avg_costs'] or 0
            avg_net = row['avg_net_pnl'] or 0
            total_net = row['total_net_pnl'] or 0

            print(f"  {row['mode']:<10} {row['n']:>6} {win_pct:>7.1f}% "
                  f"{avg_raw:>+9.4f}% {avg_costs:>9.4f}% {avg_net:>+9.4f}% {total_net:>+9.4f}%")

    except Exception as e:
        print(f"  Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="HFT Bot Healthcheck")
    parser.add_argument("--db", default="hft_trades.db", help="Database path")
    args = parser.parse_args()

    # Find database
    db_path = Path(args.db)
    if not db_path.exists():
        # Try common locations
        for path in [Path.home() / "Analize-" / "hft_trades.db",
                     Path.home() / "hft_trades.db"]:
            if path.exists():
                db_path = path
                break

    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        return

    print(f"\n🔍 HFT Bot Healthcheck - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Database: {db_path}")

    conn = get_db_connection(str(db_path))

    try:
        check_regime_distribution(conn)
        check_gate_blocks(conn)
        check_performance_summary(conn)
        check_mfe_mae_analysis(conn)
        check_causality_performance(conn)
        check_blocked_causes(conn)
        check_cost_model_impact(conn)
        check_performance_by_execution_mode(conn)
        check_trades(conn)

        print_header("END OF HEALTHCHECK")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
