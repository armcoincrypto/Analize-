#!/usr/bin/env python3
"""
Coin Audit Report Generator
===========================
Generates a comprehensive audit report for a trading symbol with deployable config.

Usage:
    python tools/coin_audit.py --db hft_trades.db --symbol XRP
    python tools/coin_audit.py --db hft_trades.db --symbol XRP --days 30 --output config_xrp.txt

Output:
    A) Regime profitability table
    B) Cause leaderboard with FULL/PROBE comparison
    C) Blocked signals counterfactual analysis
    D) Recommendations and deployable config snippet
"""

import argparse
import sqlite3
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Add parent dir for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from hft_system.symbol_utils import normalize_db_symbol
from hft_system.db import open_sqlite


@dataclass
class RegimeStats:
    """Statistics for a market regime."""
    regime: str
    trade_count: int
    wins: int
    losses: int
    win_rate: float
    total_pnl_pct: float
    avg_pnl_after_costs_pct: float
    avg_mfe: float
    avg_mae: float
    avg_hold_time_sec: float


@dataclass
class CauseStats:
    """Statistics for a primary cause."""
    cause: str
    trade_count: int
    wins: int
    losses: int
    win_rate: float
    total_pnl_pct: float
    avg_pnl_after_costs_pct: float
    avg_mfe: float
    avg_mae: float
    edge_real_pct: float
    cause_class: str  # FULL or PROBE


@dataclass
class BlockedSignalStats:
    """Statistics for blocked signals."""
    block_reason: str
    count: int
    avg_would_have_pnl: float
    positive_pnl_count: int
    positive_pnl_rate: float


@dataclass
class AuditRecommendations:
    """Audit recommendations."""
    trade_regimes: List[str]
    avoid_regimes: List[str]
    enable_causes: List[str]
    disable_causes: List[str]
    suggested_tp: float
    suggested_sl: float
    suggested_time_stop: int
    suggested_execution_mode: str
    suggested_spread_threshold: float
    min_imbalance: float


def print_header(title: str, char: str = "="):
    width = 80
    print(f"\n{char * width}")
    print(f" {title}")
    print(char * width)


def get_regime_stats(conn: sqlite3.Connection, symbol: str, days: int = None) -> List[RegimeStats]:
    """Get profitability statistics by market regime."""
    cursor = conn.cursor()

    # Build time and symbol filter
    filters = ["status = 'closed'"]
    params = []

    if symbol:
        filters.append("symbol LIKE ?")
        params.append(f"%{symbol}%")

    if days:
        cutoff_ms = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        filters.append("entry_time >= ?")
        params.append(cutoff_ms)

    where_clause = " AND ".join(filters)

    query = f"""
        SELECT
            COALESCE(regime_at_entry, 'unknown') as regime,
            COUNT(*) as trade_count,
            SUM(CASE WHEN pnl_after_costs_pct > 0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN pnl_after_costs_pct <= 0 THEN 1 ELSE 0 END) as losses,
            SUM(pnl_after_costs_pct) as total_pnl,
            AVG(pnl_after_costs_pct) as avg_pnl,
            AVG(COALESCE(mfe, 0)) as avg_mfe,
            AVG(COALESCE(mae, 0)) as avg_mae,
            AVG(COALESCE(hold_time_sec, 0)) as avg_hold_time
        FROM trades
        WHERE {where_clause}
        GROUP BY COALESCE(regime_at_entry, 'unknown')
        ORDER BY trade_count DESC
    """

    cursor.execute(query, params)
    rows = cursor.fetchall()

    stats = []
    for row in rows:
        trade_count = row[1]
        wins = row[2] or 0
        stats.append(RegimeStats(
            regime=row[0],
            trade_count=trade_count,
            wins=wins,
            losses=row[3] or 0,
            win_rate=(wins / trade_count * 100) if trade_count > 0 else 0,
            total_pnl_pct=row[4] or 0,
            avg_pnl_after_costs_pct=row[5] or 0,
            avg_mfe=row[6] or 0,
            avg_mae=row[7] or 0,
            avg_hold_time_sec=row[8] or 0
        ))

    return stats


def get_cause_stats(conn: sqlite3.Connection, symbol: str, days: int = None) -> List[CauseStats]:
    """Get profitability statistics by primary cause."""
    cursor = conn.cursor()

    # Build filters
    filters = ["t.status = 'closed'"]
    params = []

    if symbol:
        filters.append("t.symbol LIKE ?")
        params.append(f"%{symbol}%")

    if days:
        cutoff_ms = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        filters.append("t.entry_time >= ?")
        params.append(cutoff_ms)

    where_clause = " AND ".join(filters)

    # Query joining trades with trade_causality and trade_edge
    query = f"""
        SELECT
            COALESCE(tc.primary_cause, 'unknown') as cause,
            COALESCE(t.cause_class, 'FULL') as cause_class,
            COUNT(*) as trade_count,
            SUM(CASE WHEN t.pnl_after_costs_pct > 0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN t.pnl_after_costs_pct <= 0 THEN 1 ELSE 0 END) as losses,
            SUM(t.pnl_after_costs_pct) as total_pnl,
            AVG(t.pnl_after_costs_pct) as avg_pnl,
            AVG(COALESCE(t.mfe, 0)) as avg_mfe,
            AVG(COALESCE(t.mae, 0)) as avg_mae,
            AVG(COALESCE(te.real_edge_flag, 0)) * 100 as edge_real_pct
        FROM trades t
        LEFT JOIN trade_causality tc ON t.trade_id = tc.trade_id
        LEFT JOIN trade_edge te ON t.trade_id = te.trade_id
        WHERE {where_clause}
        GROUP BY COALESCE(tc.primary_cause, 'unknown'), COALESCE(t.cause_class, 'FULL')
        ORDER BY trade_count DESC
    """

    cursor.execute(query, params)
    rows = cursor.fetchall()

    stats = []
    for row in rows:
        trade_count = row[2]
        wins = row[3] or 0
        stats.append(CauseStats(
            cause=row[0],
            cause_class=row[1],
            trade_count=trade_count,
            wins=wins,
            losses=row[4] or 0,
            win_rate=(wins / trade_count * 100) if trade_count > 0 else 0,
            total_pnl_pct=row[5] or 0,
            avg_pnl_after_costs_pct=row[6] or 0,
            avg_mfe=row[7] or 0,
            avg_mae=row[8] or 0,
            edge_real_pct=row[9] or 0
        ))

    return stats


def get_blocked_signal_stats(conn: sqlite3.Connection, symbol: str, days: int = None) -> List[BlockedSignalStats]:
    """Get statistics for blocked signals - counterfactual analysis."""
    cursor = conn.cursor()

    # Check if blocked_signals table exists and has data
    try:
        cursor.execute("SELECT COUNT(*) FROM blocked_signals")
        if cursor.fetchone()[0] == 0:
            return []
    except sqlite3.OperationalError:
        return []

    # Build filters
    filters = ["1=1"]
    params = []

    if symbol:
        filters.append("symbol LIKE ?")
        params.append(f"%{symbol}%")

    if days:
        cutoff_ms = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        filters.append("timestamp >= ?")
        params.append(cutoff_ms)

    where_clause = " AND ".join(filters)

    query = f"""
        SELECT
            block_reason,
            COUNT(*) as count,
            AVG(COALESCE(would_have_pnl, 0)) as avg_would_have_pnl,
            SUM(CASE WHEN would_have_pnl > 0 THEN 1 ELSE 0 END) as positive_count
        FROM blocked_signals
        WHERE {where_clause}
        GROUP BY block_reason
        ORDER BY count DESC
    """

    cursor.execute(query, params)
    rows = cursor.fetchall()

    stats = []
    for row in rows:
        count = row[1]
        positive_count = row[3] or 0
        stats.append(BlockedSignalStats(
            block_reason=row[0],
            count=count,
            avg_would_have_pnl=row[2] or 0,
            positive_pnl_count=positive_count,
            positive_pnl_rate=(positive_count / count * 100) if count > 0 else 0
        ))

    return stats


def get_exit_reason_stats(conn: sqlite3.Connection, symbol: str, days: int = None) -> Dict:
    """Get statistics by exit reason for TP/SL/Time optimization."""
    cursor = conn.cursor()

    filters = ["status = 'closed'"]
    params = []

    if symbol:
        filters.append("symbol LIKE ?")
        params.append(f"%{symbol}%")

    if days:
        cutoff_ms = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        filters.append("entry_time >= ?")
        params.append(cutoff_ms)

    where_clause = " AND ".join(filters)

    query = f"""
        SELECT
            exit_reason,
            COUNT(*) as count,
            AVG(pnl_after_costs_pct) as avg_pnl,
            AVG(hold_time_sec) as avg_hold_time,
            AVG(mfe) as avg_mfe,
            AVG(mae) as avg_mae
        FROM trades
        WHERE {where_clause}
        GROUP BY exit_reason
        ORDER BY count DESC
    """

    cursor.execute(query, params)
    rows = cursor.fetchall()

    return {
        row[0]: {
            "count": row[1],
            "avg_pnl": row[2] or 0,
            "avg_hold_time": row[3] or 0,
            "avg_mfe": row[4] or 0,
            "avg_mae": row[5] or 0
        }
        for row in rows
    }


def get_execution_mode_stats(conn: sqlite3.Connection, symbol: str, days: int = None) -> Dict:
    """Get maker vs taker stats for execution mode recommendation."""
    cursor = conn.cursor()

    filters = ["status = 'closed'"]
    params = []

    if symbol:
        filters.append("symbol LIKE ?")
        params.append(f"%{symbol}%")

    if days:
        cutoff_ms = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        filters.append("entry_time >= ?")
        params.append(cutoff_ms)

    where_clause = " AND ".join(filters)

    query = f"""
        SELECT
            COALESCE(execution_mode, 'taker') as mode,
            COUNT(*) as count,
            AVG(pnl_after_costs_pct) as avg_pnl,
            AVG(total_costs_pct) as avg_costs,
            AVG(spread_cost_pct) as avg_spread
        FROM trades
        WHERE {where_clause}
        GROUP BY COALESCE(execution_mode, 'taker')
    """

    cursor.execute(query, params)
    rows = cursor.fetchall()

    return {
        row[0]: {
            "count": row[1],
            "avg_pnl": row[2] or 0,
            "avg_costs": row[3] or 0,
            "avg_spread": row[4] or 0
        }
        for row in rows
    }


def generate_recommendations(
    regime_stats: List[RegimeStats],
    cause_stats: List[CauseStats],
    blocked_stats: List[BlockedSignalStats],
    exit_stats: Dict,
    execution_stats: Dict
) -> AuditRecommendations:
    """Generate recommendations based on all statistics."""

    # Regime recommendations
    # Trade regimes with positive PnL and reasonable sample size
    trade_regimes = [
        r.regime for r in regime_stats
        if r.avg_pnl_after_costs_pct > 0 and r.trade_count >= 5
    ]

    # Avoid regimes with negative PnL
    avoid_regimes = [
        r.regime for r in regime_stats
        if r.avg_pnl_after_costs_pct < -0.05 and r.trade_count >= 5
    ]

    # Cause recommendations
    # Enable causes with positive PnL and decent sample
    enable_causes = []
    disable_causes = []

    for c in cause_stats:
        if c.trade_count >= 3:
            if c.avg_pnl_after_costs_pct > 0 and c.win_rate >= 40:
                enable_causes.append(c.cause)
            elif c.avg_pnl_after_costs_pct < -0.10 or c.win_rate < 25:
                disable_causes.append(c.cause)

    # TP/SL/Time recommendations based on MFE/MAE analysis
    avg_mfe = sum(c.avg_mfe for c in cause_stats if c.trade_count > 0) / max(len(cause_stats), 1)
    avg_mae = sum(c.avg_mae for c in cause_stats if c.trade_count > 0) / max(len(cause_stats), 1)

    # Suggested TP should be achievable (below avg MFE)
    suggested_tp = round(max(0.10, min(avg_mfe * 0.8, 0.30)), 2) if avg_mfe > 0 else 0.15

    # Suggested SL should limit losses (around avg MAE or tighter)
    suggested_sl = round(max(0.08, min(abs(avg_mae) * 1.2, 0.20)), 2) if avg_mae < 0 else 0.10

    # Time stop based on profitable hold times
    tp_stats = exit_stats.get("take_profit", {})
    if tp_stats.get("count", 0) > 0:
        # If TPs happen, use their avg hold time as a guide
        suggested_time_stop = int(min(max(tp_stats.get("avg_hold_time", 60) * 1.5, 30), 120))
    else:
        suggested_time_stop = 60

    # Execution mode recommendation
    taker_stats = execution_stats.get("taker", {})
    maker_stats = execution_stats.get("maker", {})

    if maker_stats.get("count", 0) >= 10 and taker_stats.get("count", 0) >= 10:
        # Compare if we have both
        if maker_stats.get("avg_pnl", 0) > taker_stats.get("avg_pnl", 0):
            suggested_execution_mode = "maker"
        else:
            suggested_execution_mode = "taker"
    elif maker_stats.get("avg_pnl", 0) > 0:
        suggested_execution_mode = "maker"
    else:
        suggested_execution_mode = "taker"

    # Spread threshold (use avg spread cost as guide)
    avg_spread = max(
        taker_stats.get("avg_spread", 0.02),
        maker_stats.get("avg_spread", 0.01)
    )
    suggested_spread_threshold = round(max(0.01, avg_spread * 1.5), 3)

    # Min imbalance from cause analysis
    ob_causes = [c for c in cause_stats if "imbalance" in c.cause.lower()]
    if ob_causes and any(c.avg_pnl_after_costs_pct > 0 for c in ob_causes):
        min_imbalance = 0.60  # Keep current
    else:
        min_imbalance = 0.65  # Tighten if OB signals not working

    return AuditRecommendations(
        trade_regimes=trade_regimes,
        avoid_regimes=avoid_regimes,
        enable_causes=enable_causes,
        disable_causes=disable_causes,
        suggested_tp=suggested_tp,
        suggested_sl=suggested_sl,
        suggested_time_stop=suggested_time_stop,
        suggested_execution_mode=suggested_execution_mode,
        suggested_spread_threshold=suggested_spread_threshold,
        min_imbalance=min_imbalance
    )


def generate_config_snippet(symbol: str, recs: AuditRecommendations) -> str:
    """Generate a deployable config snippet."""

    # Separate long/short causes (simplified - in practice need side info)
    long_causes = [c for c in recs.enable_causes if "buy" in c.lower() or "bullish" in c.lower() or "up" in c.lower()]
    short_causes = [c for c in recs.enable_causes if "sell" in c.lower() or "bearish" in c.lower() or "down" in c.lower()]

    # If we couldn't separate, put them all in both
    if not long_causes and not short_causes:
        long_causes = recs.enable_causes
        short_causes = recs.enable_causes

    snippet = f'''
# ============================================================
# CONFIG SNIPPET FOR {symbol.upper()}
# Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
# ============================================================

# === TRADE PARAMETERS ===
# Based on MFE/MAE analysis
take_profit_pct: float = {recs.suggested_tp}
stop_loss_pct: float = {recs.suggested_sl}
time_stop_seconds: int = {recs.suggested_time_stop}

# === EXECUTION MODE ===
# Recommended: {recs.suggested_execution_mode}
execution_mode: str = "{recs.suggested_execution_mode}"
auto_maker_spread_threshold: float = {recs.suggested_spread_threshold}

# === REGIME FILTER ===
# Trade only in these regimes:
allowed_regimes: List[str] = {json.dumps(recs.trade_regimes)}
# Avoid these regimes:
blocked_regimes: List[str] = {json.dumps(recs.avoid_regimes)}

# === CAUSE POLICY ===
# Causes that show positive edge (FULL size):
cause_full_allow_long: List[str] = {json.dumps(long_causes)}
cause_full_allow_short: List[str] = {json.dumps(short_causes)}

# Causes to disable (negative edge):
# {json.dumps(recs.disable_causes)}

# Orderbook imbalance threshold:
min_imbalance: float = {recs.min_imbalance}

# ============================================================
# COPY THE ABOVE INTO config.py AND ADJUST AS NEEDED
# ============================================================
'''
    return snippet


def print_regime_table(stats: List[RegimeStats]):
    """Print regime profitability table."""
    print_header("A) REGIME PROFITABILITY TABLE")

    if not stats:
        print("  No regime data available")
        return

    # Table header
    print(f"\n  {'Regime':<20} {'Trades':>7} {'WR%':>6} {'Avg PnL%':>10} {'MFE':>8} {'MAE':>8} {'Hold(s)':>8}")
    print(f"  {'-'*20} {'-'*7} {'-'*6} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")

    for r in stats:
        indicator = "✓" if r.avg_pnl_after_costs_pct > 0 else "✗"
        print(f"  {r.regime:<20} {r.trade_count:>7} {r.win_rate:>5.1f}% {r.avg_pnl_after_costs_pct:>+9.4f}% "
              f"{r.avg_mfe:>7.4f}% {r.avg_mae:>7.4f}% {r.avg_hold_time_sec:>7.1f}s {indicator}")


def print_cause_leaderboard(stats: List[CauseStats]):
    """Print cause leaderboard with FULL/PROBE comparison."""
    print_header("B) CAUSE LEADERBOARD")

    if not stats:
        print("  No cause data available")
        return

    # Separate FULL and PROBE
    full_causes = [c for c in stats if c.cause_class == "FULL"]
    probe_causes = [c for c in stats if c.cause_class == "PROBE"]

    # Table header
    print(f"\n  {'Cause':<25} {'Class':>6} {'Trades':>7} {'WR%':>6} {'Avg PnL%':>10} {'MFE':>7} {'MAE':>7} {'Edge%':>6}")
    print(f"  {'-'*25} {'-'*6} {'-'*7} {'-'*6} {'-'*10} {'-'*7} {'-'*7} {'-'*6}")

    # Print all causes sorted by trade count
    for c in sorted(stats, key=lambda x: x.trade_count, reverse=True):
        indicator = "✓" if c.avg_pnl_after_costs_pct > 0 else "✗"
        print(f"  {c.cause:<25} {c.cause_class:>6} {c.trade_count:>7} {c.win_rate:>5.1f}% "
              f"{c.avg_pnl_after_costs_pct:>+9.4f}% {c.avg_mfe:>6.4f}% {c.avg_mae:>6.4f}% {c.edge_real_pct:>5.1f}% {indicator}")

    # FULL vs PROBE summary
    if full_causes and probe_causes:
        print(f"\n  --- FULL vs PROBE SUMMARY ---")
        full_trades = sum(c.trade_count for c in full_causes)
        full_avg_pnl = sum(c.avg_pnl_after_costs_pct * c.trade_count for c in full_causes) / max(full_trades, 1)
        probe_trades = sum(c.trade_count for c in probe_causes)
        probe_avg_pnl = sum(c.avg_pnl_after_costs_pct * c.trade_count for c in probe_causes) / max(probe_trades, 1)

        print(f"  FULL:  {full_trades} trades, avg PnL {full_avg_pnl:+.4f}%")
        print(f"  PROBE: {probe_trades} trades, avg PnL {probe_avg_pnl:+.4f}%")


def print_blocked_signals(stats: List[BlockedSignalStats]):
    """Print blocked signals counterfactual analysis."""
    print_header("C) BLOCKED SIGNALS COUNTERFACTUAL")

    if not stats:
        print("  No blocked signal data available")
        print("  (Blocked signals need would_have_pnl tracking)")
        return

    print(f"\n  {'Block Reason':<25} {'Count':>7} {'Avg Would PnL%':>14} {'Positive':>10} {'Pos Rate':>10}")
    print(f"  {'-'*25} {'-'*7} {'-'*14} {'-'*10} {'-'*10}")

    for b in stats:
        indicator = "⚠" if b.avg_would_have_pnl > 0 else ""
        print(f"  {b.block_reason:<25} {b.count:>7} {b.avg_would_have_pnl:>+13.4f}% "
              f"{b.positive_pnl_count:>10} {b.positive_pnl_rate:>9.1f}% {indicator}")

    # Highlight blocks that may be too aggressive
    false_blocks = [b for b in stats if b.avg_would_have_pnl > 0 and b.positive_pnl_rate > 50]
    if false_blocks:
        print(f"\n  ⚠ WARNING: These blocks may be too aggressive (>50% would have been profitable):")
        for b in false_blocks:
            print(f"    - {b.block_reason}: {b.positive_pnl_rate:.0f}% positive")


def print_recommendations(recs: AuditRecommendations):
    """Print recommendations."""
    print_header("D) RECOMMENDATIONS")

    print(f"\n  --- REGIMES ---")
    print(f"  Trade only:  {recs.trade_regimes if recs.trade_regimes else ['(all - insufficient data)']}")
    print(f"  Avoid:       {recs.avoid_regimes if recs.avoid_regimes else ['(none identified)']}")

    print(f"\n  --- CAUSES ---")
    print(f"  Enable (FULL):  {recs.enable_causes if recs.enable_causes else ['(need more data)']}")
    print(f"  Disable:        {recs.disable_causes if recs.disable_causes else ['(none identified)']}")

    print(f"\n  --- TRADE PARAMETERS ---")
    print(f"  Take Profit:    {recs.suggested_tp}%")
    print(f"  Stop Loss:      {recs.suggested_sl}%")
    print(f"  Time Stop:      {recs.suggested_time_stop}s")

    print(f"\n  --- EXECUTION ---")
    print(f"  Mode:           {recs.suggested_execution_mode}")
    print(f"  Spread Thresh:  {recs.suggested_spread_threshold}%")
    print(f"  Min Imbalance:  {recs.min_imbalance}")


def main():
    parser = argparse.ArgumentParser(
        description="Coin Audit Report Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/coin_audit.py --db hft_trades.db --symbol XRP
  python tools/coin_audit.py --db hft_trades.db --symbol SOL --days 30
  python tools/coin_audit.py --db hft_trades.db --symbol ADA --output config_ada.txt
        """
    )
    parser.add_argument("--db", default="hft_trades.db", help="Database path")
    parser.add_argument("--symbol", required=True, help="Trading symbol (e.g., XRP, SOL, ADA)")
    parser.add_argument("--days", type=int, help="Filter to last N days")
    parser.add_argument("--output", "-o", help="Output config snippet to file")
    parser.add_argument("--quiet", "-q", action="store_true", help="Only output config snippet")
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

    # Connect (bot-safe with WAL + busy_timeout)
    conn = open_sqlite(str(db_path))

    # Normalize symbol for consistent DB queries
    symbol = normalize_db_symbol(args.symbol) if args.symbol else None

    # Get all stats
    regime_stats = get_regime_stats(conn, symbol, args.days)
    cause_stats = get_cause_stats(conn, symbol, args.days)
    blocked_stats = get_blocked_signal_stats(conn, symbol, args.days)
    exit_stats = get_exit_reason_stats(conn, symbol, args.days)
    execution_stats = get_execution_mode_stats(conn, symbol, args.days)

    # Generate recommendations
    recs = generate_recommendations(regime_stats, cause_stats, blocked_stats, exit_stats, execution_stats)

    # Generate config snippet
    config_snippet = generate_config_snippet(args.symbol, recs)

    if not args.quiet:
        # Print report header
        print(f"""
+================================================================================+
|                    COIN AUDIT REPORT: {args.symbol.upper():<40} |
+================================================================================+
| Database: {str(db_path):<66} |
| Period:   {f"Last {args.days} days" if args.days else "All time":<66} |
+================================================================================+
        """)

        # Print all sections
        print_regime_table(regime_stats)
        print_cause_leaderboard(cause_stats)
        print_blocked_signals(blocked_stats)
        print_recommendations(recs)

        # Print config snippet
        print_header("CONFIG SNIPPET", "=")
        print(config_snippet)

    # Output to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            f.write(f"# Coin Audit Report: {args.symbol.upper()}\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n")
            f.write(f"# Database: {db_path}\n")
            f.write(f"# Period: {'Last ' + str(args.days) + ' days' if args.days else 'All time'}\n")
            f.write(config_snippet)
        print(f"\n Config snippet saved to: {args.output}")

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
