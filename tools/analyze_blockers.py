#!/usr/bin/env python3
"""
Blocker Analysis Tool
=====================
Analyze which blockers cause the most trade rejections.

Quantifies the 80/20 rule: which blockers cause 80% of no-trades.

Combines data from:
- blocked_signals table (no-trade zone blocks)
- winner_gate_blocks table (winner gate blocks)

Usage:
    python tools/analyze_blockers.py --db hft_trades.db --days 7
    python tools/analyze_blockers.py --db hft_trades.db --days 1 --verbose
    python tools/analyze_blockers.py --db hft_trades.db --symbol SUIUSDT
"""

import argparse
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple


def open_sqlite(db_path: str) -> sqlite3.Connection:
    """Open SQLite with WAL mode and busy_timeout."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def get_time_cutoff_ms(days: int) -> int:
    """Get timestamp cutoff in milliseconds."""
    return int((datetime.now() - timedelta(days=days)).timestamp() * 1000)


def get_blocked_signals_stats(conn: sqlite3.Connection, cutoff_ms: int, symbol: str = None) -> Dict:
    """Get stats from blocked_signals table."""
    cursor = conn.cursor()

    symbol_filter = ""
    params = [cutoff_ms]
    if symbol:
        symbol_filter = "AND symbol = ?"
        params.append(symbol)

    # Total blocks
    cursor.execute(f"""
        SELECT COUNT(*) FROM blocked_signals
        WHERE timestamp >= ? {symbol_filter}
    """, params)
    total = cursor.fetchone()[0]

    # By block_reason
    cursor.execute(f"""
        SELECT block_reason, COUNT(*) as count
        FROM blocked_signals
        WHERE timestamp >= ? {symbol_filter}
        GROUP BY block_reason
        ORDER BY count DESC
    """, params)
    by_reason = {row[0]: row[1] for row in cursor.fetchall()}

    # By gate_name (if populated)
    cursor.execute(f"""
        SELECT gate_name, COUNT(*) as count
        FROM blocked_signals
        WHERE timestamp >= ? {symbol_filter}
        AND gate_name IS NOT NULL
        GROUP BY gate_name
        ORDER BY count DESC
    """, params)
    by_gate_name = {row[0]: row[1] for row in cursor.fetchall()}

    # By gate_param (if populated)
    cursor.execute(f"""
        SELECT gate_param, COUNT(*) as count
        FROM blocked_signals
        WHERE timestamp >= ? {symbol_filter}
        AND gate_param IS NOT NULL
        GROUP BY gate_param
        ORDER BY count DESC
    """, params)
    by_gate_param = {row[0]: row[1] for row in cursor.fetchall()}

    # By symbol
    cursor.execute(f"""
        SELECT symbol, COUNT(*) as count
        FROM blocked_signals
        WHERE timestamp >= ? {symbol_filter}
        GROUP BY symbol
        ORDER BY count DESC
    """, params)
    by_symbol = {row[0]: row[1] for row in cursor.fetchall()}

    # By regime
    cursor.execute(f"""
        SELECT regime, COUNT(*) as count
        FROM blocked_signals
        WHERE timestamp >= ? {symbol_filter}
        GROUP BY regime
        ORDER BY count DESC
    """, params)
    by_regime = {row[0]: row[1] for row in cursor.fetchall()}

    return {
        "total": total,
        "by_reason": by_reason,
        "by_gate_name": by_gate_name,
        "by_gate_param": by_gate_param,
        "by_symbol": by_symbol,
        "by_regime": by_regime
    }


def get_winner_gate_stats(conn: sqlite3.Connection, cutoff_ms: int, symbol: str = None) -> Dict:
    """Get stats from winner_gate_blocks table."""
    cursor = conn.cursor()

    # Check if table exists
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='winner_gate_blocks'
    """)
    if not cursor.fetchone():
        return {"total": 0, "by_reason": {}, "by_regime": {}, "by_tier": {}, "by_symbol": {}}

    symbol_filter = ""
    params = [cutoff_ms]
    if symbol:
        symbol_filter = "AND symbol = ?"
        params.append(symbol)

    # Total blocks
    cursor.execute(f"""
        SELECT COUNT(*) FROM winner_gate_blocks
        WHERE timestamp >= ? {symbol_filter}
    """, params)
    total = cursor.fetchone()[0]

    # By block_reason
    cursor.execute(f"""
        SELECT block_reason, COUNT(*) as count
        FROM winner_gate_blocks
        WHERE timestamp >= ? {symbol_filter}
        GROUP BY block_reason
        ORDER BY count DESC
    """, params)
    by_reason = {row[0]: row[1] for row in cursor.fetchall()}

    # By regime
    cursor.execute(f"""
        SELECT regime, COUNT(*) as count
        FROM winner_gate_blocks
        WHERE timestamp >= ? {symbol_filter}
        GROUP BY regime
        ORDER BY count DESC
    """, params)
    by_regime = {row[0]: row[1] for row in cursor.fetchall()}

    # By confidence_tier
    cursor.execute(f"""
        SELECT confidence_tier, COUNT(*) as count
        FROM winner_gate_blocks
        WHERE timestamp >= ? {symbol_filter}
        GROUP BY confidence_tier
        ORDER BY count DESC
    """, params)
    by_tier = {row[0]: row[1] for row in cursor.fetchall()}

    # By symbol
    cursor.execute(f"""
        SELECT symbol, COUNT(*) as count
        FROM winner_gate_blocks
        WHERE timestamp >= ? {symbol_filter}
        GROUP BY symbol
        ORDER BY count DESC
    """, params)
    by_symbol = {row[0]: row[1] for row in cursor.fetchall()}

    # Imbalance distribution for blocks
    cursor.execute(f"""
        SELECT
            CASE
                WHEN imbalance < 0.55 THEN '<0.55'
                WHEN imbalance < 0.60 THEN '0.55-0.60'
                WHEN imbalance < 0.65 THEN '0.60-0.65'
                WHEN imbalance < 0.70 THEN '0.65-0.70'
                WHEN imbalance < 0.75 THEN '0.70-0.75'
                ELSE '>=0.75'
            END as imb_range,
            COUNT(*) as count
        FROM winner_gate_blocks
        WHERE timestamp >= ? {symbol_filter}
        GROUP BY imb_range
        ORDER BY imb_range
    """, params)
    by_imbalance_range = {row[0]: row[1] for row in cursor.fetchall()}

    return {
        "total": total,
        "by_reason": by_reason,
        "by_regime": by_regime,
        "by_tier": by_tier,
        "by_symbol": by_symbol,
        "by_imbalance_range": by_imbalance_range
    }


def get_trade_count(conn: sqlite3.Connection, cutoff_ms: int, symbol: str = None) -> int:
    """Get number of executed trades for context."""
    cursor = conn.cursor()

    symbol_filter = ""
    params = [cutoff_ms]
    if symbol:
        symbol_filter = "AND symbol = ?"
        params.append(symbol)

    cursor.execute(f"""
        SELECT COUNT(*) FROM trades
        WHERE entry_time >= ? {symbol_filter}
    """, params)
    return cursor.fetchone()[0]


def calculate_cumulative_pct(items: Dict[str, int], total: int) -> List[Tuple[str, int, float, float]]:
    """
    Calculate cumulative percentage for 80/20 analysis.

    Returns list of (reason, count, pct, cumulative_pct)
    """
    if total == 0:
        return []

    result = []
    cumulative = 0
    for reason, count in sorted(items.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        cumulative += pct
        result.append((reason, count, pct, cumulative))
    return result


def print_section(title: str, char: str = "-"):
    """Print section header."""
    width = 70
    print()
    print(char * width)
    print(f" {title}")
    print(char * width)


def print_distribution(items: Dict[str, int], total: int, title: str, show_80_20: bool = True):
    """Print distribution with percentages."""
    if not items:
        print(f"  (no data)")
        return

    cumulative = calculate_cumulative_pct(items, total)

    print(f"\n  {'Blocker':<40} {'Count':>8} {'Pct':>8} {'Cumul':>8}")
    print(f"  {'-'*40} {'-'*8} {'-'*8} {'-'*8}")

    for reason, count, pct, cumul in cumulative:
        marker = " <-- 80% threshold" if show_80_20 and cumul >= 80 and (cumul - pct) < 80 else ""
        print(f"  {reason:<40} {count:>8} {pct:>7.1f}% {cumul:>7.1f}%{marker}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze which blockers cause the most trade rejections",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Last 7 days analysis
  python tools/analyze_blockers.py --db hft_trades.db --days 7

  # Today only, specific symbol
  python tools/analyze_blockers.py --db hft_trades.db --days 1 --symbol SUIUSDT

  # Full verbose output
  python tools/analyze_blockers.py --db hft_trades.db --days 7 --verbose
        """
    )
    parser.add_argument("--db", default="hft_trades.db", help="Database path")
    parser.add_argument("--days", type=int, default=7, help="Days to analyze (default: 7)")
    parser.add_argument("--symbol", type=str, default=None, help="Filter to specific symbol")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed breakdowns")
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

    conn = open_sqlite(str(db_path))
    cutoff_ms = get_time_cutoff_ms(args.days)

    # Header
    print("=" * 70)
    print(" BLOCKER ANALYSIS - 80/20 Rule")
    print("=" * 70)
    print(f"  Database: {db_path}")
    print(f"  Period: Last {args.days} days")
    if args.symbol:
        print(f"  Symbol filter: {args.symbol}")
    print(f"  Cutoff: {datetime.fromtimestamp(cutoff_ms/1000).isoformat()}")

    # Get stats
    blocked_stats = get_blocked_signals_stats(conn, cutoff_ms, args.symbol)
    winner_gate_stats = get_winner_gate_stats(conn, cutoff_ms, args.symbol)
    trade_count = get_trade_count(conn, cutoff_ms, args.symbol)

    total_blocks = blocked_stats["total"] + winner_gate_stats["total"]

    # Summary
    print_section("SUMMARY", "=")
    print(f"\n  Total blocks:           {total_blocks:,}")
    print(f"    - blocked_signals:    {blocked_stats['total']:,}")
    print(f"    - winner_gate_blocks: {winner_gate_stats['total']:,}")
    print(f"  Total trades executed:  {trade_count:,}")

    if total_blocks + trade_count > 0:
        block_rate = total_blocks / (total_blocks + trade_count) * 100
        print(f"  Block rate:             {block_rate:.1f}%")

    # Combined reason analysis
    print_section("TOP BLOCK REASONS (Combined)")
    combined_reasons = defaultdict(int)
    for reason, count in blocked_stats["by_reason"].items():
        combined_reasons[f"NO_TRADE_ZONE:{reason}"] += count
    for reason, count in winner_gate_stats["by_reason"].items():
        combined_reasons[f"WINNER_GATE:{reason}"] += count

    print_distribution(dict(combined_reasons), total_blocks, "Block Reasons")

    # No-trade zone details
    if blocked_stats["total"] > 0:
        print_section("NO-TRADE ZONE BLOCKS")
        print(f"  Total: {blocked_stats['total']:,}")
        print_distribution(blocked_stats["by_reason"], blocked_stats["total"], "By Reason")

        if args.verbose and blocked_stats["by_gate_name"]:
            print(f"\n  By Gate Name:")
            print_distribution(blocked_stats["by_gate_name"], blocked_stats["total"], "Gate Name", False)

        if args.verbose and blocked_stats["by_gate_param"]:
            print(f"\n  By Gate Parameter:")
            print_distribution(blocked_stats["by_gate_param"], blocked_stats["total"], "Gate Param", False)

    # Winner gate details
    if winner_gate_stats["total"] > 0:
        print_section("WINNER GATE BLOCKS")
        print(f"  Total: {winner_gate_stats['total']:,}")
        print_distribution(winner_gate_stats["by_reason"], winner_gate_stats["total"], "By Reason")

        if args.verbose:
            print(f"\n  By Regime:")
            print_distribution(winner_gate_stats["by_regime"], winner_gate_stats["total"], "Regime", False)

            print(f"\n  By Confidence Tier:")
            print_distribution(winner_gate_stats["by_tier"], winner_gate_stats["total"], "Tier", False)

            if winner_gate_stats.get("by_imbalance_range"):
                print(f"\n  By Imbalance Range:")
                print_distribution(winner_gate_stats["by_imbalance_range"],
                                 winner_gate_stats["total"], "Imbalance", False)

    # Per-symbol breakdown
    if args.verbose and not args.symbol:
        print_section("BY SYMBOL")
        combined_symbols = defaultdict(int)
        for sym, count in blocked_stats["by_symbol"].items():
            combined_symbols[sym] += count
        for sym, count in winner_gate_stats["by_symbol"].items():
            combined_symbols[sym] += count
        print_distribution(dict(combined_symbols), total_blocks, "Symbol", False)

    # Recommendations
    print_section("RECOMMENDATIONS", "=")

    if total_blocks == 0:
        print("\n  No blocks recorded - run bot with blockers enabled to collect data")
    else:
        # Find top blocker
        if combined_reasons:
            top_blocker = max(combined_reasons.items(), key=lambda x: x[1])
            top_pct = top_blocker[1] / total_blocks * 100
            print(f"\n  Top blocker: {top_blocker[0]} ({top_pct:.1f}% of blocks)")

            # Specific recommendations based on top blocker
            if "imb" in top_blocker[0].lower() or "imbalance" in top_blocker[0].lower():
                print(f"  -> Consider reducing min_imbalance threshold by 5-10%")
            elif "spread" in top_blocker[0].lower():
                print(f"  -> Consider increasing max_spread_pct by 10-20%")
            elif "regime" in top_blocker[0].lower():
                print(f"  -> Consider allowing additional regimes (e.g., low_vol_chop)")
            elif "tier" in top_blocker[0].lower() or "confidence" in top_blocker[0].lower():
                print(f"  -> Consider allowing MEDIUM tier in more conditions")
            elif "cooldown" in top_blocker[0].lower() or "rate" in top_blocker[0].lower():
                print(f"  -> Consider reducing rate limit cooldowns")

        # Find blockers that account for 80%
        cumulative = calculate_cumulative_pct(dict(combined_reasons), total_blocks)
        blockers_for_80 = [r for r, _, _, c in cumulative if c <= 80 or (c > 80 and c - _ < 80)]
        if blockers_for_80:
            print(f"\n  Blockers causing 80% of rejections ({len(blockers_for_80)} types):")
            for blocker in blockers_for_80[:5]:
                print(f"    - {blocker}")

    print()
    print("=" * 70)

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
