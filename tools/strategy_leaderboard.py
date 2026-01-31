#!/usr/bin/env python3
"""
Strategy Leaderboard Tool
=========================
Analyzes trades grouped by experiment_tag to find winning strategy configurations.

Groups trades by experiment_tag + symbol + execution_mode + exit_reason and outputs:
- Trade count
- Win rate
- Avg/Median PnL after costs
- p5 MAE (worst drawdown at 5th percentile)
- p95 MFE (best profit potential at 95th percentile)

Usage:
    python -m tools.strategy_leaderboard --days 7 --min-trades 5
    python -m tools.strategy_leaderboard --days 30 --min-trades 10 --group-by pocket
    python -m tools.strategy_leaderboard --days 14 --sort winrate --top 20
"""

import argparse
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional


def percentile(data: List[float], p: float) -> float:
    """Calculate percentile of data."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_data) else f
    return sorted_data[f] + (sorted_data[c] - sorted_data[f]) * (k - f)


def get_db_path() -> str:
    """Get database path."""
    return "hft_trades.db"


def fetch_trades(db_path: str, days: int) -> List[Dict[str, Any]]:
    """Fetch closed trades from database within the specified time window."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Calculate cutoff timestamp
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    cutoff_ms = int(cutoff.timestamp() * 1000)

    # Check which optional columns exist
    cursor.execute("PRAGMA table_info(trades)")
    columns = {row[1] for row in cursor.fetchall()}
    has_experiment_tag = "experiment_tag" in columns
    has_regime_at_entry = "regime_at_entry" in columns

    # Build SELECT clause dynamically based on available columns
    base_cols = """trade_id, symbol, side, entry_price, exit_price,
                   entry_time, exit_time, pnl, pnl_pct, hold_time_sec,
                   exit_reason, mfe, mae, pnl_after_costs_pct, pocket_id,
                   execution_mode"""

    experiment_tag_col = "experiment_tag" if has_experiment_tag else "NULL as experiment_tag"
    regime_col = "regime_at_entry" if has_regime_at_entry else "NULL as regime_at_entry"

    query = f"""
        SELECT
            {base_cols},
            {experiment_tag_col},
            {regime_col}
        FROM trades
        WHERE status = 'closed'
        AND entry_time >= ?
        AND COALESCE(exit_reason, '') != 'force_paper_trade'
    """

    cursor.execute(query, (cutoff_ms,))
    trades = [dict(row) for row in cursor.fetchall()]
    conn.close()

    return trades


def build_grouping_key(trade: Dict[str, Any], group_by: str) -> str:
    """Build grouping key based on grouping strategy."""
    experiment_tag = trade.get("experiment_tag") or ""
    symbol = trade.get("symbol", "?")
    execution_mode = trade.get("execution_mode", "taker")
    exit_reason = trade.get("exit_reason", "?")
    pocket_id = trade.get("pocket_id", "?")
    regime = trade.get("regime_at_entry", "?")

    # Parse experiment_tag if available
    # Format: "symbol|regime|pocket|tier|probe_relaxed|mode"
    tag_parts = experiment_tag.split("|") if experiment_tag else []

    if group_by == "full":
        # Group by full experiment_tag + exit_reason
        if experiment_tag:
            return f"{experiment_tag}|{exit_reason}"
        else:
            return f"{symbol}|{regime}|{pocket_id}|?|none|{execution_mode}|{exit_reason}"

    elif group_by == "tag":
        # Group by experiment_tag only (ignore exit_reason)
        if experiment_tag:
            return experiment_tag
        else:
            return f"{symbol}|{regime}|{pocket_id}|?|none|{execution_mode}"

    elif group_by == "pocket":
        # Group by symbol + pocket
        pocket = tag_parts[2] if len(tag_parts) > 2 else pocket_id
        return f"{symbol}|pocket_{pocket}"

    elif group_by == "regime":
        # Group by symbol + regime
        r = tag_parts[1] if len(tag_parts) > 1 else regime
        return f"{symbol}|{r}"

    elif group_by == "tier":
        # Group by symbol + tier
        tier = tag_parts[3] if len(tag_parts) > 3 else "?"
        return f"{symbol}|tier_{tier}"

    elif group_by == "probe":
        # Group by probe_relaxed flags
        probe_flags = tag_parts[4] if len(tag_parts) > 4 else "none"
        return f"{symbol}|probe_{probe_flags}"

    elif group_by == "mode":
        # Group by execution mode
        mode = tag_parts[5] if len(tag_parts) > 5 else execution_mode
        return f"{symbol}|{mode}"

    elif group_by == "symbol":
        # Group by symbol only
        return symbol

    elif group_by == "exit":
        # Group by exit_reason only
        return exit_reason

    else:
        # Default: full grouping
        return f"{experiment_tag or symbol}|{exit_reason}"


def analyze_groups(trades: List[Dict[str, Any]], group_by: str,
                   min_trades: int, sort_by: str) -> List[Dict[str, Any]]:
    """Group trades and calculate statistics."""
    groups: Dict[str, List[Dict[str, Any]]] = {}

    # Group trades
    for trade in trades:
        key = build_grouping_key(trade, group_by)
        if key not in groups:
            groups[key] = []
        groups[key].append(trade)

    # Calculate statistics for each group
    results = []
    for key, group_trades in groups.items():
        if len(group_trades) < min_trades:
            continue

        # Extract metrics
        pnls = [t["pnl_after_costs_pct"] or 0 for t in group_trades]
        mfes = [t["mfe"] or 0 for t in group_trades]
        maes = [t["mae"] or 0 for t in group_trades]

        # Calculate wins (positive PnL after costs)
        wins = sum(1 for p in pnls if p > 0)
        win_rate = (wins / len(pnls) * 100) if pnls else 0

        # Calculate PnL statistics
        avg_pnl = sum(pnls) / len(pnls) if pnls else 0
        median_pnl = percentile(pnls, 50) if pnls else 0
        total_pnl = sum(pnls)

        # Calculate MFE/MAE percentiles
        p95_mfe = percentile(mfes, 95) if mfes else 0
        p5_mae = percentile(maes, 5) if maes else 0  # Lower is worse for MAE

        # Extract first trade's experiment_tag for display
        sample_tag = group_trades[0].get("experiment_tag", "")

        results.append({
            "key": key,
            "experiment_tag": sample_tag,
            "trades": len(group_trades),
            "wins": wins,
            "winrate": win_rate,
            "avg_pnl": avg_pnl,
            "median_pnl": median_pnl,
            "total_pnl": total_pnl,
            "p95_mfe": p95_mfe,
            "p5_mae": p5_mae,
        })

    # Sort results
    if sort_by == "trades":
        results.sort(key=lambda x: x["trades"], reverse=True)
    elif sort_by == "winrate":
        results.sort(key=lambda x: x["winrate"], reverse=True)
    elif sort_by == "avg_pnl":
        results.sort(key=lambda x: x["avg_pnl"], reverse=True)
    elif sort_by == "total_pnl":
        results.sort(key=lambda x: x["total_pnl"], reverse=True)
    elif sort_by == "mfe":
        results.sort(key=lambda x: x["p95_mfe"], reverse=True)
    elif sort_by == "mae":
        results.sort(key=lambda x: x["p5_mae"], reverse=False)  # Lower MAE is worse
    else:
        results.sort(key=lambda x: x["trades"], reverse=True)

    return results


def print_leaderboard(results: List[Dict[str, Any]], top: int, group_by: str):
    """Print formatted leaderboard."""
    if not results:
        print("No trades found matching criteria.")
        return

    print()
    print("=" * 120)
    print(f"STRATEGY LEADERBOARD (grouped by: {group_by})")
    print("=" * 120)
    print()
    print(f"{'#':<3} {'Group Key':<50} {'Trades':>7} {'Wins':>6} {'WinRate':>8} "
          f"{'AvgPnL':>8} {'MedPnL':>8} {'TotPnL':>9} {'p95MFE':>8} {'p5MAE':>8}")
    print("-" * 120)

    for i, r in enumerate(results[:top], 1):
        key_display = r["key"][:48] + ".." if len(r["key"]) > 50 else r["key"]

        print(f"{i:<3} {key_display:<50} {r['trades']:>7} {r['wins']:>6} "
              f"{r['winrate']:>7.1f}% {r['avg_pnl']:>7.3f}% {r['median_pnl']:>7.3f}% "
              f"{r['total_pnl']:>8.3f}% {r['p95_mfe']:>7.3f}% {r['p5_mae']:>7.3f}%")

    print("-" * 120)
    print()

    # Summary statistics
    total_trades = sum(r["trades"] for r in results)
    total_wins = sum(r["wins"] for r in results)
    total_pnl = sum(r["total_pnl"] for r in results)
    overall_wr = (total_wins / total_trades * 100) if total_trades else 0

    print(f"SUMMARY: {len(results)} groups | {total_trades} total trades | "
          f"{total_wins} wins ({overall_wr:.1f}%) | Total PnL: {total_pnl:.3f}%")
    print()


def print_detailed_analysis(results: List[Dict[str, Any]], top: int):
    """Print detailed analysis with recommendations."""
    if not results:
        return

    print()
    print("=" * 80)
    print("DETAILED ANALYSIS")
    print("=" * 80)
    print()

    # Find best and worst performers
    best_wr = max(results, key=lambda x: x["winrate"]) if results else None
    best_pnl = max(results, key=lambda x: x["avg_pnl"]) if results else None
    worst_pnl = min(results, key=lambda x: x["avg_pnl"]) if results else None
    most_trades = max(results, key=lambda x: x["trades"]) if results else None

    if best_wr:
        print(f"BEST WIN RATE: {best_wr['key']}")
        print(f"  {best_wr['winrate']:.1f}% WR | {best_wr['trades']} trades | {best_wr['avg_pnl']:.3f}% avg PnL")
        print()

    if best_pnl and best_pnl != best_wr:
        print(f"BEST AVG PNL: {best_pnl['key']}")
        print(f"  {best_pnl['avg_pnl']:.3f}% avg | {best_pnl['trades']} trades | {best_pnl['winrate']:.1f}% WR")
        print()

    if worst_pnl and worst_pnl["avg_pnl"] < 0:
        print(f"WORST PERFORMER: {worst_pnl['key']}")
        print(f"  {worst_pnl['avg_pnl']:.3f}% avg | {worst_pnl['trades']} trades | {worst_pnl['winrate']:.1f}% WR")
        print(f"  RECOMMENDATION: Consider disabling this configuration")
        print()

    if most_trades:
        print(f"MOST DATA: {most_trades['key']}")
        print(f"  {most_trades['trades']} trades | {most_trades['winrate']:.1f}% WR | {most_trades['avg_pnl']:.3f}% avg")
        print()

    # Recommendations based on experiment_tag components
    print("RECOMMENDATIONS:")
    print("-" * 40)

    # Analyze by probe relaxation
    probe_groups = {}
    for r in results:
        tag = r.get("experiment_tag") or ""
        if tag and "|" in tag:
            parts = tag.split("|")
            if len(parts) > 4:
                probe_flag = parts[4]
                if probe_flag not in probe_groups:
                    probe_groups[probe_flag] = {"trades": 0, "total_pnl": 0, "wins": 0}
                probe_groups[probe_flag]["trades"] += r["trades"]
                probe_groups[probe_flag]["total_pnl"] += r["total_pnl"]
                probe_groups[probe_flag]["wins"] += r["wins"]

    if probe_groups:
        print("\nProbe Relaxation Impact:")
        for flag, data in sorted(probe_groups.items(), key=lambda x: x[1]["total_pnl"], reverse=True):
            wr = (data["wins"] / data["trades"] * 100) if data["trades"] else 0
            avg = data["total_pnl"] / data["trades"] if data["trades"] else 0
            status = "KEEP" if avg > 0 else "REVIEW"
            print(f"  {flag}: {data['trades']} trades, {wr:.1f}% WR, {avg:.3f}% avg -> {status}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Strategy Leaderboard - Analyze trading strategy performance")
    parser.add_argument("--days", type=int, default=7, help="Look back period in days (default: 7)")
    parser.add_argument("--min-trades", type=int, default=5, help="Minimum trades per group (default: 5)")
    parser.add_argument("--top", type=int, default=30, help="Show top N results (default: 30)")
    parser.add_argument("--sort", choices=["trades", "winrate", "avg_pnl", "total_pnl", "mfe", "mae"],
                        default="trades", help="Sort by metric (default: trades)")
    parser.add_argument("--group-by", choices=["full", "tag", "pocket", "regime", "tier", "probe", "mode", "symbol", "exit"],
                        default="tag", help="Grouping strategy (default: tag)")
    parser.add_argument("--db", type=str, default=None, help="Database path (default: hft_trades.db)")
    parser.add_argument("--detailed", action="store_true", help="Show detailed analysis")
    parser.add_argument("--csv", action="store_true", help="Output as CSV")

    args = parser.parse_args()

    db_path = args.db or get_db_path()

    print(f"\nStrategy Leaderboard")
    print(f"  Database: {db_path}")
    print(f"  Period: Last {args.days} days")
    print(f"  Min trades: {args.min_trades}")
    print(f"  Group by: {args.group_by}")
    print(f"  Sort by: {args.sort}")

    # Fetch trades
    try:
        trades = fetch_trades(db_path, args.days)
    except sqlite3.OperationalError as e:
        print(f"\nError: Could not read database: {e}")
        sys.exit(1)

    print(f"  Found: {len(trades)} trades")

    if not trades:
        print("\nNo trades found in the specified period.")
        sys.exit(0)

    # Analyze groups
    results = analyze_groups(trades, args.group_by, args.min_trades, args.sort)

    if args.csv:
        # CSV output
        print("\nkey,trades,wins,winrate,avg_pnl,median_pnl,total_pnl,p95_mfe,p5_mae")
        for r in results[:args.top]:
            print(f"{r['key']},{r['trades']},{r['wins']},{r['winrate']:.2f},"
                  f"{r['avg_pnl']:.4f},{r['median_pnl']:.4f},{r['total_pnl']:.4f},"
                  f"{r['p95_mfe']:.4f},{r['p5_mae']:.4f}")
    else:
        # Formatted output
        print_leaderboard(results, args.top, args.group_by)

        if args.detailed:
            print_detailed_analysis(results, args.top)


if __name__ == "__main__":
    main()
