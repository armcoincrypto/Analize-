#!/usr/bin/env python3
"""
Post-Deploy Analysis Script
===========================
Analyzes trade data to find entry conditions that predict winning exits.

Objective: Find which conditions predict micro_profit or take_profit trades.

Usage:
    python post_deploy_analysis.py [--since TIMESTAMP_MS]
"""

import sqlite3
import argparse
from datetime import datetime
from collections import defaultdict

DB_PATH = "hft_trades.db"


def get_connection():
    """Get database connection."""
    return sqlite3.connect(DB_PATH)


def analyze_exit_by_feature(conn, feature_query: str, feature_name: str,
                            buckets: list, since_ms: int = 0) -> dict:
    """
    Analyze P(exit_type | feature bucket) and expected value per bucket.

    Returns buckets with:
    - n: sample count
    - micro_profit_rate: P(micro_profit)
    - take_profit_rate: P(take_profit)
    - winner_rate: P(micro_profit OR take_profit)
    - avg_pnl: expected value
    - total_pnl: sum of pnl
    """
    cursor = conn.cursor()

    results = {}
    for bucket_name, bucket_condition in buckets:
        query = f"""
        SELECT
            COUNT(*) as n,
            SUM(CASE WHEN t.exit_reason = 'micro_profit' THEN 1 ELSE 0 END) as micro_count,
            SUM(CASE WHEN t.exit_reason = 'take_profit' THEN 1 ELSE 0 END) as tp_count,
            ROUND(AVG(t.pnl_pct), 5) as avg_pnl,
            ROUND(SUM(t.pnl_pct), 5) as total_pnl
        FROM trades t
        LEFT JOIN trade_quality_metrics tqm ON t.id = tqm.trade_id
        LEFT JOIN trade_causality tc ON t.id = tc.trade_id
        LEFT JOIN position_sizing ps ON t.id = ps.trade_id
        WHERE t.entry_time > ?
        AND t.status = 'closed'
        AND {bucket_condition}
        """

        try:
            cursor.execute(query, (since_ms,))
            row = cursor.fetchone()

            if row and row[0] and row[0] >= 1:
                n = row[0]
                micro = row[1] or 0
                tp = row[2] or 0
                results[bucket_name] = {
                    "n": n,
                    "micro_profit_rate": round(micro / n * 100, 1),
                    "take_profit_rate": round(tp / n * 100, 1),
                    "winner_rate": round((micro + tp) / n * 100, 1),
                    "avg_pnl": row[3] or 0,
                    "total_pnl": row[4] or 0
                }
        except Exception as e:
            print(f"  Error with bucket {bucket_name}: {e}")

    return results


def analyze_by_regime(conn, since_ms: int = 0) -> dict:
    """Analyze performance by regime at entry."""
    cursor = conn.cursor()

    query = """
    SELECT
        regime_at_entry,
        COUNT(*) as n,
        SUM(CASE WHEN exit_reason = 'micro_profit' THEN 1 ELSE 0 END) as micro,
        SUM(CASE WHEN exit_reason = 'take_profit' THEN 1 ELSE 0 END) as tp,
        ROUND(AVG(pnl_pct), 5) as avg_pnl,
        ROUND(SUM(pnl_pct), 5) as total_pnl
    FROM trades
    WHERE entry_time > ?
    AND status = 'closed'
    AND regime_at_entry IS NOT NULL
    GROUP BY regime_at_entry
    ORDER BY n DESC
    """

    cursor.execute(query, (since_ms,))

    results = {}
    for row in cursor.fetchall():
        if row[0] and row[1] >= 1:
            n = row[1]
            results[row[0]] = {
                "n": n,
                "micro_profit_rate": round((row[2] or 0) / n * 100, 1),
                "take_profit_rate": round((row[3] or 0) / n * 100, 1),
                "winner_rate": round(((row[2] or 0) + (row[3] or 0)) / n * 100, 1),
                "avg_pnl": row[4] or 0,
                "total_pnl": row[5] or 0
            }

    return results


def analyze_by_confidence_tier(conn, since_ms: int = 0) -> dict:
    """Analyze performance by confidence tier."""
    cursor = conn.cursor()

    query = """
    SELECT
        ps.confidence_tier,
        COUNT(*) as n,
        SUM(CASE WHEN t.exit_reason = 'micro_profit' THEN 1 ELSE 0 END) as micro,
        SUM(CASE WHEN t.exit_reason = 'take_profit' THEN 1 ELSE 0 END) as tp,
        ROUND(AVG(t.pnl_pct), 5) as avg_pnl,
        ROUND(SUM(t.pnl_pct), 5) as total_pnl
    FROM trades t
    JOIN position_sizing ps ON t.id = ps.trade_id
    WHERE t.entry_time > ?
    AND t.status = 'closed'
    GROUP BY ps.confidence_tier
    ORDER BY n DESC
    """

    cursor.execute(query, (since_ms,))

    results = {}
    for row in cursor.fetchall():
        if row[0] and row[1] >= 1:
            n = row[1]
            results[row[0]] = {
                "n": n,
                "micro_profit_rate": round((row[2] or 0) / n * 100, 1),
                "take_profit_rate": round((row[3] or 0) / n * 100, 1),
                "winner_rate": round(((row[2] or 0) + (row[3] or 0)) / n * 100, 1),
                "avg_pnl": row[4] or 0,
                "total_pnl": row[5] or 0
            }

    return results


def analyze_by_primary_cause(conn, since_ms: int = 0) -> dict:
    """Analyze performance by primary cause from causality table."""
    cursor = conn.cursor()

    query = """
    SELECT
        tc.primary_cause,
        COUNT(*) as n,
        SUM(CASE WHEN t.exit_reason = 'micro_profit' THEN 1 ELSE 0 END) as micro,
        SUM(CASE WHEN t.exit_reason = 'take_profit' THEN 1 ELSE 0 END) as tp,
        ROUND(AVG(t.pnl_pct), 5) as avg_pnl,
        ROUND(SUM(t.pnl_pct), 5) as total_pnl
    FROM trades t
    JOIN trade_causality tc ON t.id = tc.trade_id
    WHERE t.entry_time > ?
    AND t.status = 'closed'
    AND tc.primary_cause IS NOT NULL
    GROUP BY tc.primary_cause
    ORDER BY n DESC
    """

    cursor.execute(query, (since_ms,))

    results = {}
    for row in cursor.fetchall():
        if row[0] and row[1] >= 1:
            n = row[1]
            results[row[0]] = {
                "n": n,
                "micro_profit_rate": round((row[2] or 0) / n * 100, 1),
                "take_profit_rate": round((row[3] or 0) / n * 100, 1),
                "winner_rate": round(((row[2] or 0) + (row[3] or 0)) / n * 100, 1),
                "avg_pnl": row[4] or 0,
                "total_pnl": row[5] or 0
            }

    return results


def analyze_by_imbalance(conn, since_ms: int = 0) -> dict:
    """Analyze performance by orderbook imbalance buckets."""
    cursor = conn.cursor()

    # Define imbalance buckets
    buckets = [
        ("imb_0.50-0.55", "tqm.imbalance_at_entry >= 0.50 AND tqm.imbalance_at_entry < 0.55"),
        ("imb_0.55-0.60", "tqm.imbalance_at_entry >= 0.55 AND tqm.imbalance_at_entry < 0.60"),
        ("imb_0.60-0.65", "tqm.imbalance_at_entry >= 0.60 AND tqm.imbalance_at_entry < 0.65"),
        ("imb_0.65-0.70", "tqm.imbalance_at_entry >= 0.65 AND tqm.imbalance_at_entry < 0.70"),
        ("imb_0.70-0.75", "tqm.imbalance_at_entry >= 0.70 AND tqm.imbalance_at_entry < 0.75"),
        ("imb_0.75+", "tqm.imbalance_at_entry >= 0.75"),
    ]

    return analyze_exit_by_feature(conn, "", "imbalance", buckets, since_ms)


def analyze_by_spread(conn, since_ms: int = 0) -> dict:
    """Analyze performance by spread buckets."""
    buckets = [
        ("spread_<0.01", "tqm.spread_at_entry < 0.01"),
        ("spread_0.01-0.02", "tqm.spread_at_entry >= 0.01 AND tqm.spread_at_entry < 0.02"),
        ("spread_0.02-0.03", "tqm.spread_at_entry >= 0.02 AND tqm.spread_at_entry < 0.03"),
        ("spread_0.03-0.05", "tqm.spread_at_entry >= 0.03 AND tqm.spread_at_entry < 0.05"),
        ("spread_0.05+", "tqm.spread_at_entry >= 0.05"),
    ]

    return analyze_exit_by_feature(conn, "", "spread", buckets, since_ms)


def analyze_by_symbol(conn, since_ms: int = 0) -> dict:
    """Analyze performance by symbol."""
    cursor = conn.cursor()

    query = """
    SELECT
        symbol,
        COUNT(*) as n,
        SUM(CASE WHEN exit_reason = 'micro_profit' THEN 1 ELSE 0 END) as micro,
        SUM(CASE WHEN exit_reason = 'take_profit' THEN 1 ELSE 0 END) as tp,
        ROUND(AVG(pnl_pct), 5) as avg_pnl,
        ROUND(SUM(pnl_pct), 5) as total_pnl
    FROM trades
    WHERE entry_time > ?
    AND status = 'closed'
    GROUP BY symbol
    ORDER BY n DESC
    """

    cursor.execute(query, (since_ms,))

    results = {}
    for row in cursor.fetchall():
        if row[0] and row[1] >= 1:
            n = row[1]
            results[row[0]] = {
                "n": n,
                "micro_profit_rate": round((row[2] or 0) / n * 100, 1),
                "take_profit_rate": round((row[3] or 0) / n * 100, 1),
                "winner_rate": round(((row[2] or 0) + (row[3] or 0)) / n * 100, 1),
                "avg_pnl": row[4] or 0,
                "total_pnl": row[5] or 0
            }

    return results


def find_best_buckets(all_analyses: dict, min_n: int = 10) -> list:
    """
    Find top buckets by expected value (avg_pnl) with min sample size.

    Returns list of (category, bucket_name, stats) sorted by avg_pnl descending.
    """
    all_buckets = []

    for category, buckets in all_analyses.items():
        for bucket_name, stats in buckets.items():
            if stats["n"] >= min_n:
                all_buckets.append((category, bucket_name, stats))

    # Sort by avg_pnl descending
    all_buckets.sort(key=lambda x: x[2]["avg_pnl"], reverse=True)

    return all_buckets


def print_section(title: str, data: dict):
    """Print a formatted section."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

    if not data:
        print("  No data available")
        return

    # Header
    print(f"  {'Bucket':<25} {'N':>6} {'Win%':>6} {'Micro%':>7} {'TP%':>5} {'AvgPnL':>10} {'TotalPnL':>10}")
    print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*7} {'-'*5} {'-'*10} {'-'*10}")

    # Sort by avg_pnl descending
    sorted_data = sorted(data.items(), key=lambda x: x[1]["avg_pnl"], reverse=True)

    for name, stats in sorted_data:
        print(f"  {name:<25} {stats['n']:>6} {stats['winner_rate']:>5.1f}% "
              f"{stats['micro_profit_rate']:>6.1f}% {stats['take_profit_rate']:>4.1f}% "
              f"{stats['avg_pnl']:>9.4f}% {stats['total_pnl']:>9.4f}%")


def main():
    parser = argparse.ArgumentParser(description="Post-deploy trade analysis")
    parser.add_argument("--since", type=int, default=0,
                        help="Analyze trades since this timestamp (ms)")
    parser.add_argument("--min-n", type=int, default=10,
                        help="Minimum sample size for bucket recommendations")
    args = parser.parse_args()

    print(f"\n{'#'*60}")
    print(f" POST-DEPLOY TRADE ANALYSIS")
    print(f" Analyzing trades since: {args.since}")
    if args.since > 0:
        dt = datetime.utcfromtimestamp(args.since / 1000)
        print(f" ({dt.strftime('%Y-%m-%d %H:%M:%S')} UTC)")
    print(f"{'#'*60}")

    conn = get_connection()

    # Run all analyses
    analyses = {}

    print("\n[1/6] Analyzing by REGIME...")
    analyses["regime"] = analyze_by_regime(conn, args.since)
    print_section("REGIME AT ENTRY", analyses["regime"])

    print("\n[2/6] Analyzing by CONFIDENCE TIER...")
    analyses["confidence"] = analyze_by_confidence_tier(conn, args.since)
    print_section("CONFIDENCE TIER", analyses["confidence"])

    print("\n[3/6] Analyzing by PRIMARY CAUSE...")
    analyses["cause"] = analyze_by_primary_cause(conn, args.since)
    print_section("PRIMARY CAUSE (Causality)", analyses["cause"])

    print("\n[4/6] Analyzing by IMBALANCE...")
    analyses["imbalance"] = analyze_by_imbalance(conn, args.since)
    print_section("ORDERBOOK IMBALANCE AT ENTRY", analyses["imbalance"])

    print("\n[5/6] Analyzing by SPREAD...")
    analyses["spread"] = analyze_by_spread(conn, args.since)
    print_section("SPREAD AT ENTRY", analyses["spread"])

    print("\n[6/6] Analyzing by SYMBOL...")
    analyses["symbol"] = analyze_by_symbol(conn, args.since)
    print_section("SYMBOL", analyses["symbol"])

    # Find best buckets
    print(f"\n{'='*60}")
    print(f" TOP 10 BEST BUCKETS (by Avg PnL, min n={args.min_n})")
    print(f"{'='*60}")

    best = find_best_buckets(analyses, min_n=args.min_n)

    if not best:
        print("  No buckets with sufficient sample size")
    else:
        print(f"  {'Rank':<5} {'Category':<12} {'Bucket':<25} {'N':>6} {'Win%':>6} {'AvgPnL':>10}")
        print(f"  {'-'*5} {'-'*12} {'-'*25} {'-'*6} {'-'*6} {'-'*10}")

        for i, (category, bucket, stats) in enumerate(best[:10], 1):
            print(f"  {i:<5} {category:<12} {bucket:<25} {stats['n']:>6} "
                  f"{stats['winner_rate']:>5.1f}% {stats['avg_pnl']:>9.4f}%")

    # Find worst buckets (to avoid)
    print(f"\n{'='*60}")
    print(f" TOP 10 WORST BUCKETS (to AVOID, min n={args.min_n})")
    print(f"{'='*60}")

    worst = list(reversed(best[-10:])) if len(best) >= 10 else list(reversed(best))

    if not worst:
        print("  No buckets with sufficient sample size")
    else:
        print(f"  {'Rank':<5} {'Category':<12} {'Bucket':<25} {'N':>6} {'Win%':>6} {'AvgPnL':>10}")
        print(f"  {'-'*5} {'-'*12} {'-'*25} {'-'*6} {'-'*6} {'-'*10}")

        for i, (category, bucket, stats) in enumerate(worst[:10], 1):
            print(f"  {i:<5} {category:<12} {bucket:<25} {stats['n']:>6} "
                  f"{stats['winner_rate']:>5.1f}% {stats['avg_pnl']:>9.4f}%")

    # Summary recommendations
    print(f"\n{'='*60}")
    print(f" RECOMMENDED GATE SETTINGS (based on data)")
    print(f"{'='*60}")

    # Find best regime
    if analyses["regime"]:
        best_regime = max(analyses["regime"].items(),
                         key=lambda x: x[1]["avg_pnl"] if x[1]["n"] >= 5 else -999)
        worst_regime = min(analyses["regime"].items(),
                          key=lambda x: x[1]["avg_pnl"] if x[1]["n"] >= 5 else 999)
        print(f"\n  REGIMES:")
        print(f"    Best:  {best_regime[0]} (avg={best_regime[1]['avg_pnl']:.4f}%, n={best_regime[1]['n']})")
        print(f"    Worst: {worst_regime[0]} (avg={worst_regime[1]['avg_pnl']:.4f}%, n={worst_regime[1]['n']})")

    # Find best confidence tier
    if analyses["confidence"]:
        best_conf = max(analyses["confidence"].items(),
                       key=lambda x: x[1]["avg_pnl"] if x[1]["n"] >= 5 else -999)
        print(f"\n  CONFIDENCE:")
        print(f"    Best tier: {best_conf[0]} (avg={best_conf[1]['avg_pnl']:.4f}%, n={best_conf[1]['n']})")

    conn.close()

    print(f"\n{'#'*60}")
    print(f" Analysis complete. Use findings to configure WINNER_GATE.")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
