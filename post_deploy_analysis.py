#!/usr/bin/env python3
"""
Post-Deploy Analysis Script v2
==============================
Analyzes trade data to find entry conditions that predict winning exits.

FIXED: Uses 2-second timestamp tolerance to join trades with position_sizing
       and trade_causality tables (timestamps differ by ~15-30ms).

Usage:
    python post_deploy_analysis.py [--since TIMESTAMP_MS]
"""

import sqlite3
import argparse
from datetime import datetime, timezone
from collections import defaultdict

DB_PATH = "hft_trades.db"


def get_connection():
    """Get database connection."""
    return sqlite3.connect(DB_PATH)


def get_match_stats(conn, since_ms: int = 0) -> dict:
    """
    Check how well trades match with position_sizing and trade_causality.
    Uses 2-second timestamp tolerance for matching.
    """
    cursor = conn.cursor()

    # Count trades since deploy
    cursor.execute("SELECT COUNT(*) FROM trades WHERE entry_time > ? AND status = 'closed'", (since_ms,))
    trades_count = cursor.fetchone()[0]

    # Count position_sizing rows since deploy (parse timestamp from trade_id)
    cursor.execute("""
        SELECT COUNT(*) FROM position_sizing
        WHERE CAST(SUBSTR(trade_id, INSTR(trade_id, '_')+1) AS INTEGER) > ?
    """, (since_ms,))
    sizing_count = cursor.fetchone()[0]

    # Count trade_causality rows since deploy
    cursor.execute("""
        SELECT COUNT(*) FROM trade_causality
        WHERE CAST(SUBSTR(trade_id, INSTR(trade_id, '_')+1) AS INTEGER) > ?
    """, (since_ms,))
    causality_count = cursor.fetchone()[0]

    # Count matched trades (using 2s tolerance)
    cursor.execute("""
        WITH ps_parsed AS (
            SELECT
                trade_id,
                SUBSTR(trade_id, 1, INSTR(trade_id, '_')-1) AS ps_symbol,
                CAST(SUBSTR(trade_id, INSTR(trade_id, '_')+1) AS INTEGER) AS ps_ts,
                confidence_tier,
                confidence_score,
                regime,
                orderbook_imbalance,
                spread_pct,
                primary_cause
            FROM position_sizing
            WHERE CAST(SUBSTR(trade_id, INSTR(trade_id, '_')+1) AS INTEGER) > ?
        )
        SELECT COUNT(DISTINCT t.id)
        FROM trades t
        JOIN ps_parsed ps ON ps.ps_symbol = t.symbol
            AND ABS(t.entry_time - ps.ps_ts) <= 2000
        WHERE t.entry_time > ? AND t.status = 'closed'
    """, (since_ms, since_ms))
    matched_count = cursor.fetchone()[0]

    match_rate = (matched_count / trades_count * 100) if trades_count > 0 else 0

    return {
        "trades_since_deploy": trades_count,
        "sizing_rows": sizing_count,
        "causality_rows": causality_count,
        "matched_trades": matched_count,
        "match_rate": match_rate
    }


def build_enriched_trades_view(conn, since_ms: int = 0):
    """
    Create a temporary view that joins trades with position_sizing data
    using 2-second timestamp tolerance.
    """
    cursor = conn.cursor()

    # Drop if exists
    cursor.execute("DROP VIEW IF EXISTS enriched_trades")

    # Create view with proper join
    cursor.execute("""
        CREATE TEMP VIEW enriched_trades AS
        WITH ps_parsed AS (
            SELECT
                trade_id,
                SUBSTR(trade_id, 1, INSTR(trade_id, '_')-1) AS ps_symbol,
                CAST(SUBSTR(trade_id, INSTR(trade_id, '_')+1) AS INTEGER) AS ps_ts,
                confidence_tier,
                confidence_score,
                regime AS ps_regime,
                orderbook_imbalance AS ps_imbalance,
                spread_pct AS ps_spread,
                primary_cause AS ps_cause
            FROM position_sizing
        ),
        tc_parsed AS (
            SELECT
                trade_id,
                SUBSTR(trade_id, 1, INSTR(trade_id, '_')-1) AS tc_symbol,
                CAST(SUBSTR(trade_id, INSTR(trade_id, '_')+1) AS INTEGER) AS tc_ts,
                primary_cause AS tc_cause,
                ob_imbalance_ratio AS tc_imbalance,
                ob_spread_pct AS tc_spread
            FROM trade_causality
        )
        SELECT
            t.id,
            t.symbol,
            t.entry_time,
            t.exit_reason,
            t.pnl_pct,
            t.regime_at_entry,
            ps.confidence_tier,
            ps.confidence_score,
            ps.ps_regime,
            ps.ps_imbalance,
            ps.ps_spread,
            ps.ps_cause,
            tc.tc_cause,
            tc.tc_imbalance,
            tc.tc_spread,
            -- Spread bucket
            CASE
                WHEN COALESCE(ps.ps_spread, tc.tc_spread) < 0.01 THEN 'spread_<0.01'
                WHEN COALESCE(ps.ps_spread, tc.tc_spread) < 0.02 THEN 'spread_0.01-0.02'
                WHEN COALESCE(ps.ps_spread, tc.tc_spread) < 0.03 THEN 'spread_0.02-0.03'
                WHEN COALESCE(ps.ps_spread, tc.tc_spread) < 0.05 THEN 'spread_0.03-0.05'
                ELSE 'spread_0.05+'
            END AS spread_bucket,
            -- Imbalance bucket
            CASE
                WHEN COALESCE(ps.ps_imbalance, tc.tc_imbalance) < 0.55 THEN 'imb_<0.55'
                WHEN COALESCE(ps.ps_imbalance, tc.tc_imbalance) < 0.60 THEN 'imb_0.55-0.60'
                WHEN COALESCE(ps.ps_imbalance, tc.tc_imbalance) < 0.65 THEN 'imb_0.60-0.65'
                WHEN COALESCE(ps.ps_imbalance, tc.tc_imbalance) < 0.70 THEN 'imb_0.65-0.70'
                WHEN COALESCE(ps.ps_imbalance, tc.tc_imbalance) < 0.75 THEN 'imb_0.70-0.75'
                ELSE 'imb_0.75+'
            END AS imbalance_bucket
        FROM trades t
        LEFT JOIN ps_parsed ps ON ps.ps_symbol = t.symbol
            AND ABS(t.entry_time - ps.ps_ts) <= 2000
        LEFT JOIN tc_parsed tc ON tc.tc_symbol = t.symbol
            AND ABS(t.entry_time - tc.tc_ts) <= 2000
        WHERE t.status = 'closed'
    """)

    conn.commit()


def analyze_single_bucket(conn, group_col: str, since_ms: int = 0) -> dict:
    """Analyze performance by a single column/bucket."""
    cursor = conn.cursor()

    query = f"""
        SELECT
            {group_col},
            COUNT(*) as n,
            SUM(CASE WHEN exit_reason = 'micro_profit' THEN 1 ELSE 0 END) as micro,
            SUM(CASE WHEN exit_reason = 'take_profit' THEN 1 ELSE 0 END) as tp,
            ROUND(AVG(pnl_pct), 5) as avg_pnl,
            ROUND(SUM(pnl_pct), 5) as total_pnl
        FROM enriched_trades
        WHERE entry_time > ?
        AND {group_col} IS NOT NULL
        GROUP BY {group_col}
        ORDER BY n DESC
    """

    cursor.execute(query, (since_ms,))

    results = {}
    for row in cursor.fetchall():
        if row[0] and row[1] >= 1:
            n = row[1]
            results[str(row[0])] = {
                "n": n,
                "micro_profit_rate": round((row[2] or 0) / n * 100, 1),
                "take_profit_rate": round((row[3] or 0) / n * 100, 1),
                "winner_rate": round(((row[2] or 0) + (row[3] or 0)) / n * 100, 1),
                "avg_pnl": row[4] or 0,
                "total_pnl": row[5] or 0
            }

    return results


def analyze_combo_bucket(conn, col1: str, col2: str, since_ms: int = 0) -> dict:
    """Analyze performance by combination of two columns."""
    cursor = conn.cursor()

    query = f"""
        SELECT
            {col1} || ' + ' || {col2} as combo,
            COUNT(*) as n,
            SUM(CASE WHEN exit_reason = 'micro_profit' THEN 1 ELSE 0 END) as micro,
            SUM(CASE WHEN exit_reason = 'take_profit' THEN 1 ELSE 0 END) as tp,
            ROUND(AVG(pnl_pct), 5) as avg_pnl,
            ROUND(SUM(pnl_pct), 5) as total_pnl
        FROM enriched_trades
        WHERE entry_time > ?
        AND {col1} IS NOT NULL
        AND {col2} IS NOT NULL
        GROUP BY {col1}, {col2}
        HAVING COUNT(*) >= 5
        ORDER BY avg_pnl DESC
    """

    cursor.execute(query, (since_ms,))

    results = {}
    for row in cursor.fetchall():
        if row[0] and row[1] >= 1:
            n = row[1]
            results[str(row[0])] = {
                "n": n,
                "micro_profit_rate": round((row[2] or 0) / n * 100, 1),
                "take_profit_rate": round((row[3] or 0) / n * 100, 1),
                "winner_rate": round(((row[2] or 0) + (row[3] or 0)) / n * 100, 1),
                "avg_pnl": row[4] or 0,
                "total_pnl": row[5] or 0
            }

    return results


def find_best_buckets(all_analyses: dict, min_n: int = 10) -> list:
    """Find top buckets by expected value with min sample size."""
    all_buckets = []

    for category, buckets in all_analyses.items():
        for bucket_name, stats in buckets.items():
            if stats["n"] >= min_n:
                all_buckets.append((category, bucket_name, stats))

    all_buckets.sort(key=lambda x: x[2]["avg_pnl"], reverse=True)
    return all_buckets


def print_section(title: str, data: dict):
    """Print a formatted section."""
    print(f"\n{'='*70}")
    print(f" {title}")
    print(f"{'='*70}")

    if not data:
        print("  No data available")
        return

    print(f"  {'Bucket':<30} {'N':>5} {'Win%':>6} {'Micro%':>7} {'TP%':>5} {'AvgPnL':>10}")
    print(f"  {'-'*30} {'-'*5} {'-'*6} {'-'*7} {'-'*5} {'-'*10}")

    sorted_data = sorted(data.items(), key=lambda x: x[1]["avg_pnl"], reverse=True)

    for name, stats in sorted_data:
        print(f"  {name:<30} {stats['n']:>5} {stats['winner_rate']:>5.1f}% "
              f"{stats['micro_profit_rate']:>6.1f}% {stats['take_profit_rate']:>4.1f}% "
              f"{stats['avg_pnl']:>9.4f}%")


def main():
    parser = argparse.ArgumentParser(description="Post-deploy trade analysis v2")
    parser.add_argument("--since", type=int, default=0,
                        help="Analyze trades since this timestamp (ms)")
    parser.add_argument("--min-n", type=int, default=10,
                        help="Minimum sample size for recommendations")
    args = parser.parse_args()

    print(f"\n{'#'*70}")
    print(f" POST-DEPLOY TRADE ANALYSIS v2 (Fixed Joins)")
    print(f" Analyzing trades since: {args.since}")
    if args.since > 0:
        dt = datetime.fromtimestamp(args.since / 1000, tz=timezone.utc)
        print(f" ({dt.strftime('%Y-%m-%d %H:%M:%S')} UTC)")
    print(f"{'#'*70}")

    conn = get_connection()

    # Step 1: Check match stats
    print("\n[0/8] Checking join match rates...")
    match_stats = get_match_stats(conn, args.since)

    print(f"\n  MATCH STATISTICS:")
    print(f"  {'-'*40}")
    print(f"  Trades since deploy:     {match_stats['trades_since_deploy']}")
    print(f"  Position sizing rows:    {match_stats['sizing_rows']}")
    print(f"  Trade causality rows:    {match_stats['causality_rows']}")
    print(f"  Matched trades:          {match_stats['matched_trades']}")
    print(f"  Match rate:              {match_stats['match_rate']:.1f}%")

    if match_stats['match_rate'] < 95:
        print(f"\n  *** WARNING: Match rate below 95%! Results may be incomplete. ***")

    # Step 2: Build enriched view
    print("\n[1/8] Building enriched trades view...")
    build_enriched_trades_view(conn, args.since)

    analyses = {}

    # Single bucket analyses
    print("\n[2/8] Analyzing by REGIME...")
    analyses["regime"] = analyze_single_bucket(conn, "COALESCE(ps_regime, regime_at_entry)", args.since)
    print_section("REGIME AT ENTRY", analyses["regime"])

    print("\n[3/8] Analyzing by CONFIDENCE TIER...")
    analyses["confidence"] = analyze_single_bucket(conn, "confidence_tier", args.since)
    print_section("CONFIDENCE TIER", analyses["confidence"])

    print("\n[4/8] Analyzing by PRIMARY CAUSE...")
    analyses["cause"] = analyze_single_bucket(conn, "COALESCE(ps_cause, tc_cause)", args.since)
    print_section("PRIMARY CAUSE", analyses["cause"])

    print("\n[5/8] Analyzing by SPREAD BUCKET...")
    analyses["spread"] = analyze_single_bucket(conn, "spread_bucket", args.since)
    print_section("SPREAD AT ENTRY", analyses["spread"])

    print("\n[6/8] Analyzing by IMBALANCE BUCKET...")
    analyses["imbalance"] = analyze_single_bucket(conn, "imbalance_bucket", args.since)
    print_section("ORDERBOOK IMBALANCE AT ENTRY", analyses["imbalance"])

    print("\n[7/8] Analyzing by SYMBOL...")
    analyses["symbol"] = analyze_single_bucket(conn, "symbol", args.since)
    print_section("SYMBOL", analyses["symbol"])

    # Combo bucket analyses
    print("\n[8/8] Analyzing COMBO BUCKETS...")

    combos = {}
    combos["tier+regime"] = analyze_combo_bucket(conn, "confidence_tier",
                                                  "COALESCE(ps_regime, regime_at_entry)", args.since)
    combos["tier+spread"] = analyze_combo_bucket(conn, "confidence_tier", "spread_bucket", args.since)
    combos["tier+imbalance"] = analyze_combo_bucket(conn, "confidence_tier", "imbalance_bucket", args.since)
    combos["regime+spread"] = analyze_combo_bucket(conn, "COALESCE(ps_regime, regime_at_entry)",
                                                    "spread_bucket", args.since)
    combos["regime+imbalance"] = analyze_combo_bucket(conn, "COALESCE(ps_regime, regime_at_entry)",
                                                       "imbalance_bucket", args.since)

    print_section("TIER + REGIME COMBOS", combos["tier+regime"])
    print_section("TIER + SPREAD COMBOS", combos["tier+spread"])
    print_section("TIER + IMBALANCE COMBOS", combos["tier+imbalance"])
    print_section("REGIME + SPREAD COMBOS", combos["regime+spread"])
    print_section("REGIME + IMBALANCE COMBOS", combos["regime+imbalance"])

    # Merge all for ranking
    all_analyses = {**analyses, **combos}

    # Best buckets
    print(f"\n{'='*70}")
    print(f" TOP 10 BEST BUCKETS (by Avg PnL, min n={args.min_n})")
    print(f"{'='*70}")

    best = find_best_buckets(all_analyses, min_n=args.min_n)

    if not best:
        print("  No buckets with sufficient sample size")
    else:
        print(f"  {'#':<3} {'Category':<15} {'Bucket':<35} {'N':>5} {'Win%':>6} {'AvgPnL':>10}")
        print(f"  {'-'*3} {'-'*15} {'-'*35} {'-'*5} {'-'*6} {'-'*10}")

        for i, (category, bucket, stats) in enumerate(best[:10], 1):
            print(f"  {i:<3} {category:<15} {bucket:<35} {stats['n']:>5} "
                  f"{stats['winner_rate']:>5.1f}% {stats['avg_pnl']:>9.4f}%")

    # Worst buckets
    print(f"\n{'='*70}")
    print(f" TOP 10 WORST BUCKETS (to AVOID, min n={args.min_n})")
    print(f"{'='*70}")

    worst = list(reversed(best[-10:])) if len(best) >= 10 else list(reversed(best))

    if not worst:
        print("  No buckets with sufficient sample size")
    else:
        print(f"  {'#':<3} {'Category':<15} {'Bucket':<35} {'N':>5} {'Win%':>6} {'AvgPnL':>10}")
        print(f"  {'-'*3} {'-'*15} {'-'*35} {'-'*5} {'-'*6} {'-'*10}")

        for i, (category, bucket, stats) in enumerate(worst[:10], 1):
            print(f"  {i:<3} {category:<15} {bucket:<35} {stats['n']:>5} "
                  f"{stats['winner_rate']:>5.1f}% {stats['avg_pnl']:>9.4f}%")

    # Recommendations
    print(f"\n{'='*70}")
    print(f" RECOMMENDED GATE SETTINGS")
    print(f"{'='*70}")

    if analyses.get("regime"):
        sorted_regimes = sorted(analyses["regime"].items(),
                               key=lambda x: x[1]["avg_pnl"], reverse=True)
        print(f"\n  REGIMES (sorted by avg_pnl):")
        for regime, stats in sorted_regimes:
            status = "ALLOW" if stats["avg_pnl"] > -0.04 else "CONSIDER BLOCKING"
            print(f"    {regime:<20} avg={stats['avg_pnl']:>8.4f}% n={stats['n']:>3} -> {status}")

    if analyses.get("confidence"):
        sorted_tiers = sorted(analyses["confidence"].items(),
                             key=lambda x: x[1]["avg_pnl"], reverse=True)
        print(f"\n  CONFIDENCE TIERS (sorted by avg_pnl):")
        for tier, stats in sorted_tiers:
            print(f"    {tier:<20} avg={stats['avg_pnl']:>8.4f}% n={stats['n']:>3} micro_rate={stats['micro_profit_rate']:.1f}%")

    if analyses.get("spread"):
        sorted_spreads = sorted(analyses["spread"].items(),
                               key=lambda x: x[1]["avg_pnl"], reverse=True)
        print(f"\n  SPREAD BUCKETS (sorted by avg_pnl):")
        for spread, stats in sorted_spreads:
            print(f"    {spread:<20} avg={stats['avg_pnl']:>8.4f}% n={stats['n']:>3}")

    if analyses.get("imbalance"):
        sorted_imb = sorted(analyses["imbalance"].items(),
                           key=lambda x: x[1]["avg_pnl"], reverse=True)
        print(f"\n  IMBALANCE BUCKETS (sorted by avg_pnl):")
        for imb, stats in sorted_imb:
            print(f"    {imb:<20} avg={stats['avg_pnl']:>8.4f}% n={stats['n']:>3}")

    # ============================================================
    # GATE COMPLIANCE CHECK
    # ============================================================
    print(f"\n{'='*70}")
    print(f" WINNER GATE COMPLIANCE CHECK")
    print(f"{'='*70}")

    cursor = conn.cursor()

    # Check blocked signals by winner_gate
    cursor.execute("""
        SELECT COUNT(*) FROM blocked_signals
        WHERE block_reason = 'winner_gate'
        AND timestamp > ?
    """, (args.since,))
    gate_blocked = cursor.fetchone()[0]

    # Check trades that passed gate
    cursor.execute("""
        SELECT COUNT(*) FROM trades
        WHERE entry_time > ? AND status = 'closed'
    """, (args.since,))
    trades_passed = cursor.fetchone()[0]

    print(f"\n  Signals blocked by winner_gate: {gate_blocked}")
    print(f"  Trades executed (passed gate):   {trades_passed}")

    # Check for forbidden buckets in executed trades
    print(f"\n  GATE RULE VIOLATIONS (should be 0):")

    # Check mean_reversion trades
    cursor.execute("""
        SELECT COUNT(*) FROM enriched_trades
        WHERE entry_time > ?
        AND (ps_regime = 'mean_reversion' OR regime_at_entry = 'mean_reversion')
    """, (args.since,))
    mean_rev_trades = cursor.fetchone()[0]
    status = "✓ PASS" if mean_rev_trades == 0 else "✗ VIOLATION"
    print(f"    mean_reversion trades: {mean_rev_trades} {status}")

    # Check MEDIUM tier trades
    cursor.execute("""
        SELECT COUNT(*) FROM enriched_trades
        WHERE entry_time > ?
        AND confidence_tier = 'medium'
    """, (args.since,))
    medium_trades = cursor.fetchone()[0]
    status = "✓ PASS" if medium_trades == 0 else "✗ VIOLATION"
    print(f"    MEDIUM tier trades:    {medium_trades} {status}")

    # Check imbalance < 0.75 trades
    cursor.execute("""
        SELECT COUNT(*) FROM enriched_trades
        WHERE entry_time > ?
        AND COALESCE(ps_imbalance, 0) < 0.75
        AND COALESCE(ps_imbalance, 0) > 0
    """, (args.since,))
    low_imb_trades = cursor.fetchone()[0]
    status = "✓ PASS" if low_imb_trades == 0 else "✗ VIOLATION"
    print(f"    imbalance < 0.75:      {low_imb_trades} {status}")

    # Check low_vol_chop trades (blocked in strict mode)
    cursor.execute("""
        SELECT COUNT(*) FROM enriched_trades
        WHERE entry_time > ?
        AND (ps_regime = 'low_vol_chop' OR regime_at_entry = 'low_vol_chop')
    """, (args.since,))
    chop_trades = cursor.fetchone()[0]
    status = "✓ PASS" if chop_trades == 0 else "✗ VIOLATION (strict mode)"
    print(f"    low_vol_chop trades:   {chop_trades} {status}")

    # Summary
    total_violations = mean_rev_trades + medium_trades + low_imb_trades + chop_trades
    if total_violations == 0:
        print(f"\n  ✓ GATE FULLY COMPLIANT - All trades in approved pocket")
    else:
        print(f"\n  ✗ {total_violations} VIOLATIONS - Check gate implementation")

    conn.close()

    print(f"\n{'#'*70}")
    print(f" Analysis complete. Use these findings to configure WINNER_GATE.")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    main()
