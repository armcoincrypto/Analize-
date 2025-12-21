#!/usr/bin/env python3
"""
Regime Profitability Matrix Analysis
=====================================
Run this after collecting 200+ trades to determine which regimes to trade.

Usage: python regime_analysis.py
"""

import sqlite3
from datetime import datetime, timedelta
from collections import defaultdict


def analyze_regime_profitability(db_path: str = "hft_trades.db"):
    """Build the Regime Profitability Matrix."""

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    print("=" * 70)
    print("REGIME PROFITABILITY MATRIX")
    print("=" * 70)

    # Get total trade count
    total_trades = conn.execute("SELECT COUNT(*) FROM trades WHERE exit_time IS NOT NULL").fetchone()[0]
    print(f"\nTotal completed trades: {total_trades}")

    if total_trades < 50:
        print(f"\n⚠️  WARNING: Only {total_trades} trades. Need 200+ for statistical significance.")
        print("    Continue running the bot to collect more data.\n")

    # Get trades with their entry regime
    cursor = conn.execute("""
        SELECT
            t.trade_id,
            t.symbol,
            t.side,
            t.pnl_pct,
            t.exit_reason,
            t.hold_time_sec,
            mr.regime,
            mr.regime_confidence
        FROM trades t
        LEFT JOIN market_regime mr ON t.symbol = mr.symbol
            AND mr.timestamp <= t.entry_time
            AND mr.timestamp > t.entry_time - 60000
        WHERE t.exit_time IS NOT NULL
        ORDER BY t.entry_time DESC
    """)

    # Group by regime
    regime_stats = defaultdict(lambda: {
        "trades": 0,
        "wins": 0,
        "total_pnl": 0.0,
        "pnl_list": [],
        "hold_times": [],
        "symbols": defaultdict(int)
    })

    for row in cursor:
        regime = row["regime"] or "unknown"
        pnl = row["pnl_pct"] or 0

        regime_stats[regime]["trades"] += 1
        regime_stats[regime]["total_pnl"] += pnl
        regime_stats[regime]["pnl_list"].append(pnl)
        regime_stats[regime]["hold_times"].append(row["hold_time_sec"] or 0)
        regime_stats[regime]["symbols"][row["symbol"]] += 1

        if pnl > 0:
            regime_stats[regime]["wins"] += 1

    # Get edge validation stats by regime
    edge_by_regime = defaultdict(lambda: {"real": 0, "fake": 0})

    edge_cursor = conn.execute("""
        SELECT
            te.symbol,
            te.edge_duration_sec,
            te.final_pnl_pct,
            mr.regime
        FROM trade_edge te
        LEFT JOIN market_regime mr ON te.symbol = mr.symbol
            AND mr.timestamp <= te.timestamp
            AND mr.timestamp > te.timestamp - 60000
    """)

    # Calculate median edge duration for real/fake classification
    edge_data = list(edge_cursor)
    if edge_data:
        durations = [e["edge_duration_sec"] for e in edge_data if e["edge_duration_sec"]]
        if durations:
            median_duration = sorted(durations)[len(durations) // 2]

            for edge in edge_data:
                regime = edge["regime"] or "unknown"
                if edge["edge_duration_sec"] and edge["edge_duration_sec"] > median_duration:
                    edge_by_regime[regime]["real"] += 1
                else:
                    edge_by_regime[regime]["fake"] += 1

    # Print regime matrix
    print("\n" + "=" * 70)
    print(f"{'Regime':<20} {'Trades':>8} {'Avg PnL':>10} {'Win Rate':>10} {'Real Edge':>12}")
    print("=" * 70)

    # Sort by number of trades
    sorted_regimes = sorted(regime_stats.items(), key=lambda x: x[1]["trades"], reverse=True)

    recommendations = []

    for regime, stats in sorted_regimes:
        trades = stats["trades"]
        avg_pnl = stats["total_pnl"] / trades if trades > 0 else 0
        win_rate = (stats["wins"] / trades * 100) if trades > 0 else 0

        edge_stats = edge_by_regime[regime]
        total_edge = edge_stats["real"] + edge_stats["fake"]
        real_edge_pct = (edge_stats["real"] / total_edge * 100) if total_edge > 0 else 0

        # Color coding for recommendations
        pnl_str = f"{avg_pnl:+.4f}%"
        wr_str = f"{win_rate:.1f}%"
        edge_str = f"{real_edge_pct:.1f}%"

        print(f"{regime:<20} {trades:>8} {pnl_str:>10} {wr_str:>10} {edge_str:>12}")

        # Build recommendation
        if trades >= 20:  # Enough data to judge
            if avg_pnl > 0 and win_rate > 40:
                recommendations.append((regime, "TRADE", avg_pnl, win_rate))
            elif avg_pnl < -0.03 or win_rate < 30:
                recommendations.append((regime, "DISABLE", avg_pnl, win_rate))
            else:
                recommendations.append((regime, "MONITOR", avg_pnl, win_rate))
        else:
            recommendations.append((regime, "NEED DATA", avg_pnl, win_rate))

    # Print recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    for regime, action, pnl, wr in recommendations:
        if action == "TRADE":
            print(f"✅ {regime}: TRADE - Profitable ({pnl:+.4f}%, {wr:.1f}% win rate)")
        elif action == "DISABLE":
            print(f"❌ {regime}: DISABLE - Losing regime ({pnl:+.4f}%, {wr:.1f}% win rate)")
        elif action == "MONITOR":
            print(f"⚠️  {regime}: MONITOR - Borderline ({pnl:+.4f}%, {wr:.1f}% win rate)")
        else:
            print(f"📊 {regime}: NEED MORE DATA (only {regime_stats[regime]['trades']} trades)")

    # Causality analysis
    print("\n" + "=" * 70)
    print("TRADE CAUSALITY ANALYSIS")
    print("=" * 70)

    causality_cursor = conn.execute("""
        SELECT
            primary_trigger,
            COUNT(*) as count,
            AVG(trigger_strength) as avg_strength
        FROM trade_causality
        GROUP BY primary_trigger
        ORDER BY count DESC
    """)

    print(f"\n{'Trigger':<30} {'Count':>8} {'Avg Strength':>12}")
    print("-" * 50)

    for row in causality_cursor:
        print(f"{row['primary_trigger']:<30} {row['count']:>8} {row['avg_strength']:>12.4f}")

    # Blocked signals analysis
    print("\n" + "=" * 70)
    print("NO-TRADE ZONE EFFECTIVENESS")
    print("=" * 70)

    blocked_cursor = conn.execute("""
        SELECT
            block_reason,
            COUNT(*) as blocked_count,
            regime
        FROM blocked_signals
        GROUP BY block_reason, regime
        ORDER BY blocked_count DESC
        LIMIT 15
    """)

    print(f"\n{'Block Reason':<20} {'Regime':<20} {'Blocked':>10}")
    print("-" * 50)

    for row in blocked_cursor:
        print(f"{row['block_reason']:<20} {row['regime'] or 'unknown':<20} {row['blocked_count']:>10}")

    # Adaptive sizing performance
    print("\n" + "=" * 70)
    print("ADAPTIVE SIZING PERFORMANCE")
    print("=" * 70)

    sizing_cursor = conn.execute("""
        SELECT
            confidence_tier,
            COUNT(*) as trades,
            AVG(confidence_score) as avg_score,
            AVG(size_multiplier) as avg_mult
        FROM position_sizing
        GROUP BY confidence_tier
        ORDER BY avg_score DESC
    """)

    # Get PnL by tier from trades table
    print(f"\n{'Tier':<12} {'Trades':>8} {'Avg Score':>10} {'Multiplier':>12}")
    print("-" * 45)

    for row in sizing_cursor:
        print(f"{row['confidence_tier']:<12} {row['trades']:>8} {row['avg_score']:>10.1f} {row['avg_mult']:>12.2f}x")

    # Summary
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)

    if total_trades < 200:
        print(f"""
📊 Current progress: {total_trades}/200 trades ({total_trades/200*100:.0f}%)

Keep running the bot to collect more data:
- Target: 200-300 trades minimum
- Across multiple market conditions
- Over multiple trading sessions

Once you have enough data, run this script again for actionable recommendations.
""")
    else:
        print(f"""
✅ You have {total_trades} trades - enough for statistical analysis!

Based on the data above:
1. DISABLE regimes marked with ❌
2. TRADE only regimes marked with ✅
3. MONITOR borderline regimes ⚠️

To implement regime filtering, edit hft_system/config.py:
  - Add losing regimes to AVOID_REGIMES list
  - Keep only profitable regimes enabled
""")

    conn.close()


if __name__ == "__main__":
    analyze_regime_profitability()
