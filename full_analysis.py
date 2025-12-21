#!/usr/bin/env python3
"""
FULL TRADING SYSTEM ANALYSIS
============================
Run after collecting 200+ trades.

Answers:
1. Which market regimes are profitable vs unprofitable?
2. In which regimes should trading be completely disabled?
3. Which causal factors dominate REAL edge trades?
4. What patterns appear before fake edges?
5. Do blocked signals save money?
6. Final recommendations

Usage: python full_analysis.py
"""

import sqlite3
from collections import defaultdict
from datetime import datetime


def run_full_analysis(db_path: str = "hft_trades.db"):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Get trade count
    total_trades = conn.execute(
        "SELECT COUNT(*) FROM trades WHERE exit_time IS NOT NULL"
    ).fetchone()[0]

    print("=" * 70)
    print("FULL TRADING SYSTEM ANALYSIS")
    print(f"Total trades: {total_trades}")
    print(f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    if total_trades < 50:
        print(f"\n⚠️  Only {total_trades} trades. Need 200+ for reliable analysis.")
        print("   Continue running the bot to collect more data.\n")

    # ==========================================================================
    # PHASE 2: REGIME PROFITABILITY MATRIX
    # ==========================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: REGIME PROFITABILITY MATRIX")
    print("=" * 70)
    print("\nQuestion: Which regimes are profitable vs unprofitable?\n")

    # Get trades with regime at entry
    regime_trades = conn.execute("""
        SELECT
            t.trade_id,
            t.symbol,
            t.pnl_pct,
            t.exit_reason,
            COALESCE(
                (SELECT regime FROM market_regime mr
                 WHERE mr.symbol = t.symbol
                 AND mr.timestamp <= t.entry_time
                 ORDER BY mr.timestamp DESC LIMIT 1),
                'unknown'
            ) as regime
        FROM trades t
        WHERE t.exit_time IS NOT NULL
    """).fetchall()

    regime_stats = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0, "pnl_list": []})

    for row in regime_trades:
        regime = row["regime"]
        pnl = row["pnl_pct"] or 0

        regime_stats[regime]["trades"] += 1
        regime_stats[regime]["pnl"] += pnl
        regime_stats[regime]["pnl_list"].append(pnl)
        if pnl > 0:
            regime_stats[regime]["wins"] += 1

    print(f"{'Regime':<20} {'Trades':>8} {'Avg PnL':>12} {'Win Rate':>10} {'Verdict':>12}")
    print("-" * 62)

    regime_recommendations = {}

    for regime, stats in sorted(regime_stats.items(), key=lambda x: -x[1]["trades"]):
        trades = stats["trades"]
        avg_pnl = stats["pnl"] / trades if trades else 0
        win_rate = stats["wins"] / trades * 100 if trades else 0

        if trades >= 15:
            if avg_pnl > 0.01:
                verdict = "✅ TRADE"
                regime_recommendations[regime] = "trade"
            elif avg_pnl < -0.02:
                verdict = "❌ DISABLE"
                regime_recommendations[regime] = "disable"
            else:
                verdict = "⚠️ MONITOR"
                regime_recommendations[regime] = "monitor"
        else:
            verdict = "📊 LOW DATA"
            regime_recommendations[regime] = "low_data"

        print(f"{regime:<20} {trades:>8} {avg_pnl:>+11.4f}% {win_rate:>9.1f}% {verdict:>12}")

    # ==========================================================================
    # PHASE 3: CAUSAL EDGE VALIDATION
    # ==========================================================================
    print("\n" + "=" * 70)
    print("PHASE 3: CAUSAL EDGE VALIDATION")
    print("=" * 70)
    print("\nQuestion: Which causal factors dominate REAL edge trades?\n")

    # Get edge data with causality
    edge_causality = conn.execute("""
        SELECT
            te.trade_id,
            te.edge_duration_sec,
            te.final_pnl_pct,
            te.max_favorable_pct,
            te.entry_imbalance,
            te.entry_delta,
            tc.primary_trigger,
            tc.trigger_strength,
            tc.orderbook_state,
            tc.delta_direction
        FROM trade_edge te
        LEFT JOIN trade_causality tc ON te.trade_id = tc.trade_id
    """).fetchall()

    if edge_causality:
        # Calculate median for real/fake split
        durations = [e["edge_duration_sec"] for e in edge_causality if e["edge_duration_sec"]]
        median_duration = sorted(durations)[len(durations) // 2] if durations else 4.0

        real_edge_causes = defaultdict(lambda: {"count": 0, "total_pnl": 0})
        fake_edge_causes = defaultdict(lambda: {"count": 0, "total_pnl": 0})

        real_edge_patterns = {"high_imbalance": 0, "stable_delta": 0, "total": 0}
        fake_edge_patterns = {"delta_flip": 0, "imbalance_decay": 0, "total": 0}

        for edge in edge_causality:
            is_real = edge["edge_duration_sec"] and edge["edge_duration_sec"] > median_duration
            trigger = edge["primary_trigger"] or "unknown"
            pnl = edge["final_pnl_pct"] or 0

            if is_real:
                real_edge_causes[trigger]["count"] += 1
                real_edge_causes[trigger]["total_pnl"] += pnl
                real_edge_patterns["total"] += 1

                # Pattern detection
                if edge["entry_imbalance"] and abs(edge["entry_imbalance"] - 0.5) > 0.2:
                    real_edge_patterns["high_imbalance"] += 1
                if edge["delta_direction"] in ["buy", "sell"]:
                    real_edge_patterns["stable_delta"] += 1
            else:
                fake_edge_causes[trigger]["count"] += 1
                fake_edge_causes[trigger]["total_pnl"] += pnl
                fake_edge_patterns["total"] += 1

        print("REAL EDGE - Dominant Causes:")
        print(f"{'Trigger':<30} {'Count':>8} {'Avg PnL':>12}")
        print("-" * 50)

        for trigger, stats in sorted(real_edge_causes.items(), key=lambda x: -x[1]["count"]):
            avg_pnl = stats["total_pnl"] / stats["count"] if stats["count"] else 0
            print(f"{trigger:<30} {stats['count']:>8} {avg_pnl:>+11.4f}%")

        if real_edge_patterns["total"] > 0:
            imb_pct = real_edge_patterns["high_imbalance"] / real_edge_patterns["total"] * 100
            delta_pct = real_edge_patterns["stable_delta"] / real_edge_patterns["total"] * 100
            print(f"\nReal Edge Pattern: {imb_pct:.0f}% high imbalance, {delta_pct:.0f}% stable delta")

        print("\nFAKE EDGE - Warning Patterns:")
        print(f"{'Trigger':<30} {'Count':>8} {'Avg PnL':>12}")
        print("-" * 50)

        for trigger, stats in sorted(fake_edge_causes.items(), key=lambda x: -x[1]["count"]):
            avg_pnl = stats["total_pnl"] / stats["count"] if stats["count"] else 0
            print(f"{trigger:<30} {stats['count']:>8} {avg_pnl:>+11.4f}%")
    else:
        print("No edge data available yet.")

    # ==========================================================================
    # PHASE 4: BLOCKED SIGNAL ANALYSIS (YOUR BIGGEST EDGE)
    # ==========================================================================
    print("\n" + "=" * 70)
    print("PHASE 4: BLOCKED SIGNAL ANALYSIS")
    print("=" * 70)
    print("\nQuestion: Do blocked signals save money?\n")

    blocked_stats = conn.execute("""
        SELECT
            block_reason,
            regime,
            COUNT(*) as count,
            AVG(spread_at_signal) as avg_spread,
            AVG(imbalance_at_signal) as avg_imbalance
        FROM blocked_signals
        GROUP BY block_reason, regime
        ORDER BY count DESC
    """).fetchall()

    if blocked_stats:
        # Estimate would-have PnL based on similar executed trades
        avg_loss_per_trade = conn.execute("""
            SELECT AVG(pnl_pct) FROM trades
            WHERE pnl_pct < 0 AND exit_time IS NOT NULL
        """).fetchone()[0] or -0.04

        total_blocked = sum(row["count"] for row in blocked_stats)
        estimated_saved = total_blocked * abs(avg_loss_per_trade)

        print(f"{'Block Reason':<20} {'Regime':<18} {'Count':>8} {'Est. Saved':>12}")
        print("-" * 58)

        for row in blocked_stats[:10]:
            est_saved = row["count"] * abs(avg_loss_per_trade)
            print(f"{row['block_reason']:<20} {(row['regime'] or 'unknown'):<18} {row['count']:>8} {est_saved:>+11.4f}%")

        print("-" * 58)
        print(f"{'TOTAL BLOCKED':<20} {'':<18} {total_blocked:>8} {estimated_saved:>+11.4f}%")

        # Verdict
        print(f"\n📊 Blocked {total_blocked} signals")
        print(f"💰 Estimated PnL saved: {estimated_saved:+.4f}%")

        if estimated_saved > 0.1:
            print("✅ Your no-trade zones are PROTECTING capital effectively!")
        else:
            print("⚠️  Need more data to confirm protection effectiveness")
    else:
        print("No blocked signal data available yet.")

    # ==========================================================================
    # PHASE 5: FINAL RECOMMENDATIONS
    # ==========================================================================
    print("\n" + "=" * 70)
    print("PHASE 5: FINAL RECOMMENDATIONS")
    print("=" * 70)

    print("\n1. REGIMES TO TRADE:")
    trade_regimes = [r for r, v in regime_recommendations.items() if v == "trade"]
    if trade_regimes:
        for r in trade_regimes:
            print(f"   ✅ {r}")
    else:
        print("   📊 Need more data to identify profitable regimes")

    print("\n2. REGIMES TO DISABLE:")
    disable_regimes = [r for r, v in regime_recommendations.items() if v == "disable"]
    if disable_regimes:
        for r in disable_regimes:
            print(f"   ❌ {r}")
        print("\n   To disable, add to config.py:")
        print(f"   AVOID_REGIMES = {disable_regimes}")
    else:
        print("   📊 No regimes confirmed as losing yet")

    print("\n3. CAUSAL FILTERS:")
    if real_edge_causes:
        top_cause = max(real_edge_causes.items(), key=lambda x: x[1]["count"])[0]
        print(f"   ✅ Strengthen: {top_cause}")
    if fake_edge_causes:
        worst_cause = min(fake_edge_causes.items(), key=lambda x: x[1]["total_pnl"] / max(x[1]["count"], 1))[0]
        print(f"   ❌ Weaken/Remove: {worst_cause}")

    print("\n4. NO-TRADE ZONE VERDICT:")
    if total_blocked > 20:
        print("   ✅ Keep all current no-trade zone filters")
    else:
        print("   📊 Need more blocked signal data")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if total_trades >= 200:
        print(f"""
✅ You have {total_trades} trades - statistical significance achieved!

NEXT ACTIONS:
1. Update config.py with regime filters
2. Adjust causal trigger weights
3. Continue monitoring for 100 more trades
4. Re-run this analysis to confirm improvements
""")
    else:
        remaining = 200 - total_trades
        print(f"""
📊 Progress: {total_trades}/200 trades ({total_trades/200*100:.0f}%)

NEXT ACTIONS:
1. Continue running the bot
2. Collect {remaining} more trades
3. Run this analysis again when you hit 200 trades

DO NOT change any settings yet - let the data accumulate.
""")

    conn.close()


if __name__ == "__main__":
    run_full_analysis()
