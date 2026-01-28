#!/usr/bin/env python3
"""
Champion vs Challenger Comparison Tool
======================================
Compares live performance of champion and challenger strategies.

Usage:
    # Compare champion vs challenger for symbol
    python tools/champion_challenger.py --symbol XRPUSDT --days 7

    # Compare specific strategies
    python tools/champion_challenger.py --champion xrp_momentum_v1 --challenger xrp_momentum_v2

    # Get promotion recommendation
    python tools/champion_challenger.py --symbol XRPUSDT --recommend

    # Update registry with new champion (requires --force)
    python tools/champion_challenger.py --symbol XRPUSDT --promote --force

Output:
    - Side-by-side performance comparison
    - Statistical significance test
    - Promote/keep recommendation with reasons
"""

import argparse
import sqlite3
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Add parent dir for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from hft_system.db import open_sqlite

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from hft_system.strategy_loader import StrategyLoader, StrategyConfig


@dataclass
class StrategyPerformance:
    """Performance metrics for a strategy."""
    strategy_id: str
    symbol: str
    period_days: int
    trade_count: int
    wins: int
    losses: int
    win_rate_pct: float
    total_pnl_pct: float
    avg_pnl_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    profit_factor: float
    avg_hold_time_sec: float
    # Cost metrics
    avg_fees_pct: float
    avg_slippage_pct: float
    avg_total_costs_pct: float


@dataclass
class ComparisonResult:
    """Result of champion vs challenger comparison."""
    symbol: str
    period_days: int
    champion: StrategyPerformance
    challenger: StrategyPerformance
    # Comparison metrics
    pnl_diff_pct: float  # Challenger - Champion
    win_rate_diff_pct: float
    sharpe_diff: float
    # Statistical
    is_significant: bool
    p_value: float
    min_trades_for_significance: int
    # Recommendation
    recommendation: str  # "PROMOTE" | "KEEP_CHAMPION" | "INCONCLUSIVE"
    reasons: List[str]


def get_strategy_performance(
    conn: sqlite3.Connection,
    strategy_id: str,
    symbol: str,
    days: int
) -> Optional[StrategyPerformance]:
    """
    Get performance metrics for a strategy from database.

    Note: This requires trades to be tagged with strategy_id.
    For now, we use symbol + time period as proxy.
    """
    cursor = conn.cursor()
    cutoff_ms = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

    # Get trade metrics
    # In production, filter by strategy_id column when available
    cursor.execute("""
        SELECT
            COUNT(*) as trade_count,
            SUM(CASE WHEN pnl_after_costs_pct > 0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN pnl_after_costs_pct <= 0 THEN 1 ELSE 0 END) as losses,
            AVG(pnl_pct) as avg_gross_pnl,
            SUM(pnl_after_costs_pct) as total_pnl,
            AVG(pnl_after_costs_pct) as avg_pnl,
            AVG(hold_time_sec) as avg_hold_time,
            AVG(fees_paid_pct) as avg_fees,
            AVG(slippage_pct) as avg_slippage,
            AVG(total_costs_pct) as avg_costs,
            AVG(CASE WHEN pnl_after_costs_pct > 0 THEN pnl_after_costs_pct ELSE NULL END) as avg_win,
            AVG(CASE WHEN pnl_after_costs_pct <= 0 THEN ABS(pnl_after_costs_pct) ELSE NULL END) as avg_loss
        FROM trades
        WHERE symbol = ? AND status = 'closed' AND entry_time > ?
    """, (symbol, cutoff_ms))

    row = cursor.fetchone()

    if not row or row[0] == 0:
        return None

    trade_count = row[0]
    wins = row[1] or 0
    losses = row[2] or 0
    avg_win = row[10] or 0
    avg_loss = row[11] or 0.001  # Avoid division by zero

    win_rate = wins / trade_count * 100 if trade_count > 0 else 0
    profit_factor = (avg_win * wins) / (avg_loss * losses) if losses > 0 and avg_loss > 0 else 0

    # Calculate Sharpe ratio (simplified)
    # Get PnL list for std dev
    cursor.execute("""
        SELECT pnl_after_costs_pct
        FROM trades
        WHERE symbol = ? AND status = 'closed' AND entry_time > ?
    """, (symbol, cutoff_ms))
    pnls = [r[0] for r in cursor.fetchall() if r[0] is not None]

    if len(pnls) > 1:
        import statistics
        mean_pnl = statistics.mean(pnls)
        std_pnl = statistics.stdev(pnls)
        sharpe = (mean_pnl / std_pnl) if std_pnl > 0 else 0
    else:
        sharpe = 0

    # Calculate max drawdown
    cursor.execute("""
        SELECT pnl_after_costs_pct
        FROM trades
        WHERE symbol = ? AND status = 'closed' AND entry_time > ?
        ORDER BY entry_time
    """, (symbol, cutoff_ms))
    pnl_series = [r[0] for r in cursor.fetchall() if r[0] is not None]

    max_drawdown = 0
    if pnl_series:
        cumulative = 0
        peak = 0
        for pnl in pnl_series:
            cumulative += pnl
            if cumulative > peak:
                peak = cumulative
            drawdown = peak - cumulative
            if drawdown > max_drawdown:
                max_drawdown = drawdown

    return StrategyPerformance(
        strategy_id=strategy_id,
        symbol=symbol,
        period_days=days,
        trade_count=trade_count,
        wins=wins,
        losses=losses,
        win_rate_pct=win_rate,
        total_pnl_pct=row[4] or 0,
        avg_pnl_pct=row[5] or 0,
        max_drawdown_pct=max_drawdown,
        sharpe_ratio=sharpe,
        profit_factor=profit_factor,
        avg_hold_time_sec=row[6] or 0,
        avg_fees_pct=row[7] or 0,
        avg_slippage_pct=row[8] or 0,
        avg_total_costs_pct=row[9] or 0,
    )


def calculate_statistical_significance(
    champion: StrategyPerformance,
    challenger: StrategyPerformance
) -> Tuple[bool, float, int]:
    """
    Calculate if performance difference is statistically significant.

    Uses simplified t-test approximation.

    Returns:
        (is_significant, p_value_estimate, min_trades_needed)
    """
    # Minimum trades for any significance
    min_trades = 30

    if champion.trade_count < min_trades or challenger.trade_count < min_trades:
        return False, 1.0, min_trades

    # Effect size: difference in avg PnL
    effect_size = abs(challenger.avg_pnl_pct - champion.avg_pnl_pct)

    # Estimated variance (using simplified pooled estimate)
    # In production, use actual variance from trade PnLs
    est_variance = 0.1  # Rough estimate for HFT

    # Sample size needed for power = 0.8, alpha = 0.05
    # n = 2 * (z_alpha + z_beta)^2 * var / effect^2
    # Simplified: n ~ 16 * var / effect^2
    if effect_size > 0:
        min_trades_needed = int(16 * est_variance / (effect_size ** 2))
        min_trades_needed = max(min_trades, min(min_trades_needed, 500))
    else:
        min_trades_needed = 500

    # Check if we have enough trades
    total_trades = champion.trade_count + challenger.trade_count
    has_enough = total_trades >= min_trades_needed

    # Estimate p-value (very rough)
    if has_enough and effect_size > 0:
        # z = effect / (sqrt(2 * var / n))
        z = effect_size / (2 * est_variance / total_trades) ** 0.5
        # Rough p-value approximation
        p_value = max(0.001, min(1.0, 2 * (1 - min(1, z / 3))))
    else:
        p_value = 1.0

    is_significant = p_value < 0.05 and has_enough

    return is_significant, p_value, min_trades_needed


def compare_strategies(
    champion_perf: StrategyPerformance,
    challenger_perf: StrategyPerformance
) -> ComparisonResult:
    """
    Compare champion and challenger performance.

    Returns comparison result with recommendation.
    """
    # Calculate differences
    pnl_diff = challenger_perf.total_pnl_pct - champion_perf.total_pnl_pct
    win_rate_diff = challenger_perf.win_rate_pct - champion_perf.win_rate_pct
    sharpe_diff = challenger_perf.sharpe_ratio - champion_perf.sharpe_ratio

    # Statistical significance
    is_sig, p_value, min_trades = calculate_statistical_significance(
        champion_perf, challenger_perf
    )

    # Build recommendation
    reasons = []
    promote_score = 0

    # PnL comparison (weight: 3)
    if pnl_diff > 0.5:  # Challenger > 0.5% better
        reasons.append(f"Challenger PnL +{pnl_diff:.2f}% higher")
        promote_score += 3
    elif pnl_diff < -0.5:
        reasons.append(f"Champion PnL {-pnl_diff:.2f}% higher")
        promote_score -= 3
    else:
        reasons.append(f"PnL similar (diff: {pnl_diff:+.2f}%)")

    # Sharpe comparison (weight: 2)
    if sharpe_diff > 0.1:
        reasons.append(f"Challenger Sharpe +{sharpe_diff:.2f} better")
        promote_score += 2
    elif sharpe_diff < -0.1:
        reasons.append(f"Champion Sharpe {-sharpe_diff:.2f} better")
        promote_score -= 2

    # Win rate comparison (weight: 1)
    if win_rate_diff > 5:
        reasons.append(f"Challenger win rate +{win_rate_diff:.1f}% higher")
        promote_score += 1
    elif win_rate_diff < -5:
        reasons.append(f"Champion win rate {-win_rate_diff:.1f}% higher")
        promote_score -= 1

    # Drawdown comparison (weight: 2)
    dd_diff = challenger_perf.max_drawdown_pct - champion_perf.max_drawdown_pct
    if dd_diff < -0.5:  # Challenger has lower drawdown
        reasons.append(f"Challenger drawdown {-dd_diff:.2f}% lower")
        promote_score += 2
    elif dd_diff > 0.5:
        reasons.append(f"Champion drawdown {-dd_diff:.2f}% lower")
        promote_score -= 1

    # Trade count (need minimum for confidence)
    if challenger_perf.trade_count < 30:
        reasons.append(f"Challenger needs more trades ({challenger_perf.trade_count}/30)")
        promote_score -= 2

    # Statistical significance
    if not is_sig:
        reasons.append(f"Difference not statistically significant (need {min_trades} trades)")
        promote_score = min(promote_score, 1)  # Cap score if not significant

    # Determine recommendation
    if promote_score >= 4 and is_sig:
        recommendation = "PROMOTE"
    elif promote_score <= -2:
        recommendation = "KEEP_CHAMPION"
    else:
        recommendation = "INCONCLUSIVE"

    return ComparisonResult(
        symbol=champion_perf.symbol,
        period_days=champion_perf.period_days,
        champion=champion_perf,
        challenger=challenger_perf,
        pnl_diff_pct=pnl_diff,
        win_rate_diff_pct=win_rate_diff,
        sharpe_diff=sharpe_diff,
        is_significant=is_sig,
        p_value=p_value,
        min_trades_for_significance=min_trades,
        recommendation=recommendation,
        reasons=reasons,
    )


def print_comparison(result: ComparisonResult):
    """Print formatted comparison report."""
    rec_icon = {
        "PROMOTE": "[+]",
        "KEEP_CHAMPION": "[-]",
        "INCONCLUSIVE": "[?]",
    }.get(result.recommendation, "[?]")

    print(f"""
+================================================================================+
|                 CHAMPION vs CHALLENGER COMPARISON                              |
+================================================================================+
| Symbol:  {result.symbol:67s} |
| Period:  Last {result.period_days} days{' ' * 58}|
+================================================================================+

  --- PERFORMANCE COMPARISON ---

  {'Metric':<25s} {'CHAMPION':>15s} {'CHALLENGER':>15s} {'DIFF':>12s}
  {'-' * 25} {'-' * 15} {'-' * 15} {'-' * 12}
  {'Strategy ID':<25s} {result.champion.strategy_id:>15s} {result.challenger.strategy_id:>15s}
  {'Trades':<25s} {result.champion.trade_count:>15d} {result.challenger.trade_count:>15d}
  {'Win Rate':<25s} {result.champion.win_rate_pct:>14.1f}% {result.challenger.win_rate_pct:>14.1f}% {result.win_rate_diff_pct:>+11.1f}%
  {'Total PnL':<25s} {result.champion.total_pnl_pct:>+14.2f}% {result.challenger.total_pnl_pct:>+14.2f}% {result.pnl_diff_pct:>+11.2f}%
  {'Avg PnL/Trade':<25s} {result.champion.avg_pnl_pct:>+14.4f}% {result.challenger.avg_pnl_pct:>+14.4f}%
  {'Sharpe Ratio':<25s} {result.champion.sharpe_ratio:>15.3f} {result.challenger.sharpe_ratio:>15.3f} {result.sharpe_diff:>+12.3f}
  {'Max Drawdown':<25s} {result.champion.max_drawdown_pct:>14.2f}% {result.challenger.max_drawdown_pct:>14.2f}%
  {'Profit Factor':<25s} {result.champion.profit_factor:>15.2f} {result.challenger.profit_factor:>15.2f}
  {'Avg Hold Time':<25s} {result.champion.avg_hold_time_sec:>14.1f}s {result.challenger.avg_hold_time_sec:>14.1f}s

  --- COST ANALYSIS ---
  {'Avg Fees':<25s} {result.champion.avg_fees_pct:>14.4f}% {result.challenger.avg_fees_pct:>14.4f}%
  {'Avg Slippage':<25s} {result.champion.avg_slippage_pct:>14.4f}% {result.challenger.avg_slippage_pct:>14.4f}%
  {'Total Costs':<25s} {result.champion.avg_total_costs_pct:>14.4f}% {result.challenger.avg_total_costs_pct:>14.4f}%

  --- STATISTICAL ANALYSIS ---
  Significant Difference:  {'YES' if result.is_significant else 'NO'}
  Estimated p-value:       {result.p_value:.4f}
  Min Trades Needed:       {result.min_trades_for_significance}

  --- RECOMMENDATION ---
  {rec_icon} {result.recommendation}

  Reasons:
""")
    for reason in result.reasons:
        print(f"    - {reason}")

    if result.recommendation == "PROMOTE":
        print("""
  ACTION: Challenger significantly outperforms champion.
          Consider promoting challenger to champion.
          Run: python tools/champion_challenger.py --symbol {symbol} --promote --force
""".format(symbol=result.symbol))
    elif result.recommendation == "KEEP_CHAMPION":
        print("""
  ACTION: Champion performs better or challenger shows no improvement.
          Keep current champion.
""")
    else:
        print("""
  ACTION: Results inconclusive. Continue monitoring.
          Need more trades or longer observation period.
""")


def promote_challenger(
    loader: StrategyLoader,
    symbol: str,
    registry_path: str
) -> bool:
    """
    Promote challenger to champion in registry.

    Returns True if successful.
    """
    if not YAML_AVAILABLE:
        print("ERROR: PyYAML required to update registry")
        return False

    assignment = loader.assignments.get(symbol)
    if not assignment:
        print(f"ERROR: No assignment found for {symbol}")
        return False

    champion_id = assignment.get("champion")
    challenger_id = assignment.get("challenger")

    if not challenger_id:
        print(f"ERROR: No challenger defined for {symbol}")
        return False

    # Load registry
    with open(registry_path, "r") as f:
        registry = yaml.safe_load(f)

    # Update strategy statuses
    if champion_id and champion_id in registry.get("strategies", {}):
        registry["strategies"][champion_id]["status"] = "retired"

    if challenger_id in registry.get("strategies", {}):
        registry["strategies"][challenger_id]["status"] = "champion"
        registry["strategies"][challenger_id]["metadata"]["promotion_approved"] = True
        registry["strategies"][challenger_id]["metadata"]["promotion_date"] = datetime.now().isoformat()

    # Update assignment
    registry["assignments"][symbol]["champion"] = challenger_id
    registry["assignments"][symbol]["challenger"] = None
    registry["assignments"][symbol]["last_comparison"] = datetime.now().isoformat()

    # Update registry file
    registry["updated_at"] = datetime.now().isoformat()

    with open(registry_path, "w") as f:
        yaml.dump(registry, f, default_flow_style=False, sort_keys=False)

    print(f"SUCCESS: Promoted {challenger_id} to champion for {symbol}")
    print(f"         Previous champion {champion_id} retired")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Champion vs Challenger Comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare for symbol
  python tools/champion_challenger.py --symbol XRPUSDT --days 7

  # Compare specific strategies
  python tools/champion_challenger.py --champion xrp_v1 --challenger xrp_v2 --days 14

  # Get recommendation
  python tools/champion_challenger.py --symbol XRPUSDT --recommend

  # Promote challenger (requires --force)
  python tools/champion_challenger.py --symbol XRPUSDT --promote --force
        """
    )

    parser.add_argument("--db", default="hft_trades.db", help="Database path")
    parser.add_argument("--registry", default="strategies/registry.yml",
                        help="Registry file path")
    parser.add_argument("--symbol", type=str, help="Symbol to compare")
    parser.add_argument("--champion", type=str, help="Champion strategy ID")
    parser.add_argument("--challenger", type=str, help="Challenger strategy ID")
    parser.add_argument("--days", type=int, default=7, help="Comparison period (days)")
    parser.add_argument("--recommend", action="store_true",
                        help="Show promotion recommendation")
    parser.add_argument("--promote", action="store_true",
                        help="Promote challenger to champion")
    parser.add_argument("--force", action="store_true",
                        help="Force promotion (required with --promote)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    # Find database
    db_path = Path(args.db)
    if not db_path.exists():
        for try_path in [
            Path.cwd() / "hft_trades.db",
            Path.home() / "Analize-" / "hft_trades.db",
            Path("/root/Analize-/hft_trades.db"),
        ]:
            if try_path.exists():
                db_path = try_path
                break

    # Find registry
    registry_path = Path(args.registry)
    if not registry_path.exists():
        for try_path in [
            Path.cwd() / "strategies" / "registry.yml",
            Path(__file__).parent.parent / "strategies" / "registry.yml",
            Path("/root/Analize-/strategies/registry.yml"),
        ]:
            if try_path.exists():
                registry_path = try_path
                break

    if not registry_path.exists():
        print(f"ERROR: Registry not found at {args.registry}")
        return 1

    # Load registry
    loader = StrategyLoader(str(registry_path))

    # Determine champion and challenger
    champion_id = args.champion
    challenger_id = args.challenger

    if args.symbol and not (champion_id and challenger_id):
        assignment = loader.assignments.get(args.symbol, {})
        champion_id = champion_id or assignment.get("champion")
        challenger_id = challenger_id or assignment.get("challenger")

    if not champion_id or not challenger_id:
        print("ERROR: Must specify both champion and challenger")
        print("       Use --symbol to auto-detect from registry")
        print("       Or use --champion and --challenger explicitly")
        return 1

    symbol = args.symbol
    if not symbol:
        champion_strat = loader.get_strategy(champion_id)
        if champion_strat:
            symbol = champion_strat.symbol

    if not symbol:
        print("ERROR: Could not determine symbol")
        return 1

    # Promote mode
    if args.promote:
        if not args.force:
            print("ERROR: --promote requires --force flag")
            print("       This will update the registry file!")
            return 1

        return 0 if promote_challenger(loader, symbol, str(registry_path)) else 1

    # Connect to database
    if not db_path.exists():
        print(f"WARNING: Database not found at {db_path}")
        print("         Using mock data for demonstration")

        # Mock performance data
        champion_perf = StrategyPerformance(
            strategy_id=champion_id,
            symbol=symbol,
            period_days=args.days,
            trade_count=85,
            wins=42,
            losses=43,
            win_rate_pct=49.4,
            total_pnl_pct=0.72,
            avg_pnl_pct=0.0085,
            max_drawdown_pct=1.8,
            sharpe_ratio=0.58,
            profit_factor=1.12,
            avg_hold_time_sec=45.2,
            avg_fees_pct=0.026,
            avg_slippage_pct=0.008,
            avg_total_costs_pct=0.042,
        )

        challenger_perf = StrategyPerformance(
            strategy_id=challenger_id,
            symbol=symbol,
            period_days=args.days,
            trade_count=92,
            wins=48,
            losses=44,
            win_rate_pct=52.2,
            total_pnl_pct=0.95,
            avg_pnl_pct=0.0103,
            max_drawdown_pct=1.5,
            sharpe_ratio=0.71,
            profit_factor=1.24,
            avg_hold_time_sec=38.7,
            avg_fees_pct=0.015,
            avg_slippage_pct=0.012,
            avg_total_costs_pct=0.035,
        )
    else:
        conn = open_sqlite(str(db_path))

        champion_perf = get_strategy_performance(conn, champion_id, symbol, args.days)
        challenger_perf = get_strategy_performance(conn, challenger_id, symbol, args.days)

        conn.close()

        if not champion_perf:
            print(f"ERROR: No trades found for champion {champion_id}")
            return 1

        if not challenger_perf:
            print(f"ERROR: No trades found for challenger {challenger_id}")
            return 1

    # Compare
    result = compare_strategies(champion_perf, challenger_perf)

    if args.json:
        output = {
            "symbol": result.symbol,
            "period_days": result.period_days,
            "champion": {
                "strategy_id": result.champion.strategy_id,
                "trades": result.champion.trade_count,
                "win_rate_pct": result.champion.win_rate_pct,
                "total_pnl_pct": result.champion.total_pnl_pct,
                "sharpe_ratio": result.champion.sharpe_ratio,
                "max_drawdown_pct": result.champion.max_drawdown_pct,
            },
            "challenger": {
                "strategy_id": result.challenger.strategy_id,
                "trades": result.challenger.trade_count,
                "win_rate_pct": result.challenger.win_rate_pct,
                "total_pnl_pct": result.challenger.total_pnl_pct,
                "sharpe_ratio": result.challenger.sharpe_ratio,
                "max_drawdown_pct": result.challenger.max_drawdown_pct,
            },
            "comparison": {
                "pnl_diff_pct": result.pnl_diff_pct,
                "win_rate_diff_pct": result.win_rate_diff_pct,
                "sharpe_diff": result.sharpe_diff,
                "is_significant": result.is_significant,
                "p_value": result.p_value,
            },
            "recommendation": result.recommendation,
            "reasons": result.reasons,
        }
        print(json.dumps(output, indent=2))
    else:
        print_comparison(result)

    return 0


if __name__ == "__main__":
    sys.exit(main())
