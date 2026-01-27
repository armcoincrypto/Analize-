#!/usr/bin/env python3
"""
Paper → Live Promotion Check
============================
Evaluates if paper trading results meet the criteria for live promotion.

Usage:
    python tools/promotion_check.py --db hft_trades.db --days 7
    python tools/promotion_check.py --db hft_trades.db --days 14 --strict

Conditions checked:
    1. Net PnL after costs > 0
    2. Max drawdown < threshold (default 5%)
    3. Trade count >= minimum (default 50)
    4. Real edge % >= threshold (default 30%)
    5. Win rate >= minimum (default 35%)
    6. Sharpe ratio > threshold (default 0.3)

Output: PASS/FAIL with detailed reasons
"""

import argparse
import sqlite3
import sys
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

# Add parent dir for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class PromotionCriteria:
    """Criteria thresholds for promotion check."""
    min_pnl_after_costs_pct: float = 0.0  # Must be positive
    max_drawdown_pct: float = 5.0  # Max 5% drawdown
    min_trade_count: int = 50  # At least 50 trades
    min_edge_real_pct: float = 30.0  # 30% of trades have real edge
    min_win_rate_pct: float = 35.0  # At least 35% win rate
    min_sharpe_ratio: float = 0.3  # Positive risk-adjusted returns
    min_profit_factor: float = 1.0  # Must be > 1 (wins > losses)

    # Strict mode thresholds
    strict_min_pnl_after_costs_pct: float = 0.5  # Higher bar
    strict_max_drawdown_pct: float = 3.0
    strict_min_trade_count: int = 100
    strict_min_edge_real_pct: float = 40.0
    strict_min_win_rate_pct: float = 40.0
    strict_min_sharpe_ratio: float = 0.5
    strict_min_profit_factor: float = 1.2


@dataclass
class CheckResult:
    """Result of a single promotion check."""
    name: str
    passed: bool
    actual_value: float
    threshold: float
    message: str


@dataclass
class PromotionResult:
    """Overall promotion check result."""
    passed: bool
    checks: List[CheckResult]
    summary: str
    recommendation: str


def get_trading_stats(conn: sqlite3.Connection, days: int) -> dict:
    """Get comprehensive trading statistics for promotion check."""
    cursor = conn.cursor()

    cutoff_ms = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

    # Main stats query
    cursor.execute("""
        SELECT
            COUNT(*) as trade_count,
            SUM(CASE WHEN pnl_after_costs_pct > 0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN pnl_after_costs_pct <= 0 THEN 1 ELSE 0 END) as losses,
            SUM(pnl_after_costs_pct) as total_pnl,
            AVG(pnl_after_costs_pct) as avg_pnl,
            SUM(CASE WHEN pnl_after_costs_pct > 0 THEN pnl_after_costs_pct ELSE 0 END) as gross_wins,
            SUM(CASE WHEN pnl_after_costs_pct < 0 THEN ABS(pnl_after_costs_pct) ELSE 0 END) as gross_losses
        FROM trades
        WHERE status = 'closed' AND entry_time >= ?
    """, (cutoff_ms,))

    row = cursor.fetchone()

    if not row or row[0] == 0:
        return {
            "trade_count": 0,
            "wins": 0,
            "losses": 0,
            "total_pnl_pct": 0,
            "avg_pnl_pct": 0,
            "win_rate": 0,
            "profit_factor": 0,
            "max_drawdown_pct": 0,
            "sharpe_ratio": 0,
            "edge_real_pct": 0
        }

    trade_count = row[0]
    wins = row[1] or 0
    losses = row[2] or 0
    total_pnl = row[3] or 0
    avg_pnl = row[4] or 0
    gross_wins = row[5] or 0
    gross_losses = row[6] or 0

    win_rate = (wins / trade_count * 100) if trade_count > 0 else 0
    profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else (float('inf') if gross_wins > 0 else 0)

    # Get individual trade PnLs for drawdown and Sharpe calculation
    cursor.execute("""
        SELECT pnl_after_costs_pct
        FROM trades
        WHERE status = 'closed' AND entry_time >= ?
        ORDER BY entry_time
    """, (cutoff_ms,))

    pnl_list = [row[0] or 0 for row in cursor.fetchall()]

    # Calculate max drawdown from equity curve
    if pnl_list:
        equity = [100.0]
        for pnl in pnl_list:
            equity.append(equity[-1] * (1 + pnl / 100))

        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak * 100
        max_drawdown = float(np.max(drawdown))

        # Sharpe ratio
        if len(pnl_list) > 1:
            sharpe = np.mean(pnl_list) / np.std(pnl_list, ddof=1) if np.std(pnl_list, ddof=1) > 0 else 0
        else:
            sharpe = 0
    else:
        max_drawdown = 0
        sharpe = 0

    # Get edge real % from trade_edge table
    cursor.execute("""
        SELECT
            COUNT(*) as total,
            SUM(COALESCE(real_edge_flag, 0)) as real_edges
        FROM trade_edge te
        JOIN trades t ON te.trade_id = t.trade_id
        WHERE t.status = 'closed' AND t.entry_time >= ?
    """, (cutoff_ms,))

    edge_row = cursor.fetchone()
    if edge_row and edge_row[0] > 0:
        edge_real_pct = (edge_row[1] or 0) / edge_row[0] * 100
    else:
        edge_real_pct = 0  # No edge data

    return {
        "trade_count": trade_count,
        "wins": wins,
        "losses": losses,
        "total_pnl_pct": total_pnl,
        "avg_pnl_pct": avg_pnl,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "max_drawdown_pct": max_drawdown,
        "sharpe_ratio": sharpe,
        "edge_real_pct": edge_real_pct
    }


def run_promotion_checks(stats: dict, criteria: PromotionCriteria, strict: bool = False) -> PromotionResult:
    """Run all promotion checks against criteria."""
    checks = []

    # Select thresholds based on mode
    if strict:
        min_pnl = criteria.strict_min_pnl_after_costs_pct
        max_dd = criteria.strict_max_drawdown_pct
        min_trades = criteria.strict_min_trade_count
        min_edge = criteria.strict_min_edge_real_pct
        min_wr = criteria.strict_min_win_rate_pct
        min_sharpe = criteria.strict_min_sharpe_ratio
        min_pf = criteria.strict_min_profit_factor
    else:
        min_pnl = criteria.min_pnl_after_costs_pct
        max_dd = criteria.max_drawdown_pct
        min_trades = criteria.min_trade_count
        min_edge = criteria.min_edge_real_pct
        min_wr = criteria.min_win_rate_pct
        min_sharpe = criteria.min_sharpe_ratio
        min_pf = criteria.min_profit_factor

    # Check 1: Net PnL after costs
    pnl_passed = stats["total_pnl_pct"] > min_pnl
    checks.append(CheckResult(
        name="Net PnL After Costs",
        passed=pnl_passed,
        actual_value=stats["total_pnl_pct"],
        threshold=min_pnl,
        message=f"Total PnL: {stats['total_pnl_pct']:+.4f}% (need >{min_pnl}%)"
    ))

    # Check 2: Max Drawdown
    dd_passed = stats["max_drawdown_pct"] < max_dd
    checks.append(CheckResult(
        name="Max Drawdown",
        passed=dd_passed,
        actual_value=stats["max_drawdown_pct"],
        threshold=max_dd,
        message=f"Max DD: {stats['max_drawdown_pct']:.2f}% (need <{max_dd}%)"
    ))

    # Check 3: Trade Count
    trades_passed = stats["trade_count"] >= min_trades
    checks.append(CheckResult(
        name="Trade Count",
        passed=trades_passed,
        actual_value=stats["trade_count"],
        threshold=min_trades,
        message=f"Trades: {stats['trade_count']} (need >={min_trades})"
    ))

    # Check 4: Edge Real %
    edge_passed = stats["edge_real_pct"] >= min_edge or stats["edge_real_pct"] == 0  # Pass if no edge data
    checks.append(CheckResult(
        name="Real Edge %",
        passed=edge_passed,
        actual_value=stats["edge_real_pct"],
        threshold=min_edge,
        message=f"Real Edge: {stats['edge_real_pct']:.1f}% (need >={min_edge}%)" if stats["edge_real_pct"] > 0 else "No edge data (skipped)"
    ))

    # Check 5: Win Rate
    wr_passed = stats["win_rate"] >= min_wr
    checks.append(CheckResult(
        name="Win Rate",
        passed=wr_passed,
        actual_value=stats["win_rate"],
        threshold=min_wr,
        message=f"Win Rate: {stats['win_rate']:.1f}% (need >={min_wr}%)"
    ))

    # Check 6: Sharpe Ratio
    sharpe_passed = stats["sharpe_ratio"] >= min_sharpe
    checks.append(CheckResult(
        name="Sharpe Ratio",
        passed=sharpe_passed,
        actual_value=stats["sharpe_ratio"],
        threshold=min_sharpe,
        message=f"Sharpe: {stats['sharpe_ratio']:.3f} (need >={min_sharpe})"
    ))

    # Check 7: Profit Factor
    pf_passed = stats["profit_factor"] >= min_pf
    checks.append(CheckResult(
        name="Profit Factor",
        passed=pf_passed,
        actual_value=stats["profit_factor"],
        threshold=min_pf,
        message=f"Profit Factor: {stats['profit_factor']:.2f} (need >={min_pf})"
    ))

    # Overall result
    all_passed = all(c.passed for c in checks)
    failed_checks = [c for c in checks if not c.passed]

    if all_passed:
        summary = "ALL CHECKS PASSED"
        recommendation = "Strategy is ready for live promotion (with appropriate position sizing)"
    else:
        summary = f"FAILED {len(failed_checks)}/{len(checks)} CHECKS"
        recommendation = "Continue paper trading. Focus on: " + ", ".join(c.name for c in failed_checks)

    return PromotionResult(
        passed=all_passed,
        checks=checks,
        summary=summary,
        recommendation=recommendation
    )


def print_result(result: PromotionResult, stats: dict, days: int, strict: bool):
    """Print promotion check result."""
    mode = "STRICT" if strict else "STANDARD"
    status = "PASS" if result.passed else "FAIL"
    status_color = "✓" if result.passed else "✗"

    print(f"""
+================================================================================+
|                    PAPER → LIVE PROMOTION CHECK                                |
+================================================================================+
| Period:     Last {days} days
| Mode:       {mode}
| Status:     {status_color} {status}
+================================================================================+

  --- PERFORMANCE SUMMARY ---
  Trades:        {stats['trade_count']}
  Wins/Losses:   {stats['wins']}/{stats['losses']}
  Win Rate:      {stats['win_rate']:.1f}%
  Total PnL:     {stats['total_pnl_pct']:+.4f}%
  Avg PnL:       {stats['avg_pnl_pct']:+.4f}%
  Max Drawdown:  {stats['max_drawdown_pct']:.2f}%
  Sharpe Ratio:  {stats['sharpe_ratio']:.3f}
  Profit Factor: {stats['profit_factor']:.2f}
  Real Edge %:   {stats['edge_real_pct']:.1f}%

  --- CHECK RESULTS ---
""")

    for check in result.checks:
        icon = "✓" if check.passed else "✗"
        print(f"  [{icon}] {check.name:<20} {check.message}")

    print(f"""
  --- VERDICT ---
  {result.summary}

  --- RECOMMENDATION ---
  {result.recommendation}
""")

    if result.passed:
        print("""
  ⚠ BEFORE GOING LIVE:
  1. Start with minimal position size (0.1% of capital)
  2. Monitor first 10 trades manually
  3. Set up runtime alarms (see hft_bot.py)
  4. Have a kill switch ready
""")


def main():
    parser = argparse.ArgumentParser(
        description="Paper → Live Promotion Check",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/promotion_check.py --db hft_trades.db --days 7
  python tools/promotion_check.py --db hft_trades.db --days 14 --strict
  python tools/promotion_check.py --db hft_trades.db --days 30 --min-trades 100
        """
    )
    parser.add_argument("--db", default="hft_trades.db", help="Database path")
    parser.add_argument("--days", type=int, default=7, help="Evaluation period (days)")
    parser.add_argument("--strict", action="store_true", help="Use strict criteria")
    parser.add_argument("--min-trades", type=int, help="Override minimum trade count")
    parser.add_argument("--max-drawdown", type=float, help="Override max drawdown threshold")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
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

    # Connect
    conn = sqlite3.connect(str(db_path))

    # Get stats
    stats = get_trading_stats(conn, args.days)

    if stats["trade_count"] == 0:
        print(f"ERROR: No trades found in the last {args.days} days")
        return 1

    # Build criteria with overrides
    criteria = PromotionCriteria()
    if args.min_trades:
        criteria.min_trade_count = args.min_trades
        criteria.strict_min_trade_count = args.min_trades
    if args.max_drawdown:
        criteria.max_drawdown_pct = args.max_drawdown
        criteria.strict_max_drawdown_pct = args.max_drawdown

    # Run checks
    result = run_promotion_checks(stats, criteria, args.strict)

    if args.json:
        import json
        output = {
            "passed": result.passed,
            "summary": result.summary,
            "recommendation": result.recommendation,
            "stats": stats,
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "actual": c.actual_value,
                    "threshold": c.threshold,
                    "message": c.message
                }
                for c in result.checks
            ]
        }
        print(json.dumps(output, indent=2))
    else:
        print_result(result, stats, args.days, args.strict)

    conn.close()

    # Exit code: 0 for pass, 1 for fail
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
