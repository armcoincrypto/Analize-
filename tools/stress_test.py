#!/usr/bin/env python3
"""
Strategy Stress Testing Tool
============================
Tests strategy robustness under adverse conditions.

Stress scenarios:
1. Fee stress: +50%, +100% fees
2. Slippage stress: x1.5, x2.0 slippage
3. Spread stress: x1.5, x2.0 spreads
4. Combined stress: All factors increased

A strategy must pass stress tests to be eligible for LIVE trading.

Usage:
    python tools/stress_test.py --db hft_trades.db --days 7
    python tools/stress_test.py --db hft_trades.db --days 14 --scenario all
    python tools/stress_test.py --db hft_trades.db --json

Output:
    - PASS/FAIL for each stress scenario
    - Degradation metrics (how much PnL decreased)
    - Recommendation for promotion
"""

import argparse
import sqlite3
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

# Add parent dir for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class StressScenario(Enum):
    """Stress test scenarios."""
    BASELINE = "baseline"  # No stress
    FEE_PLUS_50 = "fee_plus_50"  # Fees +50%
    FEE_PLUS_100 = "fee_plus_100"  # Fees +100%
    SLIPPAGE_X1_5 = "slippage_x1_5"  # Slippage x1.5
    SLIPPAGE_X2_0 = "slippage_x2_0"  # Slippage x2.0
    SPREAD_X1_5 = "spread_x1_5"  # Spread x1.5
    SPREAD_X2_0 = "spread_x2_0"  # Spread x2.0
    COMBINED_MILD = "combined_mild"  # Fee+50%, Slip x1.5, Spread x1.5
    COMBINED_SEVERE = "combined_severe"  # Fee+100%, Slip x2.0, Spread x2.0


@dataclass
class StressConfig:
    """Stress multipliers for a scenario."""
    name: str
    fee_multiplier: float = 1.0
    slippage_multiplier: float = 1.0
    spread_multiplier: float = 1.0
    description: str = ""


# Define stress scenarios
STRESS_SCENARIOS: Dict[StressScenario, StressConfig] = {
    StressScenario.BASELINE: StressConfig(
        name="Baseline",
        fee_multiplier=1.0,
        slippage_multiplier=1.0,
        spread_multiplier=1.0,
        description="No stress - actual recorded costs"
    ),
    StressScenario.FEE_PLUS_50: StressConfig(
        name="Fees +50%",
        fee_multiplier=1.5,
        slippage_multiplier=1.0,
        spread_multiplier=1.0,
        description="Exchange raises fees by 50%"
    ),
    StressScenario.FEE_PLUS_100: StressConfig(
        name="Fees +100%",
        fee_multiplier=2.0,
        slippage_multiplier=1.0,
        spread_multiplier=1.0,
        description="Exchange doubles fees"
    ),
    StressScenario.SLIPPAGE_X1_5: StressConfig(
        name="Slippage x1.5",
        fee_multiplier=1.0,
        slippage_multiplier=1.5,
        spread_multiplier=1.0,
        description="50% worse execution than expected"
    ),
    StressScenario.SLIPPAGE_X2_0: StressConfig(
        name="Slippage x2.0",
        fee_multiplier=1.0,
        slippage_multiplier=2.0,
        spread_multiplier=1.0,
        description="100% worse execution than expected"
    ),
    StressScenario.SPREAD_X1_5: StressConfig(
        name="Spread x1.5",
        fee_multiplier=1.0,
        slippage_multiplier=1.0,
        spread_multiplier=1.5,
        description="Spreads widen by 50%"
    ),
    StressScenario.SPREAD_X2_0: StressConfig(
        name="Spread x2.0",
        fee_multiplier=1.0,
        slippage_multiplier=1.0,
        spread_multiplier=2.0,
        description="Spreads double"
    ),
    StressScenario.COMBINED_MILD: StressConfig(
        name="Combined Mild",
        fee_multiplier=1.5,
        slippage_multiplier=1.5,
        spread_multiplier=1.5,
        description="All costs +50% (market stress)"
    ),
    StressScenario.COMBINED_SEVERE: StressConfig(
        name="Combined Severe",
        fee_multiplier=2.0,
        slippage_multiplier=2.0,
        spread_multiplier=2.0,
        description="All costs doubled (crisis scenario)"
    ),
}


@dataclass
class StressResult:
    """Result of a stress test scenario."""
    scenario: StressScenario
    config: StressConfig
    # Metrics under stress
    trade_count: int = 0
    wins: int = 0
    losses: int = 0
    win_rate_pct: float = 0
    total_pnl_pct: float = 0  # Gross PnL (unchanged)
    stressed_costs_pct: float = 0  # Costs under stress
    stressed_pnl_pct: float = 0  # PnL after stressed costs
    avg_pnl_pct: float = 0
    # Comparison to baseline
    pnl_degradation_pct: float = 0  # How much worse than baseline
    cost_increase_pct: float = 0  # How much costs increased
    # Pass/Fail
    passed: bool = False
    pass_reason: str = ""


@dataclass
class StressTestResult:
    """Complete stress test result."""
    period_days: int
    baseline_pnl_pct: float
    baseline_costs_pct: float
    results: List[StressResult]
    all_passed: bool
    critical_failures: List[str]
    recommendation: str


def calculate_stressed_pnl(
    gross_pnl_pct: float,
    fees_pct: float,
    slippage_pct: float,
    spread_cost_pct: float,
    config: StressConfig
) -> Tuple[float, float]:
    """
    Calculate PnL under stressed conditions.

    Args:
        gross_pnl_pct: Gross PnL before costs
        fees_pct: Original fees paid
        slippage_pct: Original slippage
        spread_cost_pct: Original spread cost
        config: Stress configuration

    Returns:
        (stressed_costs_pct, stressed_pnl_pct)
    """
    stressed_fees = fees_pct * config.fee_multiplier
    stressed_slippage = slippage_pct * config.slippage_multiplier
    stressed_spread = spread_cost_pct * config.spread_multiplier

    stressed_costs = stressed_fees + stressed_slippage + (2 * stressed_spread)
    stressed_pnl = gross_pnl_pct - stressed_costs

    return stressed_costs, stressed_pnl


def run_stress_scenario(
    conn: sqlite3.Connection,
    days: int,
    scenario: StressScenario
) -> StressResult:
    """Run a single stress scenario."""
    config = STRESS_SCENARIOS[scenario]
    cursor = conn.cursor()
    cutoff_ms = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

    # Get all closed trades with cost breakdown
    cursor.execute("""
        SELECT
            pnl_pct,
            fees_paid_pct,
            slippage_pct,
            spread_cost_pct,
            pnl_after_costs_pct
        FROM trades
        WHERE status = 'closed' AND entry_time > ?
    """, (cutoff_ms,))

    rows = cursor.fetchall()

    if not rows:
        return StressResult(
            scenario=scenario,
            config=config,
            passed=False,
            pass_reason="No trades found"
        )

    # Calculate stressed PnL for each trade
    trade_count = len(rows)
    wins = 0
    losses = 0
    total_gross_pnl = 0
    total_original_costs = 0
    total_stressed_costs = 0
    total_stressed_pnl = 0

    for row in rows:
        gross_pnl = row[0] or 0
        fees = row[1] or 0
        slippage = row[2] or 0
        spread = row[3] or 0
        original_net = row[4] or 0

        # Calculate stressed
        stressed_costs, stressed_pnl = calculate_stressed_pnl(
            gross_pnl, fees, slippage, spread, config
        )

        total_gross_pnl += gross_pnl
        total_original_costs += (fees + slippage + 2 * spread)
        total_stressed_costs += stressed_costs
        total_stressed_pnl += stressed_pnl

        if stressed_pnl > 0:
            wins += 1
        else:
            losses += 1

    win_rate = wins / trade_count * 100 if trade_count > 0 else 0
    avg_pnl = total_stressed_pnl / trade_count if trade_count > 0 else 0

    result = StressResult(
        scenario=scenario,
        config=config,
        trade_count=trade_count,
        wins=wins,
        losses=losses,
        win_rate_pct=win_rate,
        total_pnl_pct=total_gross_pnl,
        stressed_costs_pct=total_stressed_costs,
        stressed_pnl_pct=total_stressed_pnl,
        avg_pnl_pct=avg_pnl
    )

    return result


def run_all_stress_tests(
    conn: sqlite3.Connection,
    days: int,
    scenarios: List[StressScenario] = None
) -> StressTestResult:
    """Run all stress test scenarios."""
    if scenarios is None:
        scenarios = list(StressScenario)

    results = []

    # Always run baseline first
    if StressScenario.BASELINE in scenarios:
        scenarios.remove(StressScenario.BASELINE)
        scenarios.insert(0, StressScenario.BASELINE)

    for scenario in scenarios:
        result = run_stress_scenario(conn, days, scenario)
        results.append(result)

    # Get baseline for comparison
    baseline = next((r for r in results if r.scenario == StressScenario.BASELINE), None)

    if not baseline or baseline.trade_count == 0:
        return StressTestResult(
            period_days=days,
            baseline_pnl_pct=0,
            baseline_costs_pct=0,
            results=results,
            all_passed=False,
            critical_failures=["No baseline data"],
            recommendation="Cannot run stress tests without baseline data"
        )

    # Calculate degradation and pass/fail for each scenario
    critical_failures = []

    for result in results:
        if result.scenario == StressScenario.BASELINE:
            result.pnl_degradation_pct = 0
            result.cost_increase_pct = 0
            result.passed = result.stressed_pnl_pct > 0
            result.pass_reason = "Profitable" if result.passed else "Not profitable"
        else:
            # Degradation from baseline
            if baseline.stressed_pnl_pct != 0:
                result.pnl_degradation_pct = (
                    (baseline.stressed_pnl_pct - result.stressed_pnl_pct)
                    / abs(baseline.stressed_pnl_pct) * 100
                )
            else:
                result.pnl_degradation_pct = 100 if result.stressed_pnl_pct < 0 else 0

            # Cost increase
            if baseline.stressed_costs_pct > 0:
                result.cost_increase_pct = (
                    (result.stressed_costs_pct - baseline.stressed_costs_pct)
                    / baseline.stressed_costs_pct * 100
                )

            # Pass criteria:
            # 1. Must still be profitable (stressed_pnl > 0)
            # 2. Must not lose more than 50% of baseline PnL
            if result.stressed_pnl_pct > 0:
                if result.pnl_degradation_pct <= 50:
                    result.passed = True
                    result.pass_reason = f"Profitable with {result.pnl_degradation_pct:.0f}% degradation"
                else:
                    result.passed = False
                    result.pass_reason = f"Excessive degradation ({result.pnl_degradation_pct:.0f}%)"
                    if "combined" in result.scenario.value or "severe" in result.scenario.value:
                        critical_failures.append(f"{result.config.name}: >50% degradation")
            else:
                result.passed = False
                result.pass_reason = f"Not profitable under stress"
                critical_failures.append(f"{result.config.name}: negative PnL")

    # Overall assessment
    all_passed = all(r.passed for r in results)

    # Key scenarios that MUST pass for live eligibility
    key_scenarios = [
        StressScenario.BASELINE,
        StressScenario.FEE_PLUS_50,
        StressScenario.SLIPPAGE_X1_5,
        StressScenario.COMBINED_MILD
    ]
    key_passed = all(
        r.passed for r in results
        if r.scenario in key_scenarios
    )

    if key_passed:
        recommendation = "ELIGIBLE for live trading - strategy survives key stress scenarios"
    elif all(r.passed for r in results if r.scenario in [StressScenario.BASELINE, StressScenario.FEE_PLUS_50]):
        recommendation = "PAPER ONLY - strategy profitable but vulnerable to execution stress"
    else:
        recommendation = "NOT READY - strategy fails basic stress tests, continue optimization"

    return StressTestResult(
        period_days=days,
        baseline_pnl_pct=baseline.stressed_pnl_pct,
        baseline_costs_pct=baseline.stressed_costs_pct,
        results=results,
        all_passed=all_passed,
        critical_failures=critical_failures,
        recommendation=recommendation
    )


def print_stress_report(result: StressTestResult):
    """Print formatted stress test report."""
    status = "PASS" if result.all_passed else "FAIL"
    status_icon = "+" if result.all_passed else "x"

    print(f"""
+================================================================================+
|                        STRATEGY STRESS TEST REPORT                             |
+================================================================================+
| Period:           Last {result.period_days} days
| Baseline PnL:     {result.baseline_pnl_pct:+.4f}%
| Baseline Costs:   {result.baseline_costs_pct:.4f}%
| Overall Status:   [{status_icon}] {status}
+================================================================================+

  --- STRESS SCENARIO RESULTS ---
""")

    for r in result.results:
        icon = "+" if r.passed else "x"
        degradation_str = f"{r.pnl_degradation_pct:+.0f}%" if r.scenario != StressScenario.BASELINE else "---"

        print(f"  [{icon}] {r.config.name:20s}")
        print(f"      {r.config.description}")
        print(f"      Stressed PnL:  {r.stressed_pnl_pct:+.4f}% | "
              f"Costs: {r.stressed_costs_pct:.4f}% | "
              f"Degradation: {degradation_str}")
        print(f"      Win Rate: {r.win_rate_pct:.1f}% | Trades: {r.trade_count}")
        print(f"      Status: {r.pass_reason}")
        print()

    if result.critical_failures:
        print("  --- CRITICAL FAILURES ---")
        for failure in result.critical_failures:
            print(f"  [!] {failure}")
        print()

    print(f"""  --- RECOMMENDATION ---
  {result.recommendation}
""")

    # Key thresholds explanation
    print("""  --- STRESS TEST CRITERIA ---
  PASS requires:
  1. Baseline must be profitable (PnL > 0)
  2. Fee +50% scenario must be profitable
  3. Slippage x1.5 scenario must be profitable
  4. Combined Mild must not lose >50% of baseline PnL

  For LIVE eligibility, strategy must survive:
  - All key scenarios (baseline, fee+50%, slippage x1.5, combined mild)
  - PnL degradation <= 50% from baseline
""")


def main():
    parser = argparse.ArgumentParser(
        description="Strategy Stress Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/stress_test.py --db hft_trades.db --days 7
  python tools/stress_test.py --db hft_trades.db --days 14 --scenario all
  python tools/stress_test.py --db hft_trades.db --scenario fee_plus_50,slippage_x1_5
  python tools/stress_test.py --db hft_trades.db --json

Scenarios:
  baseline        - No stress (actual costs)
  fee_plus_50     - Fees +50%
  fee_plus_100    - Fees +100%
  slippage_x1_5   - Slippage x1.5
  slippage_x2_0   - Slippage x2.0
  spread_x1_5     - Spread x1.5
  spread_x2_0     - Spread x2.0
  combined_mild   - All costs +50%
  combined_severe - All costs x2.0
        """
    )
    parser.add_argument("--db", default="hft_trades.db", help="Database path")
    parser.add_argument("--days", type=int, default=7, help="Analysis period (days)")
    parser.add_argument("--scenario", type=str, default="all",
                        help="Scenarios to run (comma-separated or 'all')")
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

    # Parse scenarios
    if args.scenario.lower() == "all":
        scenarios = list(StressScenario)
    else:
        scenario_names = [s.strip() for s in args.scenario.split(",")]
        scenarios = []
        for name in scenario_names:
            try:
                scenarios.append(StressScenario(name))
            except ValueError:
                print(f"ERROR: Unknown scenario '{name}'")
                return 1

    # Connect
    conn = sqlite3.connect(str(db_path))

    # Run stress tests
    result = run_all_stress_tests(conn, args.days, scenarios)

    if args.json:
        output = {
            "period_days": result.period_days,
            "baseline_pnl_pct": result.baseline_pnl_pct,
            "baseline_costs_pct": result.baseline_costs_pct,
            "all_passed": result.all_passed,
            "critical_failures": result.critical_failures,
            "recommendation": result.recommendation,
            "scenarios": [
                {
                    "name": r.config.name,
                    "scenario": r.scenario.value,
                    "description": r.config.description,
                    "trade_count": r.trade_count,
                    "wins": r.wins,
                    "losses": r.losses,
                    "win_rate_pct": round(r.win_rate_pct, 1),
                    "gross_pnl_pct": round(r.total_pnl_pct, 4),
                    "stressed_costs_pct": round(r.stressed_costs_pct, 4),
                    "stressed_pnl_pct": round(r.stressed_pnl_pct, 4),
                    "pnl_degradation_pct": round(r.pnl_degradation_pct, 1),
                    "cost_increase_pct": round(r.cost_increase_pct, 1),
                    "passed": r.passed,
                    "pass_reason": r.pass_reason
                }
                for r in result.results
            ]
        }
        print(json.dumps(output, indent=2))
    else:
        if result.baseline_pnl_pct == 0 and not any(r.trade_count > 0 for r in result.results):
            print(f"ERROR: No trades found in the last {args.days} days")
            return 1

        print_stress_report(result)

    conn.close()

    # Exit code: 0 for all pass, 1 for any fail
    return 0 if result.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
