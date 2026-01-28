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
    python tools/stress_test.py --db hft_trades.db --symbol XRP --days 30
    python tools/stress_test.py --db hft_trades.db --symbol XRP --days 30 --from-best reports/xrp_wf_3way.csv
    python tools/stress_test.py --db hft_trades.db --days 14 --scenario all
    python tools/stress_test.py --db hft_trades.db --json

Output:
    - PASS/FAIL for each stress scenario
    - Degradation metrics (how much PnL decreased)
    - Recommendation for promotion

Options:
    --symbol     Optional symbol filter (e.g., XRP, XRPUSDT)
    --from-best  Pick top candidate row from CSV, print params for reference
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

from hft_system.symbol_utils import normalize_db_symbol
from hft_system.db import open_sqlite


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
    scenario: StressScenario,
    symbol: Optional[str] = None,
    has_symbol_col: bool = False,
    debug: bool = False
) -> StressResult:
    """Run a single stress scenario."""
    config = STRESS_SCENARIOS[scenario]
    cursor = conn.cursor()
    cutoff_ms = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

    # Build query with optional symbol filter
    where_clause = "WHERE status = 'closed' AND entry_time > ?"
    params = [cutoff_ms]
    if symbol and has_symbol_col:
        where_clause += " AND symbol LIKE ?"
        params.append(f"%{symbol}%")

    if debug:
        print(f"\n  [DEBUG] Scenario: {scenario.value}")
        print(f"  [DEBUG] WHERE clause: {where_clause}")
        print(f"  [DEBUG] Params: {params}")

    # Get all closed trades with cost breakdown
    # Use COALESCE for NULL handling and include entry/exit prices for fallback calculation
    cursor.execute(f"""
        SELECT
            COALESCE(pnl_pct, 0) as pnl_pct,
            COALESCE(fees_paid_pct, 0) as fees_paid_pct,
            COALESCE(slippage_pct, 0) as slippage_pct,
            COALESCE(spread_cost_pct, 0) as spread_cost_pct,
            pnl_after_costs_pct,
            entry_price,
            exit_price,
            side
        FROM trades
        {where_clause}
    """, params)

    rows = cursor.fetchall()

    if debug:
        print(f"  [DEBUG] Found {len(rows)} trades")

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
    trades_with_costs = 0

    for i, row in enumerate(rows):
        gross_pnl = row[0] or 0
        fees = row[1] or 0
        slippage = row[2] or 0
        spread = row[3] or 0
        original_net = row[4]
        entry_price = row[5]
        exit_price = row[6]
        side = row[7]

        # Fallback: compute gross PnL from prices if pnl_pct is 0 or NULL
        if gross_pnl == 0 and entry_price and exit_price and entry_price > 0:
            if side and side.lower() == 'short':
                gross_pnl = (entry_price - exit_price) / entry_price * 100
            else:  # Default to long
                gross_pnl = (exit_price - entry_price) / entry_price * 100

        # Fallback: estimate costs if all cost fields are 0
        if fees == 0 and slippage == 0 and spread == 0:
            # Use a reasonable estimate: 0.1% total costs (taker fees ~0.075% + spread)
            estimated_cost = 0.10  # 0.10% per trade
            fees = estimated_cost * 0.6  # ~60% fees
            slippage = estimated_cost * 0.2  # ~20% slippage
            spread = estimated_cost * 0.1  # ~10% spread per side

        if fees > 0 or slippage > 0 or spread > 0:
            trades_with_costs += 1

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

        # Debug: show first 5 trades
        if debug and i < 5:
            print(f"  [DEBUG] Trade {i+1}: gross={gross_pnl:.4f}%, fees={fees:.4f}%, "
                  f"slip={slippage:.4f}%, spread={spread:.4f}%, stressed_pnl={stressed_pnl:.4f}%")

    if debug:
        print(f"  [DEBUG] Trades with cost data: {trades_with_costs}/{trade_count}")
        print(f"  [DEBUG] Total gross PnL: {total_gross_pnl:.4f}%")
        print(f"  [DEBUG] Total stressed PnL: {total_stressed_pnl:.4f}%")

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
    scenarios: List[StressScenario] = None,
    symbol: Optional[str] = None,
    has_symbol_col: bool = False,
    debug: bool = False
) -> StressTestResult:
    """Run all stress test scenarios."""
    if scenarios is None:
        scenarios = list(StressScenario)

    results = []

    # Always run baseline first
    if StressScenario.BASELINE in scenarios:
        scenarios.remove(StressScenario.BASELINE)
        scenarios.insert(0, StressScenario.BASELINE)

    if debug:
        print("\n=== DEBUG MODE ===")

    for scenario in scenarios:
        result = run_stress_scenario(conn, days, scenario, symbol, has_symbol_col, debug)
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


def print_stress_report(result: StressTestResult, symbol: Optional[str] = None,
                        best_params: Optional[Dict] = None):
    """Print formatted stress test report."""
    status = "PASS" if result.all_passed else "FAIL"
    status_icon = "+" if result.all_passed else "x"
    symbol_line = f"| Symbol:           {symbol}" if symbol else "| Symbol:           ALL (global)"

    print(f"""
+================================================================================+
|                        STRATEGY STRESS TEST REPORT                             |
+================================================================================+
{symbol_line}
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

    # Show best params if available
    if best_params:
        print("  --- REFERENCE PARAMS (from --from-best) ---")
        for k, v in best_params.items():
            print(f"  {k}: {v}")
        print()

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


def check_column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    """Check if a column exists in a table."""
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table})")
    columns = [row[1] for row in cursor.fetchall()]
    return column in columns


# normalize_symbol moved to hft_system/symbol_utils.py
# Use normalize_db_symbol from shared module instead


def load_best_from_csv(csv_path: str) -> Optional[Dict]:
    """
    Load best candidate from walk-forward CSV.
    Returns dict with params from top row.
    """
    import csv as csv_module
    try:
        with open(csv_path, 'r') as f:
            reader = csv_module.reader(f)
            rows = list(reader)

        # Find WINDOW DETAILS section
        header_idx = None
        for i, row in enumerate(rows):
            if row and "Window" in row[0] and "TP %" in str(row):
                header_idx = i
                break

        if header_idx is None:
            # Try alternate format - look for parameter frequency section
            for i, row in enumerate(rows):
                if row and "BEST PARAMETERS FREQUENCY" in str(row):
                    # Next non-empty row after header should have params
                    for j in range(i + 2, min(i + 10, len(rows))):
                        if rows[j] and rows[j][0] and "tp=" in rows[j][0].lower():
                            param_str = rows[j][0]
                            # Parse "tp=0.2, sl=0.15, tstop=60" format
                            params = {}
                            for part in param_str.split(","):
                                if "=" in part:
                                    k, v = part.strip().split("=")
                                    k = k.strip().lower()
                                    try:
                                        v = float(v.strip())
                                    except ValueError:
                                        v = v.strip()
                                    params[k] = v
                            if params:
                                return params
                    break

        # Parse first data row after header
        if header_idx is not None and header_idx + 1 < len(rows):
            header = rows[header_idx]
            data_row = rows[header_idx + 1]

            # Find column indices
            tp_idx = next((i for i, h in enumerate(header) if "TP" in str(h).upper()), None)
            sl_idx = next((i for i, h in enumerate(header) if "SL" in str(h).upper()), None)
            ts_idx = next((i for i, h in enumerate(header) if "TS" in str(h).upper() or "Time Stop" in str(h)), None)

            params = {}
            if tp_idx is not None and tp_idx < len(data_row):
                try:
                    params["tp"] = float(data_row[tp_idx])
                except (ValueError, IndexError):
                    pass
            if sl_idx is not None and sl_idx < len(data_row):
                try:
                    params["sl"] = float(data_row[sl_idx])
                except (ValueError, IndexError):
                    pass
            if ts_idx is not None and ts_idx < len(data_row):
                try:
                    params["tstop"] = float(data_row[ts_idx])
                except (ValueError, IndexError):
                    pass

            if params:
                return params

        return None
    except Exception as e:
        print(f"ERROR: Failed to parse CSV: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Strategy Stress Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/stress_test.py --db hft_trades.db --days 7
  python tools/stress_test.py --db hft_trades.db --symbol XRP --days 30
  python tools/stress_test.py --db hft_trades.db --symbol XRP --days 30 --from-best reports/xrp_wf_3way.csv
  python tools/stress_test.py --db hft_trades.db --days 14 --scenario all
  python tools/stress_test.py --db hft_trades.db --scenario fee_plus_50,slippage_x1_5
  python tools/stress_test.py --db hft_trades.db --json
  python tools/stress_test.py --db hft_trades.db --days 7 --debug  # Debug mode

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
    parser.add_argument("--symbol", type=str, default=None,
                        help="Symbol filter (e.g., XRP, XRPUSDT)")
    parser.add_argument("--days", type=int, default=7, help="Analysis period (days)")
    parser.add_argument("--scenario", type=str, default="all",
                        help="Scenarios to run (comma-separated or 'all')")
    parser.add_argument("--from-best", dest="from_best", type=str, default=None,
                        help="CSV file to extract best params from (e.g., reports/xrp_wf_3way.csv)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug output (SQL queries, sample trades)")
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

    # Connect (bot-safe with WAL + busy_timeout)
    conn = open_sqlite(str(db_path))

    # Check for symbol column in trades table
    has_symbol_col = check_column_exists(conn, "trades", "symbol")

    # Normalize symbol filter
    symbol = None
    if args.symbol:
        symbol = normalize_db_symbol(args.symbol)
        if not has_symbol_col:
            print(f"WARNING: --symbol {args.symbol} specified but trades table has no symbol column.")
            print("         Running global analysis (all symbols).")
            symbol = None
        else:
            print(f"Filtering by symbol: {symbol}")

    # Handle --from-best option
    best_params = None
    if args.from_best:
        csv_path = Path(args.from_best)
        if not csv_path.exists():
            print(f"ERROR: CSV file not found: {args.from_best}")
            return 1
        best_params = load_best_from_csv(str(csv_path))
        if best_params:
            print(f"\n  --- BEST PARAMS FROM CSV ---")
            print(f"  Source: {args.from_best}")
            for k, v in best_params.items():
                print(f"  {k}: {v}")
            print()
        else:
            print(f"WARNING: Could not extract params from {args.from_best}")

    # Debug: show table structure if debug mode
    if args.debug:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(trades)")
        columns = cursor.fetchall()
        print("\n  [DEBUG] trades table columns:")
        for col in columns:
            print(f"    {col[1]} ({col[2]})")

        # Show sample of trades
        cutoff_ms = int((datetime.now() - timedelta(days=args.days)).timestamp() * 1000)
        cursor.execute(f"""
            SELECT trade_id, symbol, pnl_pct, fees_paid_pct, slippage_pct, spread_cost_pct, pnl_after_costs_pct
            FROM trades
            WHERE status = 'closed' AND entry_time > ?
            LIMIT 5
        """, (cutoff_ms,))
        sample = cursor.fetchall()
        print(f"\n  [DEBUG] Sample trades (first 5):")
        for row in sample:
            print(f"    {row}")

    # Run stress tests
    result = run_all_stress_tests(conn, args.days, scenarios, symbol, has_symbol_col, args.debug)

    if args.json:
        output = {
            "symbol": symbol or "ALL",
            "period_days": result.period_days,
            "baseline_pnl_pct": result.baseline_pnl_pct,
            "baseline_costs_pct": result.baseline_costs_pct,
            "all_passed": result.all_passed,
            "critical_failures": result.critical_failures,
            "recommendation": result.recommendation,
            "best_params_from_csv": best_params,
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
            symbol_msg = f" for symbol {symbol}" if symbol else ""
            print(f"ERROR: No trades found in the last {args.days} days{symbol_msg}")
            return 1

        print_stress_report(result, symbol, best_params)

    conn.close()

    # Exit code: 0 for all pass, 1 for any fail
    return 0 if result.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
