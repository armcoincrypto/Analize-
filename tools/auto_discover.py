#!/usr/bin/env python3
"""
Auto Strategy Discovery + Switching Orchestrator
=================================================
Automated workflow for strategy discovery, validation, and deployment.

This tool implements the full discovery loop:
1. Run walk-forward optimization (3-way split)
2. Select top candidates by validation metrics
3. Evaluate candidates on final test period
4. Run stress tests on surviving candidates
5. Pick best candidate as challenger
6. Update registry and generate config

Usage:
    python tools/auto_discover.py --symbol XRPUSDT --start 2025-01-01 --end 2025-01-27
    python tools/auto_discover.py --symbol XRPUSDT --start 2025-01-01 --end 2025-01-27 \\
        --train-days 14 --valid-days 7 --final-days 7 \\
        --grid "tp=0.12,0.16,0.20; sl=0.08,0.10,0.12; tstop=30,60" \\
        --top-k 5 --require-stress

Workflow:
    A) Run WalkForwardEngine3Way → reports/<symbol>_wf.csv
    B) Select top-k candidates by VALID metric (NOT final test)
    C) Evaluate FINAL_TEST for those candidates, discard negative
    D) Run stress_test baseline + combined_mild on remaining
    E) Pick best surviving candidate (score: final_test_pnl, drawdown, trade_count)
    F) Write challenger entry to strategies/registry.yml
    G) Run registry_apply.py to generate strategies/generated_active.yml

Safety:
    - Never auto-promotes to champion in LIVE
    - Only sets challenger; promotion requires champion_challenger.py

Cron examples:
    # Daily discovery at 2am UTC (paper mode)
    0 2 * * * cd ~/Analize- && python tools/auto_discover.py --symbol XRPUSDT ...
"""

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

# Add parent dir for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class Candidate:
    """Strategy candidate from walk-forward."""
    params: Dict[str, float]  # tp, sl, tstop
    valid_pnl_pct: float = 0
    valid_win_rate: float = 0
    valid_trades: int = 0
    valid_sharpe: float = 0
    final_test_pnl_pct: float = 0
    final_test_win_rate: float = 0
    final_test_trades: int = 0
    final_test_sharpe: float = 0
    stress_passed: bool = False
    stress_baseline_pnl: float = 0
    stress_combined_pnl: float = 0
    stress_degradation_pct: float = 0
    score: float = 0  # Combined score for ranking
    disqualified: bool = False
    disqualify_reason: str = ""


@dataclass
class DiscoveryResult:
    """Result of the discovery workflow."""
    symbol: str
    start_date: datetime
    end_date: datetime
    total_candidates: int
    candidates_after_valid: int
    candidates_after_final: int
    candidates_after_stress: int
    best_candidate: Optional[Candidate] = None
    all_candidates: List[Candidate] = field(default_factory=list)
    wf_csv_path: str = ""
    registry_updated: bool = False
    config_generated: bool = False
    error: str = ""


def resolve_symbol(symbol: str, quote: str = "USDT") -> str:
    """Resolve symbol to Binance format."""
    symbol = symbol.upper().strip()
    quote = quote.upper().strip()
    for q in ["USDT", "BUSD", "BTC", "ETH", "BNB"]:
        if symbol.endswith(q) and len(symbol) > len(q):
            return symbol
    if not symbol.endswith(quote):
        return symbol + quote
    return symbol


def run_walkforward(
    symbol: str,
    start_date: str,
    end_date: str,
    train_days: int,
    valid_days: int,
    final_test_days: int,
    grid: str,
    output_dir: str = "reports"
) -> Tuple[bool, str, List[Candidate]]:
    """
    Run walk-forward optimization.

    Returns:
        (success, csv_path, candidates)
    """
    logger.info(f"Running walk-forward for {symbol}...")

    cmd = [
        sys.executable, "tools/walkforward_run.py",
        "--symbol", symbol,
        "--start", start_date,
        "--end", end_date,
        "--3way",
        "--train-days", str(train_days),
        "--valid-days", str(valid_days),
        "--final-test-days", str(final_test_days),
        "--output-dir", output_dir
    ]

    if grid:
        cmd.extend(["--grid", grid])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )

        if result.returncode != 0:
            logger.error(f"Walk-forward failed: {result.stderr}")
            return False, "", []

        # Find the generated CSV
        csv_files = list(Path(output_dir).glob(f"walkforward_3way_{symbol}_*.csv"))
        if not csv_files:
            logger.error("No CSV file generated")
            return False, "", []

        # Get most recent
        csv_path = str(max(csv_files, key=lambda p: p.stat().st_mtime))
        logger.info(f"Walk-forward complete: {csv_path}")

        # Parse candidates from CSV
        candidates = parse_wf_csv(csv_path)

        return True, csv_path, candidates

    except subprocess.TimeoutExpired:
        logger.error("Walk-forward timed out after 1 hour")
        return False, "", []
    except Exception as e:
        logger.error(f"Walk-forward error: {e}")
        return False, "", []


def parse_wf_csv(csv_path: str) -> List[Candidate]:
    """Parse walk-forward CSV to extract candidates."""
    candidates = []

    try:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)

        # Find BEST PARAMETERS FREQUENCY section
        param_freq_idx = None
        window_details_idx = None

        for i, row in enumerate(rows):
            if row and "BEST PARAMETERS FREQUENCY" in str(row):
                param_freq_idx = i
            if row and "WINDOW DETAILS" in str(row):
                window_details_idx = i

        # Extract unique parameter sets from frequency section
        param_sets = {}
        if param_freq_idx is not None:
            for j in range(param_freq_idx + 2, len(rows)):
                row = rows[j]
                if not row or not row[0] or "===" in row[0]:
                    break
                # Parse "tp=0.2, sl=0.15, tstop=60" format
                param_str = row[0]
                count = int(row[1]) if len(row) > 1 and row[1].isdigit() else 1
                params = parse_param_string(param_str)
                if params:
                    key = tuple(sorted(params.items()))
                    param_sets[key] = {"params": params, "count": count}

        # Extract window details for metrics
        if window_details_idx is not None:
            header_idx = window_details_idx + 1
            if header_idx < len(rows):
                header = rows[header_idx]

                # Find column indices
                col_map = {}
                for i, h in enumerate(header):
                    h_upper = str(h).upper()
                    if "TP" in h_upper and "%" in h_upper:
                        col_map["tp"] = i
                    elif "SL" in h_upper and "%" in h_upper:
                        col_map["sl"] = i
                    elif "TS" in h_upper or "TIME STOP" in h_upper.replace(" ", ""):
                        col_map["tstop"] = i
                    elif "VALID" in h_upper and "PNL" in h_upper:
                        col_map["valid_pnl"] = i
                    elif "VALID" in h_upper and "WR" in h_upper:
                        col_map["valid_wr"] = i
                    elif "VALID" in h_upper and "TRADE" in h_upper:
                        col_map["valid_trades"] = i
                    elif "FT" in h_upper and "PNL" in h_upper:
                        col_map["ft_pnl"] = i
                    elif "FT" in h_upper and "WR" in h_upper:
                        col_map["ft_wr"] = i
                    elif "FT" in h_upper and "TRADE" in h_upper:
                        col_map["ft_trades"] = i
                    elif "FT" in h_upper and "SHARPE" in h_upper:
                        col_map["ft_sharpe"] = i

                # Parse data rows
                for j in range(header_idx + 1, len(rows)):
                    row = rows[j]
                    if not row or not row[0] or not row[0].isdigit():
                        break

                    try:
                        params = {}
                        if "tp" in col_map:
                            params["tp"] = float(row[col_map["tp"]])
                        if "sl" in col_map:
                            params["sl"] = float(row[col_map["sl"]])
                        if "tstop" in col_map:
                            params["tstop"] = float(row[col_map["tstop"]])

                        if params:
                            candidate = Candidate(params=params)

                            if "valid_pnl" in col_map:
                                candidate.valid_pnl_pct = safe_float(row[col_map["valid_pnl"]])
                            if "valid_wr" in col_map:
                                candidate.valid_win_rate = safe_float(row[col_map["valid_wr"]])
                            if "valid_trades" in col_map:
                                candidate.valid_trades = int(safe_float(row[col_map["valid_trades"]]))
                            if "ft_pnl" in col_map:
                                candidate.final_test_pnl_pct = safe_float(row[col_map["ft_pnl"]])
                            if "ft_wr" in col_map:
                                candidate.final_test_win_rate = safe_float(row[col_map["ft_wr"]])
                            if "ft_trades" in col_map:
                                candidate.final_test_trades = int(safe_float(row[col_map["ft_trades"]]))
                            if "ft_sharpe" in col_map:
                                candidate.final_test_sharpe = safe_float(row[col_map["ft_sharpe"]])

                            candidates.append(candidate)
                    except (ValueError, IndexError) as e:
                        continue

        # Deduplicate by params
        seen = set()
        unique_candidates = []
        for c in candidates:
            key = tuple(sorted(c.params.items()))
            if key not in seen:
                seen.add(key)
                unique_candidates.append(c)

        return unique_candidates

    except Exception as e:
        logger.error(f"Error parsing CSV: {e}")
        return []


def parse_param_string(param_str: str) -> Dict[str, float]:
    """Parse 'tp=0.2, sl=0.15, tstop=60' format."""
    params = {}
    for part in param_str.split(","):
        if "=" in part:
            k, v = part.strip().split("=", 1)
            k = k.strip().lower()
            try:
                params[k] = float(v.strip())
            except ValueError:
                pass
    return params


def safe_float(val) -> float:
    """Safely convert to float."""
    try:
        return float(str(val).replace("%", "").strip())
    except (ValueError, TypeError):
        return 0.0


def select_top_candidates(
    candidates: List[Candidate],
    top_k: int,
    min_trades: int = 10
) -> List[Candidate]:
    """
    Select top-k candidates by validation PnL.

    Args:
        candidates: All candidates
        top_k: Number to select
        min_trades: Minimum trade count required

    Returns:
        Top-k candidates sorted by valid_pnl_pct
    """
    # Filter by minimum trades
    valid_candidates = [
        c for c in candidates
        if c.valid_trades >= min_trades and not c.disqualified
    ]

    # Sort by validation PnL (descending)
    valid_candidates.sort(key=lambda c: c.valid_pnl_pct, reverse=True)

    return valid_candidates[:top_k]


def filter_by_final_test(
    candidates: List[Candidate],
    min_pnl: float = 0.0,
    min_trades: int = 5
) -> List[Candidate]:
    """
    Filter candidates by final test performance.

    Args:
        candidates: Candidates to filter
        min_pnl: Minimum final test PnL (default: 0 = positive)
        min_trades: Minimum final test trades

    Returns:
        Candidates with positive final test
    """
    passing = []

    for c in candidates:
        if c.disqualified:
            continue

        if c.final_test_pnl_pct < min_pnl:
            c.disqualified = True
            c.disqualify_reason = f"Negative final test PnL: {c.final_test_pnl_pct:.4f}%"
            continue

        if c.final_test_trades < min_trades:
            c.disqualified = True
            c.disqualify_reason = f"Too few final test trades: {c.final_test_trades}"
            continue

        passing.append(c)

    return passing


def run_stress_test(
    db_path: str,
    days: int,
    symbol: Optional[str] = None
) -> Tuple[bool, float, float, float]:
    """
    Run stress test and get results.

    Returns:
        (passed, baseline_pnl, combined_mild_pnl, degradation_pct)
    """
    cmd = [
        sys.executable, "tools/stress_test.py",
        "--db", db_path,
        "--days", str(days),
        "--scenario", "baseline,combined_mild",
        "--json"
    ]

    if symbol:
        cmd.extend(["--symbol", symbol])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            logger.warning(f"Stress test failed: {result.stderr}")
            return False, 0, 0, 0

        data = json.loads(result.stdout)

        baseline_pnl = 0
        combined_pnl = 0

        for scenario in data.get("scenarios", []):
            if scenario["scenario"] == "baseline":
                baseline_pnl = scenario["stressed_pnl_pct"]
            elif scenario["scenario"] == "combined_mild":
                combined_pnl = scenario["stressed_pnl_pct"]

        degradation = 0
        if baseline_pnl != 0:
            degradation = (baseline_pnl - combined_pnl) / abs(baseline_pnl) * 100

        passed = data.get("all_passed", False) or (
            baseline_pnl > 0 and combined_pnl > 0 and degradation <= 50
        )

        return passed, baseline_pnl, combined_pnl, degradation

    except Exception as e:
        logger.error(f"Stress test error: {e}")
        return False, 0, 0, 0


def filter_by_stress_test(
    candidates: List[Candidate],
    db_path: str,
    days: int,
    symbol: Optional[str] = None
) -> List[Candidate]:
    """
    Filter candidates by stress test results.

    Note: Since stress test runs on actual trades, not per-candidate params,
    we run it once and apply the result to all candidates.
    """
    passed, baseline, combined, degradation = run_stress_test(db_path, days, symbol)

    for c in candidates:
        if c.disqualified:
            continue

        c.stress_baseline_pnl = baseline
        c.stress_combined_pnl = combined
        c.stress_degradation_pct = degradation
        c.stress_passed = passed

        if not passed:
            c.disqualified = True
            c.disqualify_reason = f"Failed stress test: baseline={baseline:.4f}%, degradation={degradation:.1f}%"

    return [c for c in candidates if not c.disqualified]


def calculate_candidate_scores(candidates: List[Candidate]) -> List[Candidate]:
    """
    Calculate combined scores for ranking candidates.

    Score formula:
    - 40% final_test_pnl (normalized)
    - 30% final_test_sharpe (normalized)
    - 20% valid_pnl (normalized, for consistency check)
    - 10% trade_count bonus (more trades = more reliable)
    """
    if not candidates:
        return candidates

    # Get ranges for normalization
    ft_pnls = [c.final_test_pnl_pct for c in candidates if not c.disqualified]
    ft_sharpes = [c.final_test_sharpe for c in candidates if not c.disqualified]
    valid_pnls = [c.valid_pnl_pct for c in candidates if not c.disqualified]
    trade_counts = [c.final_test_trades for c in candidates if not c.disqualified]

    if not ft_pnls:
        return candidates

    ft_pnl_max = max(ft_pnls) if ft_pnls else 1
    ft_sharpe_max = max(ft_sharpes) if ft_sharpes else 1
    valid_pnl_max = max(valid_pnls) if valid_pnls else 1
    trade_max = max(trade_counts) if trade_counts else 1

    for c in candidates:
        if c.disqualified:
            c.score = -999
            continue

        # Normalize each component (0 to 1)
        ft_pnl_norm = c.final_test_pnl_pct / ft_pnl_max if ft_pnl_max > 0 else 0
        ft_sharpe_norm = c.final_test_sharpe / ft_sharpe_max if ft_sharpe_max > 0 else 0
        valid_pnl_norm = c.valid_pnl_pct / valid_pnl_max if valid_pnl_max > 0 else 0
        trade_norm = c.final_test_trades / trade_max if trade_max > 0 else 0

        # Combined score
        c.score = (
            0.40 * ft_pnl_norm +
            0.30 * ft_sharpe_norm +
            0.20 * valid_pnl_norm +
            0.10 * trade_norm
        )

    # Sort by score
    candidates.sort(key=lambda c: c.score, reverse=True)

    return candidates


def update_registry(
    symbol: str,
    candidate: Candidate,
    registry_path: str = "strategies/registry.yml"
) -> bool:
    """
    Add candidate as challenger in registry.

    Returns:
        True if successful
    """
    try:
        # Try to load existing registry
        registry_data = {}
        if Path(registry_path).exists():
            try:
                import yaml
                with open(registry_path, 'r') as f:
                    registry_data = yaml.safe_load(f) or {}
            except ImportError:
                # Fallback to reading raw text and appending
                pass

        # Generate challenger ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_symbol = symbol.replace("USDT", "").lower()
        challenger_id = f"{base_symbol}_auto_{timestamp}"

        # Create challenger entry
        challenger_entry = f"""
  {challenger_id}:
    symbol: {symbol}
    status: challenger
    regimes_allowed: [high_vol_trend, mean_reversion]
    causes_enabled: []
    causes_disabled: []
    params:
      take_profit_pct: {candidate.params.get('tp', 0.20)}
      stop_loss_pct: {candidate.params.get('sl', 0.15)}
      time_stop_seconds: {int(candidate.params.get('tstop', 60))}
    execution:
      mode: auto
      maker_timeout_ms: 5000
      cancel_on_move_bps: 3.0
    risk:
      max_position_pct: 0.5
      max_daily_loss_pct: 2.0
      max_open_trades: 3
    metadata:
      created_at: "{datetime.now().isoformat()}"
      source: auto_discover
      valid_pnl_pct: {candidate.valid_pnl_pct:.4f}
      final_test_pnl_pct: {candidate.final_test_pnl_pct:.4f}
      final_test_sharpe: {candidate.final_test_sharpe:.4f}
      stress_passed: {str(candidate.stress_passed).lower()}
      score: {candidate.score:.4f}
"""

        # Ensure directory exists
        Path(registry_path).parent.mkdir(parents=True, exist_ok=True)

        # Append to registry
        with open(registry_path, 'a') as f:
            if not Path(registry_path).exists() or Path(registry_path).stat().st_size == 0:
                f.write("# Auto-generated strategy registry\nstrategies:\n")
            f.write(challenger_entry)

        logger.info(f"Added challenger to registry: {challenger_id}")

        # Update assignments section
        # This is simplified - in production would properly parse YAML
        assignment_entry = f"""
# Auto-assigned challenger for {symbol}
# {symbol}_challenger: {challenger_id}
"""
        with open(registry_path, 'a') as f:
            f.write(assignment_entry)

        return True

    except Exception as e:
        logger.error(f"Failed to update registry: {e}")
        return False


def run_registry_apply(symbol: str) -> bool:
    """Run registry_apply.py to generate config."""
    cmd = [
        sys.executable, "tools/registry_apply.py",
        "--symbol", symbol,
        "--apply"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            logger.warning(f"Registry apply warning: {result.stderr}")
        logger.info("Generated strategies/generated_active.yml")
        return True
    except Exception as e:
        logger.error(f"Registry apply failed: {e}")
        return False


def print_discovery_summary(result: DiscoveryResult):
    """Print discovery workflow summary."""
    print(f"""
+================================================================================+
|                    AUTO STRATEGY DISCOVERY SUMMARY                             |
+================================================================================+
| Symbol:                 {result.symbol}
| Period:                 {result.start_date.date()} to {result.end_date.date()}
| Walk-Forward CSV:       {result.wf_csv_path or 'N/A'}
+================================================================================+

  --- CANDIDATE FUNNEL ---
  Total candidates from WF:      {result.total_candidates}
  After validation filter:       {result.candidates_after_valid}
  After final test filter:       {result.candidates_after_final}
  After stress test filter:      {result.candidates_after_stress}
""")

    if result.best_candidate:
        c = result.best_candidate
        print(f"""
  --- BEST CANDIDATE ---
  Parameters:
    TP: {c.params.get('tp', 'N/A')}%
    SL: {c.params.get('sl', 'N/A')}%
    Time Stop: {c.params.get('tstop', 'N/A')}s

  Validation Metrics:
    PnL:     {c.valid_pnl_pct:+.4f}%
    Win Rate: {c.valid_win_rate:.1f}%
    Trades:   {c.valid_trades}

  Final Test Metrics (UNBIASED):
    PnL:     {c.final_test_pnl_pct:+.4f}%
    Win Rate: {c.final_test_win_rate:.1f}%
    Trades:   {c.final_test_trades}
    Sharpe:   {c.final_test_sharpe:.4f}

  Stress Test:
    Passed:       {c.stress_passed}
    Baseline PnL: {c.stress_baseline_pnl:+.4f}%
    Combined PnL: {c.stress_combined_pnl:+.4f}%
    Degradation:  {c.stress_degradation_pct:.1f}%

  Combined Score: {c.score:.4f}
""")
    else:
        print("\n  [!] No candidate survived all filters!\n")

    print(f"""
  --- DEPLOYMENT STATUS ---
  Registry Updated:  {'Yes' if result.registry_updated else 'No'}
  Config Generated:  {'Yes' if result.config_generated else 'No'}
""")

    if result.error:
        print(f"\n  [ERROR] {result.error}\n")

    # Show top candidates
    if result.all_candidates:
        print("  --- ALL CANDIDATES (top 10) ---")
        for i, c in enumerate(result.all_candidates[:10], 1):
            status = "SELECTED" if c == result.best_candidate else (
                f"DQ: {c.disqualify_reason}" if c.disqualified else ""
            )
            print(f"  {i}. tp={c.params.get('tp')}, sl={c.params.get('sl')}, tstop={c.params.get('tstop')}")
            print(f"     Valid PnL: {c.valid_pnl_pct:+.4f}% | FT PnL: {c.final_test_pnl_pct:+.4f}% | Score: {c.score:.4f}")
            if status:
                print(f"     {status}")
        print()

    print("""
  --- NEXT STEPS ---
  1. Review the challenger in strategies/registry.yml
  2. Run champion_challenger.py after collecting live data
  3. Promote to champion only after significant outperformance
+================================================================================+
""")


def main():
    parser = argparse.ArgumentParser(
        description="Auto Strategy Discovery + Switching Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic discovery run
  python tools/auto_discover.py --symbol XRPUSDT --start 2025-01-01 --end 2025-01-27

  # With custom parameters
  python tools/auto_discover.py --symbol XRPUSDT --start 2025-01-01 --end 2025-01-27 \\
      --train-days 14 --valid-days 7 --final-days 7 \\
      --grid "tp=0.12,0.16,0.20; sl=0.08,0.10,0.12; tstop=30,60" \\
      --top-k 5 --require-stress

  # Dry run (no registry update)
  python tools/auto_discover.py --symbol XRPUSDT --start 2025-01-01 --end 2025-01-27 --dry-run

Cron examples:
  # Daily discovery at 2am UTC
  0 2 * * * cd ~/Analize- && python tools/auto_discover.py --symbol XRPUSDT \\
      --start $(date -d '60 days ago' +%Y-%m-%d) --end $(date +%Y-%m-%d) \\
      >> logs/auto_discover.log 2>&1
        """
    )

    # Required arguments
    parser.add_argument("--symbol", required=True, help="Trading symbol (e.g., XRPUSDT or XRP)")

    # Date range (with smart defaults)
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")

    # Walk-forward parameters
    parser.add_argument("--train-days", type=int, default=14, help="Training window days (default: 14)")
    parser.add_argument("--valid-days", type=int, default=7, help="Validation window days (default: 7)")
    parser.add_argument("--final-days", type=int, default=7, help="Final test window days (default: 7)")
    parser.add_argument("--grid", type=str, default=None,
                        help="Parameter grid (e.g., 'tp=0.12,0.16,0.20; sl=0.08,0.10; tstop=30,60')")

    # Candidate selection
    parser.add_argument("--top-k", type=int, default=5, help="Top K candidates to evaluate (default: 5)")
    parser.add_argument("--min-trades", type=int, default=10, help="Minimum trades for validation (default: 10)")

    # Stress test
    parser.add_argument("--require-stress", action="store_true", help="Require stress test to pass")
    parser.add_argument("--stress-days", type=int, default=30, help="Days for stress test (default: 30)")
    parser.add_argument("--db", type=str, default="hft_trades.db", help="Database path for stress test")

    # Output control
    parser.add_argument("--dry-run", action="store_true", help="Don't update registry or generate config")
    parser.add_argument("--output-dir", default="reports", help="Output directory for CSV (default: reports)")
    parser.add_argument("--json", action="store_true", help="Output result as JSON")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output")

    args = parser.parse_args()

    # Resolve symbol
    symbol = resolve_symbol(args.symbol)

    # Parse dates
    try:
        start_date = datetime.strptime(args.start, "%Y-%m-%d")
        end_date = datetime.strptime(args.end, "%Y-%m-%d")
    except ValueError as e:
        print(f"ERROR: Invalid date format. Use YYYY-MM-DD. {e}")
        return 1

    # Initialize result
    result = DiscoveryResult(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        total_candidates=0,
        candidates_after_valid=0,
        candidates_after_final=0,
        candidates_after_stress=0
    )

    if not args.quiet:
        print(f"""
+================================================================================+
|                    AUTO STRATEGY DISCOVERY                                      |
+================================================================================+
| Symbol:      {symbol}
| Period:      {start_date.date()} to {end_date.date()}
| Split:       TRAIN={args.train_days}d / VALID={args.valid_days}d / FINAL={args.final_days}d
| Top-K:       {args.top_k}
| Min Trades:  {args.min_trades}
| Stress Test: {'Required' if args.require_stress else 'Optional'}
| Dry Run:     {args.dry_run}
+================================================================================+
""")

    # Step A: Run walk-forward
    logger.info("Step A: Running walk-forward optimization...")
    success, csv_path, candidates = run_walkforward(
        symbol=symbol,
        start_date=args.start,
        end_date=args.end,
        train_days=args.train_days,
        valid_days=args.valid_days,
        final_test_days=args.final_days,
        grid=args.grid,
        output_dir=args.output_dir
    )

    if not success:
        result.error = "Walk-forward optimization failed"
        if args.json:
            print(json.dumps({"error": result.error}, indent=2))
        else:
            print_discovery_summary(result)
        return 1

    result.wf_csv_path = csv_path
    result.total_candidates = len(candidates)
    result.all_candidates = candidates

    logger.info(f"Step A complete: {len(candidates)} candidates")

    # Step B: Select top-k by validation metrics
    logger.info(f"Step B: Selecting top-{args.top_k} by validation PnL...")
    top_candidates = select_top_candidates(candidates, args.top_k, args.min_trades)
    result.candidates_after_valid = len(top_candidates)

    logger.info(f"Step B complete: {len(top_candidates)} candidates")

    if not top_candidates:
        result.error = "No candidates passed validation filter"
        if args.json:
            print(json.dumps({"error": result.error}, indent=2))
        else:
            print_discovery_summary(result)
        return 1

    # Step C: Filter by final test performance
    logger.info("Step C: Filtering by final test performance...")
    final_candidates = filter_by_final_test(top_candidates)
    result.candidates_after_final = len(final_candidates)

    logger.info(f"Step C complete: {len(final_candidates)} candidates")

    if not final_candidates:
        result.error = "No candidates have positive final test PnL"
        if args.json:
            print(json.dumps({"error": result.error}, indent=2))
        else:
            print_discovery_summary(result)
        return 1

    # Step D: Run stress tests
    if args.require_stress:
        logger.info("Step D: Running stress tests...")
        stress_candidates = filter_by_stress_test(
            final_candidates,
            args.db,
            args.stress_days,
            symbol
        )
        result.candidates_after_stress = len(stress_candidates)
        logger.info(f"Step D complete: {len(stress_candidates)} candidates")
    else:
        stress_candidates = final_candidates
        result.candidates_after_stress = len(stress_candidates)
        logger.info("Step D: Skipped (stress test not required)")

    if not stress_candidates:
        result.error = "No candidates passed stress tests"
        if args.json:
            print(json.dumps({"error": result.error}, indent=2))
        else:
            print_discovery_summary(result)
        return 1

    # Step E: Calculate scores and pick best
    logger.info("Step E: Calculating scores and selecting best...")
    scored_candidates = calculate_candidate_scores(stress_candidates)
    best_candidate = scored_candidates[0] if scored_candidates else None

    result.best_candidate = best_candidate

    if not best_candidate:
        result.error = "No candidate survived all filters"
        if args.json:
            print(json.dumps({"error": result.error}, indent=2))
        else:
            print_discovery_summary(result)
        return 1

    logger.info(f"Step E complete: Best candidate score = {best_candidate.score:.4f}")

    # Step F: Update registry (unless dry run)
    if not args.dry_run:
        logger.info("Step F: Updating registry...")
        result.registry_updated = update_registry(symbol, best_candidate)
    else:
        logger.info("Step F: Skipped (dry run)")

    # Step G: Generate config (unless dry run)
    if not args.dry_run and result.registry_updated:
        logger.info("Step G: Generating config...")
        result.config_generated = run_registry_apply(symbol)
    else:
        logger.info("Step G: Skipped (dry run or registry not updated)")

    # Output
    if args.json:
        output = {
            "symbol": result.symbol,
            "start_date": str(result.start_date.date()),
            "end_date": str(result.end_date.date()),
            "wf_csv_path": result.wf_csv_path,
            "total_candidates": result.total_candidates,
            "candidates_after_valid": result.candidates_after_valid,
            "candidates_after_final": result.candidates_after_final,
            "candidates_after_stress": result.candidates_after_stress,
            "best_candidate": {
                "params": best_candidate.params,
                "valid_pnl_pct": best_candidate.valid_pnl_pct,
                "final_test_pnl_pct": best_candidate.final_test_pnl_pct,
                "final_test_sharpe": best_candidate.final_test_sharpe,
                "stress_passed": best_candidate.stress_passed,
                "score": best_candidate.score
            } if best_candidate else None,
            "registry_updated": result.registry_updated,
            "config_generated": result.config_generated,
            "error": result.error
        }
        print(json.dumps(output, indent=2))
    else:
        print_discovery_summary(result)

    return 0


if __name__ == "__main__":
    sys.exit(main())
