#!/usr/bin/env python3
"""
Walk-Forward Backtest CLI
=========================
Run walk-forward optimization with parameter sweep.

Usage:
    python tools/walkforward_run.py --symbol XRPUSDT --start 2024-01-01 --end 2024-02-01
    python tools/walkforward_run.py --db hft_trades.db --symbol XRP --final-days 7 --3way
    python tools/walkforward_run.py --symbol XRPUSDT --start 2024-01-01 --end 2024-02-01 \
        --train-days 14 --test-days 7 \
        --grid "tp=0.2,0.3,0.4; sl=0.1,0.15; tstop=30,60"

Output:
    - CSV leaderboard: reports/walkforward_<symbol>_<timestamp>.csv
    - Console summary with best parameters

Requirements:
    - Requires historical candle data from Binance API
    - Or cached data in a local file

Options:
    --db           Optional database path (currently uses Binance API for candles)
    --final-days   Alias for --final-test-days (3-way split mode)
"""

import argparse
import csv
import logging
import sys
import os
import time
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict

# Add parent dir for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from hft_system.research.walkforward import (
    WalkForwardEngine,
    WalkForwardEngine3Way,
    ParameterGrid,
    walk_forward_run,
    walk_forward_run_3way,
    WalkForwardResult,
    WalkForwardResult3Way,
)
from hft_system.research.metrics import metrics_to_dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def resolve_symbol(symbol: str, quote: str = "USDT") -> str:
    """
    Resolve symbol to Binance format.

    Args:
        symbol: Input symbol (e.g., "XRP", "XRPUSDT", "xrp")
        quote: Quote currency to append if missing (default: "USDT")

    Returns:
        Resolved symbol (e.g., "XRPUSDT")

    Examples:
        >>> resolve_symbol("XRP")
        'XRPUSDT'
        >>> resolve_symbol("XRPUSDT")
        'XRPUSDT'
        >>> resolve_symbol("btc", "USDT")
        'BTCUSDT'
        >>> resolve_symbol("ETH", "BTC")
        'ETHBTC'
    """
    symbol = symbol.upper().strip()
    quote = quote.upper().strip()

    # Remove common quote currencies to get base
    for q in ["USDT", "BUSD", "BTC", "ETH", "BNB"]:
        if symbol.endswith(q) and len(symbol) > len(q):
            # Already has a quote currency
            return symbol

    # Append quote if not present
    if not symbol.endswith(quote):
        return symbol + quote
    return symbol


def test_symbol_resolution():
    """Unit tests for symbol resolution."""
    test_cases = [
        ("XRP", "USDT", "XRPUSDT"),
        ("xrp", "USDT", "XRPUSDT"),
        ("XRPUSDT", "USDT", "XRPUSDT"),
        ("BTC", "USDT", "BTCUSDT"),
        ("ETH", "BTC", "ETHBTC"),
        ("ETHBTC", "BTC", "ETHBTC"),
        ("sol", "usdt", "SOLUSDT"),
        ("SUIUSDT", "USDT", "SUIUSDT"),
    ]

    all_passed = True
    for input_sym, quote, expected in test_cases:
        result = resolve_symbol(input_sym, quote)
        status = "PASS" if result == expected else "FAIL"
        if result != expected:
            all_passed = False
        print(f"  [{status}] resolve_symbol('{input_sym}', '{quote}') = '{result}' (expected: '{expected}')")

    return all_passed


def fetch_candles_from_binance(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    interval: str = "1m",
    auto_quote: bool = True
) -> List[Dict]:
    """
    Fetch historical candles from Binance API.

    Args:
        symbol: Trading pair (e.g., "XRPUSDT")
        start_date: Start datetime
        end_date: End datetime
        interval: Candle interval (1m, 5m, etc.)
        auto_quote: If True, retry with USDT suffix on 400 error

    Returns:
        List of candle dicts
    """
    base_url = "https://api.binance.com/api/v3/klines"
    candles = []

    start_ms = int(start_date.timestamp() * 1000)
    end_ms = int(end_date.timestamp() * 1000)

    logger.info(f"Fetching {symbol} candles from {start_date.date()} to {end_date.date()}...")

    # Track if we've tried the original symbol
    original_symbol = symbol
    tried_with_usdt = False

    while start_ms < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": 1000
        }

        try:
            response = requests.get(base_url, params=params, timeout=30)

            # Handle 400 error - try with USDT suffix
            if response.status_code == 400 and auto_quote and not tried_with_usdt:
                error_msg = response.json().get("msg", "")
                logger.warning(f"Binance API returned 400 for symbol '{symbol}': {error_msg}")

                # Try adding USDT if not present
                if not symbol.endswith("USDT"):
                    new_symbol = symbol + "USDT"
                    logger.info(f"Retrying with quote currency: {symbol} -> {new_symbol}")
                    symbol = new_symbol
                    tried_with_usdt = True
                    params["symbol"] = symbol
                    response = requests.get(base_url, params=params, timeout=30)

            response.raise_for_status()
            data = response.json()

            if not data:
                break

            for k in data:
                candles.append({
                    "timestamp": k[0],
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5])
                })

            # Move to next batch
            start_ms = data[-1][0] + 1

            # Rate limit
            time.sleep(0.2)

        except requests.exceptions.RequestException as e:
            logger.error(f"API error: {e}")
            if "400" in str(e) and auto_quote and not tried_with_usdt and not symbol.endswith("USDT"):
                new_symbol = symbol + "USDT"
                logger.info(f"Retrying with quote currency: {symbol} -> {new_symbol}")
                symbol = new_symbol
                tried_with_usdt = True
                continue
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            break

    if symbol != original_symbol:
        logger.info(f"Successfully fetched using resolved symbol: {symbol}")

    logger.info(f"Fetched {len(candles)} candles for {symbol}")
    return candles, symbol  # Return resolved symbol too


def export_results_to_csv(result: WalkForwardResult, filepath: str):
    """
    Export walk-forward results to CSV.

    Args:
        result: WalkForwardResult
        filepath: Output CSV path
    """
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header section
        writer.writerow(["Walk-Forward Optimization Results"])
        writer.writerow(["Symbol", result.symbol])
        writer.writerow(["Period", f"{result.start_date.date()} to {result.end_date.date()}"])
        writer.writerow(["Train Days", result.train_days])
        writer.writerow(["Test Days", result.test_days])
        writer.writerow(["Total Windows", result.total_windows])
        writer.writerow([])

        # Aggregate OOS metrics
        writer.writerow(["=== OUT-OF-SAMPLE AGGREGATE METRICS ==="])
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Total Trades", result.oos_trade_count])
        writer.writerow(["Win Rate %", f"{result.oos_win_rate:.2f}"])
        writer.writerow(["Total PnL %", f"{result.oos_total_pnl_pct:.4f}"])
        writer.writerow(["Avg PnL %", f"{result.oos_avg_pnl_pct:.4f}"])
        writer.writerow(["Max Drawdown %", f"{result.oos_max_drawdown_pct:.4f}"])
        writer.writerow(["Sharpe Ratio", f"{result.oos_sharpe_ratio:.4f}"])
        writer.writerow([])

        # Parameter frequency
        writer.writerow(["=== BEST PARAMETERS FREQUENCY ==="])
        writer.writerow(["Parameters", "Times Selected"])
        for params, count in sorted(result.best_params_frequency.items(), key=lambda x: -x[1]):
            writer.writerow([params, count])
        writer.writerow([])

        # Window details
        writer.writerow(["=== WINDOW DETAILS ==="])
        headers = [
            "Window", "Train Start", "Train End", "Test Start", "Test End",
            "TP %", "SL %", "Time Stop (s)",
            "Train Trades", "Train WR %", "Train PnL %",
            "Test Trades", "Test WR %", "Test PnL %", "Test Sharpe"
        ]
        writer.writerow(headers)

        for i, w in enumerate(result.window_results, 1):
            row = [
                i,
                w.train_start.date(),
                w.train_end.date(),
                w.test_start.date(),
                w.test_end.date(),
                w.best_params.take_profit_pct,
                w.best_params.stop_loss_pct,
                w.best_params.time_stop_seconds,
                w.train_metrics.trade_count,
                f"{w.train_metrics.win_rate:.1f}",
                f"{w.train_metrics.avg_pnl_after_costs_pct:.4f}",
                w.test_metrics.trade_count,
                f"{w.test_metrics.win_rate:.1f}",
                f"{w.test_metrics.avg_pnl_after_costs_pct:.4f}",
                f"{w.test_metrics.sharpe_ratio:.4f}"
            ]
            writer.writerow(row)

    logger.info(f"Results exported to: {filepath}")


def export_results_to_csv_3way(result: WalkForwardResult3Way, filepath: str):
    """
    Export 3-way walk-forward results to CSV.

    Args:
        result: WalkForwardResult3Way
        filepath: Output CSV path
    """
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header section
        writer.writerow(["Walk-Forward Optimization Results (3-Way Split)"])
        writer.writerow(["Symbol", result.symbol])
        writer.writerow(["Period", f"{result.start_date.date()} to {result.end_date.date()}"])
        writer.writerow(["Train Days", result.train_days])
        writer.writerow(["Valid Days", result.valid_days])
        writer.writerow(["Final Test Days", result.final_test_days])
        writer.writerow(["Total Windows", result.total_windows])
        writer.writerow([])

        # Multiple comparisons warning
        writer.writerow(["=== MULTIPLE COMPARISONS WARNING ==="])
        writer.writerow(["Total Configs Tried", result.total_configs_tried])
        writer.writerow(["Unique Configs", result.unique_configs])
        writer.writerow(["Bonferroni Alpha", f"{result.bonferroni_alpha:.6f}"])
        writer.writerow(["Warning", result.multiple_comparisons_warning])
        writer.writerow([])

        # VALIDATION metrics (used for selection)
        writer.writerow(["=== VALIDATION METRICS (used for selection) ==="])
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Total Trades", result.valid_trade_count])
        writer.writerow(["Win Rate %", f"{result.valid_win_rate:.2f}"])
        writer.writerow(["Total PnL %", f"{result.valid_total_pnl_pct:.4f}"])
        writer.writerow(["Avg PnL %", f"{result.valid_avg_pnl_pct:.4f}"])
        writer.writerow(["Max Drawdown %", f"{result.valid_max_drawdown_pct:.4f}"])
        writer.writerow(["Sharpe Ratio", f"{result.valid_sharpe_ratio:.4f}"])
        writer.writerow([])

        # FINAL TEST metrics (unbiased - NOT used for selection)
        writer.writerow(["=== FINAL TEST METRICS (UNBIASED - not used for selection) ==="])
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Total Trades", result.final_test_trade_count])
        writer.writerow(["Win Rate %", f"{result.final_test_win_rate:.2f}"])
        writer.writerow(["Total PnL %", f"{result.final_test_total_pnl_pct:.4f}"])
        writer.writerow(["Avg PnL %", f"{result.final_test_avg_pnl_pct:.4f}"])
        writer.writerow(["Max Drawdown %", f"{result.final_test_max_drawdown_pct:.4f}"])
        writer.writerow(["Sharpe Ratio", f"{result.final_test_sharpe_ratio:.4f}"])
        writer.writerow([])

        # Parameter frequency
        writer.writerow(["=== BEST PARAMETERS FREQUENCY ==="])
        writer.writerow(["Parameters", "Times Selected"])
        for params, count in sorted(result.best_params_frequency.items(), key=lambda x: -x[1]):
            writer.writerow([params, count])
        writer.writerow([])

        # Window details
        writer.writerow(["=== WINDOW DETAILS ==="])
        headers = [
            "Window", "Train Start", "Train End", "Valid Start", "Valid End", "FT Start", "FT End",
            "TP %", "SL %", "TS (s)", "Configs",
            "Train Trades", "Train WR %", "Train PnL %",
            "Valid Trades", "Valid WR %", "Valid PnL %",
            "FT Trades", "FT WR %", "FT PnL %", "FT Sharpe"
        ]
        writer.writerow(headers)

        for i, w in enumerate(result.window_results, 1):
            row = [
                i,
                w.train_start.date(),
                w.train_end.date(),
                w.valid_start.date(),
                w.valid_end.date(),
                w.final_test_start.date(),
                w.final_test_end.date(),
                w.best_params.take_profit_pct,
                w.best_params.stop_loss_pct,
                w.best_params.time_stop_seconds,
                w.configs_tried,
                w.train_metrics.trade_count,
                f"{w.train_metrics.win_rate:.1f}",
                f"{w.train_metrics.avg_pnl_after_costs_pct:.4f}",
                w.valid_metrics.trade_count,
                f"{w.valid_metrics.win_rate:.1f}",
                f"{w.valid_metrics.avg_pnl_after_costs_pct:.4f}",
                w.final_test_metrics.trade_count,
                f"{w.final_test_metrics.win_rate:.1f}",
                f"{w.final_test_metrics.avg_pnl_after_costs_pct:.4f}",
                f"{w.final_test_metrics.sharpe_ratio:.4f}"
            ]
            writer.writerow(row)

    logger.info(f"Results exported to: {filepath}")


def print_summary(result: WalkForwardResult):
    """Print summary to console."""
    print("\n" + "=" * 70)
    print(" WALK-FORWARD OPTIMIZATION RESULTS")
    print("=" * 70)
    print(f" Symbol: {result.symbol}")
    print(f" Period: {result.start_date.date()} to {result.end_date.date()}")
    print(f" Train/Test: {result.train_days}/{result.test_days} days")
    print(f" Windows: {result.total_windows}")
    print()

    print(" --- OUT-OF-SAMPLE AGGREGATE METRICS ---")
    print(f" Total Trades:    {result.oos_trade_count}")
    print(f" Win Rate:        {result.oos_win_rate:.1f}%")
    print(f" Total PnL:       {result.oos_total_pnl_pct:+.4f}%")
    print(f" Avg PnL/Trade:   {result.oos_avg_pnl_pct:+.4f}%")
    print(f" Max Drawdown:    {result.oos_max_drawdown_pct:.4f}%")
    print(f" Sharpe Ratio:    {result.oos_sharpe_ratio:.4f}")
    print()

    print(" --- BEST PARAMETERS (most frequently selected) ---")
    sorted_params = sorted(result.best_params_frequency.items(), key=lambda x: -x[1])
    for i, (params, count) in enumerate(sorted_params[:5], 1):
        pct = count / result.total_windows * 100
        print(f" {i}. {params}")
        print(f"    Selected: {count}/{result.total_windows} times ({pct:.0f}%)")
    print()

    # Evaluation
    print(" --- EVALUATION ---")
    if result.oos_avg_pnl_pct > 0:
        print(f" Positive OOS PnL: Strategy shows promise")
        if result.oos_sharpe_ratio > 0.5:
            print(f" Good Sharpe (>{0.5}): Risk-adjusted returns are reasonable")
        else:
            print(f" Low Sharpe (<0.5): High variance in returns")
    else:
        print(f" Negative OOS PnL: Strategy needs improvement")

    if result.oos_max_drawdown_pct > 5:
        print(f" WARNING: Max drawdown {result.oos_max_drawdown_pct:.1f}% is high")

    print("=" * 70)


def print_summary_3way(result: WalkForwardResult3Way):
    """Print summary for 3-way walk-forward to console."""
    print("\n" + "=" * 80)
    print(" WALK-FORWARD OPTIMIZATION RESULTS (3-WAY SPLIT)")
    print("=" * 80)
    print(f" Symbol: {result.symbol}")
    print(f" Period: {result.start_date.date()} to {result.end_date.date()}")
    print(f" Split:  TRAIN={result.train_days}d / VALID={result.valid_days}d / FINAL_TEST={result.final_test_days}d")
    print(f" Windows: {result.total_windows}")
    print()

    # Multiple comparisons warning
    print(" --- MULTIPLE COMPARISONS WARNING ---")
    print(f" Configs Tested: {result.total_configs_tried} ({result.unique_configs} unique)")
    print(f" Bonferroni Alpha: {result.bonferroni_alpha:.6f}")
    print(f" {result.multiple_comparisons_warning}")
    print()

    # Validation metrics (used for selection)
    print(" --- VALIDATION METRICS (used for param selection) ---")
    print(f" Total Trades:    {result.valid_trade_count}")
    print(f" Win Rate:        {result.valid_win_rate:.1f}%")
    print(f" Total PnL:       {result.valid_total_pnl_pct:+.4f}%")
    print(f" Avg PnL/Trade:   {result.valid_avg_pnl_pct:+.4f}%")
    print(f" Max Drawdown:    {result.valid_max_drawdown_pct:.4f}%")
    print(f" Sharpe Ratio:    {result.valid_sharpe_ratio:.4f}")
    print()

    # Final test metrics (unbiased)
    print(" --- FINAL TEST METRICS (UNBIASED - not used for selection) ---")
    print(f" Total Trades:    {result.final_test_trade_count}")
    print(f" Win Rate:        {result.final_test_win_rate:.1f}%")
    print(f" Total PnL:       {result.final_test_total_pnl_pct:+.4f}%")
    print(f" Avg PnL/Trade:   {result.final_test_avg_pnl_pct:+.4f}%")
    print(f" Max Drawdown:    {result.final_test_max_drawdown_pct:.4f}%")
    print(f" Sharpe Ratio:    {result.final_test_sharpe_ratio:.4f}")
    print()

    print(" --- BEST PARAMETERS (most frequently selected) ---")
    sorted_params = sorted(result.best_params_frequency.items(), key=lambda x: -x[1])
    for i, (params, count) in enumerate(sorted_params[:5], 1):
        pct = count / result.total_windows * 100
        print(f" {i}. {params}")
        print(f"    Selected: {count}/{result.total_windows} times ({pct:.0f}%)")
    print()

    # Evaluation
    print(" --- EVALUATION ---")

    # Compare validation vs final test (detect overfitting)
    valid_pnl = result.valid_avg_pnl_pct
    final_pnl = result.final_test_avg_pnl_pct

    if valid_pnl > 0 and final_pnl > 0:
        degradation = (valid_pnl - final_pnl) / valid_pnl * 100 if valid_pnl != 0 else 0
        print(f" Valid PnL: {valid_pnl:+.4f}% -> Final Test PnL: {final_pnl:+.4f}%")
        if degradation > 50:
            print(f" WARNING: {degradation:.0f}% degradation from validation to final test!")
            print(f"          This suggests OVERFITTING to validation data.")
        elif degradation > 20:
            print(f" CAUTION: {degradation:.0f}% degradation - some overfitting possible.")
        else:
            print(f" GOOD: Only {degradation:.0f}% degradation - results appear robust.")
    elif final_pnl > 0:
        print(f" Positive Final Test PnL: Strategy shows promise")
        if result.final_test_sharpe_ratio > 0.5:
            print(f" Good Final Test Sharpe (>{0.5}): Risk-adjusted returns reasonable")
        else:
            print(f" Low Final Test Sharpe (<0.5): High variance in returns")
    else:
        print(f" Negative Final Test PnL: Strategy needs improvement")

    if result.final_test_max_drawdown_pct > 5:
        print(f" WARNING: Final test max drawdown {result.final_test_max_drawdown_pct:.1f}% is high")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Walk-Forward Backtest with Parameter Sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 2-way split (legacy)
  python tools/walkforward_run.py --symbol XRPUSDT --start 2024-01-01 --end 2024-02-01

  # 3-way split (recommended - prevents overfitting)
  python tools/walkforward_run.py --symbol XRPUSDT --start 2024-01-01 --end 2024-03-01 --3way

  # With database path (standardized CLI)
  python tools/walkforward_run.py --db hft_trades.db --symbol XRP --final-days 7 --3way \\
      --start 2024-01-01 --end 2024-03-01

  # Custom split sizes
  python tools/walkforward_run.py --symbol BTCUSDT --start 2024-01-01 --end 2024-03-01 \\
      --3way --train-days 21 --valid-days 7 --final-test-days 7

  # Custom parameter grid
  python tools/walkforward_run.py --symbol XRPUSDT --start 2024-01-01 --end 2024-02-01 \\
      --grid "tp=0.2,0.3; sl=0.1,0.15; tstop=60,90"
        """
    )
    parser.add_argument("--db", type=str, default=None,
                        help="Database path (optional, currently candles fetched from Binance API)")
    parser.add_argument("--symbol", required=True, help="Trading symbol (e.g., XRPUSDT or XRP)")
    parser.add_argument("--quote", type=str, default="USDT",
                        help="Quote currency to append if missing (default: USDT)")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--3way", dest="three_way", action="store_true",
                        help="Use 3-way split (TRAIN/VALID/FINAL_TEST) to prevent overfitting")
    parser.add_argument("--train-days", type=int, default=14, help="Training window days (default: 14)")
    parser.add_argument("--test-days", type=int, default=7, help="Test window days (default: 7, used for 2-way)")
    parser.add_argument("--valid-days", type=int, default=7, help="Validation window days (default: 7, used for 3-way)")
    parser.add_argument("--final-test-days", type=int, default=7, help="Final test window days (default: 7, used for 3-way)")
    parser.add_argument("--final-days", type=int, default=None,
                        help="Alias for --final-test-days (for standardized CLI)")
    parser.add_argument("--top-n-validate", type=int, default=5, help="Top N candidates to validate (default: 5, used for 3-way)")
    parser.add_argument("--grid", type=str, help="Parameter grid (e.g., 'tp=0.2,0.3,0.4; sl=0.1,0.15; tstop=30,60')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--output-dir", default="reports", help="Output directory for CSV (default: reports)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output")
    args = parser.parse_args()

    # Handle --final-days alias
    if args.final_days is not None:
        args.final_test_days = args.final_days

    # Handle --db argument (currently just warns, candles come from Binance)
    if args.db:
        db_path = Path(args.db)
        if db_path.exists():
            logger.warning(f"--db specified ({args.db}) but backtest uses Binance API for candles. DB ignored.")
        else:
            logger.warning(f"--db specified ({args.db}) but file not found. Using Binance API for candles.")

    # Normalize symbol (XRP -> XRPUSDT based on --quote)
    symbol = args.symbol.upper()
    quote = args.quote.upper()
    if not symbol.endswith(quote):
        original = symbol
        symbol = symbol + quote
        logger.info(f"Normalized symbol with quote '{quote}': {original} -> {symbol}")
    args.symbol = symbol
    logger.info(f"Using Binance symbol: {symbol}")

    # Parse dates
    try:
        start_date = datetime.strptime(args.start, "%Y-%m-%d")
        end_date = datetime.strptime(args.end, "%Y-%m-%d")
    except ValueError as e:
        print(f"ERROR: Invalid date format. Use YYYY-MM-DD. {e}")
        return 1

    if start_date >= end_date:
        print("ERROR: Start date must be before end date")
        return 1

    # Parse parameter grid
    if args.grid:
        param_grid = ParameterGrid.from_string(args.grid)
    else:
        # Default grid
        param_grid = ParameterGrid(
            take_profit_pct=[0.15, 0.20, 0.25, 0.30],
            stop_loss_pct=[0.10, 0.15, 0.20],
            time_stop_seconds=[30, 60, 90],
            min_imbalance=[0.60]
        )

    if args.three_way:
        split_info = f"TRAIN={args.train_days}d / VALID={args.valid_days}d / FINAL_TEST={args.final_test_days}d"
        mode_label = "3-WAY SPLIT (prevents overfitting)"
    else:
        split_info = f"TRAIN={args.train_days}d / TEST={args.test_days}d"
        mode_label = "2-WAY SPLIT (legacy)"

    print(f"""
+======================================================================+
|              WALK-FORWARD OPTIMIZATION                               |
+======================================================================+
| Symbol:       {args.symbol:<52} |
| Period:       {start_date.date()} to {end_date.date():<40} |
| Mode:         {mode_label:<52} |
| Split:        {split_info:<52} |
| Grid Size:    {param_grid.total_combinations()} parameter combinations{' ' * 32}|
| Random Seed:  {args.seed:<52} |
+======================================================================+
    """)

    # Fetch candles
    result = fetch_candles_from_binance(args.symbol, start_date, end_date)

    # Handle tuple return (candles, resolved_symbol)
    if isinstance(result, tuple):
        candles, resolved_symbol = result
        if resolved_symbol != args.symbol:
            logger.info(f"Using resolved symbol from API: {resolved_symbol}")
            args.symbol = resolved_symbol
    else:
        candles = result

    if len(candles) < 100:
        print(f"ERROR: Not enough data ({len(candles)} candles). Need at least 100.")
        print(f"       Check if symbol '{args.symbol}' is valid on Binance.")
        return 1

    # Run walk-forward
    try:
        if args.three_way:
            result = walk_forward_run_3way(
                symbol=args.symbol,
                candles=candles,
                start_date=start_date,
                end_date=end_date,
                train_days=args.train_days,
                valid_days=args.valid_days,
                final_test_days=args.final_test_days,
                param_grid=param_grid,
                random_seed=args.seed,
                top_n_validate=args.top_n_validate
            )
        else:
            result = walk_forward_run(
                symbol=args.symbol,
                candles=candles,
                start_date=start_date,
                end_date=end_date,
                train_days=args.train_days,
                test_days=args.test_days,
                param_grid=param_grid,
                random_seed=args.seed
            )
    except Exception as e:
        logger.error(f"Walk-forward failed: {e}")
        return 1

    # Print summary
    if args.three_way:
        print_summary_3way(result)
    else:
        print_summary(result)

    # Export to CSV
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.three_way:
        csv_path = Path(args.output_dir) / f"walkforward_3way_{args.symbol}_{timestamp}.csv"
        export_results_to_csv_3way(result, str(csv_path))
    else:
        csv_path = Path(args.output_dir) / f"walkforward_{args.symbol}_{timestamp}.csv"
        export_results_to_csv(result, str(csv_path))

    print(f"\n CSV Report: {csv_path}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
