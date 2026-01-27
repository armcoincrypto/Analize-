#!/usr/bin/env python3
"""
Walk-Forward Backtest CLI
=========================
Run walk-forward optimization with parameter sweep.

Usage:
    python tools/walkforward_run.py --symbol XRPUSDT --start 2024-01-01 --end 2024-02-01
    python tools/walkforward_run.py --symbol XRPUSDT --start 2024-01-01 --end 2024-02-01 \
        --train-days 14 --test-days 7 \
        --grid "tp=0.2,0.3,0.4; sl=0.1,0.15; tstop=30,60"

Output:
    - CSV leaderboard: reports/walkforward_<symbol>_<timestamp>.csv
    - Console summary with best parameters

Requirements:
    - Requires historical candle data from Binance API
    - Or cached data in a local file
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
    ParameterGrid,
    walk_forward_run,
    WalkForwardResult,
)
from hft_system.research.metrics import metrics_to_dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def fetch_candles_from_binance(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    interval: str = "1m"
) -> List[Dict]:
    """
    Fetch historical candles from Binance API.

    Args:
        symbol: Trading pair (e.g., "XRPUSDT")
        start_date: Start datetime
        end_date: End datetime
        interval: Candle interval (1m, 5m, etc.)

    Returns:
        List of candle dicts
    """
    base_url = "https://api.binance.com/api/v3/klines"
    candles = []

    start_ms = int(start_date.timestamp() * 1000)
    end_ms = int(end_date.timestamp() * 1000)

    logger.info(f"Fetching {symbol} candles from {start_date.date()} to {end_date.date()}...")

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
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            break

    logger.info(f"Fetched {len(candles)} candles")
    return candles


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


def main():
    parser = argparse.ArgumentParser(
        description="Walk-Forward Backtest with Parameter Sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/walkforward_run.py --symbol XRPUSDT --start 2024-01-01 --end 2024-02-01
  python tools/walkforward_run.py --symbol BTCUSDT --start 2024-01-01 --end 2024-03-01 --train-days 21 --test-days 7
  python tools/walkforward_run.py --symbol XRPUSDT --start 2024-01-01 --end 2024-02-01 --grid "tp=0.2,0.3; sl=0.1,0.15; tstop=60,90"
        """
    )
    parser.add_argument("--symbol", required=True, help="Trading symbol (e.g., XRPUSDT)")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--train-days", type=int, default=14, help="Training window days (default: 14)")
    parser.add_argument("--test-days", type=int, default=7, help="Test window days (default: 7)")
    parser.add_argument("--grid", type=str, help="Parameter grid (e.g., 'tp=0.2,0.3,0.4; sl=0.1,0.15; tstop=30,60')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--output-dir", default="reports", help="Output directory for CSV (default: reports)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output")
    args = parser.parse_args()

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

    print(f"""
+======================================================================+
|              WALK-FORWARD OPTIMIZATION                               |
+======================================================================+
| Symbol:       {args.symbol:<52} |
| Period:       {start_date.date()} to {end_date.date():<40} |
| Train/Test:   {args.train_days}/{args.test_days} days{' ' * 49}|
| Grid Size:    {param_grid.total_combinations()} parameter combinations{' ' * 32}|
| Random Seed:  {args.seed:<52} |
+======================================================================+
    """)

    # Fetch candles
    candles = fetch_candles_from_binance(args.symbol, start_date, end_date)

    if len(candles) < 100:
        print(f"ERROR: Not enough data ({len(candles)} candles). Need at least 100.")
        return 1

    # Run walk-forward
    try:
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
    print_summary(result)

    # Export to CSV
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = Path(args.output_dir) / f"walkforward_{args.symbol}_{timestamp}.csv"
    export_results_to_csv(result, str(csv_path))

    print(f"\n CSV Report: {csv_path}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
