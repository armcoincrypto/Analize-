"""
Database-backed Strategy Optimizer with Parameter Grid Search.

Stores historical data in SQLite and tests thousands of parameter combinations.
"""

import sqlite3
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any
from analize import CloudAIAnalyzer
from analize.features.indicators import TechnicalIndicators


# Database path
DB_PATH = Path(__file__).parent.parent / "data" / "market_data.db"


def init_database():
    """Initialize the SQLite database."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # OHLCV data table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            interval TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL,
            UNIQUE(symbol, interval, timestamp)
        )
    """)

    # Strategy results table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS strategy_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            interval TEXT NOT NULL,
            strategy_name TEXT NOT NULL,
            parameters TEXT NOT NULL,
            start_date DATETIME,
            end_date DATETIME,
            trades INTEGER,
            win_rate REAL,
            profit_factor REAL,
            net_profit REAL,
            sharpe_ratio REAL,
            max_drawdown REAL,
            avg_win REAL,
            avg_loss REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Best parameters table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS best_parameters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            interval TEXT NOT NULL,
            strategy_name TEXT NOT NULL,
            parameters TEXT NOT NULL,
            net_profit REAL,
            sharpe_ratio REAL,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, interval, strategy_name)
        )
    """)

    conn.commit()
    conn.close()
    print(f"Database initialized at: {DB_PATH}")


def fetch_and_store_data(symbol: str = "XRPUSDT", days: int = 90, interval: str = "1h"):
    """Fetch data from Binance and store in database."""
    urls = [
        "https://api.binance.us/api/v3/klines",  # US first (works in geo-restricted regions)
        "https://api.binance.com/api/v3/klines",
    ]

    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_time,
        "endTime": end_time,
        "limit": 1000
    }

    all_data = []
    current_start = start_time

    print(f"Fetching {symbol} data ({days} days, {interval} candles)...")

    while current_start < end_time:
        params["startTime"] = current_start

        for url in urls:
            try:
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, list) and len(data) > 0:
                        all_data.extend(data)
                        # Move to next batch
                        current_start = data[-1][0] + 1
                        break
            except Exception as e:
                continue
        else:
            print(f"  Failed to fetch data starting at {current_start}")
            break

        if len(data) < 1000:
            break

    if not all_data:
        print("  No data fetched!")
        return 0

    # Store in database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    inserted = 0
    for row in all_data:
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO ohlcv
                (symbol, interval, timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol,
                interval,
                datetime.fromtimestamp(row[0] / 1000).isoformat(),
                float(row[1]),
                float(row[2]),
                float(row[3]),
                float(row[4]),
                float(row[5]),
            ))
            inserted += 1
        except Exception as e:
            pass

    conn.commit()
    conn.close()

    print(f"  Stored {inserted} candles in database")
    return inserted


def load_data_from_db(symbol: str = "XRPUSDT", interval: str = "1h", days: int = 30) -> pd.DataFrame:
    """Load data from database."""
    conn = sqlite3.connect(DB_PATH)

    start_date = (datetime.now() - timedelta(days=days)).isoformat()

    df = pd.read_sql_query("""
        SELECT timestamp, open, high, low, close, volume
        FROM ohlcv
        WHERE symbol = ? AND interval = ? AND timestamp >= ?
        ORDER BY timestamp
    """, conn, params=(symbol, interval, start_date))

    conn.close()

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def get_data_stats():
    """Get statistics about stored data."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT symbol, interval, COUNT(*) as candles,
               MIN(timestamp) as start, MAX(timestamp) as end
        FROM ohlcv
        GROUP BY symbol, interval
    """)

    results = cursor.fetchall()
    conn.close()

    return results


# ============================================================================
# STRATEGY FUNCTIONS WITH PARAMETERS
# ============================================================================

def strategy_rsi(df: pd.DataFrame, period: int = 14, buy_threshold: int = 30, sell_threshold: int = 70) -> pd.Series:
    """RSI Strategy with configurable parameters."""
    rsi = TechnicalIndicators.rsi(df["close"], period=period)
    position = 0
    entry_price = 0
    pnl_list = []

    for i in range(len(df)):
        if pd.isna(rsi.iloc[i]):
            continue
        price = df["close"].iloc[i]

        if position == 0 and rsi.iloc[i] < buy_threshold:
            position = 1
            entry_price = price
        elif position == 1 and rsi.iloc[i] > sell_threshold:
            pnl_list.append(((price - entry_price) / entry_price) * 100)
            position = 0

    return pd.Series(pnl_list)


def strategy_sma_cross(df: pd.DataFrame, fast_period: int = 10, slow_period: int = 30) -> pd.Series:
    """SMA Crossover with configurable periods."""
    sma_fast = TechnicalIndicators.sma(df["close"], period=fast_period)
    sma_slow = TechnicalIndicators.sma(df["close"], period=slow_period)

    position = 0
    entry_price = 0
    pnl_list = []

    for i in range(1, len(df)):
        if pd.isna(sma_fast.iloc[i]) or pd.isna(sma_slow.iloc[i]):
            continue
        price = df["close"].iloc[i]

        if position == 0 and sma_fast.iloc[i] > sma_slow.iloc[i] and sma_fast.iloc[i-1] <= sma_slow.iloc[i-1]:
            position = 1
            entry_price = price
        elif position == 1 and sma_fast.iloc[i] < sma_slow.iloc[i] and sma_fast.iloc[i-1] >= sma_slow.iloc[i-1]:
            pnl_list.append(((price - entry_price) / entry_price) * 100)
            position = 0

    return pd.Series(pnl_list)


def strategy_ema_cross(df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26) -> pd.Series:
    """EMA Crossover with configurable periods."""
    ema_fast = TechnicalIndicators.ema(df["close"], period=fast_period)
    ema_slow = TechnicalIndicators.ema(df["close"], period=slow_period)

    position = 0
    entry_price = 0
    pnl_list = []

    for i in range(1, len(df)):
        if pd.isna(ema_fast.iloc[i]) or pd.isna(ema_slow.iloc[i]):
            continue
        price = df["close"].iloc[i]

        if position == 0 and ema_fast.iloc[i] > ema_slow.iloc[i] and ema_fast.iloc[i-1] <= ema_slow.iloc[i-1]:
            position = 1
            entry_price = price
        elif position == 1 and ema_fast.iloc[i] < ema_slow.iloc[i] and ema_fast.iloc[i-1] >= ema_slow.iloc[i-1]:
            pnl_list.append(((price - entry_price) / entry_price) * 100)
            position = 0

    return pd.Series(pnl_list)


def strategy_bollinger(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.Series:
    """Bollinger Bands with configurable parameters."""
    upper, middle, lower = TechnicalIndicators.bollinger_bands(df["close"], period=period, std_dev=std_dev)

    position = 0
    entry_price = 0
    pnl_list = []

    for i in range(len(df)):
        if pd.isna(lower.iloc[i]) or pd.isna(upper.iloc[i]):
            continue
        price = df["close"].iloc[i]

        if position == 0 and price <= lower.iloc[i]:
            position = 1
            entry_price = price
        elif position == 1 and price >= upper.iloc[i]:
            pnl_list.append(((price - entry_price) / entry_price) * 100)
            position = 0

    return pd.Series(pnl_list)


def strategy_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3,
                        buy_threshold: int = 20, sell_threshold: int = 80) -> pd.Series:
    """Stochastic with configurable parameters."""
    k, d = TechnicalIndicators.stochastic(df["high"], df["low"], df["close"],
                                          k_period=k_period, d_period=d_period)

    position = 0
    entry_price = 0
    pnl_list = []

    for i in range(len(df)):
        if pd.isna(k.iloc[i]):
            continue
        price = df["close"].iloc[i]

        if position == 0 and k.iloc[i] < buy_threshold:
            position = 1
            entry_price = price
        elif position == 1 and k.iloc[i] > sell_threshold:
            pnl_list.append(((price - entry_price) / entry_price) * 100)
            position = 0

    return pd.Series(pnl_list)


def strategy_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """MACD with configurable parameters."""
    macd_line, signal_line, histogram = TechnicalIndicators.macd(
        df["close"], fast_period=fast, slow_period=slow, signal_period=signal
    )

    position = 0
    entry_price = 0
    pnl_list = []

    for i in range(1, len(df)):
        if pd.isna(macd_line.iloc[i]) or pd.isna(signal_line.iloc[i]):
            continue
        price = df["close"].iloc[i]

        if position == 0 and macd_line.iloc[i] > signal_line.iloc[i] and macd_line.iloc[i-1] <= signal_line.iloc[i-1]:
            position = 1
            entry_price = price
        elif position == 1 and macd_line.iloc[i] < signal_line.iloc[i] and macd_line.iloc[i-1] >= signal_line.iloc[i-1]:
            pnl_list.append(((price - entry_price) / entry_price) * 100)
            position = 0

    return pd.Series(pnl_list)


def strategy_mean_reversion(df: pd.DataFrame, lookback: int = 20, entry_std: float = 2.0, exit_std: float = 0.0) -> pd.Series:
    """Mean Reversion with configurable parameters."""
    mean = df["close"].rolling(lookback).mean()
    std = df["close"].rolling(lookback).std()

    position = 0
    entry_price = 0
    pnl_list = []

    for i in range(len(df)):
        if pd.isna(mean.iloc[i]) or pd.isna(std.iloc[i]) or std.iloc[i] == 0:
            continue
        price = df["close"].iloc[i]
        z_score = (price - mean.iloc[i]) / std.iloc[i]

        if position == 0 and z_score < -entry_std:
            position = 1
            entry_price = price
        elif position == 1 and z_score > exit_std:
            pnl_list.append(((price - entry_price) / entry_price) * 100)
            position = 0

    return pd.Series(pnl_list)


# ============================================================================
# PARAMETER GRID SEARCH
# ============================================================================

# Define parameter grids for each strategy
PARAMETER_GRIDS = {
    "RSI": {
        "func": strategy_rsi,
        "params": {
            "period": [7, 14, 21],
            "buy_threshold": [20, 25, 30, 35],
            "sell_threshold": [65, 70, 75, 80],
        }
    },
    "SMA_Cross": {
        "func": strategy_sma_cross,
        "params": {
            "fast_period": [5, 8, 10, 12, 15],
            "slow_period": [20, 25, 30, 40, 50],
        }
    },
    "EMA_Cross": {
        "func": strategy_ema_cross,
        "params": {
            "fast_period": [5, 8, 12, 15],
            "slow_period": [20, 26, 30, 40],
        }
    },
    "Bollinger": {
        "func": strategy_bollinger,
        "params": {
            "period": [10, 15, 20, 25, 30],
            "std_dev": [1.5, 2.0, 2.5, 3.0],
        }
    },
    "Stochastic": {
        "func": strategy_stochastic,
        "params": {
            "k_period": [5, 9, 14, 21],
            "d_period": [3, 5],
            "buy_threshold": [15, 20, 25, 30],
            "sell_threshold": [70, 75, 80, 85],
        }
    },
    "MACD": {
        "func": strategy_macd,
        "params": {
            "fast": [8, 12, 16],
            "slow": [20, 26, 30],
            "signal": [7, 9, 12],
        }
    },
    "Mean_Reversion": {
        "func": strategy_mean_reversion,
        "params": {
            "lookback": [10, 15, 20, 30],
            "entry_std": [1.5, 2.0, 2.5, 3.0],
            "exit_std": [-0.5, 0.0, 0.5],
        }
    },
}


def count_total_combinations():
    """Count total parameter combinations."""
    total = 0
    for strategy_name, config in PARAMETER_GRIDS.items():
        combos = 1
        for param_values in config["params"].values():
            combos *= len(param_values)
        total += combos
        print(f"  {strategy_name}: {combos} combinations")
    return total


@dataclass
class OptimizationResult:
    strategy: str
    params: dict
    trades: int
    win_rate: float
    profit_factor: float
    net_profit: float
    sharpe_ratio: float
    max_drawdown: float


def run_grid_search(df: pd.DataFrame, min_trades: int = 5) -> list[OptimizationResult]:
    """Run grid search over all strategies and parameters."""
    analyzer = CloudAIAnalyzer()
    results = []

    total_combos = sum(
        np.prod([len(v) for v in config["params"].values()])
        for config in PARAMETER_GRIDS.values()
    )

    print(f"\nRunning grid search over {total_combos} parameter combinations...")

    tested = 0
    profitable = 0

    for strategy_name, config in PARAMETER_GRIDS.items():
        func = config["func"]
        param_names = list(config["params"].keys())
        param_values = list(config["params"].values())

        for combo in product(*param_values):
            tested += 1
            params = dict(zip(param_names, combo))

            try:
                pnl = func(df, **params)

                if len(pnl) < min_trades:
                    continue

                metrics = analyzer.calculate_metrics(pnl)

                result = OptimizationResult(
                    strategy=strategy_name,
                    params=params,
                    trades=metrics["total_trades"],
                    win_rate=metrics["win_rate"],
                    profit_factor=metrics["profit_factor"],
                    net_profit=metrics["net_profit"],
                    sharpe_ratio=metrics["sharpe_ratio"],
                    max_drawdown=metrics["max_drawdown_pct"],
                )
                results.append(result)

                if result.net_profit > 0:
                    profitable += 1

            except Exception as e:
                pass

            # Progress
            if tested % 100 == 0:
                print(f"  Tested: {tested}/{total_combos} | Profitable: {profitable}")

    print(f"\nCompleted: {tested} combinations tested, {len(results)} valid, {profitable} profitable")

    # Sort by net profit
    results.sort(key=lambda x: x.net_profit, reverse=True)
    return results


def save_results_to_db(results: list[OptimizationResult], symbol: str, interval: str,
                       start_date: datetime, end_date: datetime):
    """Save optimization results to database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    for r in results:
        cursor.execute("""
            INSERT INTO strategy_results
            (symbol, interval, strategy_name, parameters, start_date, end_date,
             trades, win_rate, profit_factor, net_profit, sharpe_ratio, max_drawdown, avg_win, avg_loss)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            symbol, interval, r.strategy, str(r.params),
            start_date.isoformat(), end_date.isoformat(),
            r.trades, r.win_rate, r.profit_factor, r.net_profit,
            r.sharpe_ratio, r.max_drawdown, 0, 0
        ))

    # Update best parameters
    best_by_strategy = {}
    for r in results:
        if r.strategy not in best_by_strategy or r.net_profit > best_by_strategy[r.strategy].net_profit:
            best_by_strategy[r.strategy] = r

    for r in best_by_strategy.values():
        cursor.execute("""
            INSERT OR REPLACE INTO best_parameters
            (symbol, interval, strategy_name, parameters, net_profit, sharpe_ratio, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (symbol, interval, r.strategy, str(r.params), r.net_profit, r.sharpe_ratio))

    conn.commit()
    conn.close()
    print(f"Saved {len(results)} results to database")


def main():
    print("=" * 70)
    print("DATABASE-BACKED STRATEGY OPTIMIZER")
    print("Parameter Grid Search")
    print("=" * 70)

    # Initialize database
    init_database()

    # Configuration
    symbol = "XRPUSDT"
    interval = "1h"
    days = 30

    # Fetch and store data
    print(f"\n--- Step 1: Fetch Data ---")
    fetch_and_store_data(symbol, days=days, interval=interval)

    # Show data stats
    print(f"\n--- Database Stats ---")
    stats = get_data_stats()
    for row in stats:
        print(f"  {row[0]} ({row[1]}): {row[2]} candles from {row[3]} to {row[4]}")

    # Load data
    print(f"\n--- Step 2: Load Data ---")
    df = load_data_from_db(symbol, interval, days)
    print(f"  Loaded {len(df)} candles")
    print(f"  Range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Count combinations
    print(f"\n--- Step 3: Parameter Combinations ---")
    total = count_total_combinations()
    print(f"\n  TOTAL: {total} combinations to test")

    # Run grid search
    print(f"\n--- Step 4: Grid Search ---")
    results = run_grid_search(df, min_trades=5)

    # Save to database
    save_results_to_db(results, symbol, interval,
                       df['timestamp'].min(), df['timestamp'].max())

    # Display top results
    print("\n" + "=" * 70)
    print("TOP 10 PARAMETER COMBINATIONS")
    print("=" * 70)

    print(f"\n{'Rank':<5} {'Strategy':<15} {'Params':<40} {'Profit':<10} {'Sharpe':<8} {'Trades':<8}")
    print("-" * 90)

    for i, r in enumerate(results[:10], 1):
        params_str = ", ".join(f"{k}={v}" for k, v in r.params.items())
        if len(params_str) > 38:
            params_str = params_str[:35] + "..."
        print(f"{i:<5} {r.strategy:<15} {params_str:<40} {r.net_profit:>+8.2f}% {r.sharpe_ratio:>7.2f} {r.trades:>7}")

    # Best by strategy
    print("\n" + "=" * 70)
    print("BEST PARAMETERS BY STRATEGY")
    print("=" * 70)

    best_by_strategy = {}
    for r in results:
        if r.strategy not in best_by_strategy or r.net_profit > best_by_strategy[r.strategy].net_profit:
            best_by_strategy[r.strategy] = r

    for strategy, r in sorted(best_by_strategy.items(), key=lambda x: x[1].net_profit, reverse=True):
        print(f"\n🏆 {strategy}:")
        print(f"   Parameters: {r.params}")
        print(f"   Net Profit: {r.net_profit:+.2f}%")
        print(f"   Win Rate:   {r.win_rate:.1f}%")
        print(f"   Sharpe:     {r.sharpe_ratio:.2f}")
        print(f"   Trades:     {r.trades}")

    # Summary stats
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    profitable = [r for r in results if r.net_profit > 0]
    print(f"\n  Total combinations tested: {total}")
    print(f"  Valid strategies (≥5 trades): {len(results)}")
    print(f"  Profitable combinations: {len(profitable)} ({len(profitable)/len(results)*100:.1f}%)")

    if profitable:
        print(f"\n  Best overall: {results[0].strategy}")
        print(f"    Params: {results[0].params}")
        print(f"    Profit: {results[0].net_profit:+.2f}%")

    print("\n" + "=" * 70)
    print(f"Results saved to: {DB_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
