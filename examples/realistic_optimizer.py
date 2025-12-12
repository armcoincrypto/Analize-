"""
Realistic Strategy Optimizer with:
- 90-day backtesting
- Trading fee calculations
- Slippage simulation
- Walk-forward validation
- Out-of-sample testing
"""

import sqlite3
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any
from analize import CloudAIAnalyzer
from analize.features.indicators import TechnicalIndicators


# Configuration
DB_PATH = Path(__file__).parent.parent / "data" / "market_data.db"

# Trading costs (Binance)
TRADING_FEE_PCT = 0.1  # 0.1% per trade (taker fee)
SLIPPAGE_PCT = 0.05    # 0.05% estimated slippage
SPREAD_PCT = 0.02      # 0.02% spread cost


@dataclass
class RealisticResult:
    """Results with realistic cost calculations."""
    strategy: str
    params: dict
    trades: int
    win_rate: float
    gross_profit: float      # Before fees
    total_fees: float        # All trading costs
    net_profit: float        # After fees
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    avg_win: float
    avg_loss: float
    # Validation
    in_sample_profit: float = 0.0
    out_sample_profit: float = 0.0
    is_validated: bool = False


def init_database():
    """Initialize the SQLite database."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

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

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS validated_strategies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            interval TEXT NOT NULL,
            strategy_name TEXT NOT NULL,
            parameters TEXT NOT NULL,
            in_sample_profit REAL,
            out_sample_profit REAL,
            total_fees REAL,
            net_profit REAL,
            sharpe_ratio REAL,
            trades INTEGER,
            is_validated BOOLEAN,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


def fetch_data(symbol: str = "XRPUSDT", days: int = 90, interval: str = "1h") -> int:
    """Fetch historical data from Binance."""
    urls = [
        "https://api.binance.us/api/v3/klines",  # US first (works in geo-restricted regions)
        "https://api.binance.com/api/v3/klines",
    ]

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

    total_inserted = 0
    current_start = start_time

    print(f"Fetching {symbol} data ({days} days)...")

    while current_start < end_time:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_time,
            "limit": 1000
        }

        data = None
        for url in urls:
            try:
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, list) and len(data) > 0:
                        break
            except:
                continue

        if not data:
            break

        for row in data:
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO ohlcv
                    (symbol, interval, timestamp, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol, interval,
                    datetime.fromtimestamp(row[0] / 1000).isoformat(),
                    float(row[1]), float(row[2]), float(row[3]),
                    float(row[4]), float(row[5]),
                ))
                total_inserted += 1
            except:
                pass

        current_start = data[-1][0] + 1
        if len(data) < 1000:
            break

    conn.commit()
    conn.close()

    print(f"  Stored {total_inserted} candles")
    return total_inserted


def load_data(symbol: str, interval: str, days: int) -> pd.DataFrame:
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


# ============================================================================
# STRATEGIES WITH FEE CALCULATIONS
# ============================================================================

def calculate_trade_cost(entry_price: float, exit_price: float) -> float:
    """Calculate total cost of a round-trip trade."""
    # Entry costs
    entry_fee = entry_price * (TRADING_FEE_PCT / 100)
    entry_slippage = entry_price * (SLIPPAGE_PCT / 100)
    entry_spread = entry_price * (SPREAD_PCT / 100)

    # Exit costs
    exit_fee = exit_price * (TRADING_FEE_PCT / 100)
    exit_slippage = exit_price * (SLIPPAGE_PCT / 100)

    total_cost = entry_fee + entry_slippage + entry_spread + exit_fee + exit_slippage
    return total_cost


def strategy_stochastic_with_fees(df: pd.DataFrame, k_period: int = 14, d_period: int = 3,
                                   buy_threshold: int = 20, sell_threshold: int = 80) -> tuple[pd.Series, float]:
    """Stochastic strategy with fee tracking."""
    k, d = TechnicalIndicators.stochastic(df["high"], df["low"], df["close"],
                                          k_period=k_period, d_period=d_period)

    position = 0
    entry_price = 0
    pnl_list = []
    total_fees = 0

    for i in range(len(df)):
        if pd.isna(k.iloc[i]):
            continue
        price = df["close"].iloc[i]

        if position == 0 and k.iloc[i] < buy_threshold:
            position = 1
            entry_price = price
        elif position == 1 and k.iloc[i] > sell_threshold:
            # Calculate gross P&L
            gross_pnl_pct = ((price - entry_price) / entry_price) * 100

            # Calculate fees
            trade_cost = calculate_trade_cost(entry_price, price)
            fee_pct = (trade_cost / entry_price) * 100
            total_fees += fee_pct

            # Net P&L
            net_pnl_pct = gross_pnl_pct - fee_pct
            pnl_list.append(net_pnl_pct)
            position = 0

    return pd.Series(pnl_list), total_fees


def strategy_macd_with_fees(df: pd.DataFrame, fast: int = 12, slow: int = 26,
                            signal: int = 9) -> tuple[pd.Series, float]:
    """MACD strategy with fee tracking."""
    macd_line, signal_line, histogram = TechnicalIndicators.macd(
        df["close"], fast_period=fast, slow_period=slow, signal_period=signal
    )

    position = 0
    entry_price = 0
    pnl_list = []
    total_fees = 0

    for i in range(1, len(df)):
        if pd.isna(macd_line.iloc[i]) or pd.isna(signal_line.iloc[i]):
            continue
        price = df["close"].iloc[i]

        if position == 0 and macd_line.iloc[i] > signal_line.iloc[i] and macd_line.iloc[i-1] <= signal_line.iloc[i-1]:
            position = 1
            entry_price = price
        elif position == 1 and macd_line.iloc[i] < signal_line.iloc[i] and macd_line.iloc[i-1] >= signal_line.iloc[i-1]:
            gross_pnl_pct = ((price - entry_price) / entry_price) * 100
            trade_cost = calculate_trade_cost(entry_price, price)
            fee_pct = (trade_cost / entry_price) * 100
            total_fees += fee_pct
            net_pnl_pct = gross_pnl_pct - fee_pct
            pnl_list.append(net_pnl_pct)
            position = 0

    return pd.Series(pnl_list), total_fees


def strategy_mean_reversion_with_fees(df: pd.DataFrame, lookback: int = 20,
                                       entry_std: float = 2.0, exit_std: float = 0.0) -> tuple[pd.Series, float]:
    """Mean Reversion strategy with fee tracking."""
    mean = df["close"].rolling(lookback).mean()
    std = df["close"].rolling(lookback).std()

    position = 0
    entry_price = 0
    pnl_list = []
    total_fees = 0

    for i in range(len(df)):
        if pd.isna(mean.iloc[i]) or pd.isna(std.iloc[i]) or std.iloc[i] == 0:
            continue
        price = df["close"].iloc[i]
        z_score = (price - mean.iloc[i]) / std.iloc[i]

        if position == 0 and z_score < -entry_std:
            position = 1
            entry_price = price
        elif position == 1 and z_score > exit_std:
            gross_pnl_pct = ((price - entry_price) / entry_price) * 100
            trade_cost = calculate_trade_cost(entry_price, price)
            fee_pct = (trade_cost / entry_price) * 100
            total_fees += fee_pct
            net_pnl_pct = gross_pnl_pct - fee_pct
            pnl_list.append(net_pnl_pct)
            position = 0

    return pd.Series(pnl_list), total_fees


def strategy_ema_cross_with_fees(df: pd.DataFrame, fast_period: int = 12,
                                  slow_period: int = 26) -> tuple[pd.Series, float]:
    """EMA Crossover strategy with fee tracking."""
    ema_fast = TechnicalIndicators.ema(df["close"], period=fast_period)
    ema_slow = TechnicalIndicators.ema(df["close"], period=slow_period)

    position = 0
    entry_price = 0
    pnl_list = []
    total_fees = 0

    for i in range(1, len(df)):
        if pd.isna(ema_fast.iloc[i]) or pd.isna(ema_slow.iloc[i]):
            continue
        price = df["close"].iloc[i]

        if position == 0 and ema_fast.iloc[i] > ema_slow.iloc[i] and ema_fast.iloc[i-1] <= ema_slow.iloc[i-1]:
            position = 1
            entry_price = price
        elif position == 1 and ema_fast.iloc[i] < ema_slow.iloc[i] and ema_fast.iloc[i-1] >= ema_slow.iloc[i-1]:
            gross_pnl_pct = ((price - entry_price) / entry_price) * 100
            trade_cost = calculate_trade_cost(entry_price, price)
            fee_pct = (trade_cost / entry_price) * 100
            total_fees += fee_pct
            net_pnl_pct = gross_pnl_pct - fee_pct
            pnl_list.append(net_pnl_pct)
            position = 0

    return pd.Series(pnl_list), total_fees


# Strategy configurations
STRATEGIES = {
    "Stochastic": {
        "func": strategy_stochastic_with_fees,
        "params": {
            "k_period": [9, 14, 21],
            "d_period": [3, 5],
            "buy_threshold": [15, 20, 25],
            "sell_threshold": [75, 80, 85],
        }
    },
    "MACD": {
        "func": strategy_macd_with_fees,
        "params": {
            "fast": [8, 12, 16],
            "slow": [20, 26, 30],
            "signal": [7, 9, 12],
        }
    },
    "Mean_Reversion": {
        "func": strategy_mean_reversion_with_fees,
        "params": {
            "lookback": [15, 20, 30],
            "entry_std": [1.5, 2.0, 2.5],
            "exit_std": [-0.5, 0.0, 0.5],
        }
    },
    "EMA_Cross": {
        "func": strategy_ema_cross_with_fees,
        "params": {
            "fast_period": [5, 8, 12],
            "slow_period": [20, 26, 30],
        }
    },
}


def walk_forward_validation(df: pd.DataFrame, strategy_func, params: dict,
                            in_sample_pct: float = 0.7) -> tuple[float, float, bool]:
    """
    Walk-forward validation:
    - Optimize on first 70% of data (in-sample)
    - Test on remaining 30% (out-of-sample)
    """
    split_idx = int(len(df) * in_sample_pct)

    df_in_sample = df.iloc[:split_idx].copy()
    df_out_sample = df.iloc[split_idx:].copy()

    # In-sample performance
    pnl_in, fees_in = strategy_func(df_in_sample, **params)
    in_sample_profit = pnl_in.sum() if len(pnl_in) > 0 else 0

    # Out-of-sample performance
    pnl_out, fees_out = strategy_func(df_out_sample, **params)
    out_sample_profit = pnl_out.sum() if len(pnl_out) > 0 else 0

    # Strategy is validated if profitable in both periods
    is_validated = in_sample_profit > 0 and out_sample_profit > 0

    return in_sample_profit, out_sample_profit, is_validated


def run_realistic_optimization(df: pd.DataFrame, min_trades: int = 10) -> list[RealisticResult]:
    """Run optimization with realistic costs and validation."""
    analyzer = CloudAIAnalyzer()
    results = []

    total_combos = sum(
        np.prod([len(v) for v in config["params"].values()])
        for config in STRATEGIES.values()
    )

    print(f"\nTesting {total_combos} combinations with fees and validation...")

    tested = 0
    profitable = 0
    validated = 0

    for strategy_name, config in STRATEGIES.items():
        func = config["func"]
        param_names = list(config["params"].keys())
        param_values = list(config["params"].values())

        for combo in product(*param_values):
            tested += 1
            params = dict(zip(param_names, combo))

            try:
                # Run strategy with fees
                pnl, total_fees = func(df, **params)

                if len(pnl) < min_trades:
                    continue

                # Calculate metrics
                metrics = analyzer.calculate_metrics(pnl)

                # Walk-forward validation
                in_sample, out_sample, is_valid = walk_forward_validation(
                    df, func, params
                )

                result = RealisticResult(
                    strategy=strategy_name,
                    params=params,
                    trades=metrics["total_trades"],
                    win_rate=metrics["win_rate"],
                    gross_profit=pnl.sum() + total_fees,  # Add back fees for gross
                    total_fees=total_fees,
                    net_profit=metrics["net_profit"],
                    profit_factor=metrics["profit_factor"],
                    sharpe_ratio=metrics["sharpe_ratio"],
                    max_drawdown=metrics["max_drawdown_pct"],
                    avg_win=metrics["avg_win"],
                    avg_loss=metrics["avg_loss"],
                    in_sample_profit=in_sample,
                    out_sample_profit=out_sample,
                    is_validated=is_valid,
                )
                results.append(result)

                if result.net_profit > 0:
                    profitable += 1
                if is_valid:
                    validated += 1

            except Exception as e:
                pass

            if tested % 50 == 0:
                print(f"  Progress: {tested}/{total_combos} | Profitable: {profitable} | Validated: {validated}")

    print(f"\nCompleted: {tested} tested, {profitable} profitable, {validated} validated")

    results.sort(key=lambda x: (x.is_validated, x.net_profit), reverse=True)
    return results


def save_validated_results(results: list[RealisticResult], symbol: str, interval: str):
    """Save validated results to database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    for r in results:
        if r.is_validated:
            cursor.execute("""
                INSERT INTO validated_strategies
                (symbol, interval, strategy_name, parameters, in_sample_profit,
                 out_sample_profit, total_fees, net_profit, sharpe_ratio, trades, is_validated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol, interval, r.strategy, str(r.params),
                r.in_sample_profit, r.out_sample_profit, r.total_fees,
                r.net_profit, r.sharpe_ratio, r.trades, r.is_validated
            ))

    conn.commit()
    conn.close()


def main():
    print("=" * 70)
    print("REALISTIC STRATEGY OPTIMIZER")
    print("90-Day Backtest with Fees & Walk-Forward Validation")
    print("=" * 70)

    # Configuration
    symbol = "XRPUSDT"
    interval = "1h"
    days = 90

    # Initialize
    init_database()

    # Fetch data
    print(f"\n--- Step 1: Fetch 90 Days of Data ---")
    fetch_data(symbol, days=days, interval=interval)

    # Load data
    print(f"\n--- Step 2: Load Data ---")
    df = load_data(symbol, interval, days)
    print(f"  Loaded {len(df)} candles")
    print(f"  Range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Trading costs info
    print(f"\n--- Trading Costs (per trade) ---")
    print(f"  Trading fee:  {TRADING_FEE_PCT}%")
    print(f"  Slippage:     {SLIPPAGE_PCT}%")
    print(f"  Spread:       {SPREAD_PCT}%")
    print(f"  Total/trade:  {TRADING_FEE_PCT * 2 + SLIPPAGE_PCT * 2 + SPREAD_PCT}%")

    # Run optimization
    print(f"\n--- Step 3: Optimization with Validation ---")
    results = run_realistic_optimization(df, min_trades=10)

    # Save validated results
    save_validated_results(results, symbol, interval)

    # Display results
    print("\n" + "=" * 70)
    print("VALIDATED STRATEGIES (Profitable in both in-sample AND out-of-sample)")
    print("=" * 70)

    validated = [r for r in results if r.is_validated]

    if not validated:
        print("\n⚠️  No strategies passed validation!")
        print("   This means no strategy was profitable in both test periods.")
        print("\n   Showing top results anyway for reference:\n")
        validated = results[:10]

    print(f"\n{'Strategy':<15} {'Params':<35} {'Net%':<9} {'Fees%':<8} {'In-S%':<8} {'Out-S%':<8} {'Valid':<6}")
    print("-" * 95)

    for r in validated[:15]:
        params_str = ", ".join(f"{k}={v}" for k, v in list(r.params.items())[:3])
        if len(params_str) > 33:
            params_str = params_str[:30] + "..."
        valid_str = "✓" if r.is_validated else "✗"
        print(f"{r.strategy:<15} {params_str:<35} {r.net_profit:>+7.2f}% {r.total_fees:>6.2f}% "
              f"{r.in_sample_profit:>+7.2f}% {r.out_sample_profit:>+7.2f}% {valid_str:^6}")

    # Best validated strategy
    if validated and validated[0].is_validated:
        best = validated[0]
        print("\n" + "=" * 70)
        print(f"🏆 BEST VALIDATED STRATEGY: {best.strategy}")
        print("=" * 70)
        print(f"""
  Parameters:      {best.params}

  Performance (Full Period):
    Trades:        {best.trades}
    Win Rate:      {best.win_rate:.1f}%
    Gross Profit:  {best.gross_profit:+.2f}%
    Total Fees:    -{best.total_fees:.2f}%
    NET PROFIT:    {best.net_profit:+.2f}%

  Validation:
    In-Sample:     {best.in_sample_profit:+.2f}%  (first 70% of data)
    Out-of-Sample: {best.out_sample_profit:+.2f}%  (last 30% of data)
    VALIDATED:     ✓ Yes - Profitable in both periods!

  Risk Metrics:
    Sharpe Ratio:  {best.sharpe_ratio:.2f}
    Max Drawdown:  {best.max_drawdown:.2f}%
    Profit Factor: {best.profit_factor:.2f}
""")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_validated = len([r for r in results if r.is_validated])
    total_profitable = len([r for r in results if r.net_profit > 0])

    print(f"""
  Strategies Tested:     {len(results)}
  Profitable (net):      {total_profitable} ({total_profitable/len(results)*100:.1f}%)
  Validated (in+out):    {total_validated} ({total_validated/len(results)*100:.1f}%)

  ℹ️  Only VALIDATED strategies are recommended for live trading.
     They showed profit in both training and test periods.
""")

    print("=" * 70)


if __name__ == "__main__":
    main()
