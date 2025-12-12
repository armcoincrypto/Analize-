"""
Strategy Optimizer - Automatically test multiple strategies to find profitable ones.

Tests various trading strategies on historical data and ranks them by performance.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Callable
from analize import CloudAIAnalyzer
from analize.features.indicators import TechnicalIndicators


@dataclass
class StrategyResult:
    """Results from a strategy backtest."""
    name: str
    trades: int
    win_rate: float
    profit_factor: float
    net_profit: float
    sharpe_ratio: float
    max_drawdown: float
    avg_win: float
    avg_loss: float
    pnl_series: pd.Series


def fetch_data(symbol: str = "XRPUSDT", days: int = 30, interval: str = "1h") -> pd.DataFrame:
    """Fetch OHLCV data from Binance."""
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

    for url in urls:
        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    break
        except:
            continue
    else:
        raise Exception("Could not fetch data")

    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    return df[["timestamp", "open", "high", "low", "close", "volume"]]


# ============================================================================
# TRADING STRATEGIES
# ============================================================================

def strategy_rsi_oversold(df: pd.DataFrame, rsi_buy: int = 30, rsi_sell: int = 70) -> pd.Series:
    """RSI Oversold/Overbought Strategy - Buy when RSI < 30, sell when RSI > 70."""
    rsi = TechnicalIndicators.rsi(df["close"], period=14)
    position = 0
    entry_price = 0
    pnl_list = []

    for i in range(len(df)):
        if pd.isna(rsi.iloc[i]):
            continue
        price = df["close"].iloc[i]

        if position == 0 and rsi.iloc[i] < rsi_buy:
            position = 1
            entry_price = price
        elif position == 1 and rsi.iloc[i] > rsi_sell:
            pnl_list.append(((price - entry_price) / entry_price) * 100)
            position = 0

    return pd.Series(pnl_list)


def strategy_sma_crossover(df: pd.DataFrame, fast: int = 10, slow: int = 30) -> pd.Series:
    """SMA Crossover Strategy - Buy when fast SMA crosses above slow SMA."""
    sma_fast = TechnicalIndicators.sma(df["close"], period=fast)
    sma_slow = TechnicalIndicators.sma(df["close"], period=slow)

    position = 0
    entry_price = 0
    pnl_list = []

    for i in range(1, len(df)):
        if pd.isna(sma_fast.iloc[i]) or pd.isna(sma_slow.iloc[i]):
            continue
        price = df["close"].iloc[i]

        # Golden cross - buy
        if position == 0 and sma_fast.iloc[i] > sma_slow.iloc[i] and sma_fast.iloc[i-1] <= sma_slow.iloc[i-1]:
            position = 1
            entry_price = price
        # Death cross - sell
        elif position == 1 and sma_fast.iloc[i] < sma_slow.iloc[i] and sma_fast.iloc[i-1] >= sma_slow.iloc[i-1]:
            pnl_list.append(((price - entry_price) / entry_price) * 100)
            position = 0

    return pd.Series(pnl_list)


def strategy_bollinger_bounce(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.Series:
    """Bollinger Bands Bounce - Buy at lower band, sell at upper band."""
    upper, middle, lower = TechnicalIndicators.bollinger_bands(df["close"], period=period, std_dev=std_dev)

    position = 0
    entry_price = 0
    pnl_list = []

    for i in range(len(df)):
        if pd.isna(lower.iloc[i]) or pd.isna(upper.iloc[i]):
            continue
        price = df["close"].iloc[i]

        # Price touches lower band - buy
        if position == 0 and price <= lower.iloc[i]:
            position = 1
            entry_price = price
        # Price touches upper band - sell
        elif position == 1 and price >= upper.iloc[i]:
            pnl_list.append(((price - entry_price) / entry_price) * 100)
            position = 0

    return pd.Series(pnl_list)


def strategy_macd_crossover(df: pd.DataFrame) -> pd.Series:
    """MACD Crossover Strategy - Buy when MACD crosses above signal."""
    macd_line, signal_line, histogram = TechnicalIndicators.macd(df["close"])

    position = 0
    entry_price = 0
    pnl_list = []

    for i in range(1, len(df)):
        if pd.isna(macd_line.iloc[i]) or pd.isna(signal_line.iloc[i]):
            continue
        price = df["close"].iloc[i]

        # MACD crosses above signal - buy
        if position == 0 and macd_line.iloc[i] > signal_line.iloc[i] and macd_line.iloc[i-1] <= signal_line.iloc[i-1]:
            position = 1
            entry_price = price
        # MACD crosses below signal - sell
        elif position == 1 and macd_line.iloc[i] < signal_line.iloc[i] and macd_line.iloc[i-1] >= signal_line.iloc[i-1]:
            pnl_list.append(((price - entry_price) / entry_price) * 100)
            position = 0

    return pd.Series(pnl_list)


def strategy_stochastic(df: pd.DataFrame, k_buy: int = 20, k_sell: int = 80) -> pd.Series:
    """Stochastic Oscillator Strategy - Buy when K < 20, sell when K > 80."""
    k, d = TechnicalIndicators.stochastic(df["high"], df["low"], df["close"])

    position = 0
    entry_price = 0
    pnl_list = []

    for i in range(len(df)):
        if pd.isna(k.iloc[i]):
            continue
        price = df["close"].iloc[i]

        if position == 0 and k.iloc[i] < k_buy:
            position = 1
            entry_price = price
        elif position == 1 and k.iloc[i] > k_sell:
            pnl_list.append(((price - entry_price) / entry_price) * 100)
            position = 0

    return pd.Series(pnl_list)


def strategy_ema_trend(df: pd.DataFrame, period: int = 21) -> pd.Series:
    """EMA Trend Following - Buy when price crosses above EMA, sell when crosses below."""
    ema = TechnicalIndicators.ema(df["close"], period=period)

    position = 0
    entry_price = 0
    pnl_list = []

    for i in range(1, len(df)):
        if pd.isna(ema.iloc[i]):
            continue
        price = df["close"].iloc[i]
        prev_price = df["close"].iloc[i-1]

        # Price crosses above EMA - buy
        if position == 0 and price > ema.iloc[i] and prev_price <= ema.iloc[i-1]:
            position = 1
            entry_price = price
        # Price crosses below EMA - sell
        elif position == 1 and price < ema.iloc[i] and prev_price >= ema.iloc[i-1]:
            pnl_list.append(((price - entry_price) / entry_price) * 100)
            position = 0

    return pd.Series(pnl_list)


def strategy_mean_reversion(df: pd.DataFrame, lookback: int = 20, threshold: float = 2.0) -> pd.Series:
    """Mean Reversion - Buy when price is N std devs below mean, sell at mean."""
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

        # Price is significantly below mean - buy
        if position == 0 and z_score < -threshold:
            position = 1
            entry_price = price
        # Price returns to mean - sell
        elif position == 1 and z_score > 0:
            pnl_list.append(((price - entry_price) / entry_price) * 100)
            position = 0

    return pd.Series(pnl_list)


def strategy_breakout(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """Breakout Strategy - Buy on new high, sell on new low."""
    high_max = df["high"].rolling(lookback).max()
    low_min = df["low"].rolling(lookback).min()

    position = 0
    entry_price = 0
    pnl_list = []

    for i in range(lookback, len(df)):
        price = df["close"].iloc[i]

        # New high breakout - buy
        if position == 0 and price >= high_max.iloc[i-1]:
            position = 1
            entry_price = price
        # New low breakdown - sell
        elif position == 1 and price <= low_min.iloc[i-1]:
            pnl_list.append(((price - entry_price) / entry_price) * 100)
            position = 0

    return pd.Series(pnl_list)


# ============================================================================
# OPTIMIZER
# ============================================================================

def run_strategy(name: str, pnl: pd.Series, analyzer: CloudAIAnalyzer) -> StrategyResult | None:
    """Run a strategy and return results."""
    if len(pnl) < 3:
        return None

    metrics = analyzer.calculate_metrics(pnl)

    return StrategyResult(
        name=name,
        trades=metrics["total_trades"],
        win_rate=metrics["win_rate"],
        profit_factor=metrics["profit_factor"],
        net_profit=metrics["net_profit"],
        sharpe_ratio=metrics["sharpe_ratio"],
        max_drawdown=metrics["max_drawdown_pct"],
        avg_win=metrics["avg_win"],
        avg_loss=metrics["avg_loss"],
        pnl_series=pnl,
    )


def optimize_strategies(df: pd.DataFrame) -> list[StrategyResult]:
    """Test all strategies and return ranked results."""
    analyzer = CloudAIAnalyzer()
    results = []

    strategies = [
        ("RSI (30/70)", lambda: strategy_rsi_oversold(df, 30, 70)),
        ("RSI (20/80)", lambda: strategy_rsi_oversold(df, 20, 80)),
        ("RSI (25/75)", lambda: strategy_rsi_oversold(df, 25, 75)),
        ("SMA Cross (10/30)", lambda: strategy_sma_crossover(df, 10, 30)),
        ("SMA Cross (5/20)", lambda: strategy_sma_crossover(df, 5, 20)),
        ("SMA Cross (20/50)", lambda: strategy_sma_crossover(df, 20, 50)),
        ("Bollinger Bounce", lambda: strategy_bollinger_bounce(df)),
        ("Bollinger Bounce (1.5σ)", lambda: strategy_bollinger_bounce(df, std_dev=1.5)),
        ("MACD Crossover", lambda: strategy_macd_crossover(df)),
        ("Stochastic (20/80)", lambda: strategy_stochastic(df, 20, 80)),
        ("Stochastic (30/70)", lambda: strategy_stochastic(df, 30, 70)),
        ("EMA Trend (21)", lambda: strategy_ema_trend(df, 21)),
        ("EMA Trend (10)", lambda: strategy_ema_trend(df, 10)),
        ("Mean Reversion (2σ)", lambda: strategy_mean_reversion(df, 20, 2.0)),
        ("Mean Reversion (1.5σ)", lambda: strategy_mean_reversion(df, 20, 1.5)),
        ("Breakout (20)", lambda: strategy_breakout(df, 20)),
        ("Breakout (10)", lambda: strategy_breakout(df, 10)),
    ]

    print(f"\nTesting {len(strategies)} strategies...\n")

    for name, strategy_func in strategies:
        try:
            pnl = strategy_func()
            result = run_strategy(name, pnl, analyzer)
            if result:
                results.append(result)
                status = "✓" if result.net_profit > 0 else "✗"
                print(f"  {status} {name}: {result.trades} trades, {result.net_profit:+.2f}%")
            else:
                print(f"  - {name}: Not enough trades")
        except Exception as e:
            print(f"  ! {name}: Error - {e}")

    # Sort by net profit (descending)
    results.sort(key=lambda x: x.net_profit, reverse=True)
    return results


def main():
    print("=" * 70)
    print("STRATEGY OPTIMIZER - Find the Best Trading Strategy")
    print("=" * 70)

    # Fetch data
    symbol = "XRPUSDT"
    days = 30

    print(f"\nFetching {symbol} data ({days} days, 1h candles)...")
    try:
        df = fetch_data(symbol, days, "1h")
        print(f"  Data range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"  Candles: {len(df)}")
    except Exception as e:
        print(f"  Error: {e}")
        return

    # Price context
    print(f"\n--- Market Context ---")
    print(f"  Current Price: ${df['close'].iloc[-1]:.4f}")
    print(f"  30-Day Change: {((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100:+.2f}%")

    # Run optimizer
    results = optimize_strategies(df)

    # Display results
    print("\n" + "=" * 70)
    print("STRATEGY RANKING (by Net Profit)")
    print("=" * 70)

    if not results:
        print("\nNo strategies generated enough trades for analysis.")
        return

    # Top 5
    print("\n🏆 TOP 5 STRATEGIES:\n")
    print(f"{'Rank':<5} {'Strategy':<25} {'Trades':<8} {'Win%':<8} {'PF':<8} {'Net%':<10} {'Sharpe':<8}")
    print("-" * 70)

    for i, r in enumerate(results[:5], 1):
        pf_str = f"{r.profit_factor:.2f}" if r.profit_factor != float('inf') else "∞"
        print(f"{i:<5} {r.name:<25} {r.trades:<8} {r.win_rate:<8.1f} {pf_str:<8} {r.net_profit:<+10.2f} {r.sharpe_ratio:<8.2f}")

    # Best strategy details
    best = results[0]
    print(f"\n{'=' * 70}")
    print(f"🥇 BEST STRATEGY: {best.name}")
    print(f"{'=' * 70}")
    print(f"""
  Trades:        {best.trades}
  Win Rate:      {best.win_rate:.1f}%
  Profit Factor: {best.profit_factor:.2f}
  Net Profit:    {best.net_profit:+.2f}%
  Avg Win:       {best.avg_win:+.2f}%
  Avg Loss:      {best.avg_loss:+.2f}%
  Max Drawdown:  {best.max_drawdown:.2f}%
  Sharpe Ratio:  {best.sharpe_ratio:.2f}
""")

    # Statistical significance
    analyzer = CloudAIAnalyzer()
    wins = int(best.trades * best.win_rate / 100)
    sig_result = analyzer.test_significance(wins=wins, total=best.trades)
    print(f"  Statistical Significance:")
    print(f"    P-value:     {sig_result['p_value']:.4f}")
    print(f"    Significant: {sig_result['is_significant']}")

    # Profitable vs unprofitable
    profitable = [r for r in results if r.net_profit > 0]
    unprofitable = [r for r in results if r.net_profit <= 0]

    print(f"\n--- Summary ---")
    print(f"  Profitable Strategies:   {len(profitable)}/{len(results)}")
    print(f"  Unprofitable Strategies: {len(unprofitable)}/{len(results)}")

    if unprofitable:
        print(f"\n  ⚠️  Worst Strategy: {unprofitable[-1].name} ({unprofitable[-1].net_profit:+.2f}%)")

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
