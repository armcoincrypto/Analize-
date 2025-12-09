"""
Multi-Strategy Analysis for Top Coins

Test 10+ different strategies on XRP, ATOM, SOL to find the best approach for each.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from itertools import product
from analize.features.indicators import TechnicalIndicators


COINS = ["XRPUSDT", "SOLUSDT", "ATOMUSDT"]
FEE_PCT = 0.2  # 0.2% round trip


def fetch_data(symbol: str, days: int = 365) -> pd.DataFrame:
    """Fetch daily data."""
    urls = [
        "https://api.binance.com/api/v3/klines",
        "https://api.binance.us/api/v3/klines",
    ]

    params = {
        "symbol": symbol,
        "interval": "1d",
        "startTime": int((datetime.now() - timedelta(days=days)).timestamp() * 1000),
        "endTime": int(datetime.now().timestamp() * 1000),
        "limit": 1000
    }

    for url in urls:
        try:
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    break
        except:
            continue
    else:
        return pd.DataFrame()

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
# STRATEGIES
# ============================================================================

def strategy_bollinger(df: pd.DataFrame, period: int = 20, std: float = 1.5) -> list:
    """Bollinger Bands - Buy at lower, sell at middle."""
    upper, middle, lower = TechnicalIndicators.bollinger_bands(df["close"], period=period, std_dev=std)
    trades = []
    pos = None

    for i in range(period, len(df)):
        price = df["close"].iloc[i]
        if pos is None and price <= lower.iloc[i]:
            pos = {"entry": price, "date": df["timestamp"].iloc[i]}
        elif pos and price >= middle.iloc[i]:
            pnl = ((price - pos["entry"]) / pos["entry"]) * 100 - FEE_PCT
            trades.append({"pnl": pnl, "days": (df["timestamp"].iloc[i] - pos["date"]).days})
            pos = None
    return trades


def strategy_rsi(df: pd.DataFrame, period: int = 14, buy: int = 30, sell: int = 70) -> list:
    """RSI - Buy oversold, sell overbought."""
    rsi = TechnicalIndicators.rsi(df["close"], period=period)
    trades = []
    pos = None

    for i in range(period, len(df)):
        price = df["close"].iloc[i]
        if pd.isna(rsi.iloc[i]):
            continue
        if pos is None and rsi.iloc[i] < buy:
            pos = {"entry": price, "date": df["timestamp"].iloc[i]}
        elif pos and rsi.iloc[i] > sell:
            pnl = ((price - pos["entry"]) / pos["entry"]) * 100 - FEE_PCT
            trades.append({"pnl": pnl, "days": (df["timestamp"].iloc[i] - pos["date"]).days})
            pos = None
    return trades


def strategy_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> list:
    """MACD crossover."""
    macd, sig, _ = TechnicalIndicators.macd(df["close"], fast_period=fast, slow_period=slow, signal_period=signal)
    trades = []
    pos = None

    for i in range(1, len(df)):
        if pd.isna(macd.iloc[i]) or pd.isna(sig.iloc[i]):
            continue
        price = df["close"].iloc[i]

        # Buy when MACD crosses above signal
        if pos is None and macd.iloc[i] > sig.iloc[i] and macd.iloc[i-1] <= sig.iloc[i-1]:
            pos = {"entry": price, "date": df["timestamp"].iloc[i]}
        # Sell when MACD crosses below signal
        elif pos and macd.iloc[i] < sig.iloc[i] and macd.iloc[i-1] >= sig.iloc[i-1]:
            pnl = ((price - pos["entry"]) / pos["entry"]) * 100 - FEE_PCT
            trades.append({"pnl": pnl, "days": (df["timestamp"].iloc[i] - pos["date"]).days})
            pos = None
    return trades


def strategy_ema_cross(df: pd.DataFrame, fast: int = 9, slow: int = 21) -> list:
    """EMA crossover."""
    ema_fast = TechnicalIndicators.ema(df["close"], period=fast)
    ema_slow = TechnicalIndicators.ema(df["close"], period=slow)
    trades = []
    pos = None

    for i in range(1, len(df)):
        if pd.isna(ema_fast.iloc[i]) or pd.isna(ema_slow.iloc[i]):
            continue
        price = df["close"].iloc[i]

        if pos is None and ema_fast.iloc[i] > ema_slow.iloc[i] and ema_fast.iloc[i-1] <= ema_slow.iloc[i-1]:
            pos = {"entry": price, "date": df["timestamp"].iloc[i]}
        elif pos and ema_fast.iloc[i] < ema_slow.iloc[i] and ema_fast.iloc[i-1] >= ema_slow.iloc[i-1]:
            pnl = ((price - pos["entry"]) / pos["entry"]) * 100 - FEE_PCT
            trades.append({"pnl": pnl, "days": (df["timestamp"].iloc[i] - pos["date"]).days})
            pos = None
    return trades


def strategy_stochastic(df: pd.DataFrame, k: int = 14, buy: int = 20, sell: int = 80) -> list:
    """Stochastic oscillator."""
    k_line, _ = TechnicalIndicators.stochastic(df["high"], df["low"], df["close"], k_period=k)
    trades = []
    pos = None

    for i in range(k, len(df)):
        if pd.isna(k_line.iloc[i]):
            continue
        price = df["close"].iloc[i]

        if pos is None and k_line.iloc[i] < buy:
            pos = {"entry": price, "date": df["timestamp"].iloc[i]}
        elif pos and k_line.iloc[i] > sell:
            pnl = ((price - pos["entry"]) / pos["entry"]) * 100 - FEE_PCT
            trades.append({"pnl": pnl, "days": (df["timestamp"].iloc[i] - pos["date"]).days})
            pos = None
    return trades


def strategy_donchian(df: pd.DataFrame, period: int = 20) -> list:
    """Donchian channel breakout."""
    high_max = df["high"].rolling(period).max()
    low_min = df["low"].rolling(period).min()
    trades = []
    pos = None

    for i in range(period + 1, len(df)):
        price = df["close"].iloc[i]

        if pos is None and price > high_max.iloc[i-1]:
            pos = {"entry": price, "date": df["timestamp"].iloc[i]}
        elif pos and price < low_min.iloc[i-1]:
            pnl = ((price - pos["entry"]) / pos["entry"]) * 100 - FEE_PCT
            trades.append({"pnl": pnl, "days": (df["timestamp"].iloc[i] - pos["date"]).days})
            pos = None
    return trades


def strategy_mean_reversion(df: pd.DataFrame, period: int = 20, std_entry: float = 2.0) -> list:
    """Mean reversion - buy when N std devs below mean."""
    mean = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()
    trades = []
    pos = None

    for i in range(period, len(df)):
        if pd.isna(mean.iloc[i]) or pd.isna(std.iloc[i]) or std.iloc[i] == 0:
            continue
        price = df["close"].iloc[i]
        z = (price - mean.iloc[i]) / std.iloc[i]

        if pos is None and z < -std_entry:
            pos = {"entry": price, "date": df["timestamp"].iloc[i]}
        elif pos and z > 0:
            pnl = ((price - pos["entry"]) / pos["entry"]) * 100 - FEE_PCT
            trades.append({"pnl": pnl, "days": (df["timestamp"].iloc[i] - pos["date"]).days})
            pos = None
    return trades


def strategy_sma_trend(df: pd.DataFrame, period: int = 50) -> list:
    """SMA trend following - buy above SMA, sell below."""
    sma = TechnicalIndicators.sma(df["close"], period=period)
    trades = []
    pos = None

    for i in range(1, len(df)):
        if pd.isna(sma.iloc[i]):
            continue
        price = df["close"].iloc[i]
        prev = df["close"].iloc[i-1]

        if pos is None and price > sma.iloc[i] and prev <= sma.iloc[i-1]:
            pos = {"entry": price, "date": df["timestamp"].iloc[i]}
        elif pos and price < sma.iloc[i] and prev >= sma.iloc[i-1]:
            pnl = ((price - pos["entry"]) / pos["entry"]) * 100 - FEE_PCT
            trades.append({"pnl": pnl, "days": (df["timestamp"].iloc[i] - pos["date"]).days})
            pos = None
    return trades


def strategy_rsi_divergence(df: pd.DataFrame, rsi_period: int = 14) -> list:
    """RSI + price trend - buy when RSI oversold and price making higher lows."""
    rsi = TechnicalIndicators.rsi(df["close"], period=rsi_period)
    trades = []
    pos = None

    for i in range(rsi_period + 5, len(df)):
        if pd.isna(rsi.iloc[i]):
            continue
        price = df["close"].iloc[i]

        # Price making higher low but RSI oversold
        price_higher_low = df["low"].iloc[i] > df["low"].iloc[i-5:i].min()

        if pos is None and rsi.iloc[i] < 35 and price_higher_low:
            pos = {"entry": price, "date": df["timestamp"].iloc[i]}
        elif pos and rsi.iloc[i] > 65:
            pnl = ((price - pos["entry"]) / pos["entry"]) * 100 - FEE_PCT
            trades.append({"pnl": pnl, "days": (df["timestamp"].iloc[i] - pos["date"]).days})
            pos = None
    return trades


def strategy_golden_cross(df: pd.DataFrame) -> list:
    """Golden Cross - 50 SMA crosses above 200 SMA."""
    sma50 = TechnicalIndicators.sma(df["close"], period=50)
    sma200 = TechnicalIndicators.sma(df["close"], period=200)
    trades = []
    pos = None

    for i in range(1, len(df)):
        if pd.isna(sma50.iloc[i]) or pd.isna(sma200.iloc[i]):
            continue
        price = df["close"].iloc[i]

        # Golden cross - buy
        if pos is None and sma50.iloc[i] > sma200.iloc[i] and sma50.iloc[i-1] <= sma200.iloc[i-1]:
            pos = {"entry": price, "date": df["timestamp"].iloc[i]}
        # Death cross - sell
        elif pos and sma50.iloc[i] < sma200.iloc[i] and sma50.iloc[i-1] >= sma200.iloc[i-1]:
            pnl = ((price - pos["entry"]) / pos["entry"]) * 100 - FEE_PCT
            trades.append({"pnl": pnl, "days": (df["timestamp"].iloc[i] - pos["date"]).days})
            pos = None
    return trades


# All strategies to test
STRATEGIES = {
    "Bollinger (20,1.5)": lambda df: strategy_bollinger(df, 20, 1.5),
    "Bollinger (20,2.0)": lambda df: strategy_bollinger(df, 20, 2.0),
    "RSI (14,30,70)": lambda df: strategy_rsi(df, 14, 30, 70),
    "RSI (7,25,75)": lambda df: strategy_rsi(df, 7, 25, 75),
    "RSI (14,25,75)": lambda df: strategy_rsi(df, 14, 25, 75),
    "MACD (12,26,9)": lambda df: strategy_macd(df, 12, 26, 9),
    "MACD (8,21,5)": lambda df: strategy_macd(df, 8, 21, 5),
    "EMA Cross (9,21)": lambda df: strategy_ema_cross(df, 9, 21),
    "EMA Cross (12,26)": lambda df: strategy_ema_cross(df, 12, 26),
    "EMA Cross (5,20)": lambda df: strategy_ema_cross(df, 5, 20),
    "Stochastic (14,20,80)": lambda df: strategy_stochastic(df, 14, 20, 80),
    "Stochastic (21,20,80)": lambda df: strategy_stochastic(df, 21, 20, 80),
    "Donchian (20)": lambda df: strategy_donchian(df, 20),
    "Donchian (55)": lambda df: strategy_donchian(df, 55),
    "Mean Reversion (2σ)": lambda df: strategy_mean_reversion(df, 20, 2.0),
    "Mean Reversion (1.5σ)": lambda df: strategy_mean_reversion(df, 20, 1.5),
    "SMA Trend (50)": lambda df: strategy_sma_trend(df, 50),
    "SMA Trend (20)": lambda df: strategy_sma_trend(df, 20),
    "RSI Divergence": lambda df: strategy_rsi_divergence(df, 14),
    "Golden Cross": lambda df: strategy_golden_cross(df),
}


def analyze_strategy(trades: list) -> dict:
    """Calculate strategy metrics."""
    if len(trades) < 3:
        return None

    pnls = [t["pnl"] for t in trades]
    wins = len([p for p in pnls if p > 0])

    return {
        "trades": len(trades),
        "wins": wins,
        "win_rate": wins / len(trades) * 100,
        "total_pnl": sum(pnls),
        "avg_pnl": np.mean(pnls),
        "avg_days": np.mean([t["days"] for t in trades]),
        "best": max(pnls),
        "worst": min(pnls),
    }


def validate_strategy(df: pd.DataFrame, strategy_func) -> tuple:
    """Walk-forward validation."""
    split = int(len(df) * 0.7)

    trades_in = strategy_func(df.iloc[:split])
    trades_out = strategy_func(df.iloc[split:])

    in_pnl = sum(t["pnl"] for t in trades_in) if trades_in else -999
    out_pnl = sum(t["pnl"] for t in trades_out) if trades_out else -999

    return in_pnl, out_pnl, (in_pnl > 0 and out_pnl > 0)


def main():
    print("=" * 80)
    print("MULTI-STRATEGY ANALYSIS")
    print("Testing 20 strategies on XRP, SOL, ATOM")
    print("=" * 80)

    all_results = {}

    for symbol in COINS:
        coin = symbol.replace("USDT", "")
        print(f"\n{'='*40}")
        print(f"Analyzing {coin}...")
        print("="*40)

        df = fetch_data(symbol, days=365)
        if df.empty:
            print(f"  Failed to fetch {symbol}")
            continue

        print(f"  Loaded {len(df)} days")

        results = []

        for name, func in STRATEGIES.items():
            try:
                trades = func(df)
                metrics = analyze_strategy(trades)

                if metrics:
                    in_pnl, out_pnl, validated = validate_strategy(df, func)

                    results.append({
                        "strategy": name,
                        "trades": metrics["trades"],
                        "win_rate": metrics["win_rate"],
                        "total_pnl": metrics["total_pnl"],
                        "avg_pnl": metrics["avg_pnl"],
                        "avg_days": metrics["avg_days"],
                        "in_sample": in_pnl,
                        "out_sample": out_pnl,
                        "validated": validated,
                    })
            except Exception as e:
                pass

        # Sort by total P&L
        results.sort(key=lambda x: x["total_pnl"], reverse=True)
        all_results[coin] = results

        # Display results for this coin
        print(f"\n{'Strategy':<22} {'Trades':>7} {'Win%':>6} {'Total%':>9} {'Avg%':>7} {'Valid':>6}")
        print("-" * 65)

        for r in results[:15]:
            valid = "✓" if r["validated"] else ""
            print(f"{r['strategy']:<22} {r['trades']:>7} {r['win_rate']:>5.0f}% {r['total_pnl']:>+8.1f}% {r['avg_pnl']:>+6.1f}% {valid:>6}")

        # Best validated
        validated = [r for r in results if r["validated"]]
        if validated:
            best = validated[0]
            print(f"\n  🏆 Best Validated: {best['strategy']}")
            print(f"     Profit: {best['total_pnl']:+.1f}% | Win Rate: {best['win_rate']:.0f}%")
            print(f"     In-Sample: {best['in_sample']:+.1f}% | Out-Sample: {best['out_sample']:+.1f}%")

    # Final comparison
    print("\n" + "=" * 80)
    print("BEST STRATEGY FOR EACH COIN")
    print("=" * 80)

    for coin, results in all_results.items():
        validated = [r for r in results if r["validated"]]
        if validated:
            best = validated[0]
            print(f"""
🏆 {coin}
   Strategy:     {best['strategy']}
   Net Profit:   {best['total_pnl']:+.2f}%
   Trades:       {best['trades']}
   Win Rate:     {best['win_rate']:.0f}%
   Avg Trade:    {best['avg_pnl']:+.2f}%
   Validated:    ✓ (In: {best['in_sample']:+.1f}%, Out: {best['out_sample']:+.1f}%)
""")
        else:
            print(f"\n❌ {coin}: No validated strategy found")

    # Strategy comparison across all coins
    print("\n" + "=" * 80)
    print("STRATEGY PERFORMANCE ACROSS ALL COINS")
    print("=" * 80)

    strategy_scores = {}
    for coin, results in all_results.items():
        for r in results:
            name = r["strategy"]
            if name not in strategy_scores:
                strategy_scores[name] = {"total": 0, "validated": 0, "coins": []}
            strategy_scores[name]["total"] += r["total_pnl"]
            if r["validated"]:
                strategy_scores[name]["validated"] += 1
                strategy_scores[name]["coins"].append(coin)

    # Sort by total profit across all coins
    sorted_strategies = sorted(strategy_scores.items(), key=lambda x: x[1]["total"], reverse=True)

    print(f"\n{'Strategy':<22} {'Total Profit':>14} {'Validated':>10} {'Coins':<20}")
    print("-" * 70)

    for name, data in sorted_strategies[:10]:
        coins_str = ", ".join(data["coins"]) if data["coins"] else "-"
        print(f"{name:<22} {data['total']:>+12.1f}%  {data['validated']:>6}/3     {coins_str:<20}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Find universally good strategies
    universal = [name for name, data in strategy_scores.items() if data["validated"] >= 2]

    print(f"""
   Total Strategies Tested: {len(STRATEGIES)}
   Coins Analyzed:          {len(COINS)}

   Strategies validated on 2+ coins:
""")
    for name in universal:
        data = strategy_scores[name]
        print(f"   • {name}: {data['total']:+.1f}% total, validated on {', '.join(data['coins'])}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
