"""
Multi-Coin Bollinger Strategy Analysis

Analyze BNB, LTC, SOL, ADA, ATOM with the same Bollinger strategy.
Compare which coins work best for mean reversion trading.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from analize.features.indicators import TechnicalIndicators


COINS = ["BNBUSDT", "LTCUSDT", "SOLUSDT", "ADAUSDT", "ATOMUSDT", "XRPUSDT"]

# Bollinger parameters (best from XRP)
PERIOD = 20
STD_DEV = 1.5
FEE_PCT = 0.2  # 0.2% round trip


@dataclass
class CoinResult:
    symbol: str
    trades: int
    wins: int
    win_rate: float
    gross_profit: float
    fees: float
    net_profit: float
    avg_trade: float
    avg_days: float
    in_sample: float
    out_sample: float
    validated: bool
    current_signal: str
    distance_to_buy: float


def fetch_data(symbol: str, days: int = 365) -> pd.DataFrame:
    """Fetch daily data for a symbol."""
    urls = [
        "https://api.binance.us/api/v3/klines",  # US first (works in geo-restricted regions)
        "https://api.binance.com/api/v3/klines",
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


def run_bollinger_strategy(df: pd.DataFrame) -> tuple[list, pd.DataFrame]:
    """Run Bollinger strategy and return trades."""
    upper, middle, lower = TechnicalIndicators.bollinger_bands(
        df["close"], period=PERIOD, std_dev=STD_DEV
    )

    df = df.copy()
    df["upper"] = upper
    df["middle"] = middle
    df["lower"] = lower

    trades = []
    position = None

    for i in range(PERIOD, len(df)):
        price = df["close"].iloc[i]
        lower_band = df["lower"].iloc[i]
        middle_band = df["middle"].iloc[i]
        date = df["timestamp"].iloc[i]

        if position is None and price <= lower_band:
            position = {"entry_date": date, "entry_price": price}

        elif position is not None and price >= middle_band:
            pnl = ((price - position["entry_price"]) / position["entry_price"]) * 100
            trades.append({
                "entry_date": position["entry_date"],
                "exit_date": date,
                "entry_price": position["entry_price"],
                "exit_price": price,
                "days": (date - position["entry_date"]).days,
                "pnl": pnl - FEE_PCT,
            })
            position = None

    return trades, df


def analyze_coin(symbol: str) -> CoinResult:
    """Analyze a single coin."""
    df = fetch_data(symbol, days=365)

    if df.empty or len(df) < 50:
        return None

    trades, df_with_bands = run_bollinger_strategy(df)

    if len(trades) < 3:
        # Not enough trades
        current = df.iloc[-1]
        lower = df_with_bands["lower"].iloc[-1]
        distance = ((current["close"] - lower) / lower) * 100

        return CoinResult(
            symbol=symbol.replace("USDT", ""),
            trades=len(trades),
            wins=0,
            win_rate=0,
            gross_profit=0,
            fees=0,
            net_profit=0,
            avg_trade=0,
            avg_days=0,
            in_sample=0,
            out_sample=0,
            validated=False,
            current_signal="WAIT" if distance > 0 else "BUY",
            distance_to_buy=distance,
        )

    # Calculate metrics
    wins = len([t for t in trades if t["pnl"] > 0])
    gross = sum(t["pnl"] + FEE_PCT for t in trades)
    net = sum(t["pnl"] for t in trades)
    fees = len(trades) * FEE_PCT

    # Walk-forward validation
    split = int(len(df) * 0.7)
    df_in = df.iloc[:split]
    df_out = df.iloc[split:]

    trades_in, _ = run_bollinger_strategy(df_in)
    trades_out, _ = run_bollinger_strategy(df_out)

    in_sample = sum(t["pnl"] for t in trades_in)
    out_sample = sum(t["pnl"] for t in trades_out)
    validated = in_sample > 0 and out_sample > 0

    # Current signal
    current_price = df["close"].iloc[-1]
    lower_band = df_with_bands["lower"].iloc[-1]
    distance = ((current_price - lower_band) / lower_band) * 100

    if current_price <= lower_band:
        signal = "🔔 BUY NOW"
    else:
        signal = "WAIT"

    return CoinResult(
        symbol=symbol.replace("USDT", ""),
        trades=len(trades),
        wins=wins,
        win_rate=wins / len(trades) * 100 if trades else 0,
        gross_profit=gross,
        fees=fees,
        net_profit=net,
        avg_trade=net / len(trades) if trades else 0,
        avg_days=np.mean([t["days"] for t in trades]) if trades else 0,
        in_sample=in_sample,
        out_sample=out_sample,
        validated=validated,
        current_signal=signal,
        distance_to_buy=distance,
    )


def main():
    print("=" * 80)
    print("MULTI-COIN BOLLINGER STRATEGY ANALYSIS")
    print("=" * 80)
    print(f"\nStrategy: Bollinger Bands (period={PERIOD}, std={STD_DEV})")
    print(f"Logic: Buy at lower band, sell at middle band")
    print(f"Fees: {FEE_PCT}% per round trip")
    print(f"\nAnalyzing {len(COINS)} coins over 365 days...")

    results = []

    for symbol in COINS:
        print(f"\n  Fetching {symbol}...", end=" ")
        result = analyze_coin(symbol)
        if result:
            results.append(result)
            status = "✓" if result.validated else "✗"
            print(f"{result.trades} trades, {result.net_profit:+.1f}% {status}")
        else:
            print("Failed")

    # Sort by net profit
    results.sort(key=lambda x: x.net_profit, reverse=True)

    # Results table
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)

    print(f"\n{'Coin':<8} {'Trades':<8} {'Win%':<8} {'Net Profit':<12} {'Avg Trade':<10} {'Validated':<10} {'Signal':<12}")
    print("-" * 80)

    for r in results:
        valid = "✓ YES" if r.validated else "✗ NO"
        print(f"{r.symbol:<8} {r.trades:<8} {r.win_rate:>5.0f}%   {r.net_profit:>+10.1f}%  {r.avg_trade:>+8.1f}%   {valid:<10} {r.current_signal:<12}")

    # Validated coins
    validated = [r for r in results if r.validated]
    not_validated = [r for r in results if not r.validated]

    print("\n" + "=" * 80)
    print("VALIDATED STRATEGIES (Profitable in-sample AND out-of-sample)")
    print("=" * 80)

    if validated:
        for r in validated:
            print(f"""
🏆 {r.symbol}
   Net Profit:    {r.net_profit:+.2f}%
   Trades:        {r.trades}
   Win Rate:      {r.win_rate:.0f}%
   Avg Trade:     {r.avg_trade:+.2f}%
   Avg Duration:  {r.avg_days:.0f} days
   In-Sample:     {r.in_sample:+.2f}%
   Out-Sample:    {r.out_sample:+.2f}%
   Current:       {r.current_signal} (distance to buy: {r.distance_to_buy:+.1f}%)
""")
    else:
        print("\n⚠️  No coins passed validation with this strategy.")

    # Not validated
    if not_validated:
        print("\n" + "-" * 40)
        print("NOT VALIDATED:")
        for r in not_validated:
            print(f"  {r.symbol}: {r.net_profit:+.1f}% (In: {r.in_sample:+.1f}%, Out: {r.out_sample:+.1f}%)")

    # Current opportunities
    print("\n" + "=" * 80)
    print("CURRENT BUY SIGNALS")
    print("=" * 80)

    buy_signals = [r for r in results if "BUY" in r.current_signal]
    if buy_signals:
        for r in buy_signals:
            print(f"\n  🔔 {r.symbol}: BUY NOW! Price at lower band")
    else:
        print("\n  No buy signals currently.")
        print("\n  Closest to buy zone:")
        closest = sorted(results, key=lambda x: x.distance_to_buy)[:3]
        for r in closest:
            print(f"    {r.symbol}: needs {r.distance_to_buy:+.1f}% drop to reach buy zone")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"""
   Coins Analyzed:    {len(results)}
   Validated:         {len(validated)} ({len(validated)/len(results)*100:.0f}%)

   Best Performer:    {results[0].symbol} ({results[0].net_profit:+.1f}%)
   Worst Performer:   {results[-1].symbol} ({results[-1].net_profit:+.1f}%)

   RECOMMENDATION:
   ───────────────
   Trade the VALIDATED coins with Bollinger strategy:
""")

    for r in validated[:3]:
        print(f"   • {r.symbol}: {r.net_profit:+.1f}% profit, {r.win_rate:.0f}% win rate")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
