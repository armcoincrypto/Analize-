"""
Daily Timeframe Strategy Optimizer

Uses daily candles for fewer trades and lower fee impact.
Tests additional strategies and parameter combinations.
"""

import sqlite3
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from analize import CloudAIAnalyzer
from analize.features.indicators import TechnicalIndicators


DB_PATH = Path(__file__).parent.parent / "data" / "market_data.db"

# Lower fees for limit orders (maker)
TRADING_FEE_PCT = 0.075  # Maker fee
SLIPPAGE_PCT = 0.02      # Lower slippage on daily
SPREAD_PCT = 0.01        # Lower spread


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv_daily (
            symbol TEXT, timestamp TEXT, open REAL, high REAL,
            low REAL, close REAL, volume REAL,
            UNIQUE(symbol, timestamp)
        )
    """)
    conn.commit()
    conn.close()


def fetch_daily_data(symbol: str = "XRPUSDT", days: int = 365) -> pd.DataFrame:
    """Fetch daily candles - up to 1 year of data."""
    urls = [
        "https://api.binance.us/api/v3/klines",  # US first (works in geo-restricted regions)
        "https://api.binance.com/api/v3/klines",
    ]

    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

    params = {
        "symbol": symbol,
        "interval": "1d",
        "startTime": start_time,
        "endTime": end_time,
        "limit": 1000
    }

    print(f"Fetching {symbol} daily data ({days} days)...")

    for url in urls:
        try:
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    print(f"  Source: {url}")
                    break
        except Exception as e:
            print(f"  {url}: {e}")
            continue
    else:
        print("  Failed to fetch data!")
        return pd.DataFrame()

    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    # Save to database
    conn = sqlite3.connect(DB_PATH)
    for _, row in df.iterrows():
        conn.execute("""
            INSERT OR REPLACE INTO ohlcv_daily VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (symbol, row["timestamp"].isoformat(), row["open"], row["high"],
              row["low"], row["close"], row["volume"]))
    conn.commit()
    conn.close()

    print(f"  Loaded {len(df)} daily candles")
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def calc_fees(entry: float, exit: float) -> float:
    """Calculate round-trip fees as percentage."""
    fee = (TRADING_FEE_PCT * 2 + SLIPPAGE_PCT * 2 + SPREAD_PCT)
    return fee


# ============================================================================
# DAILY TIMEFRAME STRATEGIES
# ============================================================================

def strategy_rsi_daily(df: pd.DataFrame, period: int = 14,
                       buy: int = 30, sell: int = 70) -> tuple[pd.Series, int]:
    """RSI strategy for daily timeframe."""
    rsi = TechnicalIndicators.rsi(df["close"], period=period)

    position = 0
    entry_price = 0
    pnl_list = []

    for i in range(len(df)):
        if pd.isna(rsi.iloc[i]):
            continue
        price = df["close"].iloc[i]

        if position == 0 and rsi.iloc[i] < buy:
            position = 1
            entry_price = price
        elif position == 1 and rsi.iloc[i] > sell:
            gross = ((price - entry_price) / entry_price) * 100
            net = gross - calc_fees(entry_price, price)
            pnl_list.append(net)
            position = 0

    return pd.Series(pnl_list), len(pnl_list)


def strategy_sma_cross_daily(df: pd.DataFrame, fast: int = 10,
                              slow: int = 30) -> tuple[pd.Series, int]:
    """SMA crossover for daily timeframe."""
    sma_fast = TechnicalIndicators.sma(df["close"], period=fast)
    sma_slow = TechnicalIndicators.sma(df["close"], period=slow)

    position = 0
    entry_price = 0
    pnl_list = []

    for i in range(1, len(df)):
        if pd.isna(sma_fast.iloc[i]) or pd.isna(sma_slow.iloc[i]):
            continue
        price = df["close"].iloc[i]

        # Golden cross
        if position == 0 and sma_fast.iloc[i] > sma_slow.iloc[i] and sma_fast.iloc[i-1] <= sma_slow.iloc[i-1]:
            position = 1
            entry_price = price
        # Death cross
        elif position == 1 and sma_fast.iloc[i] < sma_slow.iloc[i]:
            gross = ((price - entry_price) / entry_price) * 100
            net = gross - calc_fees(entry_price, price)
            pnl_list.append(net)
            position = 0

    return pd.Series(pnl_list), len(pnl_list)


def strategy_ema_cross_daily(df: pd.DataFrame, fast: int = 12,
                              slow: int = 26) -> tuple[pd.Series, int]:
    """EMA crossover for daily timeframe."""
    ema_fast = TechnicalIndicators.ema(df["close"], period=fast)
    ema_slow = TechnicalIndicators.ema(df["close"], period=slow)

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
        elif position == 1 and ema_fast.iloc[i] < ema_slow.iloc[i]:
            gross = ((price - entry_price) / entry_price) * 100
            net = gross - calc_fees(entry_price, price)
            pnl_list.append(net)
            position = 0

    return pd.Series(pnl_list), len(pnl_list)


def strategy_bollinger_daily(df: pd.DataFrame, period: int = 20,
                              std: float = 2.0) -> tuple[pd.Series, int]:
    """Bollinger Bands mean reversion."""
    upper, middle, lower = TechnicalIndicators.bollinger_bands(df["close"], period=period, std_dev=std)

    position = 0
    entry_price = 0
    pnl_list = []

    for i in range(len(df)):
        if pd.isna(lower.iloc[i]):
            continue
        price = df["close"].iloc[i]

        # Buy at lower band
        if position == 0 and price <= lower.iloc[i]:
            position = 1
            entry_price = price
        # Sell at middle or upper band
        elif position == 1 and price >= middle.iloc[i]:
            gross = ((price - entry_price) / entry_price) * 100
            net = gross - calc_fees(entry_price, price)
            pnl_list.append(net)
            position = 0

    return pd.Series(pnl_list), len(pnl_list)


def strategy_macd_daily(df: pd.DataFrame, fast: int = 12, slow: int = 26,
                         signal: int = 9) -> tuple[pd.Series, int]:
    """MACD crossover strategy."""
    macd_line, signal_line, _ = TechnicalIndicators.macd(
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
        elif position == 1 and macd_line.iloc[i] < signal_line.iloc[i]:
            gross = ((price - entry_price) / entry_price) * 100
            net = gross - calc_fees(entry_price, price)
            pnl_list.append(net)
            position = 0

    return pd.Series(pnl_list), len(pnl_list)


def strategy_stochastic_daily(df: pd.DataFrame, k: int = 14, d: int = 3,
                               buy: int = 20, sell: int = 80) -> tuple[pd.Series, int]:
    """Stochastic oscillator strategy."""
    k_line, d_line = TechnicalIndicators.stochastic(
        df["high"], df["low"], df["close"], k_period=k, d_period=d
    )

    position = 0
    entry_price = 0
    pnl_list = []

    for i in range(len(df)):
        if pd.isna(k_line.iloc[i]):
            continue
        price = df["close"].iloc[i]

        if position == 0 and k_line.iloc[i] < buy:
            position = 1
            entry_price = price
        elif position == 1 and k_line.iloc[i] > sell:
            gross = ((price - entry_price) / entry_price) * 100
            net = gross - calc_fees(entry_price, price)
            pnl_list.append(net)
            position = 0

    return pd.Series(pnl_list), len(pnl_list)


def strategy_donchian_breakout(df: pd.DataFrame, period: int = 20) -> tuple[pd.Series, int]:
    """Donchian Channel breakout - trend following."""
    high_max = df["high"].rolling(period).max()
    low_min = df["low"].rolling(period).min()

    position = 0
    entry_price = 0
    pnl_list = []

    for i in range(period, len(df)):
        price = df["close"].iloc[i]

        # Breakout above high
        if position == 0 and price > high_max.iloc[i-1]:
            position = 1
            entry_price = price
        # Break below low - exit
        elif position == 1 and price < low_min.iloc[i-1]:
            gross = ((price - entry_price) / entry_price) * 100
            net = gross - calc_fees(entry_price, price)
            pnl_list.append(net)
            position = 0

    return pd.Series(pnl_list), len(pnl_list)


def strategy_adx_trend(df: pd.DataFrame, period: int = 14,
                        adx_threshold: int = 25) -> tuple[pd.Series, int]:
    """ADX trend strength + EMA direction."""
    ema = TechnicalIndicators.ema(df["close"], period=20)

    # Simple ADX approximation using ATR
    atr = TechnicalIndicators.atr(df["high"], df["low"], df["close"], period=period)

    position = 0
    entry_price = 0
    pnl_list = []

    for i in range(period + 1, len(df)):
        if pd.isna(ema.iloc[i]) or pd.isna(atr.iloc[i]):
            continue
        price = df["close"].iloc[i]

        # Trending up - price above EMA
        if position == 0 and price > ema.iloc[i] and price > df["close"].iloc[i-1]:
            position = 1
            entry_price = price
        # Exit when trend reverses
        elif position == 1 and price < ema.iloc[i]:
            gross = ((price - entry_price) / entry_price) * 100
            net = gross - calc_fees(entry_price, price)
            pnl_list.append(net)
            position = 0

    return pd.Series(pnl_list), len(pnl_list)


def strategy_support_resistance(df: pd.DataFrame, lookback: int = 20,
                                 bounce_pct: float = 2.0) -> tuple[pd.Series, int]:
    """Buy at support, sell at resistance."""
    support = df["low"].rolling(lookback).min()
    resistance = df["high"].rolling(lookback).max()

    position = 0
    entry_price = 0
    pnl_list = []

    for i in range(lookback, len(df)):
        price = df["close"].iloc[i]

        # Near support (within bounce_pct%)
        if position == 0:
            support_zone = support.iloc[i] * (1 + bounce_pct/100)
            if price <= support_zone:
                position = 1
                entry_price = price
        # Near resistance - sell
        elif position == 1:
            resistance_zone = resistance.iloc[i] * (1 - bounce_pct/100)
            if price >= resistance_zone:
                gross = ((price - entry_price) / entry_price) * 100
                net = gross - calc_fees(entry_price, price)
                pnl_list.append(net)
                position = 0

    return pd.Series(pnl_list), len(pnl_list)


def strategy_volume_breakout(df: pd.DataFrame, vol_mult: float = 2.0,
                              price_period: int = 10) -> tuple[pd.Series, int]:
    """Buy on high volume breakouts."""
    vol_avg = df["volume"].rolling(20).mean()
    price_high = df["high"].rolling(price_period).max()

    position = 0
    entry_price = 0
    pnl_list = []

    for i in range(20, len(df)):
        if pd.isna(vol_avg.iloc[i]):
            continue
        price = df["close"].iloc[i]
        volume = df["volume"].iloc[i]

        # High volume breakout
        if position == 0 and volume > vol_avg.iloc[i] * vol_mult and price > price_high.iloc[i-1]:
            position = 1
            entry_price = price
        # Exit after pullback
        elif position == 1 and price < df["close"].iloc[i-1] * 0.97:
            gross = ((price - entry_price) / entry_price) * 100
            net = gross - calc_fees(entry_price, price)
            pnl_list.append(net)
            position = 0

    return pd.Series(pnl_list), len(pnl_list)


# Strategy configurations
STRATEGIES = {
    "RSI": {
        "func": strategy_rsi_daily,
        "params": {
            "period": [7, 14, 21],
            "buy": [25, 30, 35],
            "sell": [65, 70, 75],
        }
    },
    "SMA_Cross": {
        "func": strategy_sma_cross_daily,
        "params": {
            "fast": [5, 10, 20],
            "slow": [20, 50, 100, 200],
        }
    },
    "EMA_Cross": {
        "func": strategy_ema_cross_daily,
        "params": {
            "fast": [8, 12, 21],
            "slow": [21, 50, 100],
        }
    },
    "Bollinger": {
        "func": strategy_bollinger_daily,
        "params": {
            "period": [15, 20, 25],
            "std": [1.5, 2.0, 2.5],
        }
    },
    "MACD": {
        "func": strategy_macd_daily,
        "params": {
            "fast": [8, 12],
            "slow": [21, 26],
            "signal": [7, 9],
        }
    },
    "Stochastic": {
        "func": strategy_stochastic_daily,
        "params": {
            "k": [9, 14, 21],
            "d": [3, 5],
            "buy": [15, 20, 25],
            "sell": [75, 80, 85],
        }
    },
    "Donchian": {
        "func": strategy_donchian_breakout,
        "params": {
            "period": [10, 20, 30, 55],
        }
    },
    "ADX_Trend": {
        "func": strategy_adx_trend,
        "params": {
            "period": [10, 14, 20],
            "adx_threshold": [20, 25, 30],
        }
    },
    "Support_Resistance": {
        "func": strategy_support_resistance,
        "params": {
            "lookback": [10, 20, 30],
            "bounce_pct": [1.5, 2.0, 3.0],
        }
    },
    "Volume_Breakout": {
        "func": strategy_volume_breakout,
        "params": {
            "vol_mult": [1.5, 2.0, 2.5],
            "price_period": [5, 10, 20],
        }
    },
}


@dataclass
class Result:
    strategy: str
    params: dict
    trades: int
    win_rate: float
    net_profit: float
    profit_factor: float
    sharpe: float
    in_sample: float
    out_sample: float
    validated: bool


def walk_forward_test(df: pd.DataFrame, func, params: dict) -> tuple[float, float, bool]:
    """70/30 walk-forward validation."""
    split = int(len(df) * 0.7)

    pnl_in, _ = func(df.iloc[:split], **params)
    pnl_out, _ = func(df.iloc[split:], **params)

    in_profit = pnl_in.sum() if len(pnl_in) > 0 else -999
    out_profit = pnl_out.sum() if len(pnl_out) > 0 else -999

    validated = in_profit > 0 and out_profit > 0
    return in_profit, out_profit, validated


def run_optimization(df: pd.DataFrame, min_trades: int = 5) -> list[Result]:
    """Run full optimization."""
    analyzer = CloudAIAnalyzer()
    results = []

    total = sum(np.prod([len(v) for v in c["params"].values()]) for c in STRATEGIES.values())
    print(f"\nTesting {total} parameter combinations on daily data...")

    tested = 0
    profitable = 0
    validated = 0

    for name, config in STRATEGIES.items():
        func = config["func"]
        param_names = list(config["params"].keys())
        param_values = list(config["params"].values())

        for combo in product(*param_values):
            tested += 1
            params = dict(zip(param_names, combo))

            try:
                pnl, trades = func(df, **params)

                if trades < min_trades:
                    continue

                metrics = analyzer.calculate_metrics(pnl)
                in_s, out_s, valid = walk_forward_test(df, func, params)

                result = Result(
                    strategy=name,
                    params=params,
                    trades=trades,
                    win_rate=metrics["win_rate"],
                    net_profit=metrics["net_profit"],
                    profit_factor=metrics["profit_factor"],
                    sharpe=metrics["sharpe_ratio"],
                    in_sample=in_s,
                    out_sample=out_s,
                    validated=valid,
                )
                results.append(result)

                if result.net_profit > 0:
                    profitable += 1
                if valid:
                    validated += 1

            except Exception as e:
                pass

            if tested % 100 == 0:
                print(f"  Progress: {tested}/{total} | Profitable: {profitable} | Validated: {validated}")

    print(f"\nDone: {tested} tested, {len(results)} valid, {profitable} profitable, {validated} validated")

    results.sort(key=lambda x: (x.validated, x.net_profit), reverse=True)
    return results


def main():
    print("=" * 70)
    print("DAILY TIMEFRAME STRATEGY OPTIMIZER")
    print("Lower fees, fewer trades, more reliable signals")
    print("=" * 70)

    init_db()

    # Fetch 1 year of daily data
    df = fetch_daily_data("XRPUSDT", days=365)

    if df.empty:
        print("Failed to fetch data!")
        return

    print(f"\n  Date range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
    print(f"  Days: {len(df)}")
    print(f"  Price: ${df['close'].iloc[0]:.4f} → ${df['close'].iloc[-1]:.4f}")
    print(f"  Change: {((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100:+.1f}%")

    print(f"\n--- Trading Costs (per round trip) ---")
    total_fee = TRADING_FEE_PCT * 2 + SLIPPAGE_PCT * 2 + SPREAD_PCT
    print(f"  Trading fee: {TRADING_FEE_PCT}% × 2 = {TRADING_FEE_PCT * 2}%")
    print(f"  Slippage:    {SLIPPAGE_PCT}% × 2 = {SLIPPAGE_PCT * 2}%")
    print(f"  Spread:      {SPREAD_PCT}%")
    print(f"  TOTAL:       {total_fee}% per trade")

    # Count strategies
    print(f"\n--- Strategies to Test ---")
    for name, config in STRATEGIES.items():
        combos = np.prod([len(v) for v in config["params"].values()])
        print(f"  {name}: {int(combos)} combinations")

    # Run optimization
    results = run_optimization(df, min_trades=5)

    # Results
    print("\n" + "=" * 70)
    print("TOP 20 STRATEGIES (sorted by validation + profit)")
    print("=" * 70)

    print(f"\n{'Strategy':<18} {'Parameters':<30} {'Net%':>8} {'Trades':>7} {'Win%':>6} {'Valid':>6}")
    print("-" * 80)

    for r in results[:20]:
        params_str = str(r.params).replace("'", "")[:28]
        valid_mark = "✓" if r.validated else ""
        print(f"{r.strategy:<18} {params_str:<30} {r.net_profit:>+7.1f}% {r.trades:>6} {r.win_rate:>5.0f}% {valid_mark:>6}")

    # Validated strategies
    validated = [r for r in results if r.validated]

    print("\n" + "=" * 70)
    print(f"VALIDATED STRATEGIES: {len(validated)}")
    print("=" * 70)

    if validated:
        for r in validated[:10]:
            print(f"""
🏆 {r.strategy}
   Parameters:   {r.params}
   Net Profit:   {r.net_profit:+.2f}%
   Trades:       {r.trades}
   Win Rate:     {r.win_rate:.1f}%
   Sharpe:       {r.sharpe:.2f}
   In-Sample:    {r.in_sample:+.2f}%
   Out-Sample:   {r.out_sample:+.2f}%
""")
    else:
        print("\n⚠️  No strategies passed walk-forward validation.")
        print("   Consider:")
        print("   - Different market (BTC, ETH)")
        print("   - Different timeframe")
        print("   - Lower fee tier")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    profitable_count = len([r for r in results if r.net_profit > 0])
    print(f"""
  Total combinations: {sum(np.prod([len(v) for v in c['params'].values()]) for c in STRATEGIES.values())}
  Valid (≥5 trades):  {len(results)}
  Profitable:         {profitable_count} ({profitable_count/max(len(results),1)*100:.1f}%)
  Validated:          {len(validated)} ({len(validated)/max(len(results),1)*100:.1f}%)

  Fee impact: ~{total_fee}% per trade
  With 10 trades: ~{total_fee * 10}% in fees
""")

    if validated:
        best = validated[0]
        print(f"  🏆 BEST: {best.strategy} with {best.params}")
        print(f"     Net: {best.net_profit:+.2f}% | Validated: ✓")


if __name__ == "__main__":
    main()
