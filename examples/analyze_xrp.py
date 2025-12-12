"""
Analyze real XRP price data for the last month.
Fetches data from Binance API and calculates trading metrics.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from analize import CloudAIAnalyzer
from analize.features.indicators import TechnicalIndicators


def fetch_xrp_data(days: int = 30, interval: str = "1h") -> pd.DataFrame:
    """
    Fetch XRP/USDT data from Binance.

    Args:
        days: Number of days of history
        interval: Candle interval (1m, 5m, 15m, 1h, 4h, 1d)

    Returns:
        DataFrame with OHLCV data
    """
    # Try Binance US first (works in geo-restricted regions), then Binance
    urls = [
        "https://api.binance.us/api/v3/klines",
        "https://api.binance.com/api/v3/klines",
    ]

    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

    params = {
        "symbol": "XRPUSDT",
        "interval": interval,
        "startTime": start_time,
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
                    print(f"  Using: {url}")
                    break
                else:
                    print(f"  {url}: Empty response")
            else:
                print(f"  {url}: HTTP {response.status_code}")
        except Exception as e:
            print(f"  {url}: {e}")

    if not data or len(data) == 0:
        print("\nCould not fetch data from Binance. Using sample data instead.")
        # Generate sample data for demonstration
        import numpy as np
        np.random.seed(42)
        n = 720  # 30 days * 24 hours
        dates = pd.date_range(end=datetime.now(), periods=n, freq="h")
        price = 2.0 + np.cumsum(np.random.randn(n) * 0.02)
        price = np.maximum(price, 0.5)  # Ensure positive

        return pd.DataFrame({
            "timestamp": dates,
            "open": price - np.random.rand(n) * 0.01,
            "high": price + np.random.rand(n) * 0.02,
            "low": price - np.random.rand(n) * 0.02,
            "close": price,
            "volume": np.random.rand(n) * 1000000 + 100000,
        })

    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])

    # Convert types
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def simulate_simple_strategy(df: pd.DataFrame) -> pd.Series:
    """
    Simulate a simple RSI-based strategy to generate P&L.

    Buy when RSI < 30, Sell when RSI > 70.
    This is just for demonstration - not financial advice!
    """
    # Calculate RSI
    rsi = TechnicalIndicators.rsi(df["close"], period=14)

    # Generate signals
    position = 0
    entry_price = 0
    pnl_list = []

    for i in range(len(df)):
        if pd.isna(rsi.iloc[i]):
            continue

        current_price = df["close"].iloc[i]

        # Entry: RSI < 30 (oversold)
        if position == 0 and rsi.iloc[i] < 30:
            position = 1
            entry_price = current_price

        # Exit: RSI > 70 (overbought)
        elif position == 1 and rsi.iloc[i] > 70:
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
            pnl_list.append(pnl_pct)
            position = 0
            entry_price = 0

    return pd.Series(pnl_list)


def main():
    print("=" * 60)
    print("XRP/USDT Analysis - Last 30 Days")
    print("=" * 60)

    # Fetch data
    print("\nFetching XRP data from Binance...")
    df = fetch_xrp_data(days=30, interval="1h")

    print(f"Data range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Total candles: {len(df)}")

    # Price stats
    print("\n--- Price Statistics ---")
    print(f"Current Price: ${df['close'].iloc[-1]:.4f}")
    print(f"30-Day High:   ${df['high'].max():.4f}")
    print(f"30-Day Low:    ${df['low'].min():.4f}")
    print(f"30-Day Change: {((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100:.2f}%")

    # Calculate technical indicators
    print("\n--- Technical Indicators (Current) ---")
    rsi = TechnicalIndicators.rsi(df["close"], period=14)
    sma_20 = TechnicalIndicators.sma(df["close"], period=20)
    sma_50 = TechnicalIndicators.sma(df["close"], period=50)
    upper, middle, lower = TechnicalIndicators.bollinger_bands(df["close"], period=20)

    print(f"RSI (14):      {rsi.iloc[-1]:.2f}")
    print(f"SMA (20):      ${sma_20.iloc[-1]:.4f}")
    print(f"SMA (50):      ${sma_50.iloc[-1]:.4f}")
    print(f"BB Upper:      ${upper.iloc[-1]:.4f}")
    print(f"BB Lower:      ${lower.iloc[-1]:.4f}")

    # Trend analysis
    trend = "BULLISH" if df['close'].iloc[-1] > sma_50.iloc[-1] else "BEARISH"
    print(f"\nTrend (vs SMA50): {trend}")

    # Simulate strategy and analyze
    print("\n--- Simulated RSI Strategy Analysis ---")
    pnl = simulate_simple_strategy(df)

    if len(pnl) > 0:
        analyzer = CloudAIAnalyzer()
        metrics = analyzer.calculate_metrics(pnl)

        print(f"Total Trades:    {metrics['total_trades']}")
        print(f"Win Rate:        {metrics['win_rate']:.1f}%")
        print(f"Profit Factor:   {metrics['profit_factor']:.2f}")
        print(f"Net Profit:      {metrics['net_profit']:.2f}%")
        print(f"Avg Win:         {metrics['avg_win']:.2f}%")
        print(f"Avg Loss:        {metrics['avg_loss']:.2f}%")
        print(f"Max Drawdown:    {metrics['max_drawdown_pct']:.2f}%")
        print(f"Sharpe Ratio:    {metrics['sharpe_ratio']:.2f}")

        # Significance test
        wins = metrics['wins']
        total = metrics['total_trades']
        if total >= 5:
            sig_result = analyzer.test_significance(wins=wins, total=total)
            print(f"\nStatistical Significance:")
            print(f"  P-value:     {sig_result['p_value']:.4f}")
            print(f"  Significant: {sig_result['is_significant']}")
    else:
        print("No completed trades in this period (RSI didn't trigger)")

    # Volatility analysis
    print("\n--- Volatility Analysis ---")
    returns = df["close"].pct_change().dropna()
    print(f"Daily Volatility:    {returns.std() * 100:.2f}%")
    print(f"Annualized Vol:      {returns.std() * (365 ** 0.5) * 100:.2f}%")

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
