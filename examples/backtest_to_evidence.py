#!/usr/bin/env python3
"""
Historical Backtest to Evidence Collector

Runs strategy on historical data and records ALL trades to the evidence
collector. This gives you 200+ trades in minutes instead of waiting 7-14 days.

Usage:
    python3 backtest_to_evidence.py --days 90 --symbols ATOMUSDT,XRPUSDT,SOLUSDT
"""

import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import requests

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from analize.features.indicators import TechnicalIndicators

# Import evidence collector
try:
    from evidence_collector import (
        EvidenceCollector, TradeOutcome, SignalContext, MarketRegime
    )
    HAS_EVIDENCE = True
except ImportError:
    print("ERROR: evidence_collector.py not found")
    HAS_EVIDENCE = False
    sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Strategy parameters (same as paper trader)
STRATEGY_PARAMS = {
    "k_period": 21,
    "d_period": 3,
    "buy_threshold": 20,    # Normal mode
    "sell_threshold": 75,   # Normal mode
}

DEMO_PARAMS = {
    "k_period": 21,
    "d_period": 3,
    "buy_threshold": 40,    # Demo mode - tighter
    "sell_threshold": 60,   # Demo mode - tighter
}

# Trading costs
TRADING_FEE_PCT = 0.1
SLIPPAGE_PCT = 0.05

# Symbols to test
DEFAULT_SYMBOLS = ["ATOMUSDT", "XRPUSDT", "SOLUSDT", "ETHUSDT", "BTCUSDT"]


# =============================================================================
# DATA FETCHING
# =============================================================================

def fetch_historical_data(symbol: str, days: int = 90, interval: str = "1h") -> pd.DataFrame:
    """Fetch historical OHLCV data."""
    print(f"  Fetching {symbol} ({days} days, {interval} interval)...")

    urls = [
        "https://api.binance.us/api/v3/klines",
        "https://api.binance.com/api/v3/klines",
    ]

    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

    # For hourly data over 90 days, we need multiple requests
    all_data = []
    current_start = start_time

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
                response = requests.get(url, params=params, timeout=15)
                if response.status_code == 200:
                    data = response.json()
                    if data:
                        break
            except:
                continue

        if not data:
            break

        all_data.extend(data)

        if len(data) < 1000:
            break

        # Move start time forward
        current_start = data[-1][0] + 1

    if not all_data:
        print(f"    ⚠️ No data for {symbol}")
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    print(f"    ✓ {len(df)} candles fetched")
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def detect_regime(df: pd.DataFrame, idx: int) -> str:
    """Detect market regime at given index."""
    if idx < 20:
        return "unknown"

    # Get recent data
    recent = df.iloc[max(0, idx-20):idx+1]

    # Calculate metrics
    returns = recent["close"].pct_change().dropna()
    volatility = returns.std() * 100

    # Simple trend detection
    sma_short = recent["close"].iloc[-5:].mean()
    sma_long = recent["close"].iloc[-20:].mean()

    # ADX approximation (simplified)
    high_low = recent["high"] - recent["low"]
    atr = high_low.mean()
    atr_pct = atr / recent["close"].mean() * 100

    # Classify regime
    if volatility > 3:
        return "volatile"
    elif sma_short > sma_long * 1.02:
        return "bull"
    elif sma_short < sma_long * 0.98:
        return "bear"
    else:
        return "sideways"


def calculate_indicators(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Calculate all indicators for the dataframe."""
    df = df.copy()

    # Stochastic
    k, d = TechnicalIndicators.stochastic(
        df["high"], df["low"], df["close"],
        k_period=params["k_period"],
        d_period=params["d_period"]
    )
    df["stoch_k"] = k
    df["stoch_d"] = d

    # RSI
    df["rsi"] = TechnicalIndicators.rsi(df["close"], period=14)

    # Bollinger Bands
    upper, middle, lower = TechnicalIndicators.bollinger_bands(df["close"], period=20)
    df["bb_upper"] = upper
    df["bb_middle"] = middle
    df["bb_lower"] = lower
    df["bb_touch_lower"] = df["close"] <= df["bb_lower"]
    df["bb_touch_upper"] = df["close"] >= df["bb_upper"]

    # MACD
    macd_line, signal_line, histogram = TechnicalIndicators.macd(df["close"])
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = histogram

    # ATR for volatility
    df["atr"] = TechnicalIndicators.atr(df["high"], df["low"], df["close"], period=14)
    df["atr_pct"] = df["atr"] / df["close"] * 100

    return df


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

def run_backtest(
    df: pd.DataFrame,
    symbol: str,
    params: dict,
    evidence: EvidenceCollector,
    initial_capital: float = 1000.0
) -> dict:
    """
    Run backtest and record all trades to evidence collector.
    """
    print(f"\n  Running backtest on {symbol}...")

    # Calculate indicators
    df = calculate_indicators(df, params)

    # Skip warm-up period
    start_idx = max(params["k_period"], 20) + 5

    # Trading state
    capital = initial_capital
    position = None  # {"side": "long/short", "entry_price": x, "entry_time": t, "entry_idx": i}
    trades = []
    trade_counter = 0

    # Track MFE/MAE during position
    max_favorable = 0.0
    max_adverse = 0.0

    for i in range(start_idx, len(df)):
        row = df.iloc[i]
        price = row["close"]
        timestamp = row["timestamp"]
        stoch_k = row["stoch_k"]

        if pd.isna(stoch_k):
            continue

        # Update MFE/MAE if in position
        if position:
            if position["side"] == "long":
                pnl_pct = (price - position["entry_price"]) / position["entry_price"] * 100
            else:
                pnl_pct = (position["entry_price"] - price) / position["entry_price"] * 100

            max_favorable = max(max_favorable, pnl_pct)
            max_adverse = min(max_adverse, pnl_pct)

        # Generate signal
        signal = "HOLD"
        if stoch_k < params["buy_threshold"]:
            signal = "BUY"
        elif stoch_k > params["sell_threshold"]:
            signal = "SELL"

        # Execute trades
        if position is None:
            # Open position
            if signal == "BUY":
                position = {
                    "side": "long",
                    "entry_price": price,
                    "entry_time": timestamp,
                    "entry_idx": i,
                    "entry_regime": detect_regime(df, i),
                    "entry_stoch": stoch_k,
                    "entry_rsi": row["rsi"],
                    "entry_bb_touch": row["bb_touch_lower"],
                    "entry_macd_bull": row["macd_hist"] > 0 if not pd.isna(row["macd_hist"]) else False,
                }
                max_favorable = 0.0
                max_adverse = 0.0

            elif signal == "SELL":
                position = {
                    "side": "short",
                    "entry_price": price,
                    "entry_time": timestamp,
                    "entry_idx": i,
                    "entry_regime": detect_regime(df, i),
                    "entry_stoch": stoch_k,
                    "entry_rsi": row["rsi"],
                    "entry_bb_touch": row["bb_touch_upper"],
                    "entry_macd_bull": row["macd_hist"] > 0 if not pd.isna(row["macd_hist"]) else False,
                }
                max_favorable = 0.0
                max_adverse = 0.0

        else:
            # Check for exit
            should_exit = False
            exit_reason = ""

            if position["side"] == "long" and signal == "SELL":
                should_exit = True
                exit_reason = "signal_reversal"
            elif position["side"] == "short" and signal == "BUY":
                should_exit = True
                exit_reason = "signal_reversal"

            # Time stop (max 48 hours in position)
            if not should_exit and (i - position["entry_idx"]) > 48:
                should_exit = True
                exit_reason = "time_stop"

            if should_exit:
                trade_counter += 1

                # Calculate P&L
                if position["side"] == "long":
                    gross_pnl_pct = (price - position["entry_price"]) / position["entry_price"] * 100
                else:
                    gross_pnl_pct = (position["entry_price"] - price) / position["entry_price"] * 100

                # Apply costs
                total_cost_pct = (TRADING_FEE_PCT * 2) + SLIPPAGE_PCT
                net_pnl_pct = gross_pnl_pct - total_cost_pct

                is_win = net_pnl_pct > 0

                # Build signal context
                signals = SignalContext(
                    stochastic_value=position["entry_stoch"],
                    stochastic_signal="oversold" if position["entry_stoch"] < 20 else ("overbought" if position["entry_stoch"] > 80 else "neutral"),
                    rsi_value=position["entry_rsi"] if not pd.isna(position["entry_rsi"]) else 50,
                    rsi_signal="oversold" if position.get("entry_rsi", 50) < 30 else ("overbought" if position.get("entry_rsi", 50) > 70 else "neutral"),
                    bollinger_touch=position["entry_bb_touch"],
                    macd_signal="bullish" if position["entry_macd_bull"] else "bearish",
                )

                # Create trade outcome
                trade_outcome = TradeOutcome(
                    trade_id=f"{symbol}_{trade_counter}_{timestamp.strftime('%Y%m%d%H%M')}",
                    symbol=symbol,
                    side=position["side"],
                    entry_time=position["entry_time"],
                    entry_price=position["entry_price"],
                    entry_regime=position["entry_regime"],
                    entry_signals=signals,
                    entry_confidence=0.5,
                    max_favorable_excursion=max_favorable,
                    max_adverse_excursion=max_adverse,
                    time_in_trade_seconds=int((timestamp - position["entry_time"]).total_seconds()),
                    exit_time=timestamp,
                    exit_price=price,
                    exit_reason=exit_reason,
                    slippage_pct=SLIPPAGE_PCT,
                    fees_pct=TRADING_FEE_PCT * 2,
                    gross_pnl_pct=gross_pnl_pct,
                    net_pnl_pct=net_pnl_pct,
                    is_win=is_win
                )

                # Record to evidence collector
                evidence.record_trade(trade_outcome)

                trades.append({
                    "symbol": symbol,
                    "side": position["side"],
                    "entry_time": position["entry_time"],
                    "exit_time": timestamp,
                    "entry_price": position["entry_price"],
                    "exit_price": price,
                    "net_pnl_pct": net_pnl_pct,
                    "is_win": is_win,
                    "regime": position["entry_regime"],
                    "mfe": max_favorable,
                    "mae": max_adverse,
                })

                # Update capital
                capital *= (1 + net_pnl_pct / 100)

                # Reset position
                position = None
                max_favorable = 0.0
                max_adverse = 0.0

    # Close any open position at end
    if position:
        trade_counter += 1
        price = df.iloc[-1]["close"]
        timestamp = df.iloc[-1]["timestamp"]

        if position["side"] == "long":
            gross_pnl_pct = (price - position["entry_price"]) / position["entry_price"] * 100
        else:
            gross_pnl_pct = (position["entry_price"] - price) / position["entry_price"] * 100

        total_cost_pct = (TRADING_FEE_PCT * 2) + SLIPPAGE_PCT
        net_pnl_pct = gross_pnl_pct - total_cost_pct
        is_win = net_pnl_pct > 0

        signals = SignalContext(
            stochastic_value=position["entry_stoch"],
            stochastic_signal="oversold" if position["entry_stoch"] < 20 else "neutral",
        )

        trade_outcome = TradeOutcome(
            trade_id=f"{symbol}_{trade_counter}_{timestamp.strftime('%Y%m%d%H%M')}",
            symbol=symbol,
            side=position["side"],
            entry_time=position["entry_time"],
            entry_price=position["entry_price"],
            entry_regime=position["entry_regime"],
            entry_signals=signals,
            entry_confidence=0.5,
            max_favorable_excursion=max_favorable,
            max_adverse_excursion=max_adverse,
            time_in_trade_seconds=int((timestamp - position["entry_time"]).total_seconds()),
            exit_time=timestamp,
            exit_price=price,
            exit_reason="session_end",
            slippage_pct=SLIPPAGE_PCT,
            fees_pct=TRADING_FEE_PCT * 2,
            gross_pnl_pct=gross_pnl_pct,
            net_pnl_pct=net_pnl_pct,
            is_win=is_win
        )

        evidence.record_trade(trade_outcome)

        trades.append({
            "symbol": symbol,
            "side": position["side"],
            "entry_time": position["entry_time"],
            "exit_time": timestamp,
            "net_pnl_pct": net_pnl_pct,
            "is_win": is_win,
        })

        capital *= (1 + net_pnl_pct / 100)

    # Summary
    wins = sum(1 for t in trades if t["is_win"])
    total_pnl = sum(t["net_pnl_pct"] for t in trades)

    print(f"    ✓ {len(trades)} trades | Win Rate: {wins/len(trades)*100:.1f}% | Total P&L: {total_pnl:+.2f}%")

    return {
        "symbol": symbol,
        "trades": len(trades),
        "wins": wins,
        "win_rate": wins / len(trades) * 100 if trades else 0,
        "total_pnl": total_pnl,
        "final_capital": capital,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Backtest to Evidence Collector")
    parser.add_argument("--days", type=int, default=90, help="Days of historical data")
    parser.add_argument("--symbols", default="ATOMUSDT,XRPUSDT,SOLUSDT", help="Comma-separated symbols")
    parser.add_argument("--demo", action="store_true", help="Use demo mode thresholds (40/60)")
    parser.add_argument("--interval", default="1h", help="Candle interval (1h, 4h, 1d)")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    params = DEMO_PARAMS if args.demo else STRATEGY_PARAMS

    print("=" * 70)
    print("HISTORICAL BACKTEST → EVIDENCE COLLECTOR")
    print("=" * 70)
    print(f"  Days: {args.days}")
    print(f"  Symbols: {symbols}")
    print(f"  Interval: {args.interval}")
    print(f"  Mode: {'DEMO (40/60)' if args.demo else 'NORMAL (20/75)'}")
    print(f"  Strategy: Stochastic K={params['k_period']}, D={params['d_period']}")
    print("=" * 70)

    # Initialize evidence collector
    evidence = EvidenceCollector(initial_capital=1000.0)

    # Run backtest for each symbol
    results = []
    for symbol in symbols:
        df = fetch_historical_data(symbol, days=args.days, interval=args.interval)
        if df.empty:
            continue

        result = run_backtest(df, symbol, params, evidence)
        results.append(result)

    # Print summary
    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)

    total_trades = sum(r["trades"] for r in results)
    total_wins = sum(r["wins"] for r in results)

    print(f"\n  Total Trades Recorded: {total_trades}")
    print(f"  Overall Win Rate: {total_wins/total_trades*100:.1f}%" if total_trades else "  No trades")

    print("\n  By Symbol:")
    for r in results:
        print(f"    {r['symbol']}: {r['trades']} trades, {r['win_rate']:.1f}% win, {r['total_pnl']:+.2f}% P&L")

    print("\n" + "=" * 70)
    print("NOW RUN: python3 evidence_collector.py --report")
    print("=" * 70)


if __name__ == "__main__":
    main()
