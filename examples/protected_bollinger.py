"""
Protected Bollinger Strategy with Smart Filters

Adds 5 protection layers to avoid the ~20% failed entries:
1. Trend Filter - Only buy if price > 200-day MA (avoid crashes)
2. Volume Filter - Require above-average volume (avoid fake moves)
3. First-Touch Rejection - Skip first touch after crash
4. Band Penetration Filter - Price must go 1-2% BELOW band
5. Time-Based Exit - Exit after 20-25 days max (avoid dead trades)
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from analize.features.indicators import TechnicalIndicators


# ============================================================================
# CONFIGURATION
# ============================================================================

COINS = ["XRPUSDT", "SOLUSDT", "ATOMUSDT"]

# Bollinger parameters
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 1.5

# Protection parameters
TREND_MA_PERIOD = 200           # 200-day MA for trend filter
VOLUME_MA_PERIOD = 20           # 20-day volume MA
VOLUME_MULTIPLIER = 0.8         # Require 80% of average volume minimum
PENETRATION_PCT = 1.0           # Price must go 1% BELOW lower band
CRASH_LOOKBACK = 10             # Days to look back for crash detection
CRASH_THRESHOLD = -15.0         # -15% drop = crash
FIRST_TOUCH_COOLDOWN = 5        # Days to wait after crash before buying
MAX_HOLD_DAYS = 22              # Exit after 22 days if no bounce
FEE_PCT = 0.2                   # 0.2% round trip


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ProtectionStatus:
    """Status of all protection filters."""
    trend_ok: bool = False
    volume_ok: bool = False
    not_first_touch: bool = False
    penetration_ok: bool = False

    # Details
    price: float = 0.0
    ma_200: float = 0.0
    volume_ratio: float = 0.0
    days_since_crash: int = 999
    penetration_pct: float = 0.0

    @property
    def all_passed(self) -> bool:
        return self.trend_ok and self.volume_ok and self.not_first_touch and self.penetration_ok

    @property
    def passed_count(self) -> int:
        return sum([self.trend_ok, self.volume_ok, self.not_first_touch, self.penetration_ok])


@dataclass
class Signal:
    """Enhanced trading signal with protection info."""
    symbol: str
    name: str
    signal_type: str  # "STRONG_BUY", "WEAK_BUY", "WAIT", "SELL", "TIME_EXIT"
    price: float
    lower_band: float
    middle_band: float
    protection: ProtectionStatus
    message: str
    timestamp: datetime


@dataclass
class Trade:
    """Trade record with time tracking."""
    entry_date: datetime
    entry_price: float
    exit_date: Optional[datetime] = None
    exit_price: float = 0.0
    exit_reason: str = ""  # "TARGET", "TIME_EXIT", "OPEN"
    days_held: int = 0
    pnl_pct: float = 0.0


# ============================================================================
# DATA FETCHING
# ============================================================================

def fetch_data(symbol: str, days: int = 365) -> pd.DataFrame:
    """Fetch daily data from Binance."""
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
                    df = pd.DataFrame(data, columns=[
                        "timestamp", "open", "high", "low", "close", "volume",
                        "close_time", "quote_volume", "trades", "taker_buy_base",
                        "taker_buy_quote", "ignore"
                    ])
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                    for col in ["open", "high", "low", "close", "volume"]:
                        df[col] = df[col].astype(float)
                    return df[["timestamp", "open", "high", "low", "close", "volume"]]
        except:
            continue

    return pd.DataFrame()


# ============================================================================
# PROTECTION FILTERS
# ============================================================================

class ProtectionFilters:
    """All protection filters in one class."""

    @staticmethod
    def check_trend(df: pd.DataFrame, idx: int) -> tuple[bool, float, float]:
        """
        Filter 1: Trend Filter
        Only buy if price is above 200-day MA.
        Prevents buying during crashes/downtrends.
        """
        if idx < TREND_MA_PERIOD:
            return False, 0, 0

        ma_200 = df["close"].iloc[idx-TREND_MA_PERIOD+1:idx+1].mean()
        price = df["close"].iloc[idx]

        return price > ma_200, price, ma_200

    @staticmethod
    def check_volume(df: pd.DataFrame, idx: int) -> tuple[bool, float]:
        """
        Filter 2: Volume Filter
        Require above-average volume to avoid fake/weak moves.
        Low volume dips often continue falling.
        """
        if idx < VOLUME_MA_PERIOD:
            return False, 0

        vol_ma = df["volume"].iloc[idx-VOLUME_MA_PERIOD:idx].mean()
        current_vol = df["volume"].iloc[idx]

        ratio = current_vol / vol_ma if vol_ma > 0 else 0
        return ratio >= VOLUME_MULTIPLIER, ratio

    @staticmethod
    def check_first_touch(df: pd.DataFrame, idx: int) -> tuple[bool, int]:
        """
        Filter 3: First-Touch Rejection
        Don't buy on first touch after a crash (>15% drop in 10 days).
        Price usually keeps falling after crash before stabilizing.
        """
        if idx < CRASH_LOOKBACK:
            return True, 999

        # Check for recent crash
        lookback_start = max(0, idx - CRASH_LOOKBACK)
        high_in_lookback = df["close"].iloc[lookback_start:idx].max()
        current_price = df["close"].iloc[idx]

        drop_pct = ((current_price - high_in_lookback) / high_in_lookback) * 100

        if drop_pct <= CRASH_THRESHOLD:
            # There was a crash - count days since crash peak
            # Use argmax() for position-based index (not DataFrame index)
            lookback_slice = df["close"].iloc[lookback_start:idx]
            high_pos_in_slice = lookback_slice.values.argmax()
            # Days since high = distance from high position to current
            days_since = (idx - lookback_start - 1) - high_pos_in_slice

            # Need to wait FIRST_TOUCH_COOLDOWN days after crash
            if days_since < FIRST_TOUCH_COOLDOWN:
                return False, max(0, days_since)

        return True, 999

    @staticmethod
    def check_penetration(price: float, lower_band: float) -> tuple[bool, float]:
        """
        Filter 4: Band Penetration Filter
        Price must go 1-2% BELOW the lower band, not just touch it.
        Prevents weak touches that don't indicate real oversold.
        """
        if lower_band <= 0:
            return False, 0

        penetration = ((lower_band - price) / lower_band) * 100
        return penetration >= PENETRATION_PCT, penetration

    @staticmethod
    def check_all(df: pd.DataFrame, idx: int, lower_band: float) -> ProtectionStatus:
        """Check all protection filters and return status."""
        status = ProtectionStatus()

        # Filter 1: Trend
        status.trend_ok, status.price, status.ma_200 = ProtectionFilters.check_trend(df, idx)

        # Filter 2: Volume
        status.volume_ok, status.volume_ratio = ProtectionFilters.check_volume(df, idx)

        # Filter 3: First-touch rejection
        status.not_first_touch, status.days_since_crash = ProtectionFilters.check_first_touch(df, idx)

        # Filter 4: Penetration
        status.penetration_ok, status.penetration_pct = ProtectionFilters.check_penetration(
            df["close"].iloc[idx], lower_band
        )

        return status


# ============================================================================
# PROTECTED STRATEGY
# ============================================================================

def run_protected_strategy(df: pd.DataFrame) -> tuple[List[Trade], pd.DataFrame]:
    """
    Run Bollinger strategy with all 5 protections.

    Entry rules (ALL must pass):
    1. Price below lower Bollinger band
    2. Price above 200-day MA (trend filter)
    3. Volume >= 80% of 20-day average (volume filter)
    4. Not first touch after crash (first-touch rejection)
    5. Price penetrated 1%+ below band (penetration filter)

    Exit rules (ANY triggers exit):
    1. Price reaches middle band (target)
    2. Position held > 22 days (time exit)
    """
    # Calculate indicators
    upper, middle, lower = TechnicalIndicators.bollinger_bands(
        df["close"], period=BOLLINGER_PERIOD, std_dev=BOLLINGER_STD
    )

    df = df.copy()
    df["upper"] = upper
    df["middle"] = middle
    df["lower"] = lower
    df["ma_200"] = df["close"].rolling(TREND_MA_PERIOD).mean()
    df["vol_ma"] = df["volume"].rolling(VOLUME_MA_PERIOD).mean()

    trades = []
    position = None

    start_idx = max(BOLLINGER_PERIOD, TREND_MA_PERIOD) + 5

    for i in range(start_idx, len(df)):
        price = df["close"].iloc[i]
        lower_band = df["lower"].iloc[i]
        middle_band = df["middle"].iloc[i]
        date = df["timestamp"].iloc[i]

        # If in position, check exits
        if position is not None:
            days_held = (date - position["entry_date"]).days

            # Exit 1: Target reached (middle band)
            if price >= middle_band:
                pnl = ((price - position["entry_price"]) / position["entry_price"]) * 100 - FEE_PCT
                trades.append(Trade(
                    entry_date=position["entry_date"],
                    entry_price=position["entry_price"],
                    exit_date=date,
                    exit_price=price,
                    exit_reason="TARGET",
                    days_held=days_held,
                    pnl_pct=pnl,
                ))
                position = None
                continue

            # Exit 2: Time-based exit (Filter 5)
            if days_held >= MAX_HOLD_DAYS:
                pnl = ((price - position["entry_price"]) / position["entry_price"]) * 100 - FEE_PCT
                trades.append(Trade(
                    entry_date=position["entry_date"],
                    entry_price=position["entry_price"],
                    exit_date=date,
                    exit_price=price,
                    exit_reason="TIME_EXIT",
                    days_held=days_held,
                    pnl_pct=pnl,
                ))
                position = None
                continue

        # If not in position, check entry
        if position is None and price <= lower_band:
            # Check all protections
            protection = ProtectionFilters.check_all(df, i, lower_band)

            # Only enter if ALL protections pass
            if protection.all_passed:
                position = {
                    "entry_date": date,
                    "entry_price": price,
                    "protection": protection,
                }

    return trades, df


# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_coin(symbol: str, name: str) -> dict:
    """Analyze a coin with protected strategy."""
    df = fetch_data(symbol, days=400)  # Need 400 days for 200 MA + backtest

    if df.empty or len(df) < 250:
        return None

    trades, df_with_indicators = run_protected_strategy(df)

    if not trades:
        return {
            "symbol": symbol,
            "name": name,
            "trades": 0,
            "net_profit": 0,
            "win_rate": 0,
            "target_exits": 0,
            "time_exits": 0,
        }

    # Calculate stats
    wins = [t for t in trades if t.pnl_pct > 0]
    target_exits = [t for t in trades if t.exit_reason == "TARGET"]
    time_exits = [t for t in trades if t.exit_reason == "TIME_EXIT"]

    return {
        "symbol": symbol,
        "name": name,
        "trades": len(trades),
        "wins": len(wins),
        "win_rate": len(wins) / len(trades) * 100,
        "net_profit": sum(t.pnl_pct for t in trades),
        "avg_trade": sum(t.pnl_pct for t in trades) / len(trades),
        "avg_days": np.mean([t.days_held for t in trades]),
        "target_exits": len(target_exits),
        "time_exits": len(time_exits),
        "trades_list": trades,
        "df": df_with_indicators,
    }


def get_current_signal(symbol: str, name: str) -> Optional[Signal]:
    """Get current signal with protection status."""
    df = fetch_data(symbol, days=400)

    if df.empty or len(df) < 250:
        return None

    # Calculate indicators
    upper, middle, lower = TechnicalIndicators.bollinger_bands(
        df["close"], period=BOLLINGER_PERIOD, std_dev=BOLLINGER_STD
    )

    df["lower"] = lower
    df["middle"] = middle

    idx = len(df) - 1
    price = df["close"].iloc[idx]
    lower_band = lower.iloc[idx]
    middle_band = middle.iloc[idx]

    # Check protection status
    protection = ProtectionFilters.check_all(df, idx, lower_band)

    # Determine signal
    if price <= lower_band:
        if protection.all_passed:
            signal_type = "STRONG_BUY"
            message = f"BUY: All {protection.passed_count}/4 protections passed"
        else:
            signal_type = "WEAK_BUY"
            failed = []
            if not protection.trend_ok:
                failed.append("trend")
            if not protection.volume_ok:
                failed.append("volume")
            if not protection.not_first_touch:
                failed.append("first-touch")
            if not protection.penetration_ok:
                failed.append("penetration")
            message = f"CAUTION: {protection.passed_count}/4 passed. Failed: {', '.join(failed)}"
    elif price >= middle_band:
        signal_type = "SELL"
        message = "At middle band - consider taking profits"
    else:
        signal_type = "WAIT"
        distance = ((price - lower_band) / lower_band) * 100
        message = f"Wait for entry ({distance:+.1f}% from lower band)"

    return Signal(
        symbol=symbol,
        name=name,
        signal_type=signal_type,
        price=price,
        lower_band=lower_band,
        middle_band=middle_band,
        protection=protection,
        message=message,
        timestamp=datetime.now(),
    )


# ============================================================================
# COMPARISON: PROTECTED VS UNPROTECTED
# ============================================================================

def run_unprotected_strategy(df: pd.DataFrame) -> List[Trade]:
    """Run original strategy without protections for comparison."""
    upper, middle, lower = TechnicalIndicators.bollinger_bands(
        df["close"], period=BOLLINGER_PERIOD, std_dev=BOLLINGER_STD
    )

    trades = []
    position = None

    for i in range(BOLLINGER_PERIOD, len(df)):
        price = df["close"].iloc[i]
        lower_band = lower.iloc[i]
        middle_band = middle.iloc[i]
        date = df["timestamp"].iloc[i]

        if position is None and price <= lower_band:
            position = {"entry_date": date, "entry_price": price}

        elif position is not None and price >= middle_band:
            pnl = ((price - position["entry_price"]) / position["entry_price"]) * 100 - FEE_PCT
            days_held = (date - position["entry_date"]).days
            trades.append(Trade(
                entry_date=position["entry_date"],
                entry_price=position["entry_price"],
                exit_date=date,
                exit_price=price,
                exit_reason="TARGET",
                days_held=days_held,
                pnl_pct=pnl,
            ))
            position = None

    return trades


def compare_strategies(symbol: str, name: str) -> dict:
    """Compare protected vs unprotected performance."""
    df = fetch_data(symbol, days=400)

    if df.empty or len(df) < 250:
        return None

    # Run both strategies
    protected_trades, _ = run_protected_strategy(df)
    unprotected_trades = run_unprotected_strategy(df)

    def calc_stats(trades):
        if not trades:
            return {"trades": 0, "profit": 0, "win_rate": 0}
        wins = len([t for t in trades if t.pnl_pct > 0])
        return {
            "trades": len(trades),
            "profit": sum(t.pnl_pct for t in trades),
            "win_rate": wins / len(trades) * 100,
            "avg_trade": sum(t.pnl_pct for t in trades) / len(trades),
        }

    return {
        "symbol": symbol,
        "name": name,
        "protected": calc_stats(protected_trades),
        "unprotected": calc_stats(unprotected_trades),
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("PROTECTED BOLLINGER STRATEGY")
    print("=" * 80)

    print("""
    5 PROTECTION FILTERS:
    ─────────────────────────────────────────────────────────────────────────
    1. TREND FILTER       │ Price must be > 200-day MA
                          │ Prevents buying during crashes
    ─────────────────────────────────────────────────────────────────────────
    2. VOLUME FILTER      │ Volume must be >= 80% of 20-day average
                          │ Avoids fake/low-liquidity moves
    ─────────────────────────────────────────────────────────────────────────
    3. FIRST-TOUCH        │ Don't buy first touch after >15% crash
       REJECTION          │ Price usually keeps falling initially
    ─────────────────────────────────────────────────────────────────────────
    4. BAND PENETRATION   │ Price must go 1%+ BELOW lower band
                          │ Prevents weak touches
    ─────────────────────────────────────────────────────────────────────────
    5. TIME-BASED EXIT    │ Exit after 22 days if no bounce
                          │ Prevents dead/stuck trades
    ─────────────────────────────────────────────────────────────────────────
    """)

    print("\n" + "=" * 80)
    print("STRATEGY COMPARISON: PROTECTED vs UNPROTECTED")
    print("=" * 80)

    coins = [
        ("XRPUSDT", "XRP"),
        ("SOLUSDT", "SOL"),
        ("ATOMUSDT", "ATOM"),
    ]

    print(f"\n{'Coin':<8} │ {'Strategy':<12} │ {'Trades':<8} │ {'Win Rate':<10} │ {'Net Profit':<12} │ {'Avg Trade':<10}")
    print("─" * 80)

    total_protected = {"trades": 0, "profit": 0, "wins": 0}
    total_unprotected = {"trades": 0, "profit": 0, "wins": 0}

    for symbol, name in coins:
        print(f"\n  Analyzing {name}...", end=" ")
        result = compare_strategies(symbol, name)

        if result is None:
            print("Failed")
            continue

        print("Done")

        p = result["protected"]
        u = result["unprotected"]

        print(f"{name:<8} │ {'Unprotected':<12} │ {u['trades']:<8} │ {u['win_rate']:>7.0f}%   │ {u['profit']:>+10.1f}%  │ {u.get('avg_trade', 0):>+8.1f}%")
        print(f"{'':<8} │ {'Protected':<12} │ {p['trades']:<8} │ {p['win_rate']:>7.0f}%   │ {p['profit']:>+10.1f}%  │ {p.get('avg_trade', 0):>+8.1f}%")

        improvement = p['win_rate'] - u['win_rate']
        print(f"{'':<8} │ {'Improvement':<12} │ {'':<8} │ {improvement:>+7.0f}%   │ {'':<12} │")
        print("─" * 80)

        total_protected["trades"] += p["trades"]
        total_protected["profit"] += p["profit"]
        total_unprotected["trades"] += u["trades"]
        total_unprotected["profit"] += u["profit"]

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"""
    UNPROTECTED STRATEGY:
    ─────────────────────
    Total Trades:   {total_unprotected['trades']}
    Total Profit:   {total_unprotected['profit']:+.1f}%

    PROTECTED STRATEGY:
    ─────────────────────
    Total Trades:   {total_protected['trades']}
    Total Profit:   {total_protected['profit']:+.1f}%

    IMPACT:
    ─────────────────────
    Trades filtered: {total_unprotected['trades'] - total_protected['trades']}
    """)

    # Current signals
    print("\n" + "=" * 80)
    print("CURRENT SIGNALS WITH PROTECTION STATUS")
    print("=" * 80)

    for symbol, name in coins:
        signal = get_current_signal(symbol, name)
        if signal is None:
            continue

        p = signal.protection

        print(f"""
    {signal.name} - {signal.signal_type}
    ─────────────────────────────────────
    Price:        ${signal.price:.4f}
    Lower Band:   ${signal.lower_band:.4f}
    Middle Band:  ${signal.middle_band:.4f}

    Protection Status ({p.passed_count}/4):
      Trend (>200MA):     {'PASS' if p.trend_ok else 'FAIL'} (Price ${p.price:.2f} vs MA ${p.ma_200:.2f})
      Volume (>=80%):     {'PASS' if p.volume_ok else 'FAIL'} ({p.volume_ratio:.0%} of average)
      First-Touch:        {'PASS' if p.not_first_touch else 'FAIL'} ({p.days_since_crash} days since crash)
      Penetration (>=1%): {'PASS' if p.penetration_ok else 'FAIL'} ({p.penetration_pct:.1f}% below band)

    Verdict: {signal.message}
    """)

    print("=" * 80)


if __name__ == "__main__":
    main()
