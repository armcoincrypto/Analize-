"""
Signal Generator with 5 Protection Filters

Generates trading signals based on Bollinger Band strategy with:
1. Trend Filter (200 MA)
2. Volume Filter (80% of average)
3. First-Touch Rejection (crash cooldown)
4. Band Penetration (1%+ below band)
5. Time-Based Exit (22 days max)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
from datetime import datetime
from enum import Enum


class SignalType(Enum):
    STRONG_BUY = "STRONG_BUY"
    WEAK_BUY = "WEAK_BUY"
    SELL = "SELL"
    TIME_EXIT = "TIME_EXIT"
    HOLD = "HOLD"
    NO_SIGNAL = "NO_SIGNAL"


@dataclass
class ProtectionStatus:
    """Status of all 5 protection filters."""
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

    def failed_filters(self) -> list:
        failed = []
        if not self.trend_ok:
            failed.append("TREND")
        if not self.volume_ok:
            failed.append("VOLUME")
        if not self.not_first_touch:
            failed.append("FIRST-TOUCH")
        if not self.penetration_ok:
            failed.append("PENETRATION")
        return failed


@dataclass
class Signal:
    """Trading signal with full context."""
    symbol: str
    signal_type: SignalType
    price: float
    lower_band: float
    middle_band: float
    upper_band: float
    protection: ProtectionStatus
    timestamp: datetime
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "signal_type": self.signal_type.value,
            "price": self.price,
            "lower_band": self.lower_band,
            "middle_band": self.middle_band,
            "upper_band": self.upper_band,
            "protection_passed": self.protection.passed_count,
            "protection_all_ok": self.protection.all_passed,
            "timestamp": self.timestamp.isoformat(),
            "reason": self.reason,
        }


class TechnicalIndicators:
    """Technical indicator calculations."""

    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        return series.rolling(window=period).mean()

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def bollinger_bands(
        series: pd.Series,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.

        Returns: (upper_band, middle_band, lower_band)
        """
        middle = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()

        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)

        return upper, middle, lower


class SignalGenerator:
    """
    Generates trading signals with protection filters.

    Strategy: Bollinger Mean Reversion
    Entry: Price <= Lower Band + All protections pass
    Exit: Price >= Middle Band OR Time > 22 days
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.strategy = config.get("strategy", {})

        # Strategy parameters
        self.trend_ma_period = self.strategy.get("trend_ma_period", 200)
        self.volume_ma_period = self.strategy.get("volume_ma_period", 20)
        self.volume_min_ratio = self.strategy.get("volume_min_ratio", 0.8)
        self.crash_threshold_pct = self.strategy.get("crash_threshold_pct", -15.0)
        self.first_touch_cooldown = self.strategy.get("first_touch_cooldown_days", 5)
        self.penetration_min_pct = self.strategy.get("penetration_min_pct", 1.0)
        self.max_hold_days = self.strategy.get("max_hold_days", 22)

    def _check_trend(self, df: pd.DataFrame, idx: int) -> Tuple[bool, float, float]:
        """Filter 1: Price must be above 200-day MA."""
        if idx < self.trend_ma_period:
            return False, 0, 0

        ma_200 = df["close"].iloc[idx - self.trend_ma_period + 1:idx + 1].mean()
        price = df["close"].iloc[idx]

        return price > ma_200, price, ma_200

    def _check_volume(self, df: pd.DataFrame, idx: int) -> Tuple[bool, float]:
        """Filter 2: Volume must be >= 80% of 20-day average."""
        if idx < self.volume_ma_period:
            return False, 0

        vol_ma = df["volume"].iloc[idx - self.volume_ma_period:idx].mean()
        current_vol = df["volume"].iloc[idx]

        ratio = current_vol / vol_ma if vol_ma > 0 else 0
        return ratio >= self.volume_min_ratio, ratio

    def _check_first_touch(self, df: pd.DataFrame, idx: int) -> Tuple[bool, int]:
        """Filter 3: Skip first touch after crash (>15% drop in 10 days)."""
        lookback = 10

        if idx < lookback:
            return True, 999

        lookback_start = max(0, idx - lookback)
        lookback_slice = df["close"].iloc[lookback_start:idx]
        high_in_lookback = lookback_slice.max()
        current_price = df["close"].iloc[idx]

        drop_pct = ((current_price - high_in_lookback) / high_in_lookback) * 100

        if drop_pct <= self.crash_threshold_pct:
            high_pos_in_slice = lookback_slice.values.argmax()
            days_since = (idx - lookback_start - 1) - high_pos_in_slice

            if days_since < self.first_touch_cooldown:
                return False, max(0, days_since)

        return True, 999

    def _check_penetration(self, price: float, lower_band: float) -> Tuple[bool, float]:
        """Filter 4: Price must go 1%+ below lower band."""
        if lower_band <= 0:
            return False, 0

        penetration = ((lower_band - price) / lower_band) * 100
        return penetration >= self.penetration_min_pct, penetration

    def check_protections(self, df: pd.DataFrame, idx: int, lower_band: float) -> ProtectionStatus:
        """Check all 4 entry protection filters."""
        status = ProtectionStatus()

        status.trend_ok, status.price, status.ma_200 = self._check_trend(df, idx)
        status.volume_ok, status.volume_ratio = self._check_volume(df, idx)
        status.not_first_touch, status.days_since_crash = self._check_first_touch(df, idx)
        status.penetration_ok, status.penetration_pct = self._check_penetration(
            df["close"].iloc[idx], lower_band
        )

        return status

    def analyze(
        self,
        symbol: str,
        df: pd.DataFrame,
        bb_period: int = 20,
        bb_std: float = 1.5,
    ) -> Signal:
        """
        Analyze a coin and generate trading signal.

        Args:
            symbol: Trading pair
            df: DataFrame with OHLCV data
            bb_period: Bollinger Band period
            bb_std: Bollinger Band standard deviation

        Returns:
            Signal with type and context
        """
        if df.empty or len(df) < self.trend_ma_period + 10:
            return Signal(
                symbol=symbol,
                signal_type=SignalType.NO_SIGNAL,
                price=0,
                lower_band=0,
                middle_band=0,
                upper_band=0,
                protection=ProtectionStatus(),
                timestamp=datetime.now(),
                reason="Insufficient data",
            )

        # Calculate Bollinger Bands
        upper, middle, lower = TechnicalIndicators.bollinger_bands(
            df["close"], bb_period, bb_std
        )

        idx = len(df) - 1
        price = df["close"].iloc[idx]
        lower_band = lower.iloc[idx]
        middle_band = middle.iloc[idx]
        upper_band = upper.iloc[idx]

        # Check protections
        protection = self.check_protections(df, idx, lower_band)

        # Determine signal
        if price <= lower_band:
            if protection.all_passed:
                signal_type = SignalType.STRONG_BUY
                reason = "All protections passed - Safe entry"
            else:
                signal_type = SignalType.WEAK_BUY
                reason = f"Failed: {', '.join(protection.failed_filters())}"
        elif price >= middle_band:
            signal_type = SignalType.SELL
            reason = "Price at middle band - Take profit"
        else:
            signal_type = SignalType.HOLD
            distance = ((price - lower_band) / lower_band) * 100
            reason = f"{distance:.1f}% from buy zone"

        return Signal(
            symbol=symbol,
            signal_type=signal_type,
            price=price,
            lower_band=lower_band,
            middle_band=middle_band,
            upper_band=upper_band,
            protection=protection,
            timestamp=datetime.now(),
            reason=reason,
        )

    def check_exit(
        self,
        symbol: str,
        df: pd.DataFrame,
        entry_price: float,
        entry_date: datetime,
        bb_period: int = 20,
        bb_std: float = 1.5,
    ) -> Tuple[bool, str, float]:
        """
        Check if position should be exited.

        Returns:
            (should_exit, reason, current_price)
        """
        if df.empty:
            return False, "", 0

        # Calculate Bollinger Bands
        upper, middle, lower = TechnicalIndicators.bollinger_bands(
            df["close"], bb_period, bb_std
        )

        current_price = df["close"].iloc[-1]
        middle_band = middle.iloc[-1]

        # Check target (middle band)
        if current_price >= middle_band:
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
            return True, f"TARGET: +{pnl_pct:.1f}%", current_price

        # Check time-based exit
        days_held = (datetime.now() - entry_date).days
        if days_held >= self.max_hold_days:
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
            return True, f"TIME_EXIT ({days_held} days): {pnl_pct:+.1f}%", current_price

        return False, "", current_price


# Test function
def test_signal_generator():
    """Test the signal generator."""
    import numpy as np

    print("Testing Signal Generator...")
    print("=" * 50)

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=250, freq="D")
    prices = 100 + np.cumsum(np.random.randn(250) * 2)
    volumes = np.random.randint(1000000, 5000000, 250)

    df = pd.DataFrame({
        "timestamp": dates,
        "open": prices * 0.99,
        "high": prices * 1.02,
        "low": prices * 0.98,
        "close": prices,
        "volume": volumes,
    })

    config = {
        "strategy": {
            "trend_ma_period": 200,
            "volume_ma_period": 20,
            "volume_min_ratio": 0.8,
            "crash_threshold_pct": -15.0,
            "first_touch_cooldown_days": 5,
            "penetration_min_pct": 1.0,
            "max_hold_days": 22,
        }
    }

    generator = SignalGenerator(config)
    signal = generator.analyze("TESTUSDT", df)

    print(f"\nSymbol: {signal.symbol}")
    print(f"Signal: {signal.signal_type.value}")
    print(f"Price: ${signal.price:.2f}")
    print(f"Lower Band: ${signal.lower_band:.2f}")
    print(f"Middle Band: ${signal.middle_band:.2f}")
    print(f"Protection: {signal.protection.passed_count}/4")
    print(f"Reason: {signal.reason}")


if __name__ == "__main__":
    test_signal_generator()
