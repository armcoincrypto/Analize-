"""
Technical indicator calculations.

Provides functions for computing various technical indicators
used in trading signal analysis.
"""

from typing import Any

import numpy as np
import pandas as pd


class TechnicalIndicators:
    """Computes technical indicators on OHLCV data."""

    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average."""
        return series.rolling(window=period, min_periods=1).mean()

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average."""
        return series.ewm(span=period, adjust=False, min_periods=1).mean()

    @staticmethod
    def ema_slope(series: pd.Series, period: int, lookback: int = 1) -> pd.Series:
        """
        EMA slope (rate of change).

        Args:
            series: Price series
            period: EMA period
            lookback: Number of bars to calculate slope over

        Returns:
            EMA slope as percentage
        """
        ema_values = TechnicalIndicators.ema(series, period)
        slope = (ema_values - ema_values.shift(lookback)) / ema_values.shift(lookback) * 100
        return slope

    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """
        Relative Strength Index.

        Args:
            series: Price series (typically close)
            period: RSI period

        Returns:
            RSI values (0-100)
        """
        delta = series.diff()

        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.ewm(span=period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(span=period, adjust=False, min_periods=period).mean()

        rs = avg_gain / avg_loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    def atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """
        Average True Range.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ATR period

        Returns:
            ATR values
        """
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.ewm(span=period, adjust=False, min_periods=period).mean()

        return atr

    @staticmethod
    def atr_percent(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """ATR as percentage of price."""
        atr = TechnicalIndicators.atr(high, low, close, period)
        return (atr / close) * 100

    @staticmethod
    def bollinger_bands(
        series: pd.Series,
        period: int = 20,
        std_dev: float = 2.0,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands.

        Args:
            series: Price series
            period: SMA period
            std_dev: Standard deviation multiplier

        Returns:
            Tuple of (upper, middle, lower) bands
        """
        middle = TechnicalIndicators.sma(series, period)
        std = series.rolling(window=period, min_periods=1).std()

        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)

        return upper, middle, lower

    @staticmethod
    def bb_width(
        series: pd.Series,
        period: int = 20,
        std_dev: float = 2.0,
    ) -> pd.Series:
        """
        Bollinger Band width (normalized).

        Returns:
            BB width as percentage of middle band
        """
        upper, middle, lower = TechnicalIndicators.bollinger_bands(series, period, std_dev)
        width = (upper - lower) / middle * 100
        return width

    @staticmethod
    def bb_expansion_rate(
        series: pd.Series,
        period: int = 20,
        std_dev: float = 2.0,
        lookback: int = 5,
    ) -> pd.Series:
        """
        Rate of Bollinger Band expansion.

        Args:
            series: Price series
            period: BB period
            std_dev: Standard deviation multiplier
            lookback: Periods to calculate rate over

        Returns:
            BB expansion rate (positive = expanding)
        """
        width = TechnicalIndicators.bb_width(series, period, std_dev)
        rate = (width - width.shift(lookback)) / width.shift(lookback) * 100
        return rate

    @staticmethod
    def volume_zscore(volume: pd.Series, period: int = 20) -> pd.Series:
        """
        Volume Z-score (standard deviations from mean).

        Args:
            volume: Volume series
            period: Lookback period

        Returns:
            Z-score values
        """
        mean = volume.rolling(window=period, min_periods=1).mean()
        std = volume.rolling(window=period, min_periods=1).std()
        zscore = (volume - mean) / std.replace(0, np.inf)
        return zscore

    @staticmethod
    def volume_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
        """Volume ratio to moving average."""
        avg = volume.rolling(window=period, min_periods=1).mean()
        return volume / avg.replace(0, np.inf)

    @staticmethod
    def vwap(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
    ) -> pd.Series:
        """
        Volume Weighted Average Price.

        Note: This is a cumulative VWAP. For session-based VWAP,
        you need to reset at session boundaries.
        """
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        return vwap

    @staticmethod
    def vwap_distance(
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        volume: pd.Series,
    ) -> pd.Series:
        """Distance from VWAP as percentage."""
        vwap = TechnicalIndicators.vwap(high, low, close, volume)
        return (close - vwap) / vwap * 100

    @staticmethod
    def macd(
        series: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        MACD (Moving Average Convergence Divergence).

        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        fast_ema = TechnicalIndicators.ema(series, fast)
        slow_ema = TechnicalIndicators.ema(series, slow)

        macd_line = fast_ema - slow_ema
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    @staticmethod
    def stochastic(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3,
    ) -> tuple[pd.Series, pd.Series]:
        """
        Stochastic Oscillator.

        Returns:
            Tuple of (%K, %D)
        """
        lowest_low = low.rolling(window=k_period, min_periods=1).min()
        highest_high = high.rolling(window=k_period, min_periods=1).max()

        k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.inf)
        d = TechnicalIndicators.sma(k, d_period)

        return k, d

    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume."""
        direction = np.sign(close.diff())
        obv = (volume * direction).fillna(0).cumsum()
        return obv

    @staticmethod
    def candle_body_percent(
        open_price: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> pd.Series:
        """Candle body as percentage of total range."""
        range_val = high - low
        body = abs(close - open_price)
        return (body / range_val.replace(0, np.inf)) * 100

    @staticmethod
    def candle_color(open_price: pd.Series, close: pd.Series) -> pd.Series:
        """
        Candle color/direction.

        Returns:
            1 for green (bullish), -1 for red (bearish), 0 for doji
        """

        def get_color(o: float, c: float) -> int:
            diff = c - o
            if abs(diff) < 0.0001 * o:  # Small threshold for doji
                return 0
            return 1 if diff > 0 else -1

        return pd.Series(
            [get_color(o, c) for o, c in zip(open_price, close)],
            index=open_price.index,
        )

    @staticmethod
    def price_change_percent(series: pd.Series, periods: int = 1) -> pd.Series:
        """Percentage price change over N periods."""
        return series.pct_change(periods) * 100

    @staticmethod
    def rolling_max(series: pd.Series, period: int) -> pd.Series:
        """Rolling maximum."""
        return series.rolling(window=period, min_periods=1).max()

    @staticmethod
    def rolling_min(series: pd.Series, period: int) -> pd.Series:
        """Rolling minimum."""
        return series.rolling(window=period, min_periods=1).min()

    @staticmethod
    def donchian_channel(
        high: pd.Series,
        low: pd.Series,
        period: int = 20,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Donchian Channel.

        Returns:
            Tuple of (upper, middle, lower)
        """
        upper = high.rolling(window=period, min_periods=1).max()
        lower = low.rolling(window=period, min_periods=1).min()
        middle = (upper + lower) / 2

        return upper, middle, lower

    @staticmethod
    def keltner_channel(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        ema_period: int = 20,
        atr_period: int = 10,
        multiplier: float = 2.0,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Keltner Channel.

        Returns:
            Tuple of (upper, middle, lower)
        """
        middle = TechnicalIndicators.ema(close, ema_period)
        atr = TechnicalIndicators.atr(high, low, close, atr_period)

        upper = middle + (multiplier * atr)
        lower = middle - (multiplier * atr)

        return upper, middle, lower
