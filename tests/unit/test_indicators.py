"""Tests for technical indicators."""

import numpy as np
import pandas as pd
import pytest

from analize.features.indicators import TechnicalIndicators


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Create sample OHLCV data."""
    np.random.seed(42)
    n = 100
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)

    return pd.DataFrame({
        "open": close - np.random.rand(n) * 0.5,
        "high": close + np.random.rand(n),
        "low": close - np.random.rand(n),
        "close": close,
        "volume": np.random.rand(n) * 1000 + 100,
    })


class TestTechnicalIndicators:
    """Tests for TechnicalIndicators class."""

    def test_sma(self, sample_ohlcv: pd.DataFrame) -> None:
        """Test Simple Moving Average calculation."""
        sma = TechnicalIndicators.sma(sample_ohlcv["close"], period=20)

        assert len(sma) == len(sample_ohlcv)
        assert not sma.isna().all()
        # SMA should be close to mean for stable prices
        assert abs(sma.iloc[-1] - sample_ohlcv["close"].iloc[-20:].mean()) < 0.01

    def test_ema(self, sample_ohlcv: pd.DataFrame) -> None:
        """Test Exponential Moving Average calculation."""
        ema = TechnicalIndicators.ema(sample_ohlcv["close"], period=20)

        assert len(ema) == len(sample_ohlcv)
        assert not ema.isna().all()

    def test_rsi(self, sample_ohlcv: pd.DataFrame) -> None:
        """Test RSI calculation."""
        rsi = TechnicalIndicators.rsi(sample_ohlcv["close"], period=14)

        assert len(rsi) == len(sample_ohlcv)
        # RSI should be between 0 and 100
        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()

    def test_atr(self, sample_ohlcv: pd.DataFrame) -> None:
        """Test ATR calculation."""
        atr = TechnicalIndicators.atr(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
            period=14,
        )

        assert len(atr) == len(sample_ohlcv)
        # ATR should be positive
        valid_atr = atr.dropna()
        assert (valid_atr >= 0).all()

    def test_bollinger_bands(self, sample_ohlcv: pd.DataFrame) -> None:
        """Test Bollinger Bands calculation."""
        upper, middle, lower = TechnicalIndicators.bollinger_bands(
            sample_ohlcv["close"],
            period=20,
            std_dev=2.0,
        )

        assert len(upper) == len(sample_ohlcv)
        assert len(middle) == len(sample_ohlcv)
        assert len(lower) == len(sample_ohlcv)

        # Upper should be above middle, lower should be below
        valid_idx = ~(upper.isna() | middle.isna() | lower.isna())
        assert (upper[valid_idx] >= middle[valid_idx]).all()
        assert (lower[valid_idx] <= middle[valid_idx]).all()

    def test_volume_zscore(self, sample_ohlcv: pd.DataFrame) -> None:
        """Test volume z-score calculation."""
        zscore = TechnicalIndicators.volume_zscore(sample_ohlcv["volume"], period=20)

        assert len(zscore) == len(sample_ohlcv)
        # Z-score should be roughly centered around 0
        valid_zscore = zscore.dropna()
        assert abs(valid_zscore.mean()) < 1.0

    def test_macd(self, sample_ohlcv: pd.DataFrame) -> None:
        """Test MACD calculation."""
        macd_line, signal_line, histogram = TechnicalIndicators.macd(
            sample_ohlcv["close"]
        )

        assert len(macd_line) == len(sample_ohlcv)
        assert len(signal_line) == len(sample_ohlcv)
        assert len(histogram) == len(sample_ohlcv)

    def test_stochastic(self, sample_ohlcv: pd.DataFrame) -> None:
        """Test Stochastic Oscillator calculation."""
        k, d = TechnicalIndicators.stochastic(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
        )

        assert len(k) == len(sample_ohlcv)
        assert len(d) == len(sample_ohlcv)

        # Stochastic should be between 0 and 100
        valid_k = k.dropna()
        assert (valid_k >= 0).all()
        assert (valid_k <= 100).all()

    def test_candle_body_percent(self, sample_ohlcv: pd.DataFrame) -> None:
        """Test candle body percentage calculation."""
        body_pct = TechnicalIndicators.candle_body_percent(
            sample_ohlcv["open"],
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
        )

        assert len(body_pct) == len(sample_ohlcv)
        # Body percentage should be between 0 and 100
        valid_body = body_pct.dropna()
        assert (valid_body >= 0).all()
        assert (valid_body <= 100).all()

    def test_candle_color(self, sample_ohlcv: pd.DataFrame) -> None:
        """Test candle color determination."""
        color = TechnicalIndicators.candle_color(
            sample_ohlcv["open"],
            sample_ohlcv["close"],
        )

        assert len(color) == len(sample_ohlcv)
        # Color should be -1, 0, or 1
        assert set(color.unique()).issubset({-1, 0, 1})
