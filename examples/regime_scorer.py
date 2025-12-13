#!/usr/bin/env python3
"""
Regime Confidence Scorer - Dynamic Regime Strength Grading

Phase 2 Risk Evolution Component #1

Instead of binary regime blocking (trade/don't trade), this module:
1. Grades regime strength on a 0-100 scale
2. Adjusts position sizing based on regime confidence
3. Identifies "perfect" vs "ok" vs "marginal" conditions

EVIDENCE-BASED REGIME DATA (from 311+ trades):
- Sideways: 73.7% win rate (best for mean reversion)
- Bullish:  85.7% win rate (but fewer opportunities)
- Bearish:  66.7% win rate (tradeable but cautious)

This reduces variance by trading smaller in uncertain conditions.
"""

import ssl_bypass  # Must be first!
import requests
import datetime
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum


class RegimeType(Enum):
    """Market regime classification."""
    STRONG_BULL = "strong_bull"
    BULL = "bull"
    SIDEWAYS = "sideways"
    BEAR = "bear"
    STRONG_BEAR = "strong_bear"
    VOLATILE = "volatile"
    CRISIS = "crisis"


@dataclass
class RegimeScore:
    """Complete regime assessment."""
    regime_type: RegimeType
    confidence: float           # 0-100 how confident we are in this regime
    strength: float             # 0-100 how strong the regime is
    stability: float            # 0-100 how stable (not choppy)
    size_multiplier: float      # Position size adjustment (0.0-1.0)
    trade_allowed: bool
    reason: str
    details: Dict = field(default_factory=dict)


class RegimeConfidenceScorer:
    """
    Advanced regime scoring system.

    Features:
    - Multi-timeframe analysis (1h, 4h, 1d)
    - Trend strength measurement (ADX-based)
    - Volatility regime detection (ATR-based)
    - Momentum alignment scoring
    - Stability scoring (choppy vs clean)

    Output:
    - Regime type classification
    - Confidence score (0-100)
    - Position size multiplier
    """

    def __init__(self):
        # Evidence-based regime win rates
        self.regime_win_rates = {
            RegimeType.STRONG_BULL: 85.7,
            RegimeType.BULL: 80.0,
            RegimeType.SIDEWAYS: 73.7,  # Best for mean reversion
            RegimeType.BEAR: 66.7,
            RegimeType.STRONG_BEAR: 55.0,
            RegimeType.VOLATILE: 60.0,
            RegimeType.CRISIS: 40.0,
        }

        # Size multipliers based on regime
        self.regime_size_multipliers = {
            RegimeType.STRONG_BULL: 0.75,   # Trend - not ideal for mean reversion
            RegimeType.BULL: 0.85,
            RegimeType.SIDEWAYS: 1.0,       # Optimal for our strategy
            RegimeType.BEAR: 0.7,
            RegimeType.STRONG_BEAR: 0.4,
            RegimeType.VOLATILE: 0.5,
            RegimeType.CRISIS: 0.0,         # No trading
        }

        # Cache for API data
        self._cache = {}
        self._cache_time = {}

    def _fetch_klines(self, symbol: str, interval: str, limit: int = 50) -> Optional[List]:
        """Fetch kline data from API with caching."""
        cache_key = f"{symbol}_{interval}"

        # Use cache if fresh (< 5 minutes for 1h, < 15 for 4h, < 60 for 1d)
        cache_ttl = {"1h": 300, "4h": 900, "1d": 3600}.get(interval, 300)

        if cache_key in self._cache:
            age = (datetime.datetime.utcnow() - self._cache_time.get(cache_key, datetime.datetime.min)).total_seconds()
            if age < cache_ttl:
                return self._cache[cache_key]

        urls = [
            f"https://api.binance.us/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}",
            f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}",
        ]

        for url in urls:
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    self._cache[cache_key] = data
                    self._cache_time[cache_key] = datetime.datetime.utcnow()
                    return data
            except:
                continue

        return None

    def _calculate_atr(self, klines: List, period: int = 14) -> float:
        """Calculate Average True Range."""
        if len(klines) < period + 1:
            return 0.0

        trs = []
        for i in range(1, len(klines)):
            high = float(klines[i][2])
            low = float(klines[i][3])
            prev_close = float(klines[i-1][4])
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            trs.append(tr)

        if len(trs) < period:
            return sum(trs) / len(trs) if trs else 0.0

        return sum(trs[-period:]) / period

    def _calculate_adx(self, klines: List, period: int = 14) -> float:
        """
        Calculate ADX (Average Directional Index) for trend strength.

        ADX interpretation:
        - 0-20: No trend (sideways)
        - 20-40: Weak trend
        - 40-60: Strong trend
        - 60+: Very strong trend
        """
        if len(klines) < period + 1:
            return 0.0

        plus_dm = []
        minus_dm = []
        tr_list = []

        for i in range(1, len(klines)):
            high = float(klines[i][2])
            low = float(klines[i][3])
            prev_high = float(klines[i-1][2])
            prev_low = float(klines[i-1][3])
            prev_close = float(klines[i-1][4])

            # True Range
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_list.append(tr)

            # Directional Movement
            up_move = high - prev_high
            down_move = prev_low - low

            if up_move > down_move and up_move > 0:
                plus_dm.append(up_move)
            else:
                plus_dm.append(0)

            if down_move > up_move and down_move > 0:
                minus_dm.append(down_move)
            else:
                minus_dm.append(0)

        if len(tr_list) < period:
            return 0.0

        # Smoothed averages
        atr = sum(tr_list[-period:]) / period
        if atr == 0:
            return 0.0

        plus_di = (sum(plus_dm[-period:]) / period) / atr * 100
        minus_di = (sum(minus_dm[-period:]) / period) / atr * 100

        # ADX
        di_sum = plus_di + minus_di
        if di_sum == 0:
            return 0.0

        dx = abs(plus_di - minus_di) / di_sum * 100

        return dx

    def _calculate_momentum_alignment(self, klines: List) -> Tuple[float, str]:
        """
        Calculate momentum alignment across timeframes.

        Returns:
            (alignment_score 0-100, direction "bull"/"bear"/"mixed")
        """
        if len(klines) < 20:
            return 50.0, "mixed"

        closes = [float(k[4]) for k in klines]
        current = closes[-1]

        # Short-term momentum (5 periods)
        short_ma = sum(closes[-5:]) / 5
        short_trend = "bull" if current > short_ma else "bear"

        # Medium-term momentum (10 periods)
        med_ma = sum(closes[-10:]) / 10
        med_trend = "bull" if current > med_ma else "bear"

        # Long-term momentum (20 periods)
        long_ma = sum(closes[-20:]) / 20
        long_trend = "bull" if current > long_ma else "bear"

        # Alignment score
        trends = [short_trend, med_trend, long_trend]
        bull_count = sum(1 for t in trends if t == "bull")

        if bull_count == 3:
            return 100.0, "bull"
        elif bull_count == 0:
            return 100.0, "bear"
        elif bull_count == 2:
            return 66.0, "mixed"
        else:
            return 33.0, "mixed"

    def _calculate_stability(self, klines: List, period: int = 20) -> float:
        """
        Calculate market stability (how clean vs choppy).

        Higher = more stable, cleaner trends
        Lower = choppy, unpredictable

        Returns:
            Stability score 0-100
        """
        if len(klines) < period:
            return 50.0

        closes = [float(k[4]) for k in klines[-period:]]

        # Calculate direction changes
        direction_changes = 0
        for i in range(2, len(closes)):
            prev_dir = closes[i-1] - closes[i-2]
            curr_dir = closes[i] - closes[i-1]
            if (prev_dir > 0 and curr_dir < 0) or (prev_dir < 0 and curr_dir > 0):
                direction_changes += 1

        # Maximum possible changes
        max_changes = period - 2
        if max_changes <= 0:
            return 50.0

        # Stability = fewer direction changes
        choppiness = direction_changes / max_changes
        stability = (1 - choppiness) * 100

        return max(0.0, min(100.0, stability))

    def _classify_regime(
        self,
        adx: float,
        momentum_score: float,
        momentum_dir: str,
        atr_pct: float,
        stability: float
    ) -> RegimeType:
        """Classify market regime based on indicators."""

        # Crisis detection
        if atr_pct > 8.0:
            return RegimeType.CRISIS

        # Volatile detection
        if atr_pct > 5.0 or stability < 30:
            return RegimeType.VOLATILE

        # Strong trend detection (ADX > 40)
        if adx > 40:
            if momentum_dir == "bull":
                return RegimeType.STRONG_BULL
            else:
                return RegimeType.STRONG_BEAR

        # Moderate trend detection (ADX 20-40)
        if adx > 20:
            if momentum_dir == "bull":
                return RegimeType.BULL
            else:
                return RegimeType.BEAR

        # Low ADX = sideways (best for mean reversion)
        return RegimeType.SIDEWAYS

    def score_regime(self, symbol: str = "ATOMUSDT") -> RegimeScore:
        """
        Calculate comprehensive regime score.

        Args:
            symbol: Trading symbol

        Returns:
            Complete RegimeScore with all metrics
        """
        # Fetch multi-timeframe data
        klines_1h = self._fetch_klines(symbol, "1h", 50)
        klines_4h = self._fetch_klines(symbol, "4h", 50)
        klines_1d = self._fetch_klines(symbol, "1d", 30)

        if not klines_1h:
            return RegimeScore(
                regime_type=RegimeType.SIDEWAYS,
                confidence=30.0,
                strength=50.0,
                stability=50.0,
                size_multiplier=0.5,
                trade_allowed=True,
                reason="Could not fetch data - using conservative defaults",
                details={}
            )

        # Calculate indicators on 1h timeframe
        atr = self._calculate_atr(klines_1h)
        current_price = float(klines_1h[-1][4])
        atr_pct = (atr / current_price * 100) if current_price > 0 else 0

        adx = self._calculate_adx(klines_1h)
        momentum_score, momentum_dir = self._calculate_momentum_alignment(klines_1h)
        stability = self._calculate_stability(klines_1h)

        # Multi-timeframe confirmation
        mtf_score = 50.0
        if klines_4h and klines_1d:
            _, dir_4h = self._calculate_momentum_alignment(klines_4h)
            _, dir_1d = self._calculate_momentum_alignment(klines_1d)

            # Score higher if all timeframes align
            if dir_4h == dir_1d == momentum_dir:
                mtf_score = 100.0
            elif dir_4h == momentum_dir or dir_1d == momentum_dir:
                mtf_score = 75.0
            else:
                mtf_score = 25.0

        # Classify regime
        regime_type = self._classify_regime(adx, momentum_score, momentum_dir, atr_pct, stability)

        # Calculate confidence based on multiple factors
        confidence = (
            stability * 0.3 +           # Stable = more confident
            mtf_score * 0.3 +           # Aligned = more confident
            (100 - min(atr_pct * 10, 100)) * 0.2 +  # Low vol = more confident
            (100 if adx > 15 else adx * 5) * 0.2   # Clear regime = more confident
        )
        confidence = max(0.0, min(100.0, confidence))

        # Get regime-specific multiplier
        base_multiplier = self.regime_size_multipliers.get(regime_type, 0.5)

        # Adjust multiplier based on confidence
        if confidence >= 80:
            size_multiplier = base_multiplier * 1.0
        elif confidence >= 60:
            size_multiplier = base_multiplier * 0.85
        elif confidence >= 40:
            size_multiplier = base_multiplier * 0.6
        else:
            size_multiplier = base_multiplier * 0.4

        # Determine if trading is allowed
        trade_allowed = regime_type != RegimeType.CRISIS and confidence >= 30

        # Build reason
        win_rate = self.regime_win_rates.get(regime_type, 50)
        if regime_type == RegimeType.SIDEWAYS:
            reason = f"OPTIMAL: Sideways regime ({win_rate}% historical win rate)"
        elif regime_type == RegimeType.CRISIS:
            reason = f"BLOCKED: Crisis conditions detected"
        elif regime_type in [RegimeType.STRONG_BULL, RegimeType.STRONG_BEAR]:
            reason = f"CAUTION: Strong trend - reduced size for mean reversion"
        elif regime_type == RegimeType.VOLATILE:
            reason = f"CAUTION: High volatility - reduced size"
        else:
            reason = f"OK: {regime_type.value} regime ({win_rate}% historical win rate)"

        return RegimeScore(
            regime_type=regime_type,
            confidence=round(confidence, 1),
            strength=round(adx, 1),
            stability=round(stability, 1),
            size_multiplier=round(size_multiplier, 2),
            trade_allowed=trade_allowed,
            reason=reason,
            details={
                "adx": round(adx, 1),
                "atr_pct": round(atr_pct, 2),
                "momentum_score": round(momentum_score, 1),
                "momentum_direction": momentum_dir,
                "mtf_alignment": round(mtf_score, 1),
                "stability": round(stability, 1),
                "historical_win_rate": win_rate,
            }
        )

    def print_regime_analysis(self, symbol: str = "ATOMUSDT"):
        """Print detailed regime analysis."""
        score = self.score_regime(symbol)

        print("\n" + "=" * 60)
        print(f"REGIME CONFIDENCE ANALYSIS: {symbol}")
        print("=" * 60)

        print(f"\n[REGIME TYPE]")
        print(f"  Classification: {score.regime_type.value.upper()}")
        print(f"  Reason: {score.reason}")

        print(f"\n[SCORES]")
        print(f"  Confidence:  {score.confidence:.0f}/100")
        print(f"  Strength:    {score.strength:.0f}/100 (ADX)")
        print(f"  Stability:   {score.stability:.0f}/100")

        print(f"\n[POSITION SIZING]")
        print(f"  Size Multiplier: {score.size_multiplier:.2f}x")
        print(f"  Trade Allowed: {'YES' if score.trade_allowed else 'NO'}")

        print(f"\n[DETAILS]")
        for key, value in score.details.items():
            print(f"  {key}: {value}")

        print("\n[SIZE INTERPRETATION]")
        if score.size_multiplier >= 0.9:
            print("  FULL SIZE - Optimal conditions")
        elif score.size_multiplier >= 0.7:
            print("  MODERATE SIZE - Good but not perfect")
        elif score.size_multiplier >= 0.5:
            print("  REDUCED SIZE - Uncertain conditions")
        elif score.size_multiplier > 0:
            print("  MINIMAL SIZE - Marginal conditions")
        else:
            print("  NO TRADE - Conditions blocked")

        print("=" * 60)

        return score


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_regime_size_multiplier(symbol: str = "ATOMUSDT") -> Tuple[float, str]:
    """
    Quick function to get current regime size multiplier.

    Returns:
        Tuple of (multiplier, reason)
    """
    scorer = RegimeConfidenceScorer()
    score = scorer.score_regime(symbol)
    return score.size_multiplier, score.reason


def is_optimal_regime(symbol: str = "ATOMUSDT") -> bool:
    """Check if current regime is optimal for trading."""
    scorer = RegimeConfidenceScorer()
    score = scorer.score_regime(symbol)
    return score.regime_type == RegimeType.SIDEWAYS and score.confidence >= 60


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("REGIME CONFIDENCE SCORER TEST")
    print("=" * 60)

    scorer = RegimeConfidenceScorer()

    # Test with primary symbol
    symbols = ["ATOMUSDT", "SOLUSDT", "BTCUSDT"]

    for symbol in symbols:
        scorer.print_regime_analysis(symbol)
        print()

    # Quick check example
    print("\n[QUICK CHECK]")
    print("-" * 40)
    multiplier, reason = get_regime_size_multiplier("ATOMUSDT")
    print(f"  ATOMUSDT: {multiplier:.2f}x - {reason}")

    print(f"\n  Is optimal regime? {is_optimal_regime('ATOMUSDT')}")
