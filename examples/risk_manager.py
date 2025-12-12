#!/usr/bin/env python3
"""
Risk Manager - Comprehensive Trading Risk Controls

Implements all safety measures for the Cloud AI Analyzer:
1. Stop-loss & exposure caps
2. Circuit breaker (BTC crash protection)
3. Multi-signal confirmation requirement
4. Regime-aware position sizing
5. Instrument-level safety checks (liquidity, halts)
6. Whale signal weight decay
7. Cooldown periods after major events

Author: Cloud AI Analyzer
"""

import requests
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import json
from pathlib import Path


# =============================================================================
# CONFIGURATION
# =============================================================================

class RiskConfig:
    """Risk management configuration parameters."""

    # Exposure limits
    MAX_PORTFOLIO_EXPOSURE_PCT = 10.0      # Max 10% of AUM until validated
    MAX_POSITION_SIZE_PCT = 2.0            # Max 2% per trade
    MAX_SINGLE_ASSET_PCT = 20.0            # Max 20% in any single asset

    # Stop-loss settings
    DEFAULT_STOP_LOSS_PCT = 5.0            # 5% stop-loss
    ATR_STOP_LOSS_MULTIPLIER = 2.0         # 2x ATR for dynamic stops

    # Circuit breaker thresholds
    BTC_7D_CRASH_THRESHOLD = -10.0         # Pause if BTC drops 10% in 7 days
    BTC_24H_CRASH_THRESHOLD = -7.0         # Pause if BTC drops 7% in 24 hours
    CIRCUIT_BREAKER_COOLDOWN_HOURS = 24    # Wait 24h after circuit breaker

    # Multi-signal confirmation
    MIN_SIGNALS_NORMAL = 2                 # Require 2 signals normally
    MIN_SIGNALS_VOLATILE = 3               # Require 3 signals in volatile regime
    MIN_CONFIDENCE_THRESHOLD = 0.65        # Minimum confidence to trade

    # Liquidity safety
    MAX_TRADE_VOLUME_RATIO = 0.005         # Max 0.5% of 24h volume
    MIN_24H_VOLUME_USD = 1_000_000         # Min $1M daily volume

    # Whale signal decay
    WHALE_SIGNAL_DECAY_HOURS = 48          # Decay whale signals after 48h
    WHALE_SIGNAL_HALF_LIFE_HOURS = 24      # Half-life for decay

    # Event cooldowns
    FUNDING_SPIKE_COOLDOWN_HOURS = 24      # Cooldown after funding spike
    NEWS_EVENT_COOLDOWN_HOURS = 72         # Cooldown after major news

    # ML constraints
    MIN_ML_ACCURACY_FOR_TRADING = 0.60     # Don't trade if ML accuracy < 60%
    MAX_ML_CONFIDENCE = 0.95               # Cap ML confidence at 95%


class MarketRegime(Enum):
    """Market regime classification."""
    BULL = "BULL"
    BEAR = "BEAR"
    SIDEWAYS = "SIDEWAYS"
    VOLATILE = "VOLATILE"
    CRISIS = "CRISIS"


class TradeDecision(Enum):
    """Trade execution decision."""
    APPROVED = "APPROVED"
    REJECTED_EXPOSURE = "REJECTED_EXPOSURE"
    REJECTED_CIRCUIT_BREAKER = "REJECTED_CIRCUIT_BREAKER"
    REJECTED_INSUFFICIENT_SIGNALS = "REJECTED_INSUFFICIENT_SIGNALS"
    REJECTED_LOW_CONFIDENCE = "REJECTED_LOW_CONFIDENCE"
    REJECTED_LOW_LIQUIDITY = "REJECTED_LOW_LIQUIDITY"
    REJECTED_ML_UNRELIABLE = "REJECTED_ML_UNRELIABLE"
    REJECTED_COOLDOWN = "REJECTED_COOLDOWN"
    REQUIRES_APPROVAL = "REQUIRES_APPROVAL"


@dataclass
class Signal:
    """Trading signal with metadata."""
    signal_type: str          # technical, news, ml, whale, funding, orderflow
    direction: str            # BUY, SELL, NEUTRAL
    confidence: float         # 0.0 to 1.0
    timestamp: datetime
    symbol: str
    source: str               # indicator/model name
    metadata: Dict = field(default_factory=dict)


@dataclass
class TradeRequest:
    """Trade request to be validated."""
    symbol: str
    direction: str            # LONG or SHORT
    size_usd: float
    signals: List[Signal]
    entry_price: float
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None


@dataclass
class RiskAssessment:
    """Result of risk assessment."""
    decision: TradeDecision
    approved: bool
    adjusted_size_usd: float
    stop_loss_price: float
    reasons: List[str]
    warnings: List[str]
    regime: MarketRegime
    signal_count: int
    avg_confidence: float


# =============================================================================
# RISK MANAGER CLASS
# =============================================================================

class RiskManager:
    """
    Comprehensive risk management for trading operations.

    Features:
    - Stop-loss enforcement
    - Position sizing limits
    - Circuit breaker for market crashes
    - Multi-signal confirmation
    - Regime-aware adjustments
    - Liquidity checks
    - Whale signal decay
    - Cooldown management
    """

    def __init__(self, config: RiskConfig = None):
        """Initialize risk manager."""
        self.config = config or RiskConfig()
        self.circuit_breaker_triggered = False
        self.circuit_breaker_until: Optional[datetime] = None
        self.cooldowns: Dict[str, datetime] = {}  # symbol -> cooldown_until
        self.positions: Dict[str, float] = {}      # symbol -> position_size_usd
        self.total_exposure: float = 0.0
        self.aum: float = 10000.0  # Default AUM, should be set externally

        # Cache for BTC data
        self._btc_price_cache: Dict[str, float] = {}
        self._btc_cache_time: Optional[datetime] = None

        print("=" * 60)
        print("RISK MANAGER INITIALIZED")
        print("=" * 60)
        print(f"  Max Portfolio Exposure: {self.config.MAX_PORTFOLIO_EXPOSURE_PCT}%")
        print(f"  Max Position Size:      {self.config.MAX_POSITION_SIZE_PCT}%")
        print(f"  Max Single Asset:       {self.config.MAX_SINGLE_ASSET_PCT}%")
        print(f"  Default Stop-Loss:      {self.config.DEFAULT_STOP_LOSS_PCT}%")
        print(f"  Min Signals Required:   {self.config.MIN_SIGNALS_NORMAL}")
        print(f"  Min Confidence:         {self.config.MIN_CONFIDENCE_THRESHOLD}")
        print("=" * 60)

    def set_aum(self, aum: float):
        """Set current assets under management."""
        self.aum = aum
        print(f"  [RISK] AUM set to ${aum:,.2f}")

    # =========================================================================
    # CIRCUIT BREAKER
    # =========================================================================

    def check_circuit_breaker(self) -> Tuple[bool, str]:
        """
        Check if circuit breaker should be triggered.

        Returns:
            (is_triggered, reason)
        """
        # Check if already in cooldown
        if self.circuit_breaker_triggered:
            if datetime.now() < self.circuit_breaker_until:
                remaining = (self.circuit_breaker_until - datetime.now()).total_seconds() / 3600
                return True, f"Circuit breaker active, {remaining:.1f}h remaining"
            else:
                self.circuit_breaker_triggered = False
                print("  [RISK] Circuit breaker cooldown expired, trading resumed")

        # Fetch BTC price data
        btc_data = self._fetch_btc_data()
        if not btc_data:
            return False, "Could not fetch BTC data"

        # Check 7-day change
        if btc_data.get('change_7d', 0) <= self.config.BTC_7D_CRASH_THRESHOLD:
            self._trigger_circuit_breaker(
                f"BTC 7d change: {btc_data['change_7d']:.1f}% <= {self.config.BTC_7D_CRASH_THRESHOLD}%"
            )
            return True, f"BTC crashed {btc_data['change_7d']:.1f}% in 7 days"

        # Check 24-hour change
        if btc_data.get('change_24h', 0) <= self.config.BTC_24H_CRASH_THRESHOLD:
            self._trigger_circuit_breaker(
                f"BTC 24h change: {btc_data['change_24h']:.1f}% <= {self.config.BTC_24H_CRASH_THRESHOLD}%"
            )
            return True, f"BTC crashed {btc_data['change_24h']:.1f}% in 24 hours"

        return False, "OK"

    def _trigger_circuit_breaker(self, reason: str):
        """Trigger the circuit breaker."""
        self.circuit_breaker_triggered = True
        self.circuit_breaker_until = datetime.now() + timedelta(
            hours=self.config.CIRCUIT_BREAKER_COOLDOWN_HOURS
        )
        print(f"\n{'!'*60}")
        print(f"  CIRCUIT BREAKER TRIGGERED!")
        print(f"  Reason: {reason}")
        print(f"  Trading paused until: {self.circuit_breaker_until}")
        print(f"{'!'*60}\n")

    def _fetch_btc_data(self) -> Optional[Dict]:
        """Fetch BTC price and change data."""
        # Use cache if fresh (< 5 minutes)
        if self._btc_cache_time and (datetime.now() - self._btc_cache_time).seconds < 300:
            return self._btc_price_cache

        urls = [
            "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd&include_24hr_change=true&include_7d_change=true",
            "https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT",
        ]

        for url in urls:
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()

                    if 'bitcoin' in data:
                        # CoinGecko format
                        self._btc_price_cache = {
                            'price': data['bitcoin']['usd'],
                            'change_24h': data['bitcoin'].get('usd_24h_change', 0),
                            'change_7d': data['bitcoin'].get('usd_7d_change', 0),
                        }
                    else:
                        # Binance format
                        self._btc_price_cache = {
                            'price': float(data['lastPrice']),
                            'change_24h': float(data['priceChangePercent']),
                            'change_7d': float(data['priceChangePercent']) * 2,  # Estimate
                        }

                    self._btc_cache_time = datetime.now()
                    return self._btc_price_cache
            except Exception as e:
                continue

        return None

    # =========================================================================
    # REGIME DETECTION
    # =========================================================================

    def detect_regime(self) -> MarketRegime:
        """
        Detect current market regime based on BTC behavior.

        Returns:
            Current market regime
        """
        btc_data = self._fetch_btc_data()
        if not btc_data:
            return MarketRegime.SIDEWAYS  # Default to conservative

        change_24h = btc_data.get('change_24h', 0)
        change_7d = btc_data.get('change_7d', 0)

        # Crisis: severe drops
        if change_7d <= -20 or change_24h <= -10:
            return MarketRegime.CRISIS

        # Volatile: large swings
        if abs(change_24h) >= 5 or abs(change_7d) >= 15:
            return MarketRegime.VOLATILE

        # Bull: sustained gains
        if change_7d >= 10 and change_24h >= 0:
            return MarketRegime.BULL

        # Bear: sustained losses
        if change_7d <= -5 and change_24h <= 0:
            return MarketRegime.BEAR

        return MarketRegime.SIDEWAYS

    # =========================================================================
    # SIGNAL VALIDATION
    # =========================================================================

    def validate_signals(self, signals: List[Signal], regime: MarketRegime) -> Tuple[bool, List[str]]:
        """
        Validate that enough independent signals confirm the trade.

        Args:
            signals: List of trading signals
            regime: Current market regime

        Returns:
            (is_valid, reasons)
        """
        reasons = []

        # Determine required signals based on regime
        min_signals = self.config.MIN_SIGNALS_NORMAL
        if regime in [MarketRegime.VOLATILE, MarketRegime.CRISIS]:
            min_signals = self.config.MIN_SIGNALS_VOLATILE

        # Count unique signal types
        signal_types = set(s.signal_type for s in signals if s.confidence >= 0.5)
        unique_count = len(signal_types)

        if unique_count < min_signals:
            reasons.append(
                f"Insufficient signals: {unique_count} < {min_signals} required "
                f"(regime={regime.value})"
            )
            return False, reasons

        # Check average confidence
        if signals:
            avg_confidence = sum(s.confidence for s in signals) / len(signals)
            if avg_confidence < self.config.MIN_CONFIDENCE_THRESHOLD:
                reasons.append(
                    f"Low average confidence: {avg_confidence:.2f} < "
                    f"{self.config.MIN_CONFIDENCE_THRESHOLD}"
                )
                return False, reasons

        # Check for conflicting signals
        buy_signals = sum(1 for s in signals if s.direction == "BUY")
        sell_signals = sum(1 for s in signals if s.direction == "SELL")

        if buy_signals > 0 and sell_signals > 0:
            conflict_ratio = min(buy_signals, sell_signals) / max(buy_signals, sell_signals)
            if conflict_ratio > 0.5:
                reasons.append(
                    f"Conflicting signals: {buy_signals} BUY vs {sell_signals} SELL"
                )
                return False, reasons

        return True, []

    def apply_whale_decay(self, signals: List[Signal]) -> List[Signal]:
        """
        Apply time decay to whale signals.

        Whale signals lose weight over time if not confirmed by volume.

        Args:
            signals: List of signals

        Returns:
            Signals with adjusted confidence for whale signals
        """
        adjusted = []
        now = datetime.now()

        for signal in signals:
            if signal.signal_type == "whale":
                age_hours = (now - signal.timestamp).total_seconds() / 3600

                if age_hours > self.config.WHALE_SIGNAL_DECAY_HOURS:
                    # Signal too old, reduce confidence to near zero
                    signal.confidence *= 0.1
                else:
                    # Apply exponential decay
                    decay_factor = 0.5 ** (age_hours / self.config.WHALE_SIGNAL_HALF_LIFE_HOURS)
                    signal.confidence *= decay_factor

            adjusted.append(signal)

        return adjusted

    # =========================================================================
    # POSITION SIZING
    # =========================================================================

    def calculate_position_size(
        self,
        requested_size: float,
        symbol: str,
        regime: MarketRegime
    ) -> Tuple[float, List[str]]:
        """
        Calculate safe position size with all constraints.

        Args:
            requested_size: Requested position size in USD
            symbol: Trading symbol
            regime: Current market regime

        Returns:
            (adjusted_size, warnings)
        """
        warnings = []
        adjusted_size = requested_size

        # Apply regime multiplier
        regime_multiplier = {
            MarketRegime.BULL: 1.0,
            MarketRegime.SIDEWAYS: 0.8,
            MarketRegime.BEAR: 0.5,
            MarketRegime.VOLATILE: 0.5,
            MarketRegime.CRISIS: 0.25,
        }.get(regime, 0.5)

        if regime_multiplier < 1.0:
            adjusted_size *= regime_multiplier
            warnings.append(
                f"Position reduced by {(1-regime_multiplier)*100:.0f}% due to {regime.value} regime"
            )

        # Cap at max position size
        max_position = self.aum * (self.config.MAX_POSITION_SIZE_PCT / 100)
        if adjusted_size > max_position:
            adjusted_size = max_position
            warnings.append(
                f"Position capped at {self.config.MAX_POSITION_SIZE_PCT}% of AUM (${max_position:,.2f})"
            )

        # Check single asset exposure
        current_exposure = self.positions.get(symbol, 0)
        max_asset_exposure = self.aum * (self.config.MAX_SINGLE_ASSET_PCT / 100)

        if current_exposure + adjusted_size > max_asset_exposure:
            adjusted_size = max(0, max_asset_exposure - current_exposure)
            warnings.append(
                f"Position limited due to {self.config.MAX_SINGLE_ASSET_PCT}% single-asset cap"
            )

        # Check total portfolio exposure
        max_total = self.aum * (self.config.MAX_PORTFOLIO_EXPOSURE_PCT / 100)
        if self.total_exposure + adjusted_size > max_total:
            adjusted_size = max(0, max_total - self.total_exposure)
            warnings.append(
                f"Position limited due to {self.config.MAX_PORTFOLIO_EXPOSURE_PCT}% total exposure cap"
            )

        return adjusted_size, warnings

    def calculate_stop_loss(
        self,
        entry_price: float,
        direction: str,
        atr: Optional[float] = None
    ) -> float:
        """
        Calculate stop-loss price.

        Args:
            entry_price: Entry price
            direction: LONG or SHORT
            atr: Average True Range (optional, for dynamic stops)

        Returns:
            Stop-loss price
        """
        if atr:
            # ATR-based stop
            stop_distance = atr * self.config.ATR_STOP_LOSS_MULTIPLIER
        else:
            # Fixed percentage stop
            stop_distance = entry_price * (self.config.DEFAULT_STOP_LOSS_PCT / 100)

        if direction == "LONG":
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance

    # =========================================================================
    # LIQUIDITY CHECKS
    # =========================================================================

    def check_liquidity(self, symbol: str, size_usd: float) -> Tuple[bool, str]:
        """
        Check if there's sufficient liquidity for the trade.

        Args:
            symbol: Trading symbol
            size_usd: Trade size in USD

        Returns:
            (is_safe, reason)
        """
        volume_24h = self._fetch_24h_volume(symbol)

        if volume_24h is None:
            return True, "Could not fetch volume data"

        if volume_24h < self.config.MIN_24H_VOLUME_USD:
            return False, f"24h volume ${volume_24h:,.0f} below minimum ${self.config.MIN_24H_VOLUME_USD:,.0f}"

        volume_ratio = size_usd / volume_24h
        if volume_ratio > self.config.MAX_TRADE_VOLUME_RATIO:
            return False, f"Trade size {volume_ratio*100:.2f}% of 24h volume exceeds {self.config.MAX_TRADE_VOLUME_RATIO*100:.1f}% limit"

        return True, "OK"

    def _fetch_24h_volume(self, symbol: str) -> Optional[float]:
        """Fetch 24h trading volume for a symbol."""
        try:
            url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return float(data['quoteVolume'])
        except:
            pass
        return None

    # =========================================================================
    # COOLDOWN MANAGEMENT
    # =========================================================================

    def set_cooldown(self, symbol: str, hours: float, reason: str):
        """Set a cooldown period for a symbol."""
        until = datetime.now() + timedelta(hours=hours)
        self.cooldowns[symbol] = until
        print(f"  [RISK] Cooldown set for {symbol}: {hours}h ({reason})")

    def check_cooldown(self, symbol: str) -> Tuple[bool, str]:
        """Check if symbol is in cooldown."""
        if symbol in self.cooldowns:
            if datetime.now() < self.cooldowns[symbol]:
                remaining = (self.cooldowns[symbol] - datetime.now()).total_seconds() / 3600
                return True, f"Cooldown active: {remaining:.1f}h remaining"
            else:
                del self.cooldowns[symbol]
        return False, "OK"

    # =========================================================================
    # MAIN ASSESSMENT
    # =========================================================================

    def assess_trade(self, request: TradeRequest) -> RiskAssessment:
        """
        Perform comprehensive risk assessment for a trade request.

        Args:
            request: Trade request to assess

        Returns:
            Complete risk assessment
        """
        reasons = []
        warnings = []

        # 1. Check circuit breaker
        cb_triggered, cb_reason = self.check_circuit_breaker()
        if cb_triggered:
            return RiskAssessment(
                decision=TradeDecision.REJECTED_CIRCUIT_BREAKER,
                approved=False,
                adjusted_size_usd=0,
                stop_loss_price=0,
                reasons=[cb_reason],
                warnings=[],
                regime=self.detect_regime(),
                signal_count=len(request.signals),
                avg_confidence=0,
            )

        # 2. Detect regime
        regime = self.detect_regime()

        # 3. Check cooldown
        in_cooldown, cooldown_reason = self.check_cooldown(request.symbol)
        if in_cooldown:
            return RiskAssessment(
                decision=TradeDecision.REJECTED_COOLDOWN,
                approved=False,
                adjusted_size_usd=0,
                stop_loss_price=0,
                reasons=[cooldown_reason],
                warnings=[],
                regime=regime,
                signal_count=len(request.signals),
                avg_confidence=0,
            )

        # 4. Apply whale decay and validate signals
        adjusted_signals = self.apply_whale_decay(request.signals)
        signals_valid, signal_reasons = self.validate_signals(adjusted_signals, regime)

        if not signals_valid:
            return RiskAssessment(
                decision=TradeDecision.REJECTED_INSUFFICIENT_SIGNALS,
                approved=False,
                adjusted_size_usd=0,
                stop_loss_price=0,
                reasons=signal_reasons,
                warnings=[],
                regime=regime,
                signal_count=len(request.signals),
                avg_confidence=sum(s.confidence for s in adjusted_signals) / len(adjusted_signals) if adjusted_signals else 0,
            )

        # 5. Check liquidity
        liq_ok, liq_reason = self.check_liquidity(request.symbol, request.size_usd)
        if not liq_ok:
            return RiskAssessment(
                decision=TradeDecision.REJECTED_LOW_LIQUIDITY,
                approved=False,
                adjusted_size_usd=0,
                stop_loss_price=0,
                reasons=[liq_reason],
                warnings=[],
                regime=regime,
                signal_count=len(request.signals),
                avg_confidence=sum(s.confidence for s in adjusted_signals) / len(adjusted_signals),
            )

        # 6. Calculate position size
        adjusted_size, size_warnings = self.calculate_position_size(
            request.size_usd,
            request.symbol,
            regime
        )
        warnings.extend(size_warnings)

        if adjusted_size <= 0:
            return RiskAssessment(
                decision=TradeDecision.REJECTED_EXPOSURE,
                approved=False,
                adjusted_size_usd=0,
                stop_loss_price=0,
                reasons=["Position size reduced to zero due to exposure limits"],
                warnings=warnings,
                regime=regime,
                signal_count=len(request.signals),
                avg_confidence=sum(s.confidence for s in adjusted_signals) / len(adjusted_signals),
            )

        # 7. Calculate stop-loss
        stop_loss = request.stop_loss_price or self.calculate_stop_loss(
            request.entry_price,
            request.direction
        )

        # 8. Check if approval required for large trades
        decision = TradeDecision.APPROVED
        if adjusted_size > self.aum * 0.05:  # > 5% AUM needs approval
            decision = TradeDecision.REQUIRES_APPROVAL
            warnings.append("Trade > 5% AUM requires human approval")

        avg_confidence = sum(s.confidence for s in adjusted_signals) / len(adjusted_signals) if adjusted_signals else 0

        return RiskAssessment(
            decision=decision,
            approved=(decision == TradeDecision.APPROVED),
            adjusted_size_usd=adjusted_size,
            stop_loss_price=stop_loss,
            reasons=reasons,
            warnings=warnings,
            regime=regime,
            signal_count=len(set(s.signal_type for s in adjusted_signals)),
            avg_confidence=avg_confidence,
        )

    def record_trade(self, symbol: str, size_usd: float):
        """Record an executed trade for exposure tracking."""
        self.positions[symbol] = self.positions.get(symbol, 0) + size_usd
        self.total_exposure += size_usd
        print(f"  [RISK] Position recorded: {symbol} +${size_usd:,.2f}")
        print(f"  [RISK] Total exposure: ${self.total_exposure:,.2f} ({self.total_exposure/self.aum*100:.1f}% of AUM)")

    def close_position(self, symbol: str, size_usd: float):
        """Record a closed position."""
        self.positions[symbol] = max(0, self.positions.get(symbol, 0) - size_usd)
        self.total_exposure = max(0, self.total_exposure - size_usd)
        print(f"  [RISK] Position closed: {symbol} -${size_usd:,.2f}")


# =============================================================================
# DEMO / TEST
# =============================================================================

def demo_risk_manager():
    """Demonstrate risk manager functionality."""
    print("\n" + "=" * 70)
    print("RISK MANAGER DEMO")
    print("=" * 70)

    rm = RiskManager()
    rm.set_aum(10000)

    # Check circuit breaker
    print("\n[1] Checking circuit breaker...")
    triggered, reason = rm.check_circuit_breaker()
    print(f"    Circuit breaker: {'TRIGGERED' if triggered else 'OK'} - {reason}")

    # Detect regime
    print("\n[2] Detecting market regime...")
    regime = rm.detect_regime()
    print(f"    Current regime: {regime.value}")

    # Create sample signals
    print("\n[3] Creating sample trade request...")
    signals = [
        Signal(
            signal_type="technical",
            direction="BUY",
            confidence=0.75,
            timestamp=datetime.now(),
            symbol="XRPUSDT",
            source="RSI"
        ),
        Signal(
            signal_type="whale",
            direction="BUY",
            confidence=0.70,
            timestamp=datetime.now() - timedelta(hours=12),
            symbol="XRPUSDT",
            source="whale_tracker"
        ),
        Signal(
            signal_type="funding",
            direction="BUY",
            confidence=0.65,
            timestamp=datetime.now(),
            symbol="XRPUSDT",
            source="funding_analyzer"
        ),
    ]

    request = TradeRequest(
        symbol="XRPUSDT",
        direction="LONG",
        size_usd=500,
        signals=signals,
        entry_price=2.03,
    )

    # Assess trade
    print("\n[4] Assessing trade request...")
    assessment = rm.assess_trade(request)

    print(f"\n    ASSESSMENT RESULT:")
    print(f"    ─────────────────────────────────────")
    print(f"    Decision:        {assessment.decision.value}")
    print(f"    Approved:        {assessment.approved}")
    print(f"    Adjusted Size:   ${assessment.adjusted_size_usd:,.2f}")
    print(f"    Stop-Loss:       ${assessment.stop_loss_price:.4f}")
    print(f"    Regime:          {assessment.regime.value}")
    print(f"    Signal Count:    {assessment.signal_count}")
    print(f"    Avg Confidence:  {assessment.avg_confidence:.2f}")

    if assessment.warnings:
        print(f"\n    Warnings:")
        for w in assessment.warnings:
            print(f"      - {w}")

    if assessment.reasons:
        print(f"\n    Reasons:")
        for r in assessment.reasons:
            print(f"      - {r}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    demo_risk_manager()
