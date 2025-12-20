"""
Signal Engine for HFT System - MICROSTRUCTURE
==============================================
Based on 141K signal analysis:

PROVEN BEST: Orderbook alone (+0.012% at 5m with 55,781 signals)
- 1/5 condition is OPTIMAL - more conditions = WORSE
- RSI adds no value when combined with Orderbook

CONDITIONS (in order of effectiveness):
1. Order Book Imbalance - Bid/Ask ratio > 60% [PRIMARY - USE ALONE]
2. Volume Spike - Current volume > 2.5x average
3. Micro Indicator - RSI (less effective than orderbook)
4. Price Movement - Sharp drop/spike
5. Derivatives Signal - Funding rate

Entry: 1 condition required (Orderbook imbalance preferred)
Exit: 0.15% TP, 0.1% SL, 30s time stop
"""

import logging
import time
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import numpy as np

from .config import SYSTEM_CONFIG, get_asset_config, AssetConfig
from .websocket_manager import WebSocketManager, KlineData

logger = logging.getLogger(__name__)


class SignalType(Enum):
    LONG = "LONG"
    SHORT = "SHORT"  # Future use
    NONE = "NONE"


@dataclass
class ConditionResult:
    """Result of a single condition check."""
    name: str
    triggered: bool
    value: float
    threshold: float
    direction: SignalType = SignalType.NONE
    message: str = ""


@dataclass
class Signal:
    """
    Complete signal with all condition results.

    MICROSTRUCTURE: Entry validity based on ORDERBOOK ONLY.
    """
    symbol: str
    signal_type: SignalType
    conditions_met: int
    total_conditions: int
    confidence: float
    entry_price: float
    timestamp: int
    conditions: List[ConditionResult] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Entry valid ONLY when orderbook imbalance triggered."""
        # Check if orderbook_imbalance condition was triggered
        orderbook_triggered = any(
            c.name == "orderbook_imbalance" and c.triggered
            for c in self.conditions
        )
        return orderbook_triggered and self.signal_type != SignalType.NONE


class RSICalculator:
    """Calculate RSI from price data."""

    def __init__(self, period: int = 14):
        self.period = period
        self.prices: deque = deque(maxlen=period + 50)  # Extra buffer

    def add_price(self, price: float):
        self.prices.append(price)

    def calculate(self) -> Optional[float]:
        if len(self.prices) < self.period + 1:
            return None

        prices = list(self.prices)
        deltas = np.diff(prices[-self.period - 1:])

        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi


class SignalEngine:
    """
    Multi-condition signal generator.

    Monitors all enabled assets and generates signals
    when 3+ of 5 conditions are met.
    """

    def __init__(self, ws_manager: WebSocketManager):
        self.ws = ws_manager

        # RSI calculators per symbol
        self.rsi_calculators: Dict[str, RSICalculator] = {}
        for symbol in SYSTEM_CONFIG.enabled_assets:
            config = get_asset_config(symbol)
            self.rsi_calculators[config.exchange_symbol] = RSICalculator(period=14)

        # Track last signal time per symbol to avoid spam
        self.last_signal_time: Dict[str, float] = {}

        # Signal callback
        self.on_signal = None

        # Register for kline updates to feed RSI
        self.ws.on_kline = self._on_kline

    def _on_kline(self, kline: KlineData):
        """Update RSI calculator with new candle close price."""
        if kline.is_closed and kline.symbol in self.rsi_calculators:
            self.rsi_calculators[kline.symbol].add_price(kline.close)

    def check_price_movement(self, symbol: str, config: AssetConfig) -> ConditionResult:
        """
        Condition 1: Sharp price movement.

        Checks if price has dropped 0.8-1.2% in the last 60 seconds.
        This indicates a potential mean-reversion opportunity.
        """
        exchange_symbol = config.exchange_symbol
        price_change = self.ws.get_price_change(symbol, config.impulse_window_sec)

        if price_change is None:
            return ConditionResult(
                name="price_movement",
                triggered=False,
                value=0,
                threshold=config.min_price_drop_pct,
                message="Insufficient price data"
            )

        # Check for drop (negative change)
        if price_change <= -config.min_price_drop_pct:
            if abs(price_change) <= config.max_price_drop_pct:
                return ConditionResult(
                    name="price_movement",
                    triggered=True,
                    value=abs(price_change),
                    threshold=config.min_price_drop_pct,
                    direction=SignalType.LONG,
                    message=f"Sharp drop: {price_change:.2f}% in {config.impulse_window_sec}s"
                )
            else:
                return ConditionResult(
                    name="price_movement",
                    triggered=False,
                    value=abs(price_change),
                    threshold=config.max_price_drop_pct,
                    message=f"Drop too large (knife): {price_change:.2f}%"
                )

        # Check for spike (positive change) - potential short
        if price_change >= config.min_price_spike_pct:
            return ConditionResult(
                name="price_movement",
                triggered=True,
                value=price_change,
                threshold=config.min_price_spike_pct,
                direction=SignalType.SHORT,
                message=f"Sharp spike: +{price_change:.2f}%"
            )

        return ConditionResult(
            name="price_movement",
            triggered=False,
            value=abs(price_change),
            threshold=config.min_price_drop_pct,
            message=f"Price stable: {price_change:.2f}%"
        )

    def check_volume_spike(self, symbol: str, config: AssetConfig) -> ConditionResult:
        """
        Condition 2: Volume spike.

        Checks if recent volume is significantly higher than average.
        High volume confirms the price movement is significant.
        """
        volume_ratio = self.ws.get_volume_spike(symbol, short_window=60, long_window=300)

        if volume_ratio is None:
            return ConditionResult(
                name="volume_spike",
                triggered=False,
                value=0,
                threshold=config.volume_spike_multiplier,
                message="Insufficient volume data"
            )

        triggered = volume_ratio >= config.volume_spike_multiplier

        return ConditionResult(
            name="volume_spike",
            triggered=triggered,
            value=volume_ratio,
            threshold=config.volume_spike_multiplier,
            message=f"Volume {volume_ratio:.1f}x average" if triggered else f"Volume normal: {volume_ratio:.1f}x"
        )

    def check_orderbook_imbalance(self, symbol: str, config: AssetConfig) -> ConditionResult:
        """
        Condition 3: Order book imbalance.

        Checks if there's a significant imbalance between bids and asks.
        More bids = bullish pressure, more asks = bearish pressure.
        """
        orderbook = self.ws.get_orderbook(symbol)

        if orderbook is None:
            return ConditionResult(
                name="orderbook_imbalance",
                triggered=False,
                value=0.5,
                threshold=config.orderbook_imbalance_ratio,
                message="No orderbook data"
            )

        imbalance = orderbook.imbalance_ratio

        # For LONG: we want more bids (buyers) = imbalance > threshold
        # For SHORT: we want more asks (sellers) = imbalance < (1 - threshold)

        if imbalance >= config.orderbook_imbalance_ratio:
            return ConditionResult(
                name="orderbook_imbalance",
                triggered=True,
                value=imbalance,
                threshold=config.orderbook_imbalance_ratio,
                direction=SignalType.LONG,
                message=f"Bullish imbalance: {imbalance:.1%} bids"
            )
        elif imbalance <= (1 - config.orderbook_imbalance_ratio):
            return ConditionResult(
                name="orderbook_imbalance",
                triggered=True,
                value=imbalance,
                threshold=1 - config.orderbook_imbalance_ratio,
                direction=SignalType.SHORT,
                message=f"Bearish imbalance: {1 - imbalance:.1%} asks"
            )

        return ConditionResult(
            name="orderbook_imbalance",
            triggered=False,
            value=imbalance,
            threshold=config.orderbook_imbalance_ratio,
            message=f"Balanced book: {imbalance:.1%} bids"
        )

    def check_derivatives_signal(self, symbol: str, config: AssetConfig) -> ConditionResult:
        """
        Condition 4: Derivatives market signal.

        Checks funding rate:
        - Negative funding = shorts paying longs = bullish
        - Highly positive funding = longs paying shorts = bearish
        """
        funding_rate = self.ws.get_funding_rate(symbol)

        if funding_rate is None:
            return ConditionResult(
                name="derivatives_signal",
                triggered=False,
                value=0,
                threshold=0,
                message="No funding rate data"
            )

        # Convert to percentage (funding rate is typically in decimal form)
        funding_pct = funding_rate * 100

        # Negative funding = shorts paying longs = bullish
        if funding_rate < -0.01:  # -0.01% threshold
            return ConditionResult(
                name="derivatives_signal",
                triggered=True,
                value=funding_pct,
                threshold=-0.01,
                direction=SignalType.LONG,
                message=f"Negative funding: {funding_pct:.4f}% (bullish)"
            )

        # Highly positive funding = bearish
        if funding_rate > 0.05:  # 0.05% threshold
            return ConditionResult(
                name="derivatives_signal",
                triggered=True,
                value=funding_pct,
                threshold=0.05,
                direction=SignalType.SHORT,
                message=f"High positive funding: {funding_pct:.4f}% (bearish)"
            )

        return ConditionResult(
            name="derivatives_signal",
            triggered=False,
            value=funding_pct,
            threshold=0.01,
            message=f"Neutral funding: {funding_pct:.4f}%"
        )

    def check_micro_indicator(self, symbol: str, config: AssetConfig) -> ConditionResult:
        """
        Condition 5: Micro indicator (RSI).

        Checks RSI for oversold/overbought conditions:
        - RSI < 25 = oversold = bullish (LONG)
        - RSI > 75 = overbought = bearish (SHORT)
        """
        exchange_symbol = config.exchange_symbol
        calculator = self.rsi_calculators.get(exchange_symbol)

        if calculator is None:
            return ConditionResult(
                name="micro_indicator",
                triggered=False,
                value=50,
                threshold=config.rsi_oversold,
                message="No RSI calculator"
            )

        rsi = calculator.calculate()

        if rsi is None:
            return ConditionResult(
                name="micro_indicator",
                triggered=False,
                value=50,
                threshold=config.rsi_oversold,
                message="Insufficient data for RSI"
            )

        if rsi <= config.rsi_oversold:
            return ConditionResult(
                name="micro_indicator",
                triggered=True,
                value=rsi,
                threshold=config.rsi_oversold,
                direction=SignalType.LONG,
                message=f"RSI oversold: {rsi:.1f}"
            )

        if rsi >= config.rsi_overbought:
            return ConditionResult(
                name="micro_indicator",
                triggered=True,
                value=rsi,
                threshold=config.rsi_overbought,
                direction=SignalType.SHORT,
                message=f"RSI overbought: {rsi:.1f}"
            )

        return ConditionResult(
            name="micro_indicator",
            triggered=False,
            value=rsi,
            threshold=config.rsi_oversold,
            message=f"RSI neutral: {rsi:.1f}"
        )

    def check_all_conditions(self, symbol: str) -> Signal:
        """
        MICROSTRUCTURE ENTRY - Orderbook imbalance ONLY.

        Based on 141K signal analysis:
        - Orderbook alone: +0.012% at 5m (BEST)
        - RSI adds NO value - removed from entry decision
        - More conditions = worse performance

        Entry triggers ONLY on orderbook imbalance.
        Other conditions logged for data collection only.
        """
        config = get_asset_config(symbol)
        conditions = []

        # Check orderbook FIRST - this is the ONLY entry condition
        orderbook_cond = self.check_orderbook_imbalance(symbol, config)
        conditions.append(orderbook_cond)

        # Log other conditions for data collection (NOT used for entry)
        conditions.append(self.check_price_movement(symbol, config))
        conditions.append(self.check_volume_spike(symbol, config))
        conditions.append(self.check_derivatives_signal(symbol, config))
        # RSI removed from entry logic - proven to add no value
        # conditions.append(self.check_micro_indicator(symbol, config))

        # ENTRY DECISION: Orderbook imbalance ONLY
        orderbook_triggered = orderbook_cond.triggered

        # Count for logging purposes
        triggered = [c for c in conditions if c.triggered]
        conditions_met = len(triggered)

        # Signal direction from orderbook imbalance
        if orderbook_triggered:
            signal_type = orderbook_cond.direction
        else:
            signal_type = SignalType.NONE

        # Get current price
        current_price = self.ws.get_current_price(symbol) or 0

        # Confidence based on orderbook strength
        if orderbook_triggered:
            # Orderbook imbalance value is between 0-1, higher = stronger
            imbalance = orderbook_cond.value
            confidence = abs(imbalance - 0.5) * 2  # Convert to 0-1 scale
        else:
            confidence = 0

        signal = Signal(
            symbol=symbol,
            signal_type=signal_type,
            conditions_met=conditions_met,
            total_conditions=len(conditions),
            confidence=confidence,
            entry_price=current_price,
            timestamp=int(time.time() * 1000),
            conditions=conditions
        )

        return signal

    def scan_all_assets(self) -> List[Signal]:
        """
        Scan all enabled assets for signals.

        Returns list of valid signals (3+ conditions met).
        """
        valid_signals = []

        for symbol in SYSTEM_CONFIG.enabled_assets:
            signal = self.check_all_conditions(symbol)

            if signal.is_valid:
                # Check cooldown
                last_time = self.last_signal_time.get(symbol, 0)
                if time.time() - last_time < 30:  # 30s cooldown
                    continue

                self.last_signal_time[symbol] = time.time()
                valid_signals.append(signal)

                # Log signal
                logger.info(
                    f"SIGNAL: {signal.symbol} {signal.signal_type.value} | "
                    f"Conditions: {signal.conditions_met}/5 | "
                    f"Price: ${signal.entry_price:.4f} | "
                    f"Confidence: {signal.confidence:.0%}"
                )

                for c in signal.conditions:
                    status = "OK" if c.triggered else "--"
                    logger.debug(f"  [{status}] {c.name}: {c.message}")

                # Callback
                if self.on_signal:
                    self.on_signal(signal)

        return valid_signals

    def get_status(self) -> Dict:
        """Get current status of signal engine."""
        status = {}

        for symbol in SYSTEM_CONFIG.enabled_assets:
            config = get_asset_config(symbol)
            exchange_symbol = config.exchange_symbol

            price = self.ws.get_current_price(symbol)
            price_change = self.ws.get_price_change(symbol, 60)
            volume_spike = self.ws.get_volume_spike(symbol)
            orderbook = self.ws.get_orderbook(symbol)
            funding = self.ws.get_funding_rate(symbol)
            rsi_calc = self.rsi_calculators.get(exchange_symbol)
            rsi = rsi_calc.calculate() if rsi_calc else None

            status[symbol] = {
                "price": price,
                "price_change_60s": price_change,
                "volume_spike": volume_spike,
                "orderbook_imbalance": orderbook.imbalance_ratio if orderbook else None,
                "funding_rate": funding,
                "rsi": rsi
            }

        return status
