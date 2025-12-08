"""
Realistic execution model for accurate backtesting.

Simulates real-world trading conditions including:
- Depth-based slippage (market impact)
- Exchange constraints (min notional, lot sizes, tick sizes)
- Commission and fee structures
- Latency simulation
- Partial fills
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


class OrderType(str, Enum):
    """Order type."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_MARKET = "STOP_MARKET"
    TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"


class OrderSide(str, Enum):
    """Order side."""

    BUY = "BUY"
    SELL = "SELL"


@dataclass
class ExchangeRules:
    """Exchange-specific trading rules for a symbol."""

    symbol: str

    # Price constraints
    tick_size: float = 0.01  # Minimum price increment
    min_price: float = 0.0
    max_price: float = float("inf")

    # Quantity constraints
    lot_size: float = 0.001  # Minimum quantity increment
    min_qty: float = 0.001
    max_qty: float = float("inf")

    # Notional constraints
    min_notional: float = 10.0  # Minimum order value in quote currency
    max_notional: float = float("inf")

    # Market constraints
    max_num_orders: int = 200
    max_position: float = float("inf")

    # Fees (in percentage)
    maker_fee: float = 0.02  # 0.02% = 2 bps
    taker_fee: float = 0.04  # 0.04% = 4 bps

    def round_price(self, price: float) -> float:
        """Round price to valid tick size."""
        return round(price / self.tick_size) * self.tick_size

    def round_qty(self, qty: float) -> float:
        """Round quantity to valid lot size."""
        return round(qty / self.lot_size) * self.lot_size

    def validate_order(
        self,
        price: float,
        qty: float,
        side: OrderSide,
    ) -> tuple[bool, list[str]]:
        """
        Validate an order against exchange rules.

        Returns (is_valid, list of violations).
        """
        violations = []

        # Check price
        if price < self.min_price:
            violations.append(f"Price {price} below minimum {self.min_price}")
        if price > self.max_price:
            violations.append(f"Price {price} above maximum {self.max_price}")

        # Check quantity
        if qty < self.min_qty:
            violations.append(f"Quantity {qty} below minimum {self.min_qty}")
        if qty > self.max_qty:
            violations.append(f"Quantity {qty} above maximum {self.max_qty}")

        # Check notional
        notional = price * qty
        if notional < self.min_notional:
            violations.append(f"Notional {notional} below minimum {self.min_notional}")

        return len(violations) == 0, violations


@dataclass
class OrderbookDepth:
    """Orderbook depth for slippage calculation."""

    # List of (price, quantity) tuples
    bids: list[tuple[float, float]] = field(default_factory=list)
    asks: list[tuple[float, float]] = field(default_factory=list)

    @property
    def best_bid(self) -> float:
        """Best bid price."""
        return self.bids[0][0] if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        """Best ask price."""
        return self.asks[0][0] if self.asks else 0.0

    @property
    def mid_price(self) -> float:
        """Mid price."""
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return self.best_bid or self.best_ask

    @property
    def spread(self) -> float:
        """Bid-ask spread."""
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return 0.0

    @property
    def spread_bps(self) -> float:
        """Spread in basis points."""
        mid = self.mid_price
        if mid > 0:
            return (self.spread / mid) * 10000
        return 0.0

    def calculate_market_impact(
        self,
        qty: float,
        side: OrderSide,
    ) -> tuple[float, float]:
        """
        Calculate market impact (slippage) for a market order.

        Args:
            qty: Order quantity
            side: Order side (BUY or SELL)

        Returns:
            (average_fill_price, slippage_pct)
        """
        levels = self.asks if side == OrderSide.BUY else self.bids
        if not levels:
            return 0.0, 0.0

        remaining = qty
        total_cost = 0.0
        filled_qty = 0.0

        for price, level_qty in levels:
            fill_qty = min(remaining, level_qty)
            total_cost += fill_qty * price
            filled_qty += fill_qty
            remaining -= fill_qty

            if remaining <= 0:
                break

        if filled_qty == 0:
            return 0.0, 0.0

        avg_price = total_cost / filled_qty
        best_price = self.best_ask if side == OrderSide.BUY else self.best_bid

        # Slippage as percentage
        if best_price > 0:
            if side == OrderSide.BUY:
                slippage_pct = (avg_price - best_price) / best_price * 100
            else:
                slippage_pct = (best_price - avg_price) / best_price * 100
        else:
            slippage_pct = 0.0

        return avg_price, slippage_pct


@dataclass
class ExecutionResult:
    """Result of order execution simulation."""

    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    requested_qty: float
    filled_qty: float
    requested_price: float | None  # None for market orders
    avg_fill_price: float
    slippage_pct: float
    commission: float
    commission_asset: str
    execution_time: datetime
    latency_ms: float
    is_partial: bool
    is_rejected: bool
    rejection_reason: str | None = None

    @property
    def fill_ratio(self) -> float:
        """Percentage of order filled."""
        if self.requested_qty == 0:
            return 0.0
        return self.filled_qty / self.requested_qty

    @property
    def total_cost(self) -> float:
        """Total cost including commission."""
        return self.avg_fill_price * self.filled_qty + self.commission


class RealisticExecutionModel:
    """
    Simulates realistic order execution with market microstructure effects.

    Key features:
    - Depth-based slippage calculation
    - Exchange rule enforcement
    - Latency simulation
    - Partial fill modeling
    - Commission calculation
    """

    # Default exchange rules for common symbols
    DEFAULT_RULES = {
        "BTCUSDT": ExchangeRules(
            symbol="BTCUSDT",
            tick_size=0.01,
            lot_size=0.001,
            min_qty=0.001,
            min_notional=10.0,
            maker_fee=0.02,
            taker_fee=0.04,
        ),
        "ETHUSDT": ExchangeRules(
            symbol="ETHUSDT",
            tick_size=0.01,
            lot_size=0.001,
            min_qty=0.001,
            min_notional=10.0,
            maker_fee=0.02,
            taker_fee=0.04,
        ),
    }

    def __init__(
        self,
        exchange_rules: dict[str, ExchangeRules] | None = None,
        latency_mean_ms: float = 50.0,
        latency_std_ms: float = 20.0,
        partial_fill_probability: float = 0.05,
    ):
        """
        Initialize execution model.

        Args:
            exchange_rules: Dict of symbol -> ExchangeRules
            latency_mean_ms: Mean execution latency
            latency_std_ms: Latency standard deviation
            partial_fill_probability: Probability of partial fills
        """
        self.rules = {**self.DEFAULT_RULES, **(exchange_rules or {})}
        self.latency_mean = latency_mean_ms
        self.latency_std = latency_std_ms
        self.partial_fill_prob = partial_fill_probability

        # Running order ID counter
        self._order_counter = 0

    def get_rules(self, symbol: str) -> ExchangeRules:
        """Get exchange rules for a symbol."""
        if symbol in self.rules:
            return self.rules[symbol]

        # Return default rules for unknown symbols
        return ExchangeRules(symbol=symbol)

    def simulate_latency(self) -> float:
        """Simulate execution latency in milliseconds."""
        latency = np.random.normal(self.latency_mean, self.latency_std)
        return max(1.0, latency)  # Minimum 1ms

    def estimate_slippage_from_volatility(
        self,
        price: float,
        qty: float,
        position_size_usd: float,
        atr_pct: float,
        volume_24h: float | None = None,
    ) -> float:
        """
        Estimate slippage when orderbook depth is not available.

        Uses ATR and order size relative to typical volume.

        Args:
            price: Current price
            qty: Order quantity
            position_size_usd: Order value in USD
            atr_pct: ATR as percentage of price
            volume_24h: 24h trading volume (optional)

        Returns:
            Estimated slippage as percentage
        """
        # Base slippage from spread (estimate as fraction of ATR)
        spread_estimate = atr_pct * 0.1  # Spread typically ~10% of ATR

        # Market impact from order size
        if volume_24h and volume_24h > 0:
            # Order size relative to hourly volume
            hourly_volume = volume_24h / 24
            size_ratio = position_size_usd / (hourly_volume * price) if hourly_volume > 0 else 0

            # Square root market impact model
            market_impact = spread_estimate * np.sqrt(size_ratio) * 10
        else:
            # Default: larger orders have more impact
            market_impact = spread_estimate * (position_size_usd / 10000) ** 0.5

        # Total slippage
        total_slippage = spread_estimate / 2 + market_impact

        # Cap at reasonable maximum (5% for extreme cases)
        return min(total_slippage, 5.0)

    def simulate_market_order(
        self,
        symbol: str,
        side: OrderSide,
        qty: float,
        orderbook: OrderbookDepth | None = None,
        current_price: float | None = None,
        atr_pct: float = 0.5,
        timestamp: datetime | None = None,
    ) -> ExecutionResult:
        """
        Simulate a market order execution.

        Args:
            symbol: Trading symbol
            side: Order side
            qty: Order quantity
            orderbook: Orderbook depth (for accurate slippage)
            current_price: Current price (if no orderbook)
            atr_pct: ATR percentage (for slippage estimation)
            timestamp: Order timestamp

        Returns:
            ExecutionResult
        """
        self._order_counter += 1
        order_id = f"SIM-{self._order_counter:06d}"
        timestamp = timestamp or datetime.utcnow()
        rules = self.get_rules(symbol)

        # Round quantity
        qty = rules.round_qty(qty)

        # Calculate fill price and slippage
        if orderbook and orderbook.mid_price > 0:
            avg_price, slippage_pct = orderbook.calculate_market_impact(qty, side)
            if avg_price == 0:
                avg_price = orderbook.best_ask if side == OrderSide.BUY else orderbook.best_bid
        elif current_price:
            # Estimate slippage
            position_value = qty * current_price
            slippage_pct = self.estimate_slippage_from_volatility(
                current_price, qty, position_value, atr_pct
            )
            # Apply slippage
            if side == OrderSide.BUY:
                avg_price = current_price * (1 + slippage_pct / 100)
            else:
                avg_price = current_price * (1 - slippage_pct / 100)
        else:
            return ExecutionResult(
                order_id=order_id,
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                requested_qty=qty,
                filled_qty=0,
                requested_price=None,
                avg_fill_price=0,
                slippage_pct=0,
                commission=0,
                commission_asset=symbol[-4:],  # e.g., "USDT"
                execution_time=timestamp,
                latency_ms=self.simulate_latency(),
                is_partial=False,
                is_rejected=True,
                rejection_reason="No price data available",
            )

        # Validate order
        is_valid, violations = rules.validate_order(avg_price, qty, side)
        if not is_valid:
            return ExecutionResult(
                order_id=order_id,
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                requested_qty=qty,
                filled_qty=0,
                requested_price=None,
                avg_fill_price=0,
                slippage_pct=0,
                commission=0,
                commission_asset=symbol[-4:],
                execution_time=timestamp,
                latency_ms=self.simulate_latency(),
                is_partial=False,
                is_rejected=True,
                rejection_reason="; ".join(violations),
            )

        # Simulate partial fill
        is_partial = np.random.random() < self.partial_fill_prob
        if is_partial:
            fill_ratio = np.random.uniform(0.7, 0.99)
            filled_qty = rules.round_qty(qty * fill_ratio)
        else:
            filled_qty = qty

        # Calculate commission (taker fee for market orders)
        notional = avg_price * filled_qty
        commission = notional * (rules.taker_fee / 100)

        return ExecutionResult(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            requested_qty=qty,
            filled_qty=filled_qty,
            requested_price=None,
            avg_fill_price=rules.round_price(avg_price),
            slippage_pct=round(slippage_pct, 4),
            commission=round(commission, 8),
            commission_asset=symbol[-4:],
            execution_time=timestamp + timedelta(milliseconds=self.simulate_latency()),
            latency_ms=self.simulate_latency(),
            is_partial=is_partial,
            is_rejected=False,
        )

    def simulate_trade_sequence(
        self,
        symbol: str,
        entry_price: float,
        entry_qty: float,
        tp_price: float,
        sl_price: float,
        price_series: pd.DataFrame,
        atr_pct: float = 0.5,
    ) -> dict[str, Any]:
        """
        Simulate a complete trade (entry -> exit).

        Args:
            symbol: Trading symbol
            entry_price: Intended entry price
            entry_qty: Entry quantity
            tp_price: Take profit price
            sl_price: Stop loss price
            price_series: DataFrame with timestamp, high, low, close
            atr_pct: ATR percentage

        Returns:
            Dictionary with trade results
        """
        rules = self.get_rules(symbol)

        # Simulate entry
        entry_result = self.simulate_market_order(
            symbol=symbol,
            side=OrderSide.BUY,
            qty=entry_qty,
            current_price=entry_price,
            atr_pct=atr_pct,
        )

        if entry_result.is_rejected:
            return {
                "status": "rejected",
                "rejection_reason": entry_result.rejection_reason,
                "entry": entry_result,
            }

        actual_entry = entry_result.avg_fill_price
        filled_qty = entry_result.filled_qty

        # Adjust TP/SL based on actual entry
        entry_diff_pct = (actual_entry - entry_price) / entry_price * 100
        tp_adjusted = tp_price * (1 + entry_diff_pct / 100)
        sl_adjusted = sl_price * (1 + entry_diff_pct / 100)

        # Simulate price path
        exit_result = None
        exit_reason = None
        mfe = 0.0
        mae = 0.0

        for _, row in price_series.iterrows():
            high = row["high"]
            low = row["low"]
            close = row["close"]
            timestamp = row.get("timestamp", datetime.utcnow())

            # Track MFE/MAE
            high_pnl = (high - actual_entry) / actual_entry * 100
            low_pnl = (low - actual_entry) / actual_entry * 100
            mfe = max(mfe, high_pnl)
            mae = min(mae, low_pnl)

            # Check TP (assumes TP hit at high)
            if high >= tp_adjusted:
                exit_result = self.simulate_market_order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    qty=filled_qty,
                    current_price=tp_adjusted,
                    atr_pct=atr_pct,
                    timestamp=timestamp,
                )
                exit_reason = "TP"
                break

            # Check SL (assumes SL hit at low)
            if low <= sl_adjusted:
                exit_result = self.simulate_market_order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    qty=filled_qty,
                    current_price=sl_adjusted,
                    atr_pct=atr_pct,
                    timestamp=timestamp,
                )
                exit_reason = "SL"
                break

        # If no exit, use last price
        if exit_result is None:
            last_close = price_series["close"].iloc[-1]
            exit_result = self.simulate_market_order(
                symbol=symbol,
                side=OrderSide.SELL,
                qty=filled_qty,
                current_price=last_close,
                atr_pct=atr_pct,
            )
            exit_reason = "TIMEOUT"

        # Calculate final PnL
        gross_pnl = (exit_result.avg_fill_price - actual_entry) * filled_qty
        total_commission = entry_result.commission + exit_result.commission
        net_pnl = gross_pnl - total_commission
        net_pnl_pct = net_pnl / (actual_entry * filled_qty) * 100

        return {
            "status": "completed",
            "entry": entry_result,
            "exit": exit_result,
            "exit_reason": exit_reason,
            "entry_price": actual_entry,
            "exit_price": exit_result.avg_fill_price,
            "filled_qty": filled_qty,
            "gross_pnl": round(gross_pnl, 2),
            "total_commission": round(total_commission, 4),
            "net_pnl": round(net_pnl, 2),
            "net_pnl_pct": round(net_pnl_pct, 4),
            "entry_slippage_pct": entry_result.slippage_pct,
            "exit_slippage_pct": exit_result.slippage_pct,
            "total_slippage_pct": entry_result.slippage_pct + exit_result.slippage_pct,
            "mfe_pct": round(mfe, 4),
            "mae_pct": round(mae, 4),
        }
