"""
Execution Engine for HFT System
===============================
Handles trade execution and exit management.

EXIT CONDITIONS:
- Take Profit: +2% from entry
- Stop Loss: -1% from entry
- Time Stop: 90 seconds with <0.5% profit

Also monitors open positions every tick for exit opportunities.
"""

import asyncio
import logging
import time
from typing import Dict, Optional, Callable
from dataclasses import dataclass
from enum import Enum

from .config import SYSTEM_CONFIG, TradingMode, get_asset_config
from .signal_engine import Signal, SignalType
from .risk_controller import RiskController, Position, RiskDecision
from .websocket_manager import WebSocketManager

logger = logging.getLogger(__name__)


class ExitReason(Enum):
    """Reasons for exiting a position."""
    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    TIME_STOP = "time_stop"
    MANUAL = "manual"
    EMERGENCY = "emergency"
    SIGNAL_EXIT = "signal_exit"
    # MICROSTRUCTURE event-based exits
    MICRO_PROFIT = "micro_profit"    # Small profit + OB weakening
    OB_FLIP = "ob_flip"              # Orderbook imbalance flipped against us
    DELTA_NEGATIVE = "delta_negative"  # Trade flow turned against us
    SPREAD_WIDEN = "spread_widen"    # Spread widened >2x entry spread
    NO_MOVEMENT = "no_movement"       # No favorable movement after 3s
    FLOW_NEUTRAL = "flow_neutral"    # Trade flow went neutral after 3s


@dataclass
class TradeResult:
    """Result of a trade execution."""
    success: bool
    symbol: str
    side: str
    entry_price: float
    quantity: float
    timestamp: int
    order_id: str = ""
    error: str = ""


@dataclass
class ExitResult:
    """Result of position exit."""
    success: bool
    symbol: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    hold_time_sec: float
    reason: ExitReason
    timestamp: int
    entry_time: int = 0  # For linking to trade_id


class ExecutionEngine:
    """
    Manages trade execution and position monitoring.

    In PAPER mode: Simulates trades instantly
    In LIVE mode: Would submit orders to exchange (not implemented)
    """

    def __init__(
        self,
        ws_manager: WebSocketManager,
        risk_controller: RiskController,
        mode: TradingMode = None
    ):
        self.ws = ws_manager
        self.risk = risk_controller
        self.mode = mode or SYSTEM_CONFIG.mode

        # Callbacks for logging
        self.on_trade: Optional[Callable[[TradeResult], None]] = None
        self.on_exit: Optional[Callable[[ExitResult], None]] = None

        # Position monitoring
        self.monitoring = False
        self.monitor_interval = 0.1  # Check every 100ms

        # Track MFE/MAE for analysis
        self.position_mfe: Dict[str, float] = {}  # Max favorable excursion
        self.position_mae: Dict[str, float] = {}  # Max adverse excursion

        # MICROSTRUCTURE: Track entry conditions for event-based exits
        self.entry_spread: Dict[str, float] = {}      # Spread at entry
        self.entry_imbalance: Dict[str, float] = {}   # OB imbalance at entry
        self.entry_delta: Dict[str, float] = {}       # Trade flow delta at entry
        self.entry_price_check: Dict[str, float] = {} # Price at 3s for movement check
        self.entry_check_time: Dict[str, float] = {}  # Time of 3s check

        logger.info(f"ExecutionEngine initialized in {self.mode.value} mode")

    async def execute_entry(self, signal: Signal, risk_decision: RiskDecision) -> TradeResult:
        """
        Execute entry for a signal.

        In PAPER mode, instantly fills at signal price.
        """
        if not risk_decision.approved:
            return TradeResult(
                success=False,
                symbol=signal.symbol,
                side=signal.signal_type.value,
                entry_price=signal.entry_price,
                quantity=0,
                timestamp=int(time.time() * 1000),
                error=risk_decision.message
            )

        if self.mode == TradingMode.PAPER:
            # Paper trading - instant fill
            return await self._paper_entry(signal, risk_decision)
        elif self.mode == TradingMode.LIVE:
            # Live trading - would submit to exchange
            return await self._live_entry(signal, risk_decision)
        else:
            return TradeResult(
                success=False,
                symbol=signal.symbol,
                side=signal.signal_type.value,
                entry_price=signal.entry_price,
                quantity=0,
                timestamp=int(time.time() * 1000),
                error=f"Unknown mode: {self.mode}"
            )

    async def _paper_entry(self, signal: Signal, risk_decision: RiskDecision) -> TradeResult:
        """Execute paper trade entry."""
        # Simulate small slippage (0.01-0.03%)
        slippage = signal.entry_price * 0.0002  # 0.02% average
        if signal.signal_type == SignalType.LONG:
            fill_price = signal.entry_price + slippage
        else:
            fill_price = signal.entry_price - slippage

        # Create position in risk controller
        signal.entry_price = fill_price
        position = self.risk.open_position(signal, risk_decision.position_size)

        # Initialize MFE/MAE tracking
        self.position_mfe[signal.symbol] = 0
        self.position_mae[signal.symbol] = 0

        # MICROSTRUCTURE: Capture entry conditions for event-based exits
        orderbook = self.ws.get_orderbook(signal.symbol)
        trade_flow = self.ws.get_trade_flow(signal.symbol)
        self.entry_spread[signal.symbol] = orderbook.spread if orderbook else 0
        self.entry_imbalance[signal.symbol] = orderbook.imbalance_ratio if orderbook else 0.5
        self.entry_delta[signal.symbol] = trade_flow.get("net_delta", 0)
        self.entry_price_check[signal.symbol] = fill_price  # Will check movement after 3s
        self.entry_check_time[signal.symbol] = 0  # Not checked yet

        result = TradeResult(
            success=True,
            symbol=signal.symbol,
            side=signal.signal_type.value,
            entry_price=fill_price,
            quantity=risk_decision.position_size,
            timestamp=int(time.time() * 1000),
            order_id=position.position_id
        )

        logger.info(
            f"PAPER ENTRY: {signal.symbol} {signal.signal_type.value} | "
            f"Price: ${fill_price:.4f} | Qty: {risk_decision.position_size:.4f} | "
            f"Value: ${risk_decision.position_value:.2f}"
        )

        if self.on_trade:
            self.on_trade(result)

        return result

    async def _live_entry(self, signal: Signal, risk_decision: RiskDecision) -> TradeResult:
        """Execute live trade entry (placeholder)."""
        # TODO: Implement actual exchange order submission
        logger.warning("LIVE trading not implemented - using paper mode")
        return await self._paper_entry(signal, risk_decision)

    async def check_exits(self):
        """
        Check all open positions for exit conditions.

        Called continuously by the monitor loop.
        """
        positions = self.risk.get_all_positions()

        for position in positions:
            exit_reason = self._check_exit_conditions(position)

            if exit_reason:
                await self.execute_exit(position.symbol, exit_reason)

    def _check_exit_conditions(self, position: Position) -> Optional[ExitReason]:
        """
        Check if position should be exited.

        MICROSTRUCTURE EXITS (priority order):
        1. Take Profit: +0.15%
        2. Stop Loss: -0.1%
        3. OB Flip: Orderbook imbalance flipped against position
        4. Delta Negative: Trade flow turned against position
        5. Spread Widen: Spread widened 2x+ from entry
        6. No Movement: No favorable movement after 3s
        7. Time Stop (fallback): 30s max hold
        """
        config = get_asset_config(position.symbol)
        current_price = self.ws.get_current_price(position.symbol)

        if current_price is None:
            return None

        # Calculate current P&L
        if position.side == SignalType.LONG:
            pnl_pct = (current_price - position.entry_price) / position.entry_price * 100
        else:
            pnl_pct = (position.entry_price - current_price) / position.entry_price * 100

        # Update MFE/MAE
        if position.symbol in self.position_mfe:
            if pnl_pct > self.position_mfe[position.symbol]:
                self.position_mfe[position.symbol] = pnl_pct
            if pnl_pct < self.position_mae[position.symbol]:
                self.position_mae[position.symbol] = pnl_pct

        hold_time = (time.time() * 1000 - position.entry_time) / 1000

        # 1. Check Take Profit
        if pnl_pct >= config.take_profit_pct:
            logger.info(f"{position.symbol} hit TAKE PROFIT: {pnl_pct:.2f}%")
            return ExitReason.TAKE_PROFIT

        # 2. Check Stop Loss
        if pnl_pct <= -config.stop_loss_pct:
            logger.info(f"{position.symbol} hit STOP LOSS: {pnl_pct:.2f}%")
            return ExitReason.STOP_LOSS

        # === MICROSTRUCTURE EVENT-BASED EXITS ===
        orderbook = self.ws.get_orderbook(position.symbol)
        trade_flow = self.ws.get_trade_flow(position.symbol)

        # 3. MICRO PROFIT: Exit with small profit (+0.05% to +0.12%) if OB is weakening
        # This captures edge before it disappears
        if 0.05 <= pnl_pct < config.take_profit_pct:
            if orderbook:
                current_imbalance = orderbook.imbalance_ratio
                entry_imbalance = self.entry_imbalance.get(position.symbol, 0.5)

                if position.side == SignalType.LONG:
                    # For LONG: OB weakening = imbalance dropping toward 0.5
                    imbalance_drop = entry_imbalance - current_imbalance
                    if imbalance_drop > 0.1:  # Lost 10%+ of bullish edge
                        logger.info(
                            f"{position.symbol} MICRO PROFIT: +{pnl_pct:.3f}% | "
                            f"OB weakening {entry_imbalance:.1%} -> {current_imbalance:.1%}"
                        )
                        return ExitReason.MICRO_PROFIT
                else:
                    # For SHORT: OB weakening = imbalance rising toward 0.5
                    imbalance_rise = current_imbalance - entry_imbalance
                    if imbalance_rise > 0.1:  # Lost 10%+ of bearish edge
                        logger.info(
                            f"{position.symbol} MICRO PROFIT: +{pnl_pct:.3f}% | "
                            f"OB weakening {entry_imbalance:.1%} -> {current_imbalance:.1%}"
                        )
                        return ExitReason.MICRO_PROFIT

        # 4. OB Flip: Orderbook imbalance flipped against us
        if orderbook:
            current_imbalance = orderbook.imbalance_ratio
            entry_imbalance = self.entry_imbalance.get(position.symbol, 0.5)

            if position.side == SignalType.LONG:
                # For LONG: We entered with bullish imbalance (>0.5)
                # Exit if imbalance flips bearish (<0.4) = sellers taking over
                if entry_imbalance > 0.5 and current_imbalance < 0.4:
                    logger.info(
                        f"{position.symbol} OB FLIP: imbalance {entry_imbalance:.1%} -> {current_imbalance:.1%}"
                    )
                    return ExitReason.OB_FLIP
            else:
                # For SHORT: We entered with bearish imbalance (<0.5)
                # Exit if imbalance flips bullish (>0.6) = buyers taking over
                if entry_imbalance < 0.5 and current_imbalance > 0.6:
                    logger.info(
                        f"{position.symbol} OB FLIP: imbalance {entry_imbalance:.1%} -> {current_imbalance:.1%}"
                    )
                    return ExitReason.OB_FLIP

        # 4. Delta Negative: Trade flow turned against us
        if trade_flow:
            current_delta = trade_flow.get("delta_1s", 0)
            entry_delta = self.entry_delta.get(position.symbol, 0)

            if position.side == SignalType.LONG:
                # For LONG: Exit if delta turned significantly negative (sellers winning)
                if entry_delta > 0 and current_delta < -abs(entry_delta) * 0.5:
                    logger.info(
                        f"{position.symbol} DELTA NEGATIVE: {entry_delta:.0f} -> {current_delta:.0f}"
                    )
                    return ExitReason.DELTA_NEGATIVE
            else:
                # For SHORT: Exit if delta turned significantly positive (buyers winning)
                if entry_delta < 0 and current_delta > abs(entry_delta) * 0.5:
                    logger.info(
                        f"{position.symbol} DELTA NEGATIVE: {entry_delta:.0f} -> {current_delta:.0f}"
                    )
                    return ExitReason.DELTA_NEGATIVE

        # 5. Spread Widen: Spread widened >2x from entry
        if orderbook:
            current_spread = orderbook.spread
            entry_spread = self.entry_spread.get(position.symbol, 0)

            if entry_spread > 0 and current_spread > entry_spread * 2:
                logger.info(
                    f"{position.symbol} SPREAD WIDEN: {entry_spread:.4f}% -> {current_spread:.4f}%"
                )
                return ExitReason.SPREAD_WIDEN

        # 6. No Movement: Check after 3 seconds
        if hold_time >= 3:
            check_time = self.entry_check_time.get(position.symbol, 0)
            if check_time == 0:
                # First check at 3s - record price
                self.entry_check_time[position.symbol] = time.time()
                self.entry_price_check[position.symbol] = current_price
            elif time.time() - check_time >= 1:  # Check again after 1s
                entry_check_price = self.entry_price_check.get(position.symbol, position.entry_price)
                price_movement = (current_price - entry_check_price) / entry_check_price * 100

                if position.side == SignalType.LONG:
                    # For LONG: No positive movement or slightly negative
                    if price_movement <= 0.02:  # Less than 0.02% improvement
                        logger.info(
                            f"{position.symbol} NO MOVEMENT: {price_movement:.3f}% in {hold_time:.1f}s"
                        )
                        return ExitReason.NO_MOVEMENT
                else:
                    # For SHORT: No negative movement (price should go down)
                    if price_movement >= -0.02:
                        logger.info(
                            f"{position.symbol} NO MOVEMENT: {price_movement:.3f}% in {hold_time:.1f}s"
                        )
                        return ExitReason.NO_MOVEMENT

        # 7. Flow Neutral: Trade flow went neutral after 3s (no edge left)
        if hold_time >= 3 and trade_flow:
            delta_1s = trade_flow.get("delta_1s", 0)
            delta_pct = trade_flow.get("delta_pct", 0)

            # Check if flow is essentially neutral (within ±10%)
            if abs(delta_pct) < 10:
                # Also check OB hasn't improved
                if orderbook:
                    current_imbalance = orderbook.imbalance_ratio
                    entry_imbalance = self.entry_imbalance.get(position.symbol, 0.5)

                    if position.side == SignalType.LONG:
                        # For LONG: OB should be improving (higher imbalance)
                        if current_imbalance <= entry_imbalance:
                            logger.info(
                                f"{position.symbol} FLOW NEUTRAL: delta_pct={delta_pct:.1f}% | "
                                f"OB not improved {entry_imbalance:.1%} -> {current_imbalance:.1%}"
                            )
                            return ExitReason.FLOW_NEUTRAL
                    else:
                        # For SHORT: OB should be improving (lower imbalance)
                        if current_imbalance >= entry_imbalance:
                            logger.info(
                                f"{position.symbol} FLOW NEUTRAL: delta_pct={delta_pct:.1f}% | "
                                f"OB not improved {entry_imbalance:.1%} -> {current_imbalance:.1%}"
                            )
                            return ExitReason.FLOW_NEUTRAL

        # 8. Time Stop (fallback): Still keep as ultimate backstop
        if hold_time >= config.time_stop_seconds:
            if pnl_pct < config.min_profit_for_time_check:
                logger.info(
                    f"{position.symbol} hit TIME STOP: {hold_time:.0f}s, "
                    f"only {pnl_pct:.2f}% profit"
                )
                return ExitReason.TIME_STOP

        return None

    async def execute_exit(self, symbol: str, reason: ExitReason) -> Optional[ExitResult]:
        """
        Execute exit for a position.
        """
        position = self.risk.get_position(symbol)
        if not position:
            logger.warning(f"No position found for {symbol}")
            return None

        current_price = self.ws.get_current_price(symbol)
        if current_price is None:
            logger.error(f"Cannot get current price for {symbol}")
            return None

        if self.mode == TradingMode.PAPER:
            return await self._paper_exit(position, current_price, reason)
        else:
            return await self._paper_exit(position, current_price, reason)

    async def _paper_exit(
        self,
        position: Position,
        exit_price: float,
        reason: ExitReason
    ) -> ExitResult:
        """Execute paper trade exit."""
        # Simulate small slippage
        slippage = exit_price * 0.0002
        if position.side == SignalType.LONG:
            fill_price = exit_price - slippage
        else:
            fill_price = exit_price + slippage

        # Calculate final P&L
        if position.side == SignalType.LONG:
            pnl = (fill_price - position.entry_price) * position.quantity
            pnl_pct = (fill_price - position.entry_price) / position.entry_price * 100
        else:
            pnl = (position.entry_price - fill_price) * position.quantity
            pnl_pct = (position.entry_price - fill_price) / position.entry_price * 100

        hold_time = (time.time() * 1000 - position.entry_time) / 1000

        # Get MFE/MAE before removing
        mfe = self.position_mfe.pop(position.symbol, 0)
        mae = self.position_mae.pop(position.symbol, 0)

        # Clean up entry tracking data
        self.entry_spread.pop(position.symbol, None)
        self.entry_imbalance.pop(position.symbol, None)
        self.entry_delta.pop(position.symbol, None)
        self.entry_price_check.pop(position.symbol, None)
        self.entry_check_time.pop(position.symbol, None)

        # Close in risk controller
        self.risk.close_position(position.symbol, fill_price, reason.value)

        result = ExitResult(
            success=True,
            symbol=position.symbol,
            entry_price=position.entry_price,
            exit_price=fill_price,
            quantity=position.quantity,
            pnl=pnl,
            pnl_pct=pnl_pct,
            hold_time_sec=hold_time,
            reason=reason,
            timestamp=int(time.time() * 1000),
            entry_time=position.entry_time
        )

        emoji = "WIN" if pnl > 0 else "LOSS"

        logger.info(
            f"PAPER EXIT [{emoji}]: {position.symbol} | "
            f"Entry: ${position.entry_price:.4f} -> Exit: ${fill_price:.4f} | "
            f"PnL: ${pnl:.2f} ({pnl_pct:+.2f}%) | "
            f"Hold: {hold_time:.0f}s | Reason: {reason.value} | "
            f"MFE: {mfe:.2f}% MAE: {mae:.2f}%"
        )

        if self.on_exit:
            self.on_exit(result)

        return result

    async def start_monitoring(self):
        """Start position monitoring loop."""
        self.monitoring = True
        logger.info("Position monitoring started")

        while self.monitoring:
            try:
                await self.check_exits()
                await asyncio.sleep(self.monitor_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(1)

    def stop_monitoring(self):
        """Stop position monitoring."""
        self.monitoring = False
        logger.info("Position monitoring stopped")

    async def emergency_close_all(self, reason: str = "Emergency"):
        """Close all positions immediately."""
        positions = self.risk.get_all_positions()

        for position in positions:
            await self.execute_exit(position.symbol, ExitReason.EMERGENCY)

        logger.critical(f"Emergency close all: {len(positions)} positions closed - {reason}")

    def get_open_positions_status(self) -> Dict:
        """Get status of all open positions."""
        positions = self.risk.get_all_positions()
        status = {}

        for position in positions:
            current_price = self.ws.get_current_price(position.symbol)
            if current_price:
                if position.side == SignalType.LONG:
                    pnl_pct = (current_price - position.entry_price) / position.entry_price * 100
                else:
                    pnl_pct = (position.entry_price - current_price) / position.entry_price * 100

                hold_time = (time.time() * 1000 - position.entry_time) / 1000
                config = get_asset_config(position.symbol)

                status[position.symbol] = {
                    "side": position.side.value,
                    "entry_price": position.entry_price,
                    "current_price": current_price,
                    "quantity": position.quantity,
                    "pnl_pct": pnl_pct,
                    "hold_time_sec": hold_time,
                    "time_stop_at": config.time_stop_seconds,
                    "tp_at": config.take_profit_pct,
                    "sl_at": -config.stop_loss_pct,
                    "mfe": self.position_mfe.get(position.symbol, 0),
                    "mae": self.position_mae.get(position.symbol, 0)
                }

        return status
