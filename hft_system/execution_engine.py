"""
Execution Engine for HFT System
===============================
Handles trade execution and exit management.

# ============================================================
# STRATEGY LOCKED — DO NOT MODIFY
# Validation phase: edge preservation
# Last update: 2025-12-31
# ============================================================

EXIT PRIORITY (data-validated):
1. Take Profit: +0.20% (rare but dominant)
2. Micro Profit: +0.05% with OB weakening (primary income - 100% win rate)
3. OB Flip: Orderbook flipped against position (edge decay)
4. Stop Loss: -0.12% (risk cap)
5. Time Stop: 60s fallback (last resort)

DISABLED EXITS (data-proven negative expectancy):
- DELTA_NEGATIVE: -4.70% weekly loss (115 trades) - DISABLED
- NO_MOVEMENT: 0% win rate in 54 trades - DISABLED
- FLOW_NEUTRAL: Low win rate - DISABLED
- CONFIRMATION_TIMEOUT: Covered by time_stop - DISABLED

Also monitors open positions every tick for exit opportunities.
"""

import asyncio
import logging
import time
from typing import Dict, Optional, Callable
from dataclasses import dataclass
from enum import Enum

from .config import (
    SYSTEM_CONFIG, TradingMode, get_asset_config, COST_MODEL,
    BASE_MIN_PROFIT_PCT, COST_MULTIPLIER, COST_BUFFER_PCT,
)
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
    mfe: float = 0.0  # Max favorable excursion (%)
    mae: float = 0.0  # Max adverse excursion (%)


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

        # TASK 9: Edge validation tracking
        self.time_of_mfe: Dict[str, float] = {}       # Time when MFE was reached
        self.time_of_ob_decay: Dict[str, float] = {}  # Time when OB imbalance decayed
        self.time_of_delta_flip: Dict[str, float] = {}  # Time when delta turned against us
        self.ob_decay_amount: Dict[str, float] = {}   # How much OB decayed
        self.entry_time: Dict[str, float] = {}        # Entry time for edge duration calc
        self.exit_edge_data: Dict[str, Dict] = {}     # Captured edge data at exit for callback

        # 2-PHASE ENTRY: Probe then Confirm
        self.is_probe_position: Dict[str, bool] = {}  # symbol -> True if in probe phase
        self.probe_confirmed: Dict[str, bool] = {}    # symbol -> True if probe confirmed
        self.probe_start_time: Dict[str, float] = {}  # When probe started
        self.probe_check_interval: float = 10.0       # Seconds to wait before confirming

        # SOFTENED DELTA EXIT: Track sustained negative delta
        self.delta_negative_start: Dict[str, float] = {}  # When delta first turned negative
        self.delta_negative_duration: float = 3.0  # Require 3 seconds of negative delta

        # Adaptive minimum take-profit
        self.min_profit_pct = self._compute_min_profit_pct()
        logger.info(f"Adaptive min profit: {self.min_profit_pct:.4f}% "
                    f"(base={BASE_MIN_PROFIT_PCT}, K={COST_MULTIPLIER}, "
                    f"buffer={COST_BUFFER_PCT})")
        logger.info(f"ExecutionEngine initialized in {self.mode.value} mode")

    @staticmethod
    def _compute_min_profit_pct() -> float:
        """Compute adaptive minimum take-profit threshold.

        min_profit_pct = max(BASE_MIN_PROFIT_PCT, K * est_cost_pct + buffer)
        """
        if COST_MODEL.enabled:
            est_cost_pct = (
                COST_MODEL.entry_fee_pct + COST_MODEL.exit_fee_pct
                + 2 * COST_MODEL.spread_cost_pct + COST_MODEL.base_slippage_pct
            )
        else:
            est_cost_pct = 0.0
        cost_based = COST_MULTIPLIER * est_cost_pct + COST_BUFFER_PCT
        return max(BASE_MIN_PROFIT_PCT, cost_based)

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
        """Execute paper trade entry with 2-phase probe logic."""
        # Simulate small slippage (0.01-0.03%)
        slippage = signal.entry_price * 0.0002  # 0.02% average
        if signal.signal_type == SignalType.LONG:
            fill_price = signal.entry_price + slippage
        else:
            fill_price = signal.entry_price - slippage

        # 2-PHASE ENTRY: Start with 0.25x probe size
        # Will scale up to full size if confirmed after 10s
        probe_multiplier = 0.25
        probe_size = risk_decision.position_size * probe_multiplier

        # Create position in risk controller with probe size
        signal.entry_price = fill_price
        position = self.risk.open_position(signal, probe_size)

        # Track probe state
        self.is_probe_position[signal.symbol] = True
        self.probe_confirmed[signal.symbol] = False
        self.probe_start_time[signal.symbol] = time.time()

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

        # TASK 9: Initialize edge tracking
        self.entry_time[signal.symbol] = time.time()
        self.time_of_mfe[signal.symbol] = 0  # Will be set when MFE is updated
        self.time_of_ob_decay[signal.symbol] = 0  # Will be set when OB decays
        self.time_of_delta_flip[signal.symbol] = 0  # Will be set when delta flips
        self.ob_decay_amount[signal.symbol] = 0

        result = TradeResult(
            success=True,
            symbol=signal.symbol,
            side=signal.signal_type.value,
            entry_price=fill_price,
            quantity=probe_size,  # Use probe size initially
            timestamp=int(time.time() * 1000),
            order_id=position.position_id
        )

        probe_value = probe_size * fill_price
        logger.info(
            f"PAPER ENTRY [PROBE 0.25x]: {signal.symbol} {signal.signal_type.value} | "
            f"Price: ${fill_price:.4f} | Qty: {probe_size:.4f} | "
            f"Value: ${probe_value:.2f} (full: ${risk_decision.position_value:.2f})"
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

        # ============================================================
        # STRATEGY LOCKED - EXIT PRIORITY (data-validated)
        # ============================================================
        1. Take Profit: +0.20%
        2. Stop Loss: -0.12%
        3. MICRO PROFIT: +0.05% with OB weakening (100% win rate - PROMOTED)
        4. OB Flip: Orderbook flipped against position
        5. Delta Negative: SOFTENED - requires 3s sustained OR OB flip combo
        6. Spread Widen: 2x entry spread
        7. Time Stop: 60s fallback

        DISABLED (0% or low win rate):
        - NO_MOVEMENT: COMPLETELY DISABLED
        - FLOW_NEUTRAL: DISABLED
        - CONFIRMATION_TIMEOUT: DISABLED
        """
        config = get_asset_config(position.symbol)
        current_price = self.ws.get_current_price(position.symbol)

        if current_price is None:
            return None

        # 2-PHASE ENTRY: Check probe confirmation at 10s
        # NOTE: Probe abort now uses TIME_STOP instead of NO_MOVEMENT
        if self.is_probe_position.get(position.symbol, False):
            probe_start = self.probe_start_time.get(position.symbol, time.time())
            probe_age = time.time() - probe_start

            if probe_age >= self.probe_check_interval and not self.probe_confirmed.get(position.symbol, False):
                # Time to confirm or abort the probe
                orderbook = self.ws.get_orderbook(position.symbol)
                trade_flow = self.ws.get_trade_flow(position.symbol)

                confirmations = 0

                # Check 1: CVD continues in entry direction
                if trade_flow:
                    current_delta = trade_flow.get("delta_1s", 0)
                    if position.side == SignalType.LONG and current_delta > 0:
                        confirmations += 1
                    elif position.side == SignalType.SHORT and current_delta < 0:
                        confirmations += 1

                # Check 2: OB imbalance holds or strengthens
                if orderbook:
                    current_imbalance = orderbook.imbalance_ratio
                    entry_imbalance = self.entry_imbalance.get(position.symbol, 0.5)
                    if position.side == SignalType.LONG:
                        if current_imbalance >= entry_imbalance - 0.1:  # Within 10%
                            confirmations += 1
                    else:
                        if current_imbalance <= entry_imbalance + 0.1:
                            confirmations += 1

                # Check 3: Spread hasn't widened significantly
                if orderbook:
                    current_spread = orderbook.spread
                    entry_spread = self.entry_spread.get(position.symbol, 0)
                    if entry_spread > 0 and current_spread < entry_spread * 1.5:
                        confirmations += 1

                # Need 2/3 confirmations to continue
                if confirmations >= 2:
                    self.probe_confirmed[position.symbol] = True
                    logger.info(
                        f"{position.symbol} PROBE CONFIRMED: {confirmations}/3 checks passed | "
                        f"Continuing position"
                    )
                # If probe fails, let it continue to time_stop (don't force exit here)

        # Calculate current P&L (guard against division by zero)
        if not position.entry_price or position.entry_price <= 0:
            return None  # Can't calculate P&L without valid entry price

        if position.side == SignalType.LONG:
            pnl_pct = (current_price - position.entry_price) / position.entry_price * 100
        else:
            pnl_pct = (position.entry_price - current_price) / position.entry_price * 100

        # Update MFE/MAE
        if position.symbol in self.position_mfe:
            if pnl_pct > self.position_mfe[position.symbol]:
                self.position_mfe[position.symbol] = pnl_pct
                # TASK 9: Record time of MFE (max favorable excursion)
                if position.symbol in self.entry_time:
                    self.time_of_mfe[position.symbol] = time.time() - self.entry_time[position.symbol]
            if pnl_pct < self.position_mae[position.symbol]:
                self.position_mae[position.symbol] = pnl_pct

        hold_time = (time.time() * 1000 - position.entry_time) / 1000

        # TASK 9: Track OB decay and delta flip for edge validation
        orderbook = self.ws.get_orderbook(position.symbol)
        trade_flow = self.ws.get_trade_flow(position.symbol)

        if orderbook and position.symbol in self.entry_imbalance:
            current_imbalance = orderbook.imbalance_ratio
            entry_imbalance = self.entry_imbalance[position.symbol]

            # Track OB decay (first time imbalance drops significantly)
            if self.time_of_ob_decay.get(position.symbol, 0) == 0:
                if position.side == SignalType.LONG:
                    decay = entry_imbalance - current_imbalance
                    if decay > 0.1:  # Lost 10%+ of edge
                        self.time_of_ob_decay[position.symbol] = time.time() - self.entry_time.get(position.symbol, time.time())
                        self.ob_decay_amount[position.symbol] = decay
                else:
                    decay = current_imbalance - entry_imbalance
                    if decay > 0.1:
                        self.time_of_ob_decay[position.symbol] = time.time() - self.entry_time.get(position.symbol, time.time())
                        self.ob_decay_amount[position.symbol] = decay

        if trade_flow and position.symbol in self.entry_delta:
            current_delta = trade_flow.get("delta_1s", 0)
            entry_delta = self.entry_delta[position.symbol]

            # Track delta flip (first time delta turns against us)
            if self.time_of_delta_flip.get(position.symbol, 0) == 0:
                if position.side == SignalType.LONG:
                    if entry_delta > 0 and current_delta < 0:
                        self.time_of_delta_flip[position.symbol] = time.time() - self.entry_time.get(position.symbol, time.time())
                else:
                    if entry_delta < 0 and current_delta > 0:
                        self.time_of_delta_flip[position.symbol] = time.time() - self.entry_time.get(position.symbol, time.time())

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

        # 3. MICRO PROFIT: Exit with small profit above adaptive min threshold
        # min_profit_pct = max(BASE_MIN_PROFIT_PCT, K * est_cost + buffer)
        # This ensures we never take a micro-profit below breakeven after costs.
        if self.min_profit_pct <= pnl_pct < config.take_profit_pct:
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

        # 5. Delta Negative: SOFTENED - requires SUSTAINED negative delta (3s) OR OB flip combo
        # Data showed 6.3% win rate when triggering too early - now require confirmation
        if trade_flow:
            current_delta = trade_flow.get("delta_1s", 0)
            # ============================================================
            # DELTA_NEGATIVE: DISABLED (2025-12-31)
            # Reason: -4.70% weekly loss on 115 trades
            # Data proves negative expectancy - do not re-enable
            # ============================================================
            # entry_delta = self.entry_delta.get(position.symbol, 0)
            # delta_against logic removed - was causing majority of losses
            pass  # Placeholder to maintain code structure

        # 5. Spread Widen: Spread widened >2x from entry
        if orderbook:
            current_spread = orderbook.spread
            entry_spread = self.entry_spread.get(position.symbol, 0)

            if entry_spread > 0 and current_spread > entry_spread * 2:
                logger.info(
                    f"{position.symbol} SPREAD WIDEN: {entry_spread:.4f}% -> {current_spread:.4f}%"
                )
                return ExitReason.SPREAD_WIDEN

        # ============================================================
        # DISABLED EXITS (data-proven negative expectancy - DO NOT RE-ENABLE)
        # ============================================================
        # DELTA_NEGATIVE: -4.70% weekly, 115 trades - DISABLED 2025-12-31
        # NO_MOVEMENT: 0% win rate in 54 trades - DISABLED
        # FLOW_NEUTRAL: Low win rate - DISABLED
        # CONFIRMATION_TIMEOUT: 0% win rate - DISABLED
        # ============================================================

        # 7. Time Stop (fallback): Keep as ultimate backstop
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

        # Calculate final P&L (guard against division by zero)
        if position.entry_price and position.entry_price > 0:
            if position.side == SignalType.LONG:
                pnl = (fill_price - position.entry_price) * position.quantity
                pnl_pct = (fill_price - position.entry_price) / position.entry_price * 100
            else:
                pnl = (position.entry_price - fill_price) * position.quantity
                pnl_pct = (position.entry_price - fill_price) / position.entry_price * 100
        else:
            pnl = 0
            pnl_pct = 0

        hold_time = (time.time() * 1000 - position.entry_time) / 1000

        # Get MFE/MAE before removing
        mfe = self.position_mfe.pop(position.symbol, 0)
        mae = self.position_mae.pop(position.symbol, 0)

        # TASK 9: Capture edge metrics before cleanup
        trade_id = f"{position.symbol}_{position.entry_time}"
        entry_time_val = self.entry_time.get(position.symbol, time.time())
        edge_duration = time.time() - entry_time_val

        # Compute delta persistence: time delta stayed favorable (before flip or exit)
        delta_flip_time = self.time_of_delta_flip.get(position.symbol, 0)
        if delta_flip_time > 0:
            delta_persistence = delta_flip_time
        else:
            delta_persistence = hold_time  # Delta never flipped, persisted entire trade

        self.exit_edge_data[trade_id] = {
            "seconds_to_max_favorable": self.time_of_mfe.get(position.symbol, 0),
            "max_favorable_pct": mfe,
            "seconds_to_ob_decay": self.time_of_ob_decay.get(position.symbol, 0),
            "ob_decay_amount": self.ob_decay_amount.get(position.symbol, 0),
            "delta_persistence_sec": delta_persistence,
            "entry_imbalance": self.entry_imbalance.get(position.symbol, 0),
            "entry_delta": self.entry_delta.get(position.symbol, 0),
            "entry_spread": self.entry_spread.get(position.symbol, 0),
            "edge_duration_sec": edge_duration,
            "side": position.side.value
        }

        # Clean up entry tracking data
        self.entry_spread.pop(position.symbol, None)
        self.entry_imbalance.pop(position.symbol, None)
        self.entry_delta.pop(position.symbol, None)
        self.entry_price_check.pop(position.symbol, None)
        self.entry_check_time.pop(position.symbol, None)

        # Clean up edge tracking data
        self.time_of_mfe.pop(position.symbol, None)
        self.time_of_ob_decay.pop(position.symbol, None)
        self.time_of_delta_flip.pop(position.symbol, None)
        self.ob_decay_amount.pop(position.symbol, None)
        self.entry_time.pop(position.symbol, None)

        # Clean up probe tracking data
        self.is_probe_position.pop(position.symbol, None)
        self.probe_confirmed.pop(position.symbol, None)
        self.probe_start_time.pop(position.symbol, None)

        # Clean up softened delta tracking
        self.delta_negative_start.pop(position.symbol, None)

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
            entry_time=position.entry_time,
            mfe=mfe,  # Pass MFE to result (before it was popped)
            mae=mae   # Pass MAE to result (before it was popped)
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

    def get_edge_data(self, trade_id: str) -> Optional[Dict]:
        """Get captured edge data for a trade and remove from cache."""
        return self.exit_edge_data.pop(trade_id, None)

    def get_open_positions_status(self) -> Dict:
        """Get status of all open positions."""
        positions = self.risk.get_all_positions()
        status = {}

        for position in positions:
            current_price = self.ws.get_current_price(position.symbol)
            if current_price and position.entry_price and position.entry_price > 0:
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
