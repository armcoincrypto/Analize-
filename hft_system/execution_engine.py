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
import uuid
import random
from typing import Dict, Optional, Callable
from dataclasses import dataclass
from enum import Enum

from .config import SYSTEM_CONFIG, TradingMode, get_asset_config, COST_MODEL, MAKER_SCALPER
from .signal_engine import Signal, SignalType
from .risk_controller import RiskController, Position, RiskDecision
from .websocket_manager import WebSocketManager

logger = logging.getLogger(__name__)


class TakerViolationError(Exception):
    """Raised when a taker fill occurs in maker-only mode."""
    pass


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
    execution_mode: str = "taker"  # "maker" or "taker"


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
        mode: TradingMode = None,
        trade_logger = None
    ):
        self.ws = ws_manager
        self.risk = risk_controller
        self.mode = mode or SYSTEM_CONFIG.mode
        self.trade_logger = trade_logger  # For maker order telemetry

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
        self.time_of_mae: Dict[str, float] = {}       # Time when MAE was reached
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

        # ============================================================
        # MAKER-ONLY MODE TRACKING
        # ============================================================
        self.taker_violation_active: bool = False  # True if entries are paused
        self.taker_violation_time: float = 0  # When violation occurred
        self.taker_violation_count: int = 0  # Total violations this session

        # ============================================================
        # ENHANCED MFE/MAE TRACKING
        # ============================================================
        # Track actual max/min prices seen for accurate MFE/MAE
        self.position_max_price: Dict[str, float] = {}  # Max price seen during position
        self.position_min_price: Dict[str, float] = {}  # Min price seen during position
        self.position_last_update: Dict[str, float] = {}  # Last update timestamp

        # ============================================================
        # MAKER SCALPER STRATEGY TRACKING
        # ============================================================
        self.pending_entry_orders: Dict[str, dict] = {}  # symbol -> order info
        self.pending_tp_orders: Dict[str, dict] = {}  # symbol -> TP order info
        self.entry_order_placed_time: Dict[str, float] = {}  # symbol -> timestamp
        self.tp_order_placed_time: Dict[str, float] = {}  # symbol -> timestamp
        self.exit_attempt_start: Dict[str, float] = {}  # symbol -> timestamp for maker exit attempts

        logger.info(f"ExecutionEngine initialized in {self.mode.value} mode")
        if COST_MODEL.maker_only:
            logger.info("MAKER_ONLY mode: ENABLED - all entries must be post-only maker orders")
        if COST_MODEL.cost_gating_enabled:
            logger.info(f"COST_GATING: ENABLED - max expected costs: {COST_MODEL.max_expected_total_costs_pct*100:.4f}%")

    def check_taker_violation_cooldown(self) -> tuple:
        """Check if entries are paused due to taker violation.

        Returns:
            (is_paused, remaining_sec, message)
        """
        if not self.taker_violation_active:
            return False, 0, None

        elapsed = time.time() - self.taker_violation_time
        remaining = COST_MODEL.taker_violation_cooldown_sec - elapsed

        if remaining <= 0:
            # Cooldown expired
            self.taker_violation_active = False
            logger.info("TAKER_VIOLATION: Cooldown expired, entries resumed")
            return False, 0, None

        return True, remaining, f"Taker violation cooldown: {remaining:.0f}s remaining"

    def estimate_trade_costs(self, symbol: str, position_value: float) -> dict:
        """Estimate pre-trade costs for cost gating.

        Args:
            symbol: Trading symbol
            position_value: Size of position in USD

        Returns:
            Cost estimation dict from COST_MODEL.estimate_pre_trade_costs()
        """
        # Get current orderbook data
        orderbook = self.ws.get_orderbook(symbol)

        if orderbook:
            current_spread_pct = orderbook.spread
            # Estimate top-of-book depth in USD
            top_bid_depth = sum(qty * price for price, qty in orderbook.bids[:3]) if orderbook.bids else 0
            top_ask_depth = sum(qty * price for price, qty in orderbook.asks[:3]) if orderbook.asks else 0
            top_of_book_depth = (top_bid_depth + top_ask_depth) / 2
        else:
            current_spread_pct = 0.05  # Conservative fallback
            top_of_book_depth = 0

        return COST_MODEL.estimate_pre_trade_costs(
            current_spread_pct=current_spread_pct,
            order_size_usd=position_value,
            top_of_book_depth_usd=top_of_book_depth
        )

    async def execute_entry(
        self,
        signal: Signal,
        risk_decision: RiskDecision,
        is_cause_probe: bool = False
    ) -> TradeResult:
        """
        Execute entry for a signal.

        Args:
            signal: The trading signal
            risk_decision: Position sizing decision
            is_cause_probe: If True, this is already a PROBE cause trade
                           (skip execution probe to avoid double-reduction)

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

        # ============================================================
        # MAKER-ONLY CHECK: Block entries during taker violation cooldown
        # ============================================================
        if COST_MODEL.maker_only and COST_MODEL.pause_entries_on_taker_violation:
            is_paused, remaining, msg = self.check_taker_violation_cooldown()
            if is_paused:
                logger.warning(f"ENTRY_BLOCKED: {signal.symbol} - {msg}")
                return TradeResult(
                    success=False,
                    symbol=signal.symbol,
                    side=signal.signal_type.value,
                    entry_price=signal.entry_price,
                    quantity=0,
                    timestamp=int(time.time() * 1000),
                    error=msg
                )

        # ============================================================
        # COST GATING: Estimate costs and block if too high
        # ============================================================
        if COST_MODEL.cost_gating_enabled:
            position_value = risk_decision.position_size * signal.entry_price
            cost_estimate = self.estimate_trade_costs(signal.symbol, position_value)

            if not cost_estimate["should_trade"]:
                logger.warning(
                    f"COST_GATE_BLOCK: {signal.symbol} | "
                    f"expected_costs={cost_estimate['expected_total_costs_pct']*100:.4f}% > "
                    f"max={cost_estimate['max_allowed_costs_pct']*100:.4f}% | "
                    f"fee={cost_estimate['expected_fee_sum_pct']*100:.4f}%, "
                    f"spread={cost_estimate['estimated_spread_cost_pct']*100:.4f}%, "
                    f"slip={cost_estimate['estimated_slippage_pct']*100:.4f}%"
                )
                return TradeResult(
                    success=False,
                    symbol=signal.symbol,
                    side=signal.signal_type.value,
                    entry_price=signal.entry_price,
                    quantity=0,
                    timestamp=int(time.time() * 1000),
                    error=f"Cost gating: expected {cost_estimate['expected_total_costs_pct']*100:.4f}% > max {cost_estimate['max_allowed_costs_pct']*100:.4f}%"
                )

        if self.mode == TradingMode.PAPER:
            # Paper trading - instant fill
            return await self._paper_entry(signal, risk_decision, is_cause_probe)
        elif self.mode == TradingMode.LIVE:
            # Live trading - would submit to exchange
            return await self._live_entry(signal, risk_decision, is_cause_probe)
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

    async def _paper_entry(
        self,
        signal: Signal,
        risk_decision: RiskDecision,
        is_cause_probe: bool = False
    ) -> TradeResult:
        """Execute paper trade entry with conditional 2-phase probe logic.

        Args:
            is_cause_probe: If True, this is already a PROBE cause trade.
                           Skip the execution probe (0.25x) to avoid double-reduction.
                           PROBE cause already applied 0.10x in hft_bot.py.
        """
        # Determine execution mode (taker vs maker)
        orderbook = self.ws.get_orderbook(signal.symbol)
        current_spread_pct = orderbook.spread if orderbook else 0.05
        execution_mode = COST_MODEL.get_effective_mode(current_spread_pct)

        # FORCE_MAKER_IN_PAPER: Always use maker path in paper mode for testing
        if COST_MODEL.execution_mode == "maker" or (
            hasattr(COST_MODEL, 'force_maker_in_paper') and COST_MODEL.force_maker_in_paper
        ):
            logger.info(
                f"MAKER_FORCED_PAPER: {signal.symbol} {signal.signal_type.value} | "
                f"spread_bps={current_spread_pct*100:.1f}"
            )
            return await self._paper_maker_entry(signal, risk_decision, is_cause_probe, orderbook)

        # Use maker execution if auto mode decides maker
        if execution_mode == "maker":
            return await self._paper_maker_entry(signal, risk_decision, is_cause_probe, orderbook)

        # === TAKER EXECUTION (default) ===
        # Simulate small slippage (0.01-0.03%)
        slippage = signal.entry_price * 0.0002  # 0.02% average
        if signal.signal_type == SignalType.LONG:
            fill_price = signal.entry_price + slippage
        else:
            fill_price = signal.entry_price - slippage

        # ============================================================
        # SINGLE-PROBE LOGIC: Only one probe multiplier should apply
        # ============================================================
        # If is_cause_probe=True: PROBE cause already applied 0.10x in hft_bot.py
        #   -> Use risk_decision.position_size directly (no additional reduction)
        # If is_cause_probe=False: Apply 2-phase execution probe (0.25x)
        #   -> Will scale up to full size if confirmed after 10s
        # ============================================================
        if is_cause_probe:
            # PROBE cause: size already reduced to 0.10x - use as-is
            final_size = risk_decision.position_size
            probe_label = "CAUSE_PROBE"
            exec_multiplier = 1.0  # No additional reduction
        else:
            # Normal trade: apply 2-phase execution probe
            exec_multiplier = 0.25
            final_size = risk_decision.position_size * exec_multiplier
            probe_label = "EXEC_PROBE 0.25x"

        # Safety clamp: minimum 0.10x of base position
        # (prevents accidentally tiny positions from bugs)
        min_size = risk_decision.position_size * 0.10 if not is_cause_probe else final_size
        if final_size < min_size:
            logger.warning(f"Safety clamp: {final_size:.4f} < min {min_size:.4f}, using min")
            final_size = min_size

        # Create position in risk controller
        signal.entry_price = fill_price
        position = self.risk.open_position(signal, final_size)

        # Track probe state
        self.is_probe_position[signal.symbol] = not is_cause_probe  # Only track exec probes
        self.probe_confirmed[signal.symbol] = is_cause_probe  # Cause probes are "pre-confirmed"
        self.probe_start_time[signal.symbol] = time.time()

        # Store the actual multiplier applied for logging
        self.applied_exec_multiplier = exec_multiplier if not is_cause_probe else 1.0

        # Initialize MFE/MAE tracking
        self.position_mfe[signal.symbol] = 0
        self.position_mae[signal.symbol] = 0

        # ENHANCED MFE/MAE: Track actual max/min prices seen
        self.position_max_price[signal.symbol] = fill_price
        self.position_min_price[signal.symbol] = fill_price
        self.position_last_update[signal.symbol] = time.time()

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
        self.time_of_mae[signal.symbol] = 0  # Will be set when MAE is updated
        self.time_of_ob_decay[signal.symbol] = 0  # Will be set when OB decays
        self.time_of_delta_flip[signal.symbol] = 0  # Will be set when delta flips
        self.ob_decay_amount[signal.symbol] = 0

        result = TradeResult(
            success=True,
            symbol=signal.symbol,
            side=signal.signal_type.value,
            entry_price=fill_price,
            quantity=final_size,
            timestamp=int(time.time() * 1000),
            order_id=position.position_id,
            execution_mode="taker"  # TAKER execution path
        )

        final_value = final_size * fill_price
        logger.info(
            f"PAPER ENTRY [{probe_label}]: {signal.symbol} {signal.signal_type.value} | "
            f"Price: ${fill_price:.4f} | Qty: {final_size:.4f} | "
            f"Value: ${final_value:.2f} (full: ${risk_decision.position_value:.2f})"
        )

        if self.on_trade:
            self.on_trade(result)

        return result

    async def _paper_maker_entry(
        self,
        signal: Signal,
        risk_decision: RiskDecision,
        is_cause_probe: bool = False,
        orderbook = None
    ) -> TradeResult:
        """Execute paper trade entry with maker (limit order) simulation.

        Simulates posting a limit order, waiting for fill, and logging telemetry.

        MAKER-ONLY MODE:
        - Uses post-only order type
        - If order would immediately fill as taker, it's rejected
        - If maker-only mode and order times out, NO fallback to taker
        """
        # Get order book data
        if orderbook is None:
            orderbook = self.ws.get_orderbook(signal.symbol)

        best_bid = orderbook.best_bid if orderbook else signal.entry_price * 0.9999
        best_ask = orderbook.best_ask if orderbook else signal.entry_price * 1.0001
        mid_price = (best_bid + best_ask) / 2
        spread_pct = ((best_ask - best_bid) / mid_price * 100) if mid_price > 0 else 0.02

        # Calculate limit price with offset inside the spread
        # For LONG: place bid slightly above best bid (inside spread)
        # For SHORT: place ask slightly below best ask (inside spread)
        offset_bps = MAKER_SCALPER.entry_price_offset_bps if MAKER_SCALPER.enabled else 0.5
        if signal.signal_type == SignalType.LONG:
            limit_price = best_bid + (best_ask - best_bid) * (offset_bps / 100)
            # POST-ONLY CHECK: Ensure we don't cross the spread
            if COST_MODEL.post_only_enabled and limit_price >= best_ask:
                logger.warning(
                    f"POST_ONLY_REJECT: {signal.symbol} LONG limit ${limit_price:.6f} >= ask ${best_ask:.6f}"
                )
                limit_price = best_ask - (best_ask - best_bid) * 0.001  # Stay just inside
        else:
            limit_price = best_ask - (best_ask - best_bid) * (offset_bps / 100)
            # POST-ONLY CHECK: Ensure we don't cross the spread
            if COST_MODEL.post_only_enabled and limit_price <= best_bid:
                logger.warning(
                    f"POST_ONLY_REJECT: {signal.symbol} SHORT limit ${limit_price:.6f} <= bid ${best_bid:.6f}"
                )
                limit_price = best_bid + (best_ask - best_bid) * 0.001  # Stay just inside

        # Generate order ID
        order_id = f"MKR_{signal.symbol}_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}"

        # Calculate position size (same logic as taker)
        if is_cause_probe:
            final_size = risk_decision.position_size
            probe_label = "CAUSE_PROBE MAKER"
            exec_multiplier = 1.0
        else:
            exec_multiplier = 0.25
            final_size = risk_decision.position_size * exec_multiplier
            probe_label = "EXEC_PROBE MAKER 0.25x"

        min_size = risk_decision.position_size * 0.10 if not is_cause_probe else final_size
        if final_size < min_size:
            final_size = min_size

        # Log maker order posted
        market_data = {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid_price": mid_price,
            "spread_pct": spread_pct,
            "queue_position_estimate": 0,
            "post_only": COST_MODEL.post_only_enabled
        }
        if self.trade_logger:
            self.trade_logger.log_maker_order_posted(
                order_id=order_id,
                symbol=signal.symbol,
                side=signal.signal_type.value,
                limit_price=limit_price,
                quantity=final_size,
                market_data=market_data
            )

        # Track entry order for timeout handling
        self.entry_order_placed_time[signal.symbol] = time.time()
        self.pending_entry_orders[signal.symbol] = {
            "order_id": order_id,
            "limit_price": limit_price,
            "quantity": final_size,
            "side": signal.signal_type.value
        }

        logger.info(
            f"MAKER ORDER POSTED: {signal.symbol} {signal.signal_type.value} | "
            f"Limit: ${limit_price:.6f} | Best bid/ask: ${best_bid:.6f}/${best_ask:.6f} | "
            f"Spread: {spread_pct:.4f}% | post_only={COST_MODEL.post_only_enabled}"
        )

        # Simulate wait for fill with entry timeout
        entry_timeout = MAKER_SCALPER.entry_timeout_sec if MAKER_SCALPER.enabled else COST_MODEL.maker_wait_seconds
        fill_probability = COST_MODEL.maker_fill_probability

        # Simulate fill check - random based on fill probability
        await asyncio.sleep(min(entry_timeout, 0.5))  # Don't actually wait full time in paper

        filled = random.random() < fill_probability
        filled_as_taker = False  # Track if we accidentally filled as taker

        if filled:
            # Check if the fill was as taker (price moved through our limit)
            # In paper mode, simulate ~5% chance of accidental taker fill
            if random.random() < 0.05:
                filled_as_taker = True
                logger.warning(f"TAKER_VIOLATION_DETECTED: {signal.symbol} order filled as taker")

            if filled_as_taker and COST_MODEL.maker_only and COST_MODEL.reject_taker_fills:
                # Reject the taker fill - mark as violation
                self.taker_violation_active = True
                self.taker_violation_time = time.time()
                self.taker_violation_count += 1

                if self.trade_logger:
                    self.trade_logger.log_maker_order_cancelled(
                        order_id=order_id,
                        cancel_reason="taker_fill_rejected",
                        market_data=market_data,
                        symbol=signal.symbol,
                        side=signal.signal_type.value,
                        limit_price=limit_price
                    )

                logger.error(
                    f"TAKER_VIOLATION: {signal.symbol} | Order filled as taker, rejected | "
                    f"Entries paused for {COST_MODEL.taker_violation_cooldown_sec}s | "
                    f"Total violations: {self.taker_violation_count}"
                )

                return TradeResult(
                    success=False,
                    symbol=signal.symbol,
                    side=signal.signal_type.value,
                    entry_price=signal.entry_price,
                    quantity=0,
                    timestamp=int(time.time() * 1000),
                    error="Taker fill rejected in maker-only mode"
                )

            # Order filled - use limit price (no slippage for maker)
            fill_price = limit_price
            actual_execution_mode = "taker" if filled_as_taker else "maker"

            # Log maker order filled
            if self.trade_logger:
                self.trade_logger.log_maker_order_filled(
                    order_id=order_id,
                    filled_quantity=final_size,
                    avg_fill_price=fill_price,
                    market_data=market_data,
                    price_crossed_limit=filled_as_taker,
                    cross_depth_bps=0,
                    symbol=signal.symbol,
                    side=signal.signal_type.value,
                    limit_price=limit_price
                )

            logger.info(
                f"MAKER ORDER FILLED: {order_id} @ ${fill_price:.6f} | "
                f"Qty: {final_size:.4f} | mode={actual_execution_mode}"
            )
        else:
            # Order not filled within timeout
            if self.trade_logger:
                self.trade_logger.log_maker_order_cancelled(
                    order_id=order_id,
                    cancel_reason="entry_timeout_no_fill",
                    market_data=market_data,
                    symbol=signal.symbol,
                    side=signal.signal_type.value,
                    limit_price=limit_price
                )

            # MAKER-ONLY MODE: Do NOT fall back to taker
            if COST_MODEL.maker_only:
                logger.info(
                    f"MAKER ORDER CANCELLED (no fill): {order_id} | "
                    f"MAKER_ONLY mode - no taker fallback"
                )
                # Clean up pending order
                self.pending_entry_orders.pop(signal.symbol, None)
                self.entry_order_placed_time.pop(signal.symbol, None)

                return TradeResult(
                    success=False,
                    symbol=signal.symbol,
                    side=signal.signal_type.value,
                    entry_price=signal.entry_price,
                    quantity=0,
                    timestamp=int(time.time() * 1000),
                    error="Entry timeout - no maker fill"
                )

            # Legacy behavior: Fall back to taker execution (only if not maker-only)
            logger.info(
                f"MAKER ORDER CANCELLED (no fill): {order_id} | Falling back to taker"
            )
            slippage = signal.entry_price * 0.0002
            if signal.signal_type == SignalType.LONG:
                fill_price = signal.entry_price + slippage
            else:
                fill_price = signal.entry_price - slippage
            probe_label = probe_label.replace("MAKER", "MAKER->TAKER")
            filled = True
            actual_execution_mode = "taker"

        # Create position in risk controller
        signal.entry_price = fill_price
        position = self.risk.open_position(signal, final_size)

        # Track probe state
        self.is_probe_position[signal.symbol] = not is_cause_probe
        self.probe_confirmed[signal.symbol] = is_cause_probe
        self.probe_start_time[signal.symbol] = time.time()
        self.applied_exec_multiplier = exec_multiplier if not is_cause_probe else 1.0

        # Initialize MFE/MAE tracking
        self.position_mfe[signal.symbol] = 0
        self.position_mae[signal.symbol] = 0

        # ENHANCED MFE/MAE: Track actual max/min prices seen
        self.position_max_price[signal.symbol] = fill_price
        self.position_min_price[signal.symbol] = fill_price
        self.position_last_update[signal.symbol] = time.time()

        # Capture entry conditions
        trade_flow = self.ws.get_trade_flow(signal.symbol)
        self.entry_spread[signal.symbol] = orderbook.spread if orderbook else 0
        self.entry_imbalance[signal.symbol] = orderbook.imbalance_ratio if orderbook else 0.5
        self.entry_delta[signal.symbol] = trade_flow.get("net_delta", 0) if trade_flow else 0
        self.entry_price_check[signal.symbol] = fill_price
        self.entry_check_time[signal.symbol] = 0

        # Initialize edge tracking
        self.entry_time[signal.symbol] = time.time()
        self.time_of_mfe[signal.symbol] = 0
        self.time_of_mae[signal.symbol] = 0  # Will be set when MAE is updated
        self.time_of_ob_decay[signal.symbol] = 0
        self.time_of_delta_flip[signal.symbol] = 0
        self.ob_decay_amount[signal.symbol] = 0

        # Clean up pending entry order tracking
        self.pending_entry_orders.pop(signal.symbol, None)
        self.entry_order_placed_time.pop(signal.symbol, None)

        # actual_execution_mode is already set above

        result = TradeResult(
            success=True,
            symbol=signal.symbol,
            side=signal.signal_type.value,
            entry_price=fill_price,
            quantity=final_size,
            timestamp=int(time.time() * 1000),
            order_id=position.position_id,
            execution_mode=actual_execution_mode
        )

        final_value = final_size * fill_price
        logger.info(
            f"PAPER ENTRY [{probe_label}]: {signal.symbol} {signal.signal_type.value} | "
            f"Price: ${fill_price:.4f} | Qty: {final_size:.4f} | "
            f"Value: ${final_value:.2f} (full: ${risk_decision.position_value:.2f})"
        )

        if self.on_trade:
            self.on_trade(result)

        return result

    async def _live_entry(
        self,
        signal: Signal,
        risk_decision: RiskDecision,
        is_cause_probe: bool = False
    ) -> TradeResult:
        """Execute live trade entry (placeholder)."""
        # TODO: Implement actual exchange order submission
        logger.warning("LIVE trading not implemented - using paper mode")
        return await self._paper_entry(signal, risk_decision, is_cause_probe)

    def update_mfe_mae_from_price(self, symbol: str, current_price: float):
        """
        Update MFE/MAE tracking from a price tick.

        This method should be called on every price update from websocket
        for accurate MFE/MAE tracking. This fixes the zero MFE/MAE issue.

        Args:
            symbol: Trading symbol
            current_price: Current price from websocket tick
        """
        position = self.risk.get_position(symbol)
        if not position:
            return

        # Update max/min price tracking
        if symbol in self.position_max_price:
            if current_price > self.position_max_price[symbol]:
                self.position_max_price[symbol] = current_price
            if current_price < self.position_min_price[symbol]:
                self.position_min_price[symbol] = current_price
            self.position_last_update[symbol] = time.time()

            # Recalculate MFE/MAE from price extremes
            max_seen = self.position_max_price[symbol]
            min_seen = self.position_min_price[symbol]

            if position.side == SignalType.LONG:
                # LONG: MFE = (max_seen - entry) / entry
                price_based_mfe = (max_seen - position.entry_price) / position.entry_price * 100
                price_based_mae = (min_seen - position.entry_price) / position.entry_price * 100
            else:
                # SHORT: MFE = (entry - min_seen) / entry
                price_based_mfe = (position.entry_price - min_seen) / position.entry_price * 100
                price_based_mae = (position.entry_price - max_seen) / position.entry_price * 100

            # Update MFE if more favorable
            if symbol in self.position_mfe and price_based_mfe > self.position_mfe[symbol]:
                self.position_mfe[symbol] = price_based_mfe
                if symbol in self.entry_time:
                    self.time_of_mfe[symbol] = time.time() - self.entry_time[symbol]

            # Update MAE if more adverse
            if symbol in self.position_mae and price_based_mae < self.position_mae[symbol]:
                self.position_mae[symbol] = price_based_mae
                if symbol in self.entry_time:
                    self.time_of_mae[symbol] = time.time() - self.entry_time[symbol]

    async def poll_mfe_mae_fallback(self, symbols: list, interval_sec: float = 1.0):
        """
        REST polling fallback for MFE/MAE tracking when websocket is unavailable.

        Should be run as a background task if websocket price updates aren't reliable.
        """
        while True:
            try:
                for symbol in symbols:
                    if self.risk.has_position(symbol):
                        current_price = self.ws.get_current_price(symbol)
                        if current_price:
                            self.update_mfe_mae_from_price(symbol, current_price)
                await asyncio.sleep(interval_sec)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"MFE/MAE polling error: {e}")
                await asyncio.sleep(interval_sec)

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

        # ============================================================
        # ENHANCED MFE/MAE TRACKING
        # ============================================================
        # Track both:
        # 1. Max/min prices seen (for accurate price-based MFE/MAE)
        # 2. PnL-based MFE/MAE (as percentage)
        # This fixes the zero MFE/MAE issue by properly tracking price extremes.
        # ============================================================

        # Update max/min price tracking
        if position.symbol in self.position_max_price:
            if current_price > self.position_max_price[position.symbol]:
                self.position_max_price[position.symbol] = current_price
            if current_price < self.position_min_price[position.symbol]:
                self.position_min_price[position.symbol] = current_price
            self.position_last_update[position.symbol] = time.time()

        # Calculate MFE/MAE from actual price extremes
        if position.symbol in self.position_max_price and position.symbol in self.position_min_price:
            max_seen = self.position_max_price[position.symbol]
            min_seen = self.position_min_price[position.symbol]

            if position.side == SignalType.LONG:
                # LONG: MFE = (max_seen - entry) / entry, MAE = (min_seen - entry) / entry (negative)
                price_based_mfe = (max_seen - position.entry_price) / position.entry_price * 100
                price_based_mae = (min_seen - position.entry_price) / position.entry_price * 100
            else:
                # SHORT: MFE = (entry - min_seen) / entry, MAE = (entry - max_seen) / entry (negative)
                price_based_mfe = (position.entry_price - min_seen) / position.entry_price * 100
                price_based_mae = (position.entry_price - max_seen) / position.entry_price * 100

            # Update MFE/MAE if price-based values are more extreme
            if position.symbol in self.position_mfe:
                if price_based_mfe > self.position_mfe[position.symbol]:
                    self.position_mfe[position.symbol] = price_based_mfe
                    if position.symbol in self.entry_time:
                        self.time_of_mfe[position.symbol] = time.time() - self.entry_time[position.symbol]

            if position.symbol in self.position_mae:
                if price_based_mae < self.position_mae[position.symbol]:
                    self.position_mae[position.symbol] = price_based_mae
                    if position.symbol in self.entry_time:
                        self.time_of_mae[position.symbol] = time.time() - self.entry_time[position.symbol]

        # Fallback: Also update from current PnL (defensive)
        if position.symbol in self.position_mfe and position.symbol in self.position_mae:
            if pnl_pct > self.position_mfe[position.symbol]:
                self.position_mfe[position.symbol] = pnl_pct
                if position.symbol in self.entry_time:
                    self.time_of_mfe[position.symbol] = time.time() - self.entry_time[position.symbol]
            if pnl_pct < self.position_mae[position.symbol]:
                self.position_mae[position.symbol] = pnl_pct
                if position.symbol in self.entry_time:
                    self.time_of_mae[position.symbol] = time.time() - self.entry_time[position.symbol]

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

        # 3. MICRO PROFIT: Exit with profit above costs if OB is weakening
        # ECONOMIC GUARDRAIL: Dynamically calculate required profit based on execution mode
        # required_profit = max(min_profit_floor_pct, cost * min_profit_multiple_of_cost)
        required_profit = COST_MODEL.get_required_profit_pct()
        estimated_cost = COST_MODEL.get_total_round_trip_cost()

        if pnl_pct > 0 and pnl_pct < config.take_profit_pct:
            if orderbook:
                current_imbalance = orderbook.imbalance_ratio
                entry_imbalance = self.entry_imbalance.get(position.symbol, 0.5)

                # Check if OB is weakening (edge decaying)
                ob_weakening = False
                if position.side == SignalType.LONG:
                    imbalance_drop = entry_imbalance - current_imbalance
                    ob_weakening = imbalance_drop > 0.1  # Lost 10%+ of bullish edge
                else:
                    imbalance_rise = current_imbalance - entry_imbalance
                    ob_weakening = imbalance_rise > 0.1  # Lost 10%+ of bearish edge

                if ob_weakening:
                    # ECONOMIC GUARDRAIL: Only exit if profit clears costs
                    if pnl_pct >= required_profit:
                        logger.info(
                            f"{position.symbol} MICRO PROFIT: +{pnl_pct:.3f}% | "
                            f"OB weakening {entry_imbalance:.1%} -> {current_imbalance:.1%} | "
                            f"required={required_profit:.2f}% cost={estimated_cost:.2f}%"
                        )
                        return ExitReason.MICRO_PROFIT
                    else:
                        # Log skipped micro_profit - profit doesn't cover costs
                        logger.debug(
                            f"{position.symbol} MICRO_PROFIT_SKIPPED: target={pnl_pct:.3f}% "
                            f"required={required_profit:.2f}% est_cost={estimated_cost:.2f}% | "
                            f"OB weakening but waiting for higher profit"
                        )

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
            "seconds_to_max_adverse": self.time_of_mae.get(position.symbol, 0),
            "max_adverse_pct": mae,
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
        self.time_of_mae.pop(position.symbol, None)
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

        # Clean up enhanced MFE/MAE tracking
        self.position_max_price.pop(position.symbol, None)
        self.position_min_price.pop(position.symbol, None)
        self.position_last_update.pop(position.symbol, None)

        # Clean up maker scalper tracking
        self.pending_entry_orders.pop(position.symbol, None)
        self.pending_tp_orders.pop(position.symbol, None)
        self.entry_order_placed_time.pop(position.symbol, None)
        self.tp_order_placed_time.pop(position.symbol, None)
        self.exit_attempt_start.pop(position.symbol, None)

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

        try:
            while self.monitoring:
                try:
                    await self.check_exits()
                    await asyncio.sleep(self.monitor_interval)
                except asyncio.CancelledError:
                    break  # Exit loop on cancellation
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    await asyncio.sleep(1)
        except asyncio.CancelledError:
            logger.info("Position monitoring cancelled")
            raise

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
