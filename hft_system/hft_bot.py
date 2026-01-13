"""
HFT Trading Bot
===============
Main entry point for the HFT trading system.

# ============================================================
# STRATEGY LOCKED — DO NOT MODIFY
# Validation phase: edge preservation
# Last update: 2025-12-31
# ============================================================

Combines all components:
- WebSocket Manager: Real-time data feeds
- Signal Engine: Multi-condition signal detection
- Risk Controller: Position and risk management
- Execution Engine: Trade execution with TP/SL/Time stops
- Trade Logger: Persistent logging for analysis

LOCKED CONFIGURATION (data-validated):
- Enabled exits: TP, MICRO_PROFIT, OB_FLIP, TIME_STOP, SL
- Disabled exits: DELTA_NEGATIVE (-4.70% weekly), NO_MOVEMENT (0% win), FLOW_NEUTRAL
- Allowed regimes: low_vol_chop, high_vol_trend, mean_reversion
- Blocked regimes: liquidity_vacuum, news_spike, unknown
- Allowed symbols: SUI, XRP (ATOM disabled - 0% win rate)
- Confidence: HIGH always, MEDIUM if daily_pnl >= 0, LOW disabled

Usage:
    python -m hft_system.hft_bot [--live] [--capital 10000]
"""

import asyncio
import logging
import signal
import sys
import time
import argparse
from datetime import datetime

from .config import SYSTEM_CONFIG, ASSETS, TradingMode, get_asset_config, WINNER_GATE, COST_MODEL, REGIME_HYSTERESIS
from .websocket_manager import WebSocketManager
from .signal_engine import SignalEngine
from .risk_controller import RiskController, ConfidenceTier, ConfidenceScore
from .execution_engine import ExecutionEngine
from .trade_logger import TradeLogger

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


class HFTBot:
    """
    High-Frequency Trading Bot.

    Monitors multiple assets via WebSocket and executes trades
    when 3+ of 5 conditions are met.
    """

    def __init__(self, mode: TradingMode = TradingMode.PAPER, capital: float = None):
        self.mode = mode
        self.capital = capital or SYSTEM_CONFIG.initial_capital
        self.running = False

        # Initialize components
        self.ws_manager = WebSocketManager()
        self.risk_controller = RiskController(initial_capital=self.capital)
        self.signal_engine = SignalEngine(self.ws_manager)
        self.execution_engine = ExecutionEngine(
            self.ws_manager,
            self.risk_controller,
            mode=self.mode
        )
        self.trade_logger = TradeLogger()

        # Wire up callbacks (don't use async callback for signal - handle in scan loop)
        self.execution_engine.on_trade = self._on_trade
        self.execution_engine.on_exit = self._on_exit

        # Status tracking
        self.signals_generated = 0
        self.trades_executed = 0
        self.start_time = None

        # TASK 6: Track entry conditions for quality metrics
        self.entry_conditions: dict = {}  # trade_id -> {flow, orderbook, signal_price}

        # TASK 10: Market regime tracking
        self.current_regime: dict = {}  # symbol -> regime type
        self.regime_confidence: dict = {}  # symbol -> confidence
        # Data-backed regime configuration (from 86 trade analysis)
        # liquidity_vacuum: 13 trades, 0 wins, -0.04% avg = DISABLE
        # news_spike: 4 trades, 0 wins, -0.038% avg = DISABLE
        # unknown: 7 trades, 0 wins, -0.034% avg = DISABLE
        self.preferred_regimes = ["low_vol_chop", "high_vol_trend"]  # Only profitable regimes
        self.avoid_regimes = ["liquidity_vacuum", "news_spike", "unknown"]  # HARD BLOCK
        self.filter_by_regime = True  # ENABLED - skip signals in bad regimes

        # Regime cooldown: DISABLED for now - need more data collection
        # Will re-enable once we have 200+ trades with positive expectancy
        self.regime_loss_count: dict = {}  # regime -> consecutive loss count
        self.regime_paused: dict = {}  # regime -> True if paused
        self.enable_regime_cooldown = False  # DISABLED - collect more data first

        # TASK 11: No-trade zone tracking
        self.enable_no_trade_zones = True  # Set False to disable blocking
        self.signals_blocked = 0

        # TASK 12: Adaptive position sizing
        self.enable_adaptive_sizing = True  # Use confidence-based sizing
        self.sizing_decisions: dict = {}  # symbol -> last sizing decision

        # RESEARCH_GATE: Probe protections for paper mode
        # Set research mode based on paper/live trading mode
        self.research_mode = (self.mode == TradingMode.PAPER)
        self.last_trade_times: dict = {}  # symbol -> timestamp of last trade
        self.hourly_trade_counts: dict = {}  # symbol -> {hour: count}

        # REGIME HYSTERESIS: Stable regime classification
        # Prevents regime from flipping on every tick
        self.regime_raw: dict = {}  # symbol -> current raw regime detection
        self.regime_stable: dict = {}  # symbol -> stable regime (after hysteresis)
        self.regime_confirmations: dict = {}  # symbol -> {regime: consecutive_count}
        self.regime_stable_since: dict = {}  # symbol -> timestamp when stable regime started
        self.regime_failure_count: dict = {}  # symbol -> consecutive failures of current regime

        # POCKET TRACKING: Which gate pocket allowed the trade
        self.active_pocket: dict = {}  # symbol -> pocket_id for active trade

        logger.info("=" * 60)
        logger.info("STRATEGY LOCKED - EDGE PRESERVATION PHASE")
        logger.info("=" * 60)

        # DEPLOY_MARKER for tracking deployments
        import subprocess
        try:
            git_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'],
                                                stderr=subprocess.DEVNULL).decode().strip()
        except:
            git_hash = "unknown"
        deploy_time = int(datetime.utcnow().timestamp() * 1000)
        enabled_exits = "take_profit,micro_profit,ob_flip,spread_widen,time_stop,stop_loss"
        gate_status = f"winner_gate={'ON' if WINNER_GATE.enabled else 'OFF'}"
        if WINNER_GATE.enabled:
            gate_status += f",dual_pocket=A+B"
            gate_status += f",pocket_a={WINNER_GATE.pocket_a_regimes},imb>={WINNER_GATE.pocket_a_min_imbalance}"
            gate_status += f",pocket_b={'ON' if WINNER_GATE.pocket_b_enabled else 'OFF'}"
            if WINNER_GATE.pocket_b_enabled:
                gate_status += f",pocket_b_imb>={WINNER_GATE.pocket_b_min_imbalance}"
            gate_status += f",rate_limit={WINNER_GATE.max_trades_per_symbol_per_hour}/hr"
        cost_status = f"cost_model={'ON' if COST_MODEL.enabled else 'OFF'}"
        if COST_MODEL.enabled:
            total_cost = COST_MODEL.entry_fee_pct + COST_MODEL.exit_fee_pct + 2*COST_MODEL.spread_cost_pct + COST_MODEL.base_slippage_pct
            cost_status += f",est_cost={total_cost:.3f}%"
        hysteresis_status = f"hysteresis={'ON' if REGIME_HYSTERESIS.enabled else 'OFF'}"
        if REGIME_HYSTERESIS.enabled:
            hysteresis_status += f",enter={REGIME_HYSTERESIS.enter_confirmations},exit={REGIME_HYSTERESIS.exit_failures}"
        logger.info(f"DEPLOY_MARKER commit={git_hash} time={deploy_time} exits=[{enabled_exits}] {gate_status} {cost_status} {hysteresis_status}")

        logger.info(f"HFT Bot initialized")
        logger.info(f"  Mode: {self.mode.value}")
        logger.info(f"  Capital: ${self.capital:,.2f}")

        # Asset configuration
        logger.info(f"SYMBOLS:")
        logger.info(f"  Enabled: {', '.join(SYSTEM_CONFIG.enabled_assets)}")
        logger.info(f"  Disabled: ATOM (0% win rate in 19 trades)")

        # Exit configuration
        asset_config = get_asset_config(SYSTEM_CONFIG.enabled_assets[0])
        logger.info(f"EXITS (enabled):")
        logger.info(f"  Take Profit: +{asset_config.take_profit_pct}% (rare but dominant)")
        logger.info(f"  Micro Profit: +0.05% with OB weakening (primary income)")
        logger.info(f"  OB Flip: Orderbook flipped against position")
        logger.info(f"  Time Stop: {asset_config.time_stop_seconds}s fallback")
        logger.info(f"  Stop Loss: -{asset_config.stop_loss_pct}% (risk cap)")
        logger.info(f"EXITS (disabled - negative expectancy):")
        logger.info(f"  DELTA_NEGATIVE: DISABLED (-4.70% weekly, 115 trades)")
        logger.info(f"  NO_MOVEMENT: DISABLED (0% win in 54 trades)")
        logger.info(f"  FLOW_NEUTRAL: DISABLED")
        logger.info(f"  CONFIRMATION_TIMEOUT: DISABLED")

        # Regime configuration
        logger.info(f"REGIMES:")
        logger.info(f"  Allowed: {', '.join(self.preferred_regimes)}")
        logger.info(f"  Blocked: {', '.join(self.avoid_regimes)}")

        # Confidence configuration - PROFESSIONAL OPTION B
        logger.info(f"CONFIDENCE GATING (Professional Option B):")
        logger.info(f"  HIGH (27.3% win): ALWAYS ALLOWED")
        logger.info(f"  MEDIUM (15.9% win):")
        logger.info(f"    - Day start (PnL=0): ALLOWED (exploration)")
        logger.info(f"    - After loss (PnL<0): BLOCKED (contraction)")
        logger.info(f"    - After win (PnL>0): ALLOWED (expansion)")
        logger.info(f"  LOW (14.0% win): ALWAYS BLOCKED")

        # Winner Gate v1 configuration
        logger.info(f"WINNER GATE v1 (Data-validated):")
        logger.info(f"  Enabled: {WINNER_GATE.enabled}")
        if WINNER_GATE.enabled:
            logger.info(f"  Strict Mode: {WINNER_GATE.strict_mode} (only high_vol_trend)")
            logger.info(f"  Min Confidence: {WINNER_GATE.min_confidence_tier}")
            logger.info(f"  Min Imbalance: {WINNER_GATE.min_imbalance}")
            logger.info(f"  Max Spread: {WINNER_GATE.max_spread_pct}%")
            logger.info(f"  Expected: Only trades in [high_vol_trend + imb>=0.75 + HIGH tier]")
        logger.info("=" * 60)

    def _apply_regime_hysteresis(self, symbol: str, raw_regime: str) -> str:
        """
        Apply hysteresis to regime classification to prevent rapid flipping.

        Returns stable regime that requires N confirmations to enter
        and M failures to exit.
        """
        if not REGIME_HYSTERESIS.enabled:
            return raw_regime

        current_time = time.time()

        # Store raw regime for logging
        self.regime_raw[symbol] = raw_regime

        # Get current stable regime (or use raw if no stable yet)
        stable_regime = self.regime_stable.get(symbol, raw_regime)

        # Initialize tracking dicts if needed
        if symbol not in self.regime_confirmations:
            self.regime_confirmations[symbol] = {}
        if symbol not in self.regime_failure_count:
            self.regime_failure_count[symbol] = 0

        # Check if raw matches stable
        if raw_regime == stable_regime:
            # Reset failure count since we're confirming current regime
            self.regime_failure_count[symbol] = 0
            return stable_regime

        # Raw differs from stable - check if we should switch
        # Increment confirmation count for new regime
        if raw_regime not in self.regime_confirmations[symbol]:
            self.regime_confirmations[symbol][raw_regime] = 0
        self.regime_confirmations[symbol][raw_regime] += 1

        # Increment failure count for current stable regime
        self.regime_failure_count[symbol] += 1

        # Check if new regime has enough confirmations AND old regime has enough failures
        new_regime_confirmed = (
            self.regime_confirmations[symbol][raw_regime] >= REGIME_HYSTERESIS.enter_confirmations
        )
        old_regime_failed = (
            self.regime_failure_count[symbol] >= REGIME_HYSTERESIS.exit_failures
        )

        # Check minimum duration in current regime
        stable_since = self.regime_stable_since.get(symbol, 0)
        min_duration_met = (current_time - stable_since) >= REGIME_HYSTERESIS.min_regime_duration_sec

        # Switch if all conditions met
        if new_regime_confirmed and old_regime_failed and min_duration_met:
            old_stable = stable_regime
            self.regime_stable[symbol] = raw_regime
            self.regime_stable_since[symbol] = current_time
            self.regime_failure_count[symbol] = 0
            # Reset all confirmation counts
            self.regime_confirmations[symbol] = {}

            if REGIME_HYSTERESIS.log_raw_regime:
                logger.info(f"REGIME SWITCH [{symbol}]: {old_stable} -> {raw_regime} "
                           f"(confirmed after {REGIME_HYSTERESIS.enter_confirmations} samples)")

            return raw_regime

        # Not enough confirmations yet - return stable regime
        return stable_regime

    async def _on_signal(self, signal):
        """Handle new signal from signal engine."""
        self.signals_generated += 1

        # TASK 10: Check market regime filter
        current_regime = self.current_regime.get(signal.symbol, "unknown")

        # Check if regime is paused due to consecutive losses (if cooldown enabled)
        if self.enable_regime_cooldown and current_regime in self.regime_paused:
            logger.info(f"Signal SKIPPED: {signal.symbol} - regime={current_regime} (COOLED DOWN)")
            market_conditions = {"spread_pct": 0, "spread_change_1s": 0,
                                "delta_variance": 0, "ob_volume_instability": 0, "liquidity_depth": 0}
            self.trade_logger.log_blocked_signal(
                symbol=signal.symbol,
                signal_type=signal.signal_type.value,
                entry_price=signal.entry_price,
                block_reason="regime_cooldown",
                block_details=f"Regime {current_regime} paused after losses",
                market_conditions=market_conditions,
                regime=current_regime
            )
            self.signals_blocked += 1
            return

        if self.filter_by_regime and current_regime in self.avoid_regimes:
            logger.info(f"Signal SKIPPED: {signal.symbol} - regime={current_regime} (avoided)")
            # Log as blocked signal with regime reason
            market_conditions = {"spread_pct": 0, "spread_change_1s": 0,
                                "delta_variance": 0, "ob_volume_instability": 0, "liquidity_depth": 0}
            self.trade_logger.log_blocked_signal(
                symbol=signal.symbol,
                signal_type=signal.signal_type.value,
                entry_price=signal.entry_price,
                block_reason="regime_avoided",
                block_details=f"Avoided regime: {current_regime}",
                market_conditions=market_conditions,
                regime=current_regime
            )
            self.signals_blocked += 1
            return

        # TASK 11: Check no-trade zone before entry
        if self.enable_no_trade_zones:
            is_blocked, block_reason, block_details, market_conditions = \
                self.ws_manager.detect_no_trade_zone(signal.symbol)

            if is_blocked:
                logger.info(f"NO-TRADE ZONE: {signal.symbol} - {block_reason}: {block_details}")
                self.trade_logger.log_blocked_signal(
                    symbol=signal.symbol,
                    signal_type=signal.signal_type.value,
                    entry_price=signal.entry_price,
                    block_reason=block_reason,
                    block_details=block_details,
                    market_conditions=market_conditions,
                    regime=current_regime
                )
                self.signals_blocked += 1
                return

        # Check risk
        risk_decision = self.risk_controller.check_risk(signal)

        # TASK 12: Calculate confidence score and apply adaptive sizing
        orderbook = self.ws_manager.get_orderbook(signal.symbol)
        trade_flow = self.ws_manager.get_trade_flow(signal.symbol)

        orderbook_data = {
            "imbalance": orderbook.imbalance_ratio if orderbook else 0.5,
            "spread_pct": orderbook.spread if orderbook else 0.05
        }
        regime_data = {
            "regime": current_regime,
            "confidence": self.regime_confidence.get(signal.symbol, 0.5)
        }
        causality_data = {
            "primary_cause": "orderbook_imbalance",  # Primary driver for this system
            "strength": abs(orderbook.imbalance_ratio - 0.5) if orderbook else 0
        }
        edge_data = {
            "expected_duration_sec": 15  # Default expectation
        }

        confidence = self.risk_controller.calculate_confidence_score(
            orderbook_data=orderbook_data,
            regime_data=regime_data,
            causality_data=causality_data,
            edge_data=edge_data
        )

        # Store sizing decision
        self.sizing_decisions[signal.symbol] = confidence

        # ============================================================
        # WINNER GATE v2 - Dual-Pocket Entry Filter
        # ============================================================
        # POCKET A: high_vol_trend + relaxed tier/imbalance (primary edge)
        # POCKET B: low_vol_chop + stricter requirements (secondary)
        # ============================================================
        if WINNER_GATE.enabled:
            current_imbalance = orderbook_data.get("imbalance", 0.5)
            current_spread = orderbook_data.get("spread_pct", 0)
            tier_str = confidence.tier.value if hasattr(confidence.tier, 'value') else str(confidence.tier).lower()
            current_time = time.time()
            symbol = signal.symbol

            # Get orderbook depth for Pocket B
            ob_depth = 0
            if orderbook:
                ob_depth = (orderbook.bid_volume + orderbook.ask_volume) * signal.entry_price

            gate_blocked = True  # Default to blocked, try each pocket
            gate_reason = ""
            pocket_id = None

            # === PROBE PROTECTION: Rate limiting per symbol (applies to all pockets) ===
            rate_limited = False

            # Check minimum time between trades
            last_trade = self.last_trade_times.get(symbol, 0)
            seconds_since_last = current_time - last_trade
            if seconds_since_last < WINNER_GATE.min_seconds_between_trades_per_symbol:
                rate_limited = True
                gate_reason = f"cooldown={seconds_since_last:.0f}s (min {WINNER_GATE.min_seconds_between_trades_per_symbol}s)"

            # Check hourly trade count
            if not rate_limited:
                current_hour = int(current_time // 3600)
                if symbol not in self.hourly_trade_counts:
                    self.hourly_trade_counts[symbol] = {}
                hourly_counts = self.hourly_trade_counts[symbol]
                # Clean old hours
                hourly_counts = {h: c for h, c in hourly_counts.items() if h >= current_hour - 1}
                self.hourly_trade_counts[symbol] = hourly_counts
                current_hour_count = hourly_counts.get(current_hour, 0)
                if current_hour_count >= WINNER_GATE.max_trades_per_symbol_per_hour:
                    rate_limited = True
                    gate_reason = f"rate_limit={current_hour_count}/{WINNER_GATE.max_trades_per_symbol_per_hour} trades/hr"

            if rate_limited:
                gate_blocked = True
            else:
                # === TRY POCKET A: high_vol_trend (primary edge) ===
                if WINNER_GATE.pocket_a_enabled and current_regime in WINNER_GATE.pocket_a_regimes:
                    pocket_a_pass = (
                        tier_str in WINNER_GATE.pocket_a_tiers and
                        current_imbalance >= WINNER_GATE.pocket_a_min_imbalance and
                        current_spread <= WINNER_GATE.pocket_a_max_spread_pct
                    )
                    if pocket_a_pass:
                        gate_blocked = False
                        pocket_id = "A"
                    else:
                        # Log why Pocket A failed
                        if tier_str not in WINNER_GATE.pocket_a_tiers:
                            gate_reason = f"pocket_a: tier={tier_str} not in {WINNER_GATE.pocket_a_tiers}"
                        elif current_imbalance < WINNER_GATE.pocket_a_min_imbalance:
                            gate_reason = f"pocket_a: imb={current_imbalance:.3f} < {WINNER_GATE.pocket_a_min_imbalance}"
                        else:
                            gate_reason = f"pocket_a: spread={current_spread:.4f}% > {WINNER_GATE.pocket_a_max_spread_pct}%"

                # === TRY POCKET B: low_vol_chop with stricter requirements ===
                if gate_blocked and WINNER_GATE.pocket_b_enabled and current_regime in WINNER_GATE.pocket_b_regimes:
                    pocket_b_pass = (
                        tier_str in WINNER_GATE.pocket_b_tiers and
                        current_imbalance >= WINNER_GATE.pocket_b_min_imbalance and
                        current_spread <= WINNER_GATE.pocket_b_max_spread_pct and
                        ob_depth >= WINNER_GATE.pocket_b_min_depth
                    )
                    if pocket_b_pass:
                        gate_blocked = False
                        pocket_id = "B"
                    else:
                        # Log why Pocket B failed
                        if tier_str not in WINNER_GATE.pocket_b_tiers:
                            gate_reason = f"pocket_b: tier={tier_str} not in {WINNER_GATE.pocket_b_tiers}"
                        elif current_imbalance < WINNER_GATE.pocket_b_min_imbalance:
                            gate_reason = f"pocket_b: imb={current_imbalance:.3f} < {WINNER_GATE.pocket_b_min_imbalance}"
                        elif current_spread > WINNER_GATE.pocket_b_max_spread_pct:
                            gate_reason = f"pocket_b: spread={current_spread:.4f}% > {WINNER_GATE.pocket_b_max_spread_pct}%"
                        else:
                            gate_reason = f"pocket_b: depth=${ob_depth:.0f} < ${WINNER_GATE.pocket_b_min_depth:.0f}"

                # === NO POCKET MATCHED: Check if regime is blocked entirely ===
                if gate_blocked and not gate_reason:
                    if current_regime in WINNER_GATE.blocked_regimes:
                        gate_reason = f"regime={current_regime} (blocked)"
                    else:
                        gate_reason = f"regime={current_regime} (no pocket matches)"

            if gate_blocked:
                logger.info(f"WINNER_GATE BLOCK: {signal.symbol} - {gate_reason}")
                # Log to winner_gate_blocks table for analysis
                self.trade_logger.log_winner_gate_block(
                    symbol=signal.symbol,
                    signal_type=signal.signal_type.value,
                    entry_price=signal.entry_price,
                    block_reason=gate_reason,
                    regime=current_regime,
                    confidence_tier=tier_str,
                    imbalance=current_imbalance,
                    spread_pct=current_spread,
                    research_mode=self.research_mode
                )
                self.signals_blocked += 1
                return

            # === GATE PASSED: Update rate limiting trackers and store pocket_id ===
            self.last_trade_times[symbol] = current_time
            current_hour = int(current_time // 3600)
            if symbol not in self.hourly_trade_counts:
                self.hourly_trade_counts[symbol] = {}
            self.hourly_trade_counts[symbol][current_hour] = self.hourly_trade_counts[symbol].get(current_hour, 0) + 1
            self.active_pocket[symbol] = pocket_id
            logger.info(f"WINNER_GATE PASS [Pocket {pocket_id}]: {signal.symbol} | regime={current_regime}, tier={tier_str}, imb={current_imbalance:.3f}")

        # ============================================================
        # PROFESSIONAL OPTION B - CONFIDENCE GATING (data-validated)
        # ============================================================
        # This creates: Exploration when safe, Contraction when hurt,
        #               Expansion when proven
        #
        # HIGH confidence (27.3% win): ALWAYS ALLOWED
        # MEDIUM confidence (15.9% win):
        #   - PnL == 0 (day start): ALLOWED (exploration)
        #   - PnL < 0 (after loss): BLOCKED (contraction)
        #   - PnL > 0 (after win): ALLOWED (expansion)
        # LOW confidence (14.0% win): ALWAYS BLOCKED
        # SKIP tier: ALWAYS BLOCKED
        # ============================================================

        # Get daily PnL for MEDIUM tier decision
        daily_pnl = self.risk_controller.get_status()['capital']['daily_pnl']

        # LOW confidence: Always blocked (14% win rate too low)
        if confidence.tier == ConfidenceTier.LOW:
            logger.info(f"Signal SKIPPED: {signal.symbol} - LOW confidence={confidence.total_score:.0f} (disabled)")
            self.trade_logger.log_blocked_signal(
                symbol=signal.symbol,
                signal_type=signal.signal_type.value,
                entry_price=signal.entry_price,
                block_reason="low_confidence_disabled",
                block_details=f"LOW confidence tier disabled (14% win rate)",
                market_conditions=orderbook_data,
                regime=current_regime
            )
            self.signals_blocked += 1
            return

        # MEDIUM confidence: CONTRACTION mode when PnL < 0
        # (Allowed at day start PnL=0, blocked after losses, allowed after wins)
        if confidence.tier == ConfidenceTier.MEDIUM and daily_pnl < 0:
            logger.info(f"Signal SKIPPED: {signal.symbol} - MEDIUM blocked [CONTRACTION mode] (PnL=${daily_pnl:.2f})")
            self.trade_logger.log_blocked_signal(
                symbol=signal.symbol,
                signal_type=signal.signal_type.value,
                entry_price=signal.entry_price,
                block_reason="medium_contraction_mode",
                block_details=f"MEDIUM blocked in contraction mode (PnL=${daily_pnl:.2f} < 0)",
                market_conditions=orderbook_data,
                regime=current_regime
            )
            self.signals_blocked += 1
            return

        # Skip if confidence is SKIP tier
        if confidence.tier == ConfidenceTier.SKIP:
            logger.info(f"Signal SKIPPED: {signal.symbol} - confidence={confidence.total_score:.0f} (too low)")
            self.trade_logger.log_blocked_signal(
                symbol=signal.symbol,
                signal_type=signal.signal_type.value,
                entry_price=signal.entry_price,
                block_reason="skip_confidence",
                block_details=f"Confidence score {confidence.total_score:.0f} < 20 threshold",
                market_conditions=orderbook_data,
                regime=current_regime
            )
            self.signals_blocked += 1
            return

        # Apply confidence-based sizing
        if self.enable_adaptive_sizing:
            adjusted_qty, adjusted_value, adjustment_reason = \
                self.risk_controller.calculate_position_size_with_confidence(signal, confidence)
            risk_decision.position_size = adjusted_qty
            risk_decision.position_value = adjusted_value
            risk_decision.confidence = confidence
            risk_decision.size_adjustment_reason = adjustment_reason

        # Log signal
        self.trade_logger.log_signal(signal, risk_decision)

        if risk_decision.approved:
            # Execute trade with adjusted sizing
            result = await self.execution_engine.execute_entry(signal, risk_decision)

            if result.success:
                self.trades_executed += 1
                self.trade_logger.log_trade_entry(result, signal)

                # Log MICROSTRUCTURE trade flow at entry
                trade_id = result.order_id or f"{result.symbol}_{result.timestamp}"
                orderbook = self.ws_manager.get_orderbook(signal.symbol)
                self.trade_logger.log_trade_entry_flow(
                    trade_id=trade_id,
                    symbol=signal.symbol,
                    trade_flow=trade_flow,
                    orderbook_imbalance=orderbook.imbalance_ratio if orderbook else None,
                    spread_pct=orderbook.spread if orderbook else None,
                    snapshot_type="entry"
                )

                # TASK 6: Store entry conditions for quality metrics
                self.entry_conditions[trade_id] = {
                    "flow": trade_flow.copy(),
                    "orderbook": {
                        "imbalance": orderbook.imbalance_ratio if orderbook else 0,
                        "spread_pct": orderbook.spread if orderbook else 0
                    },
                    "signal_price": signal.entry_price,  # Expected entry price
                    "actual_price": result.entry_price,   # Actual fill price
                    "side": signal.signal_type.value
                }

                # TASK 8: Log trade causality - WHY this signal triggered
                ob_causality = self.ws_manager.get_orderbook_causality_data(signal.symbol)
                cvd_data = {
                    "delta_1s": trade_flow.get("delta_1s", 0),
                    "delta_3s": trade_flow.get("delta_3s", 0),
                    "delta_5s": trade_flow.get("net_delta", 0)  # Use net_delta as 5s proxy
                }
                price_velocity = self.ws_manager.get_price_velocity(signal.symbol)
                btc_data = self.ws_manager.get_btc_data()

                primary_cause = self.trade_logger.log_trade_causality(
                    trade_id=trade_id,
                    symbol=signal.symbol,
                    orderbook_data=ob_causality,
                    cvd_data=cvd_data,
                    price_velocity=price_velocity,
                    btc_data=btc_data
                )

                # Log primary cause in trade entry message
                logger.info(f"  Primary cause: {primary_cause}")

                # TASK 10: Log regime at entry
                entry_regime = self.current_regime.get(signal.symbol, "unknown")
                self.trade_logger.update_trade_regime(signal.symbol, entry_regime, is_entry=True)
                logger.info(f"  Market regime: {entry_regime}")

                # TASK 12: Log position sizing decision
                if self.enable_adaptive_sizing and confidence:
                    base_qty, base_value = self.risk_controller.calculate_position_size(signal)
                    confidence_data = {
                        "total_score": confidence.total_score,
                        "tier": confidence.tier.value,
                        "orderbook_score": confidence.orderbook_score,
                        "regime_score": confidence.regime_score,
                        "causality_score": confidence.causality_score,
                        "edge_persistence_score": confidence.edge_persistence_score,
                        "size_multiplier": confidence.size_multiplier,
                        "adjustment_reason": risk_decision.size_adjustment_reason,
                        "reasoning": confidence.reasoning
                    }
                    # Use same snapshot values from gate check, not live orderbook
                    context_data = {
                        "regime": entry_regime,
                        "orderbook_imbalance": orderbook_data.get("imbalance", 0),
                        "spread_pct": orderbook_data.get("spread_pct", 0),
                        "primary_cause": "orderbook_imbalance"
                    }
                    self.trade_logger.log_position_sizing(
                        trade_id=trade_id,
                        symbol=signal.symbol,
                        base_quantity=base_qty,
                        base_value=base_value,
                        final_quantity=result.quantity,
                        final_value=result.quantity * result.entry_price,
                        confidence_data=confidence_data,
                        context_data=context_data
                    )
                    logger.info(f"  Position sizing: {confidence.tier.value} ({confidence.size_multiplier}x)")
        else:
            logger.info(f"Signal rejected: {signal.symbol} - {risk_decision.message}")

    def _on_trade(self, result):
        """Handle trade execution result."""
        pass  # Already logged in _on_signal

    def _on_exit(self, result):
        """Handle position exit."""
        mfe = self.execution_engine.position_mfe.get(result.symbol, 0)
        mae = self.execution_engine.position_mae.get(result.symbol, 0)

        # Get market conditions at exit for cost model
        orderbook = self.ws_manager.get_orderbook(result.symbol)
        spread_at_exit = orderbook.spread if orderbook else 0
        imbalance_at_exit = orderbook.imbalance_ratio if orderbook else 0.5

        # Get pocket_id that allowed this trade
        pocket_id = self.active_pocket.pop(result.symbol, None)

        # Log trade exit with cost model and pocket tracking
        self.trade_logger.log_trade_exit(
            result, mfe, mae,
            spread_at_exit=spread_at_exit,
            imbalance_at_exit=imbalance_at_exit,
            pocket_id=pocket_id
        )

        # TASK 10: Log regime at exit (before status changes to closed)
        exit_regime = self.current_regime.get(result.symbol, "unknown")
        self.trade_logger.update_trade_regime(result.symbol, exit_regime, is_entry=False)

        # REGIME COOLDOWN: Track losses per regime, pause after 2 consecutive (if enabled)
        if self.enable_regime_cooldown:
            if result.pnl_pct < 0:  # Loss
                current_count = self.regime_loss_count.get(exit_regime, 0) + 1
                self.regime_loss_count[exit_regime] = current_count

                if current_count >= 2 and exit_regime not in self.avoid_regimes:
                    self.regime_paused[exit_regime] = True
                    logger.warning(
                        f"REGIME COOLDOWN: {exit_regime} paused after {current_count} consecutive losses"
                    )
            else:  # Win - reset loss count
                self.regime_loss_count[exit_regime] = 0
                if exit_regime in self.regime_paused:
                    del self.regime_paused[exit_regime]
                    logger.info(f"REGIME COOLDOWN RESET: {exit_regime} back to active")

        # Log MICROSTRUCTURE trade flow at exit
        trade_id = f"{result.symbol}_{result.entry_time}"
        exit_flow = self.ws_manager.get_trade_flow(result.symbol)
        orderbook = self.ws_manager.get_orderbook(result.symbol)
        self.trade_logger.log_trade_entry_flow(
            trade_id=trade_id,
            symbol=result.symbol,
            trade_flow=exit_flow,
            orderbook_imbalance=orderbook.imbalance_ratio if orderbook else None,
            spread_pct=orderbook.spread if orderbook else None,
            snapshot_type="exit"
        )

        # TASK 6: Log consolidated trade quality metrics
        entry_cond = self.entry_conditions.pop(trade_id, None)
        if entry_cond:
            exit_orderbook = {
                "imbalance": orderbook.imbalance_ratio if orderbook else 0,
                "spread_pct": orderbook.spread if orderbook else 0
            }
            self.trade_logger.log_trade_quality_metrics(
                trade_id=trade_id,
                symbol=result.symbol,
                side=entry_cond.get("side", "unknown"),
                entry_flow=entry_cond.get("flow", {}),
                exit_flow=exit_flow,
                entry_orderbook=entry_cond.get("orderbook", {}),
                exit_orderbook=exit_orderbook,
                time_in_trade_sec=result.hold_time_sec,
                exit_reason=result.reason.value,
                pnl_pct=result.pnl_pct,
                expected_entry_price=entry_cond.get("signal_price", result.entry_price),
                actual_entry_price=entry_cond.get("actual_price", result.entry_price),
                expected_exit_price=result.exit_price,  # For paper, expected=actual
                actual_exit_price=result.exit_price
            )

        # TASK 9: Log edge validation metrics
        edge_data = self.execution_engine.get_edge_data(trade_id)
        if edge_data:
            self.trade_logger.log_trade_edge(
                trade_id=trade_id,
                symbol=result.symbol,
                side=edge_data.get("side", "unknown"),
                seconds_to_max_favorable=edge_data.get("seconds_to_max_favorable", 0),
                max_favorable_pct=edge_data.get("max_favorable_pct", 0),
                seconds_to_ob_decay=edge_data.get("seconds_to_ob_decay", 0),
                ob_decay_amount=edge_data.get("ob_decay_amount", 0),
                delta_persistence_sec=edge_data.get("delta_persistence_sec", 0),
                entry_imbalance=edge_data.get("entry_imbalance", 0),
                entry_delta=edge_data.get("entry_delta", 0),
                entry_spread=edge_data.get("entry_spread", 0),
                edge_duration_sec=edge_data.get("edge_duration_sec", 0),
                final_pnl_pct=result.pnl_pct,
                exit_reason=result.reason.value
            )

    async def _signal_scan_loop(self):
        """Periodically scan for signals."""
        # Wait for WebSocket data to populate
        logger.info("Waiting 30s for WebSocket data to populate...")
        await asyncio.sleep(30)

        # TASK 10: Initialize market regime classification for all assets
        logger.info("Initializing market regime classification...")
        for symbol in SYSTEM_CONFIG.enabled_assets:
            try:
                raw_regime, confidence, metrics = self.ws_manager.classify_market_regime(symbol)
                # Apply hysteresis (will initialize stable regime on first call)
                stable_regime = self._apply_regime_hysteresis(symbol, raw_regime)
                self.current_regime[symbol] = stable_regime
                self.regime_confidence[symbol] = confidence
                # Initialize stable regime tracking
                self.regime_stable[symbol] = stable_regime
                self.regime_stable_since[symbol] = time.time()
                # Skip DB logging at startup - let the scan loop handle it
                logger.info(f"  {symbol}: regime={stable_regime} (raw={raw_regime}, confidence={confidence:.2f})")
            except Exception as e:
                logger.warning(f"  {symbol}: Failed to classify regime: {e}")
                self.current_regime[symbol] = "unknown"
                self.regime_confidence[symbol] = 0.0

        logger.info("Starting signal scan loop")
        scan_count = 0

        while self.running:
            try:
                # Scan all assets for signals
                signals = self.signal_engine.scan_all_assets()

                # Process any valid signals
                for signal in signals:
                    await self._on_signal(signal)

                # Log indicator snapshot every scan (1 second)
                for symbol in SYSTEM_CONFIG.enabled_assets:
                    signal = self.signal_engine.check_all_conditions(symbol)
                    status = self.signal_engine.get_status().get(symbol, {})

                    # Add conditions_met to status
                    status["conditions_met"] = signal.conditions_met

                    # Log indicator values
                    self.trade_logger.log_indicator_snapshot(symbol, status)

                    # Log strategy signal for multi-strategy analysis
                    conditions = {c.name: c.triggered for c in signal.conditions}
                    self.trade_logger.log_strategy_signal(symbol, signal.entry_price, conditions)

                    # Log orderbook snapshot
                    orderbook = self.ws_manager.get_orderbook(symbol)
                    if orderbook:
                        self.trade_logger.log_orderbook_snapshot(symbol, {
                            "best_bid": orderbook.best_bid,
                            "best_ask": orderbook.best_ask,
                            "bid_volume": orderbook.bid_volume,
                            "ask_volume": orderbook.ask_volume,
                            "spread_pct": orderbook.spread,
                            "imbalance": orderbook.imbalance_ratio
                        })

                    # Update old signals with current price (for outcome tracking)
                    if signal.entry_price and signal.entry_price > 0:
                        self.trade_logger.update_signal_outcomes(symbol, signal.entry_price)

                    # Log CVD (Cumulative Volume Delta) snapshot
                    cvd_data = self.ws_manager.get_cvd(symbol)
                    self.trade_logger.log_cvd_snapshot(symbol, cvd_data)

                # Batch commit every 10 scans
                scan_count += 1
                if scan_count % 10 == 0:
                    self.trade_logger.batch_commit()

                # TASK 10: Classify market regime every 30 scans (30 seconds)
                # Apply hysteresis to prevent rapid regime flipping
                if scan_count % 30 == 0:
                    for symbol in SYSTEM_CONFIG.enabled_assets:
                        try:
                            raw_regime, confidence, metrics = self.ws_manager.classify_market_regime(symbol)
                            # Apply hysteresis - only changes stable regime after N confirmations
                            stable_regime = self._apply_regime_hysteresis(symbol, raw_regime)
                            old_regime = self.current_regime.get(symbol, "unknown")
                            self.current_regime[symbol] = stable_regime
                            self.regime_confidence[symbol] = confidence
                            # Log with both raw and stable for analysis
                            self.trade_logger.log_market_regime(symbol, stable_regime, metrics, confidence)
                        except Exception as e:
                            logger.warning(f"Regime classification failed for {symbol}: {e}")

                await asyncio.sleep(1)  # Scan every second

            except Exception as e:
                logger.error(f"Error in signal scan: {e}")
                await asyncio.sleep(5)

    async def _status_loop(self):
        """Periodically print status."""
        while self.running:
            try:
                await asyncio.sleep(60)  # Every minute
                self._print_status()
            except Exception as e:
                logger.error(f"Error in status loop: {e}")

    def _print_status(self):
        """Print current bot status with condition debug info."""
        uptime = (datetime.now() - self.start_time).total_seconds() / 3600

        risk_status = self.risk_controller.get_status()
        ws_stats = self.ws_manager.get_connection_stats()
        positions = self.execution_engine.get_open_positions_status()

        logger.info("=" * 60)
        logger.info(f"STATUS UPDATE | Uptime: {uptime:.1f}h")
        logger.info(f"  Signals: {self.signals_generated} | Trades: {self.trades_executed} | Blocked: {self.signals_blocked}")
        logger.info(f"  Capital: ${risk_status['capital']['current']:,.2f} "
                   f"(Daily PnL: ${risk_status['capital']['daily_pnl']:+,.2f})")
        logger.info(f"  Daily: {risk_status['daily_stats']['wins']}W / "
                   f"{risk_status['daily_stats']['losses']}L | "
                   f"Losses remaining: {risk_status['daily_stats']['max_losses_remaining']}")
        logger.info(f"  WebSocket: {ws_stats['message_count']} msgs | "
                   f"Reconnects: {ws_stats['reconnect_count']}")

        # Show condition status for each asset
        logger.info("  CONDITIONS (need 3/5):")
        for symbol in SYSTEM_CONFIG.enabled_assets:
            signal = self.signal_engine.check_all_conditions(symbol)
            triggered = [c.name[:6] for c in signal.conditions if c.triggered]
            price = signal.entry_price
            regime = self.current_regime.get(symbol, "unknown")
            logger.info(f"    {symbol}: ${price:.4f} | {signal.conditions_met}/5 | {triggered} | regime={regime}")

        if positions:
            logger.info(f"  Open positions:")
            for symbol, pos in positions.items():
                logger.info(f"    {symbol}: {pos['pnl_pct']:+.2f}% | "
                          f"Hold: {pos['hold_time_sec']:.0f}s / {pos['time_stop_at']}s")

        # TASK 7: Microstructure performance metrics
        micro_stats = self.trade_logger.get_microstructure_stats()
        if "total_trades" in micro_stats and micro_stats["total_trades"] > 0:
            logger.info("  MICROSTRUCTURE METRICS:")
            logger.info(f"    Avg PnL: {micro_stats['avg_pnl_pct']:+.4f}% | "
                       f"Net Edge: {micro_stats['net_edge_after_costs_pct']:+.4f}%")
            logger.info(f"    Avg Hold: {micro_stats['avg_hold_time_sec']:.1f}s | "
                       f"Exit <5s: {micro_stats['pct_exited_within_5s']:.1f}%")
            logger.info(f"    Invalidation Exits: {micro_stats['pct_exited_by_invalidation']:.1f}% | "
                       f"Costs: {micro_stats['avg_total_cost_pct']:.4f}%")
            if "evaluation" in micro_stats:
                logger.info(f"    {micro_stats['evaluation']}")

        # TASK 9: Edge validation stats
        edge_stats = self.trade_logger.get_edge_stats()
        if "total_trades" in edge_stats and edge_stats["total_trades"] > 0:
            logger.info("  EDGE VALIDATION:")
            logger.info(f"    Real Edge: {edge_stats['pct_real_edge']:.1f}% | "
                       f"Fake Edge: {edge_stats['pct_fake_edge']:.1f}%")
            logger.info(f"    Avg Edge Duration: {edge_stats['avg_edge_duration_sec']:.1f}s | "
                       f"Time to MFE: {edge_stats['avg_time_to_max_favorable_sec']:.1f}s")
            if "edge_pnl_comparison" in edge_stats:
                real_pnl = edge_stats["edge_pnl_comparison"].get("real", {}).get("avg_pnl_pct", 0)
                fake_pnl = edge_stats["edge_pnl_comparison"].get("fake", {}).get("avg_pnl_pct", 0)
                logger.info(f"    Real Edge Avg PnL: {real_pnl:+.4f}% | Fake Edge Avg PnL: {fake_pnl:+.4f}%")

        # TASK 11: No-trade zone / blocked signal stats
        if self.signals_blocked > 0:
            blocked_stats = self.trade_logger.get_blocked_signal_stats()
            logger.info("  NO-TRADE ZONE STATS:")
            logger.info(f"    Blocked: {blocked_stats.get('total_blocked', 0)} | "
                       f"Block Rate: {blocked_stats.get('block_rate_pct', 0):.1f}%")
            if blocked_stats.get("reason_breakdown"):
                reasons = [f"{k}: {v['count']}" for k, v in blocked_stats["reason_breakdown"].items()]
                logger.info(f"    Reasons: {', '.join(reasons)}")

        # TASK 12: Position sizing / confidence stats
        sizing_stats = self.trade_logger.get_position_sizing_stats()
        if sizing_stats.get("tier_distribution"):
            logger.info("  ADAPTIVE SIZING:")
            for tier, data in sizing_stats["tier_distribution"].items():
                logger.info(f"    {tier}: {data['count']} trades | "
                           f"avg score {data['avg_score']:.0f} | "
                           f"multiplier {data['avg_multiplier']:.2f}x")
            if sizing_stats.get("tier_performance"):
                for tier, perf in sizing_stats["tier_performance"].items():
                    logger.info(f"    {tier} performance: {perf['avg_pnl_pct']:+.4f}% | "
                               f"win rate {perf['win_rate']:.1f}%")
        logger.info("=" * 60)

    async def start(self):
        """Start the HFT bot."""
        self.running = True
        self.start_time = datetime.now()

        logger.info("=" * 60)
        logger.info("HFT BOT STARTING")
        logger.info("=" * 60)

        # Start all tasks concurrently
        tasks = [
            asyncio.create_task(self.ws_manager.connect()),
            asyncio.create_task(self.execution_engine.start_monitoring()),
            asyncio.create_task(self._signal_scan_loop()),
            asyncio.create_task(self._status_loop())
        ]

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Bot tasks cancelled")
        except Exception as e:
            logger.error(f"Bot error: {e}")
        finally:
            await self.stop()

    async def stop(self):
        """Stop the HFT bot gracefully."""
        logger.info("Stopping HFT bot...")
        self.running = False

        # Stop components
        self.execution_engine.stop_monitoring()
        await self.ws_manager.disconnect()

        # Close any open positions
        positions = self.risk_controller.get_all_positions()
        if positions:
            logger.warning(f"Closing {len(positions)} open positions...")
            await self.execution_engine.emergency_close_all("Bot shutdown")

        # Print final summary
        summary = self.trade_logger.get_performance_summary()
        logger.info("=" * 60)
        logger.info("FINAL SUMMARY")
        logger.info("=" * 60)

        if "message" not in summary:
            logger.info(f"  Total trades: {summary['total_trades']}")
            logger.info(f"  Win rate: {summary['win_rate']:.1f}%")
            logger.info(f"  Total PnL: ${summary['total_pnl']:.2f}")
            logger.info(f"  Profit factor: {summary['profit_factor']:.2f}")
            logger.info(f"  Avg hold time: {summary['avg_hold_time_sec']:.0f}s")
            logger.info(f"  Exit reasons: {summary['exit_reasons']}")
        else:
            logger.info(f"  {summary['message']}")

        # TASK 7: Microstructure final stats
        micro_stats = self.trade_logger.get_microstructure_stats()
        if "total_trades" in micro_stats and micro_stats["total_trades"] > 0:
            logger.info("-" * 60)
            logger.info("MICROSTRUCTURE PERFORMANCE (what matters)")
            logger.info(f"  Avg PnL per trade: {micro_stats['avg_pnl_pct']:+.4f}%")
            logger.info(f"  Net edge after costs: {micro_stats['net_edge_after_costs_pct']:+.4f}%")
            logger.info(f"  Avg hold time: {micro_stats['avg_hold_time_sec']:.1f}s")
            logger.info(f"  Exited within 5s: {micro_stats['pct_exited_within_5s']:.1f}%")
            logger.info(f"  Exited by invalidation: {micro_stats['pct_exited_by_invalidation']:.1f}%")
            logger.info(f"  Avg total cost: {micro_stats['avg_total_cost_pct']:.4f}%")
            logger.info(f"  Exit breakdown: {micro_stats.get('exit_breakdown', {})}")
            if "evaluation" in micro_stats:
                logger.info(f"  EVALUATION: {micro_stats['evaluation']}")

        # TASK 8: Causality analysis - WHY trades happened
        causality_stats = self.trade_logger.get_causality_stats()
        if "cause_breakdown" in causality_stats and causality_stats["cause_breakdown"]:
            logger.info("-" * 60)
            logger.info("TRADE CAUSALITY (WHY signals triggered)")
            for cause, data in causality_stats["cause_breakdown"].items():
                logger.info(f"  {cause}: {data['count']} trades (strength: {data['avg_strength']:.4f})")

        # TASK 9: Edge validation - WHEN edge is real vs fake
        edge_stats = self.trade_logger.get_edge_stats()
        if "total_trades" in edge_stats and edge_stats["total_trades"] > 0:
            logger.info("-" * 60)
            logger.info("EDGE VALIDATION (real vs fake signals)")
            logger.info(f"  Real edge: {edge_stats['pct_real_edge']:.1f}%")
            logger.info(f"  Fake edge: {edge_stats['pct_fake_edge']:.1f}%")
            logger.info(f"  Avg edge duration: {edge_stats['avg_edge_duration_sec']:.1f}s")
            logger.info(f"  Avg time to max favorable: {edge_stats['avg_time_to_max_favorable_sec']:.1f}s")
            logger.info(f"  Avg OB decay time: {edge_stats['avg_ob_decay_time_sec']:.1f}s")
            logger.info(f"  Avg delta persistence: {edge_stats['avg_delta_persistence_sec']:.1f}s")

            # Show PnL comparison between real and fake edge trades
            if "edge_pnl_comparison" in edge_stats:
                pnl_comp = edge_stats["edge_pnl_comparison"]
                if "real" in pnl_comp:
                    logger.info(f"  Real edge trades: {pnl_comp['real']['count']} @ avg {pnl_comp['real']['avg_pnl_pct']:+.4f}%")
                if "fake" in pnl_comp:
                    logger.info(f"  Fake edge trades: {pnl_comp['fake']['count']} @ avg {pnl_comp['fake']['avg_pnl_pct']:+.4f}%")

            # Show per-symbol breakdown
            if "symbol_breakdown" in edge_stats and edge_stats["symbol_breakdown"]:
                logger.info("  Per-symbol edge quality:")
                for symbol, data in edge_stats["symbol_breakdown"].items():
                    logger.info(f"    {symbol}: {data['pct_real_edge']:.1f}% real | "
                               f"avg {data['avg_edge_duration']:.1f}s duration")

        # TASK 10: Market regime analysis - WHICH market context
        regime_stats = self.trade_logger.get_regime_stats()
        if regime_stats.get("total_trades_with_regime", 0) > 0:
            logger.info("-" * 60)
            logger.info("MARKET REGIME ANALYSIS (which conditions work best)")

            # Show performance by regime
            if "regime_performance" in regime_stats and regime_stats["regime_performance"]:
                logger.info("  Performance by regime:")
                for regime, data in regime_stats["regime_performance"].items():
                    logger.info(f"    {regime}: {data['trades']} trades | "
                               f"avg PnL {data['avg_pnl_pct']:+.4f}% | "
                               f"win rate {data['win_rate']:.1f}%")

            # Show recent regime distribution
            if "recent_regime_distribution" in regime_stats and regime_stats["recent_regime_distribution"]:
                logger.info("  Recent regime distribution (last 1h):")
                for regime, count in regime_stats["recent_regime_distribution"].items():
                    logger.info(f"    {regime}: {count} samples")

        # TASK 11: No-trade zone analysis - WHEN NOT to trade
        blocked_stats = self.trade_logger.get_blocked_signal_stats()
        if blocked_stats.get("total_blocked", 0) > 0:
            logger.info("-" * 60)
            logger.info("NO-TRADE ZONE ANALYSIS (blocked signals)")
            logger.info(f"  Total blocked: {blocked_stats['total_blocked']}")
            logger.info(f"  Block rate: {blocked_stats['block_rate_pct']:.1f}%")

            # Show breakdown by reason
            if "reason_breakdown" in blocked_stats and blocked_stats["reason_breakdown"]:
                logger.info("  Block reasons:")
                for reason, data in blocked_stats["reason_breakdown"].items():
                    pnl_str = f" (would have been {data['avg_would_have_pnl']:+.4f}%)" if data["avg_would_have_pnl"] else ""
                    logger.info(f"    {reason}: {data['count']} signals{pnl_str}")

            # Show estimated savings
            if blocked_stats.get("bad_trades_avoided", 0) > 0:
                logger.info(f"  Bad trades avoided: {blocked_stats['bad_trades_avoided']}")
                logger.info(f"  Avg loss avoided: {blocked_stats['avg_loss_avoided_pct']:+.4f}%")
                logger.info(f"  Estimated total savings: {blocked_stats['estimated_savings_pct']:+.4f}%")

        # TASK 12: Adaptive position sizing analysis - HOW MUCH to trade
        sizing_stats = self.trade_logger.get_position_sizing_stats()
        if sizing_stats.get("tier_distribution"):
            logger.info("-" * 60)
            logger.info("ADAPTIVE SIZING ANALYSIS (confidence-based position sizing)")

            # Show tier distribution
            if sizing_stats["tier_distribution"]:
                logger.info("  Tier distribution:")
                for tier, data in sizing_stats["tier_distribution"].items():
                    logger.info(f"    {tier}: {data['count']} trades | "
                               f"avg score {data['avg_score']:.0f} | "
                               f"multiplier {data['avg_multiplier']:.2f}x")

            # Show performance by tier
            if sizing_stats.get("tier_performance"):
                logger.info("  Performance by confidence tier:")
                for tier, perf in sizing_stats["tier_performance"].items():
                    logger.info(f"    {tier}: {perf['trades']} trades @ "
                               f"avg {perf['avg_pnl_pct']:+.4f}% | "
                               f"win rate {perf['win_rate']:.1f}%")

            # Show component averages
            if sizing_stats.get("component_averages"):
                comp = sizing_stats["component_averages"]
                logger.info(f"  Component avg scores: OB={comp['orderbook_score']:.2f} "
                           f"Regime={comp['regime_score']:.2f} "
                           f"Cause={comp['causality_score']:.2f} "
                           f"Edge={comp['edge_persistence_score']:.2f}")

            # Show sizing summary
            if sizing_stats.get("sizing_summary"):
                summary = sizing_stats["sizing_summary"]
                logger.info(f"  Sizing impact: base ${summary['avg_base_value']:.0f} -> "
                           f"final ${summary['avg_final_value']:.0f} "
                           f"({summary['avg_adjustment_pct']:+.1f}% avg adjustment)")

        logger.info("=" * 60)

        # Close database
        self.trade_logger.close()

        logger.info("HFT bot stopped")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="HFT Trading Bot")
    parser.add_argument("--live", action="store_true", help="Run in live mode (default: paper)")
    parser.add_argument("--capital", type=float, default=10000, help="Initial capital")
    args = parser.parse_args()

    mode = TradingMode.LIVE if args.live else TradingMode.PAPER

    bot = HFTBot(mode=mode, capital=args.capital)

    # Handle shutdown signals
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def shutdown_handler():
        logger.info("Shutdown signal received")
        for task in asyncio.all_tasks(loop):
            task.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown_handler)

    try:
        loop.run_until_complete(bot.start())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        loop.close()


if __name__ == "__main__":
    main()
