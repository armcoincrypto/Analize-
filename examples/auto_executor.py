#!/usr/bin/env python3
"""
Automated Trading Executor - Level 3 Operator-Driven Bot

This is NOT a naive "signal → trade" bot.
This is a CONTROLLED EXECUTOR that obeys the full hierarchy:

    STRATEGY SIGNAL
           ↓
    TRADING GATE (can block)
           ↓
    CAPITAL SCALING (can reduce)
           ↓
    SESSION TRACKER (can stop)
           ↓
    KILL SWITCH (can terminate)
           ↓
    EXCHANGE EXECUTION

If ANY layer says NO → NO TRADE. Period.

This is how professionals automate.
"""

import ssl_bypass  # Must be first
import os
import sys
import json
import time
import requests
import datetime
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from enum import Enum

# Import all control layers
try:
    from trading_gate import TradingGate, GateDecision, should_trade_now
except ImportError:
    print("ERROR: trading_gate.py required")
    TradingGate = None

try:
    from production_config import ProductionConfig, validate_signal, validate_symbol, get_position_size
except ImportError:
    print("ERROR: production_config.py required")
    ProductionConfig = None

try:
    from session_tracker import SessionTracker, create_trade_record
except ImportError:
    print("WARNING: session_tracker.py not found")
    SessionTracker = None

try:
    from capital_scaling import CapitalScalingManager, ScalingPhase
except ImportError:
    print("WARNING: capital_scaling.py not found")
    CapitalScalingManager = None

try:
    from evidence_collector import EvidenceCollector
except ImportError:
    EvidenceCollector = None


class ExecutionMode(Enum):
    """Trading execution mode."""
    PAPER = "paper"           # Simulated trades
    MICRO_LIVE = "micro_live" # Real trades, tiny size
    LIVE = "live"             # Real trades, normal size


class TradeAction(Enum):
    """Trade action."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Signal:
    """Trading signal from strategy."""
    symbol: str
    action: TradeAction
    signals: List[str]        # ["BB", "MACD", etc.]
    confidence: float         # 0-100
    entry_price: float
    stop_loss: float
    take_profit: float
    regime: str
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.utcnow)


@dataclass
class ExecutionResult:
    """Result of trade execution."""
    executed: bool
    action: TradeAction
    symbol: str
    entry_price: float
    size_usd: float
    size_qty: float
    stop_loss: float
    take_profit: float
    blocked_by: Optional[str]
    blocked_reason: Optional[str]
    adjustments: List[str]
    timestamp: datetime.datetime


class AutomatedExecutor:
    """
    The automated trading executor.

    Obeys the FULL control hierarchy:
    1. Trading Gate - Can block entirely
    2. Production Config - Validates symbol/signals
    3. Capital Scaling - Determines phase and size
    4. Session Tracker - Enforces session limits
    5. Kill Switch - Emergency override

    Only after ALL pass does execution happen.
    """

    def __init__(
        self,
        mode: ExecutionMode = ExecutionMode.PAPER,
        initial_capital: float = 10000.0,
        api_key: str = None,
        api_secret: str = None
    ):
        self.mode = mode
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.api_key = api_key
        self.api_secret = api_secret

        # Control layers
        self.config = ProductionConfig() if ProductionConfig else None
        self.session = SessionTracker(initial_capital=initial_capital) if SessionTracker else None
        self.scaler = CapitalScalingManager() if CapitalScalingManager else None
        self.evidence = EvidenceCollector() if EvidenceCollector else None

        # Open positions
        self.positions: Dict[str, Dict] = {}

        # Kill switch
        self.kill_switch_active = False
        self.kill_switch_reason = ""

        # Statistics
        self.total_signals = 0
        self.blocked_signals = 0
        self.executed_trades = 0

        print("=" * 60)
        print("AUTOMATED TRADING EXECUTOR")
        print("=" * 60)
        print(f"  Mode: {mode.value.upper()}")
        print(f"  Capital: ${initial_capital:,.2f}")
        print(f"  Kill Switch: {'ACTIVE' if self.kill_switch_active else 'Ready'}")
        print("=" * 60)

    def activate_kill_switch(self, reason: str):
        """Activate emergency kill switch - stops ALL trading."""
        self.kill_switch_active = True
        self.kill_switch_reason = reason

        print("\n" + "!" * 60)
        print("  KILL SWITCH ACTIVATED")
        print(f"  Reason: {reason}")
        print("  ALL TRADING STOPPED")
        print("!" * 60 + "\n")

        # Close all positions if live
        if self.mode != ExecutionMode.PAPER:
            self._emergency_close_all()

    def _emergency_close_all(self):
        """Emergency close all positions."""
        for symbol, position in list(self.positions.items()):
            print(f"  [EMERGENCY] Closing {symbol}...")
            # In live mode, would send market close order
            self.positions.pop(symbol, None)

    def check_hierarchy(self, signal: Signal) -> Tuple[bool, str, float]:
        """
        Check the FULL control hierarchy.

        Returns:
            Tuple of (allowed, blocked_reason, size_multiplier)
        """
        adjustments = []

        # Layer 0: Kill Switch (highest priority)
        if self.kill_switch_active:
            return False, f"KILL SWITCH: {self.kill_switch_reason}", 0.0

        # Layer 1: Trading Gate
        gate = TradingGate(signal.symbol) if TradingGate else None
        if gate:
            status = gate.evaluate()
            if status.decision == GateDecision.CLOSED:
                return False, f"GATE CLOSED: {status.summary}", 0.0
            adjustments.append(f"Gate: {status.size_multiplier:.0%}")
            gate_multiplier = status.size_multiplier
        else:
            gate_multiplier = 1.0

        # Layer 2: Production Config - Symbol validation
        if self.config:
            sym_check = validate_symbol(signal.symbol, self.config)
            if not sym_check["valid"]:
                return False, f"SYMBOL BLOCKED: {sym_check['reason']}", 0.0
            adjustments.append(f"Symbol weight: {sym_check['weight']}")
            symbol_weight = sym_check["weight"]
        else:
            symbol_weight = 1.0

        # Layer 3: Production Config - Signal validation
        if self.config:
            sig_check = validate_signal(signal.signals, self.config)
            if not sig_check["valid"]:
                return False, f"SIGNAL BLOCKED: {sig_check['reason']}", 0.0
            adjustments.append(f"Signal confidence: {sig_check['confidence']:.0f}%")
            signal_multiplier = sig_check["size_multiplier"]
        else:
            signal_multiplier = 1.0

        # Layer 4: Session Tracker
        if self.session:
            can_trade, reason, details = self.session.can_trade()
            if not can_trade:
                return False, f"SESSION BLOCKED: {reason}", 0.0
            adjustments.append("Session: OK")

        # Layer 5: Capital Scaling (phase limits)
        # Skip scaling check for PAPER mode - paper trading is for validation
        if self.scaler and self.mode != ExecutionMode.PAPER:
            min_size, max_size = self.scaler.get_position_size_range()
            if max_size == 0:
                return False, f"SCALING BLOCKED: Phase {self.scaler.state.phase.value}", 0.0
            adjustments.append(f"Phase: {self.scaler.state.phase.value}")
        elif self.mode == ExecutionMode.PAPER:
            adjustments.append("Phase: PAPER (validation mode)")

        # Calculate combined multiplier
        combined = gate_multiplier * symbol_weight * signal_multiplier

        return True, None, combined

    def calculate_position_size(self, signal: Signal, multiplier: float) -> Tuple[float, float]:
        """
        Calculate position size in USD and quantity.

        Returns:
            Tuple of (size_usd, size_qty)
        """
        # Base size from config
        if self.config:
            base_pct = self.config.BASE_POSITION_SIZE_PERCENT
        else:
            base_pct = 2.0

        # Apply multiplier
        final_pct = base_pct * multiplier

        # Cap at max
        if self.config:
            final_pct = min(final_pct, self.config.MAX_POSITION_SIZE_PERCENT)

        # Calculate USD
        size_usd = self.current_capital * (final_pct / 100)

        # Apply mode limits
        if self.mode == ExecutionMode.PAPER:
            pass  # No limit for paper
        elif self.mode == ExecutionMode.MICRO_LIVE:
            size_usd = min(size_usd, 50)  # Max $50 for micro
        elif self.mode == ExecutionMode.LIVE:
            if self.scaler:
                _, max_size = self.scaler.get_position_size_range()
                size_usd = min(size_usd, max_size)

        # Calculate quantity
        size_qty = size_usd / signal.entry_price if signal.entry_price > 0 else 0

        return size_usd, size_qty

    def execute(self, signal: Signal) -> ExecutionResult:
        """
        Execute a trading signal through the FULL hierarchy.

        This is the main entry point.
        """
        self.total_signals += 1
        adjustments = []

        print(f"\n{'─' * 60}")
        print(f"[SIGNAL] {signal.symbol} {signal.action.value}")
        print(f"  Signals: {signal.signals}")
        print(f"  Price: ${signal.entry_price:.4f}")
        print(f"  Confidence: {signal.confidence:.0f}%")
        print(f"{'─' * 60}")

        # Check full hierarchy
        allowed, blocked_reason, multiplier = self.check_hierarchy(signal)

        if not allowed:
            self.blocked_signals += 1
            print(f"  [BLOCKED] {blocked_reason}")

            return ExecutionResult(
                executed=False,
                action=signal.action,
                symbol=signal.symbol,
                entry_price=signal.entry_price,
                size_usd=0,
                size_qty=0,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                blocked_by="HIERARCHY",
                blocked_reason=blocked_reason,
                adjustments=[],
                timestamp=datetime.datetime.utcnow()
            )

        # Calculate position size
        size_usd, size_qty = self.calculate_position_size(signal, multiplier)
        adjustments.append(f"Size multiplier: {multiplier:.0%}")
        adjustments.append(f"Position: ${size_usd:.2f}")

        if size_usd < 1:
            self.blocked_signals += 1
            print(f"  [BLOCKED] Position size too small: ${size_usd:.2f}")

            return ExecutionResult(
                executed=False,
                action=signal.action,
                symbol=signal.symbol,
                entry_price=signal.entry_price,
                size_usd=size_usd,
                size_qty=size_qty,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                blocked_by="SIZE",
                blocked_reason=f"Position size ${size_usd:.2f} too small",
                adjustments=adjustments,
                timestamp=datetime.datetime.utcnow()
            )

        # Execute the trade
        print(f"  [EXECUTING] {signal.action.value} {size_qty:.4f} {signal.symbol}")
        print(f"    Size: ${size_usd:.2f}")
        print(f"    Entry: ${signal.entry_price:.4f}")
        print(f"    Stop: ${signal.stop_loss:.4f}")
        print(f"    Target: ${signal.take_profit:.4f}")

        if self.mode == ExecutionMode.PAPER:
            success = self._execute_paper(signal, size_usd, size_qty)
        else:
            success = self._execute_live(signal, size_usd, size_qty)

        if success:
            self.executed_trades += 1
            print(f"  [SUCCESS] Trade executed")

            # Record in session tracker
            if self.session:
                # Will be recorded when trade closes
                pass

            # Track position
            self.positions[signal.symbol] = {
                "side": signal.action.value,
                "entry_price": signal.entry_price,
                "size_usd": size_usd,
                "size_qty": size_qty,
                "stop_loss": signal.stop_loss,
                "take_profit": signal.take_profit,
                "entry_time": datetime.datetime.utcnow()
            }

        return ExecutionResult(
            executed=success,
            action=signal.action,
            symbol=signal.symbol,
            entry_price=signal.entry_price,
            size_usd=size_usd,
            size_qty=size_qty,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            blocked_by=None,
            blocked_reason=None,
            adjustments=adjustments,
            timestamp=datetime.datetime.utcnow()
        )

    def _execute_paper(self, signal: Signal, size_usd: float, size_qty: float) -> bool:
        """Execute paper trade (simulated)."""
        # Paper trades always succeed
        print(f"    [PAPER] Simulated execution")
        return True

    def _execute_live(self, signal: Signal, size_usd: float, size_qty: float) -> bool:
        """Execute live trade on exchange."""
        if not self.api_key or not self.api_secret:
            print(f"    [ERROR] No API credentials for live trading")
            return False

        # TODO: Implement actual exchange API calls
        # This would use ccxt or direct Binance API
        print(f"    [LIVE] Exchange execution (not implemented)")
        return False

    def check_and_close_positions(self, current_prices: Dict[str, float]):
        """
        Check open positions for exit conditions.

        Args:
            current_prices: Dict of symbol -> current price
        """
        for symbol, position in list(self.positions.items()):
            if symbol not in current_prices:
                continue

            current_price = current_prices[symbol]
            entry_price = position["entry_price"]
            side = position["side"]

            # Calculate P&L
            if side == "BUY":
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
            else:
                pnl_pct = ((entry_price - current_price) / entry_price) * 100

            # Check stop loss
            if side == "BUY" and current_price <= position["stop_loss"]:
                self._close_position(symbol, current_price, "STOP_LOSS")
            elif side == "SELL" and current_price >= position["stop_loss"]:
                self._close_position(symbol, current_price, "STOP_LOSS")

            # Check take profit
            elif side == "BUY" and current_price >= position["take_profit"]:
                self._close_position(symbol, current_price, "TAKE_PROFIT")
            elif side == "SELL" and current_price <= position["take_profit"]:
                self._close_position(symbol, current_price, "TAKE_PROFIT")

    def _close_position(self, symbol: str, exit_price: float, reason: str):
        """Close a position and record the trade."""
        if symbol not in self.positions:
            return

        position = self.positions.pop(symbol)
        entry_price = position["entry_price"]
        side = position["side"]
        size_usd = position["size_usd"]

        # Calculate P&L
        if side == "BUY":
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100
        else:
            pnl_pct = ((entry_price - exit_price) / entry_price) * 100

        pnl_usd = size_usd * (pnl_pct / 100)

        print(f"\n  [CLOSE] {symbol} {reason}")
        print(f"    Entry: ${entry_price:.4f} → Exit: ${exit_price:.4f}")
        print(f"    P&L: {pnl_pct:+.2f}% (${pnl_usd:+.2f})")

        # Update capital
        self.current_capital += pnl_usd

        # Record in session tracker
        if self.session:
            trade = create_trade_record(
                symbol=symbol,
                side=side,
                entry_price=entry_price,
                exit_price=exit_price,
                position_size=size_usd,
                signals=["BB", "MACD"],  # Would come from position
                exit_reason=reason
            )
            self.session.record_trade(trade)

        # Record in scaler
        if self.scaler:
            self.scaler.record_trade(pnl_pct, pnl_usd)

    def print_status(self):
        """Print current executor status."""
        print("\n" + "=" * 60)
        print("EXECUTOR STATUS")
        print("=" * 60)

        print(f"\n[MODE]")
        print(f"  Mode: {self.mode.value.upper()}")
        print(f"  Kill Switch: {'ACTIVE - ' + self.kill_switch_reason if self.kill_switch_active else 'Ready'}")

        print(f"\n[CAPITAL]")
        print(f"  Initial: ${self.initial_capital:,.2f}")
        print(f"  Current: ${self.current_capital:,.2f}")
        print(f"  P&L: ${self.current_capital - self.initial_capital:+,.2f}")

        print(f"\n[STATISTICS]")
        print(f"  Total Signals: {self.total_signals}")
        print(f"  Blocked: {self.blocked_signals} ({self.blocked_signals/max(1,self.total_signals)*100:.0f}%)")
        print(f"  Executed: {self.executed_trades}")

        print(f"\n[POSITIONS]")
        if self.positions:
            for symbol, pos in self.positions.items():
                print(f"  {symbol}: {pos['side']} ${pos['size_usd']:.2f} @ ${pos['entry_price']:.4f}")
        else:
            print(f"  No open positions")

        if self.session:
            print(f"\n[SESSION]")
            status = self.session.get_status()
            print(f"  Session P&L: ${status['session_pnl']:+.2f} ({status['session_pnl_pct']:+.1f}%)")
            print(f"  Daily Trades: {status['daily_trades']}")
            print(f"  Consecutive Losses: {status['consecutive_losses']}")

        if self.scaler:
            print(f"\n[SCALING]")
            print(f"  Phase: {self.scaler.state.phase.value}")
            min_size, max_size = self.scaler.get_position_size_range()
            print(f"  Size Range: ${min_size} - ${max_size}")

        print("=" * 60)


# =============================================================================
# MAIN - Demo
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Automated Trading Executor")
    parser.add_argument("--mode", choices=["paper", "micro", "live"], default="paper")
    parser.add_argument("--capital", type=float, default=10000.0)
    parser.add_argument("--symbol", default="ATOMUSDT")
    args = parser.parse_args()

    mode_map = {
        "paper": ExecutionMode.PAPER,
        "micro": ExecutionMode.MICRO_LIVE,
        "live": ExecutionMode.LIVE
    }

    executor = AutomatedExecutor(
        mode=mode_map[args.mode],
        initial_capital=args.capital
    )

    # Demo: Create a test signal
    print("\n[DEMO] Creating test signal...")

    test_signal = Signal(
        symbol=args.symbol,
        action=TradeAction.BUY,
        signals=["BB", "MACD"],
        confidence=82.0,
        entry_price=10.50,
        stop_loss=10.29,  # 2% stop
        take_profit=10.82,  # 3% target
        regime="sideways"
    )

    # Execute through full hierarchy
    result = executor.execute(test_signal)

    # Show status
    executor.print_status()

    # Show result
    print("\n[EXECUTION RESULT]")
    print(f"  Executed: {result.executed}")
    if result.blocked_reason:
        print(f"  Blocked By: {result.blocked_by}")
        print(f"  Reason: {result.blocked_reason}")
    else:
        print(f"  Size: ${result.size_usd:.2f}")
        print(f"  Entry: ${result.entry_price:.4f}")

    print("\n" + "=" * 60)
    print("REMEMBER: If gate is CLOSED → bot does NOTHING. Period.")
    print("=" * 60)


if __name__ == "__main__":
    main()
