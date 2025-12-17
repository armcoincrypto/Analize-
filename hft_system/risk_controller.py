"""
Risk Controller for HFT System
==============================
Enforces strict risk limits to protect capital.

LIMITS:
- Max 1 position per symbol
- Max 2 total positions
- Max 3 daily losses
- Max 3% daily drawdown
- Min 30 seconds between trades
- BTC volatility filter (pause if BTC moves >4% in 5min)
- Exchange latency filter (pause if >500ms)

All trades MUST pass risk check before execution.
"""

import logging
import time
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum

from .config import RISK_CONFIG, SYSTEM_CONFIG, get_asset_config
from .signal_engine import Signal, SignalType

logger = logging.getLogger(__name__)


class RiskRejectReason(Enum):
    """Reasons for rejecting a trade."""
    APPROVED = "approved"
    MAX_POSITION_PER_SYMBOL = "max_position_per_symbol"
    MAX_TOTAL_POSITIONS = "max_total_positions"
    MAX_DAILY_LOSSES = "max_daily_losses"
    MAX_DAILY_DRAWDOWN = "max_daily_drawdown"
    MIN_TIME_BETWEEN_TRADES = "min_time_between_trades"
    BTC_VOLATILITY = "btc_high_volatility"
    EXCHANGE_LATENCY = "exchange_high_latency"
    INSUFFICIENT_CAPITAL = "insufficient_capital"
    TRADING_DISABLED = "trading_disabled"


@dataclass
class Position:
    """Active position."""
    symbol: str
    entry_price: float
    quantity: float
    side: SignalType
    entry_time: int
    position_id: str

    @property
    def notional_value(self) -> float:
        return self.entry_price * self.quantity


@dataclass
class DailyStats:
    """Daily trading statistics."""
    date: date
    trades: int = 0
    wins: int = 0
    losses: int = 0
    pnl: float = 0.0
    max_drawdown: float = 0.0
    peak_capital: float = 0.0


@dataclass
class RiskDecision:
    """Result of risk check."""
    approved: bool
    reason: RiskRejectReason
    message: str
    position_size: float = 0.0
    position_value: float = 0.0


class RiskController:
    """
    Central risk management system.

    Tracks positions, daily stats, and enforces all limits.
    """

    def __init__(self, initial_capital: float = None):
        self.initial_capital = initial_capital or SYSTEM_CONFIG.initial_capital
        self.current_capital = self.initial_capital
        self.peak_capital = self.initial_capital

        # Active positions
        self.positions: Dict[str, Position] = {}

        # Daily statistics
        self.daily_stats = DailyStats(
            date=date.today(),
            peak_capital=self.initial_capital
        )

        # Trade timing
        self.last_trade_time: Dict[str, float] = {}
        self.last_global_trade_time: float = 0

        # Market state
        self.btc_volatility_5m: float = 0.0
        self.exchange_latency_ms: int = 0

        # Emergency controls
        self.trading_enabled: bool = True
        self.manual_pause: bool = False

        logger.info(f"RiskController initialized with ${self.initial_capital:.2f} capital")

    def reset_daily_stats(self):
        """Reset daily stats at start of new day."""
        if date.today() != self.daily_stats.date:
            logger.info(f"New trading day - resetting daily stats")
            logger.info(
                f"Yesterday: {self.daily_stats.trades} trades, "
                f"{self.daily_stats.wins}W/{self.daily_stats.losses}L, "
                f"PnL: ${self.daily_stats.pnl:.2f}"
            )

            self.daily_stats = DailyStats(
                date=date.today(),
                peak_capital=self.current_capital
            )

    def update_market_state(self, btc_volatility: float = None, latency_ms: int = None):
        """Update market state for risk filters."""
        if btc_volatility is not None:
            self.btc_volatility_5m = btc_volatility
        if latency_ms is not None:
            self.exchange_latency_ms = latency_ms

    def get_position_count(self) -> int:
        """Get total number of open positions."""
        return len(self.positions)

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol."""
        return self.positions.get(symbol)

    def has_position(self, symbol: str) -> bool:
        """Check if we have a position in symbol."""
        return symbol in self.positions

    def calculate_position_size(self, signal: Signal) -> tuple:
        """
        Calculate position size based on risk parameters.

        Returns (quantity, notional_value).
        """
        config = get_asset_config(signal.symbol)

        # Position size as % of capital
        position_value = self.current_capital * (RISK_CONFIG.position_size_pct / 100)

        # Don't exceed max per position
        max_position_value = self.current_capital * (config.max_position_pct / 100)
        position_value = min(position_value, max_position_value)

        # Calculate quantity
        if signal.entry_price > 0:
            quantity = position_value / signal.entry_price
        else:
            quantity = 0

        return quantity, position_value

    def check_risk(self, signal: Signal) -> RiskDecision:
        """
        Check if a trade passes all risk controls.

        Returns RiskDecision with approval status and position size if approved.
        """
        self.reset_daily_stats()

        # Emergency disable
        if not self.trading_enabled or self.manual_pause:
            return RiskDecision(
                approved=False,
                reason=RiskRejectReason.TRADING_DISABLED,
                message="Trading is disabled"
            )

        # Check max losses
        if self.daily_stats.losses >= RISK_CONFIG.max_daily_losses:
            return RiskDecision(
                approved=False,
                reason=RiskRejectReason.MAX_DAILY_LOSSES,
                message=f"Max daily losses reached: {self.daily_stats.losses}/{RISK_CONFIG.max_daily_losses}"
            )

        # Check max drawdown
        drawdown_pct = ((self.daily_stats.peak_capital - self.current_capital) /
                        self.daily_stats.peak_capital * 100)
        if drawdown_pct >= RISK_CONFIG.max_daily_drawdown_pct:
            return RiskDecision(
                approved=False,
                reason=RiskRejectReason.MAX_DAILY_DRAWDOWN,
                message=f"Max daily drawdown reached: {drawdown_pct:.2f}%/{RISK_CONFIG.max_daily_drawdown_pct}%"
            )

        # Check position per symbol
        if self.has_position(signal.symbol):
            return RiskDecision(
                approved=False,
                reason=RiskRejectReason.MAX_POSITION_PER_SYMBOL,
                message=f"Already have position in {signal.symbol}"
            )

        # Check total positions
        if self.get_position_count() >= RISK_CONFIG.max_total_positions:
            return RiskDecision(
                approved=False,
                reason=RiskRejectReason.MAX_TOTAL_POSITIONS,
                message=f"Max positions reached: {self.get_position_count()}/{RISK_CONFIG.max_total_positions}"
            )

        # Check time between trades
        now = time.time()
        symbol_last = self.last_trade_time.get(signal.symbol, 0)
        if now - symbol_last < RISK_CONFIG.min_time_between_trades_sec:
            wait_time = RISK_CONFIG.min_time_between_trades_sec - (now - symbol_last)
            return RiskDecision(
                approved=False,
                reason=RiskRejectReason.MIN_TIME_BETWEEN_TRADES,
                message=f"Must wait {wait_time:.0f}s before trading {signal.symbol}"
            )

        # Check BTC volatility
        if self.btc_volatility_5m > RISK_CONFIG.btc_max_volatility_5m:
            return RiskDecision(
                approved=False,
                reason=RiskRejectReason.BTC_VOLATILITY,
                message=f"BTC too volatile: {self.btc_volatility_5m:.2f}%"
            )

        # Check exchange latency
        if self.exchange_latency_ms > RISK_CONFIG.max_exchange_latency_ms:
            return RiskDecision(
                approved=False,
                reason=RiskRejectReason.EXCHANGE_LATENCY,
                message=f"Exchange latency too high: {self.exchange_latency_ms}ms"
            )

        # Calculate position size
        quantity, position_value = self.calculate_position_size(signal)

        # Check we have enough capital
        if position_value > self.current_capital * 0.95:  # Leave 5% buffer
            return RiskDecision(
                approved=False,
                reason=RiskRejectReason.INSUFFICIENT_CAPITAL,
                message=f"Insufficient capital for position"
            )

        # All checks passed
        return RiskDecision(
            approved=True,
            reason=RiskRejectReason.APPROVED,
            message="All risk checks passed",
            position_size=quantity,
            position_value=position_value
        )

    def open_position(self, signal: Signal, quantity: float) -> Position:
        """
        Record a new position.

        Called after trade execution is confirmed.
        """
        position_id = f"{signal.symbol}_{int(time.time() * 1000)}"

        position = Position(
            symbol=signal.symbol,
            entry_price=signal.entry_price,
            quantity=quantity,
            side=signal.signal_type,
            entry_time=signal.timestamp,
            position_id=position_id
        )

        self.positions[signal.symbol] = position
        self.last_trade_time[signal.symbol] = time.time()
        self.last_global_trade_time = time.time()
        self.daily_stats.trades += 1

        logger.info(
            f"Position opened: {signal.symbol} {signal.signal_type.value} | "
            f"Entry: ${signal.entry_price:.4f} | Qty: {quantity:.4f}"
        )

        return position

    def close_position(self, symbol: str, exit_price: float, reason: str) -> Optional[float]:
        """
        Close a position and record PnL.

        Returns PnL amount.
        """
        position = self.positions.get(symbol)
        if not position:
            logger.warning(f"No position found for {symbol}")
            return None

        # Calculate PnL
        if position.side == SignalType.LONG:
            pnl = (exit_price - position.entry_price) * position.quantity
        else:
            pnl = (position.entry_price - exit_price) * position.quantity

        pnl_pct = (exit_price - position.entry_price) / position.entry_price * 100

        # Update stats
        self.current_capital += pnl
        self.daily_stats.pnl += pnl

        if pnl > 0:
            self.daily_stats.wins += 1
            if self.current_capital > self.peak_capital:
                self.peak_capital = self.current_capital
            if self.current_capital > self.daily_stats.peak_capital:
                self.daily_stats.peak_capital = self.current_capital
        else:
            self.daily_stats.losses += 1

        # Remove position
        del self.positions[symbol]

        hold_time = (time.time() * 1000 - position.entry_time) / 1000

        logger.info(
            f"Position closed: {symbol} | "
            f"Exit: ${exit_price:.4f} | "
            f"PnL: ${pnl:.2f} ({pnl_pct:+.2f}%) | "
            f"Reason: {reason} | "
            f"Hold: {hold_time:.0f}s"
        )

        return pnl

    def pause_trading(self, reason: str = "Manual pause"):
        """Temporarily pause all trading."""
        self.manual_pause = True
        logger.warning(f"Trading PAUSED: {reason}")

    def resume_trading(self):
        """Resume trading after pause."""
        self.manual_pause = False
        logger.info("Trading RESUMED")

    def emergency_stop(self, reason: str = "Emergency stop"):
        """Emergency stop - requires manual restart."""
        self.trading_enabled = False
        logger.critical(f"EMERGENCY STOP: {reason}")

    def get_status(self) -> Dict:
        """Get current risk controller status."""
        drawdown_pct = 0
        if self.daily_stats.peak_capital > 0:
            drawdown_pct = ((self.daily_stats.peak_capital - self.current_capital) /
                           self.daily_stats.peak_capital * 100)

        return {
            "trading_enabled": self.trading_enabled and not self.manual_pause,
            "capital": {
                "initial": self.initial_capital,
                "current": self.current_capital,
                "peak": self.peak_capital,
                "daily_pnl": self.daily_stats.pnl,
                "daily_drawdown_pct": drawdown_pct
            },
            "positions": {
                "open": self.get_position_count(),
                "max": RISK_CONFIG.max_total_positions,
                "symbols": list(self.positions.keys())
            },
            "daily_stats": {
                "date": str(self.daily_stats.date),
                "trades": self.daily_stats.trades,
                "wins": self.daily_stats.wins,
                "losses": self.daily_stats.losses,
                "max_losses_remaining": RISK_CONFIG.max_daily_losses - self.daily_stats.losses
            },
            "market_state": {
                "btc_volatility_5m": self.btc_volatility_5m,
                "exchange_latency_ms": self.exchange_latency_ms
            }
        }

    def get_all_positions(self) -> List[Position]:
        """Get all open positions."""
        return list(self.positions.values())
