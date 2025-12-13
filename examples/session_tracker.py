#!/usr/bin/env python3
"""
Session Tracker - Real-time Session Risk Controls

Addresses expert-identified concerns:
1. Session loss limits (stop after 5% session loss)
2. Consecutive loss protection (stop after 3 losses)
3. Daily trade limits
4. Drawdown kill switches
5. Trade interval enforcement

Integrates with production_config.py for evidence-based rules.
"""

import ssl_bypass  # Must be first!
import os
import json
import requests
import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path


@dataclass
class TradeRecord:
    """Record of a single trade for session tracking."""
    timestamp: datetime.datetime
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    pnl_percent: float
    pnl_absolute: float
    position_size: float
    signals: List[str]
    exit_reason: str


class SessionTracker:
    """
    Real-time session tracking and risk control.

    Features:
    - Session P&L tracking with loss limits
    - Consecutive loss detection
    - Daily trade counting
    - Automatic pause triggers
    - Drawdown monitoring
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        state_file: str = None,
        max_session_loss_pct: float = 5.0,
        max_consecutive_losses: int = 3,
        max_daily_trades: int = 10,
        max_daily_loss_pct: float = 8.0,
        max_drawdown_pct: float = 15.0,
        cooldown_hours: float = 4.0,
        min_trade_interval_minutes: int = 30
    ):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital

        # Risk limits
        self.max_session_loss_pct = max_session_loss_pct
        self.max_consecutive_losses = max_consecutive_losses
        self.max_daily_trades = max_daily_trades
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.cooldown_hours = cooldown_hours
        self.min_trade_interval_minutes = min_trade_interval_minutes

        # State file
        if state_file is None:
            state_file = str(Path(__file__).parent / "session_state.json")
        self.state_file = state_file

        # Session state
        self.session_start = datetime.datetime.utcnow()
        self.session_trades: List[TradeRecord] = []
        self.session_pnl = 0.0
        self.consecutive_losses = 0

        # Daily state
        self.daily_trades: List[TradeRecord] = []
        self.daily_pnl = 0.0
        self.daily_date = datetime.datetime.utcnow().date()

        # Pause state
        self.is_paused = False
        self.pause_reason = ""
        self.pause_until: Optional[datetime.datetime] = None

        # Last trade time
        self.last_trade_time: Optional[datetime.datetime] = None

        # Load persisted state
        self._load_state()

    def _load_state(self):
        """Load persisted session state."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    state = json.load(f)

                self.current_capital = state.get("current_capital", self.initial_capital)
                self.peak_capital = state.get("peak_capital", self.current_capital)
                self.consecutive_losses = state.get("consecutive_losses", 0)

                # Check if state is from today
                state_date = state.get("date", "")
                today = str(datetime.datetime.utcnow().date())
                if state_date != today:
                    # New day - reset daily limits
                    self.daily_trades = []
                    self.daily_pnl = 0.0
                else:
                    self.daily_pnl = state.get("daily_pnl", 0.0)

                # Check if paused
                pause_until = state.get("pause_until")
                if pause_until:
                    pause_dt = datetime.datetime.fromisoformat(pause_until)
                    if pause_dt > datetime.datetime.utcnow():
                        self.is_paused = True
                        self.pause_reason = state.get("pause_reason", "")
                        self.pause_until = pause_dt

                # Load last trade time
                last_trade = state.get("last_trade_time")
                if last_trade:
                    self.last_trade_time = datetime.datetime.fromisoformat(last_trade)

            except Exception as e:
                print(f"Warning: Could not load session state: {e}")

    def _save_state(self):
        """Persist session state."""
        try:
            state = {
                "date": str(datetime.datetime.utcnow().date()),
                "current_capital": self.current_capital,
                "peak_capital": self.peak_capital,
                "daily_pnl": self.daily_pnl,
                "daily_trade_count": len(self.daily_trades),
                "consecutive_losses": self.consecutive_losses,
                "pause_until": self.pause_until.isoformat() if self.pause_until else None,
                "pause_reason": self.pause_reason,
                "last_trade_time": self.last_trade_time.isoformat() if self.last_trade_time else None,
            }

            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=2)

        except Exception as e:
            print(f"Warning: Could not save session state: {e}")

    def get_drawdown_pct(self) -> float:
        """Get current drawdown percentage from peak."""
        if self.peak_capital <= 0:
            return 0.0
        return ((self.peak_capital - self.current_capital) / self.peak_capital) * 100

    def get_session_pnl_pct(self) -> float:
        """Get session P&L as percentage of initial capital."""
        if self.initial_capital <= 0:
            return 0.0
        return (self.session_pnl / self.initial_capital) * 100

    def get_daily_pnl_pct(self) -> float:
        """Get daily P&L as percentage of initial capital."""
        if self.initial_capital <= 0:
            return 0.0
        return (self.daily_pnl / self.initial_capital) * 100

    def _pause_trading(self, reason: str, hours: float = None):
        """Pause trading for specified hours."""
        if hours is None:
            hours = self.cooldown_hours

        self.is_paused = True
        self.pause_reason = reason
        self.pause_until = datetime.datetime.utcnow() + datetime.timedelta(hours=hours)
        self._save_state()

        print(f"\n{'!'*60}")
        print(f"  TRADING PAUSED!")
        print(f"  Reason: {reason}")
        print(f"  Resume at: {self.pause_until.strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"{'!'*60}\n")

    def can_trade(self) -> Tuple[bool, str, Dict]:
        """
        Check if trading is currently allowed.

        Returns:
            Tuple of (allowed, reason, details)
        """
        details = {
            "checks_passed": [],
            "checks_failed": [],
            "warnings": [],
            "current_capital": self.current_capital,
            "drawdown_pct": self.get_drawdown_pct(),
            "session_pnl_pct": self.get_session_pnl_pct(),
            "daily_pnl_pct": self.get_daily_pnl_pct(),
            "consecutive_losses": self.consecutive_losses,
            "daily_trades": len(self.daily_trades),
        }

        # Check 1: Is trading paused?
        if self.is_paused:
            if self.pause_until and self.pause_until > datetime.datetime.utcnow():
                time_left = int((self.pause_until - datetime.datetime.utcnow()).total_seconds() / 60)
                details["checks_failed"].append(f"Paused: {self.pause_reason}")
                return False, f"Trading paused for {time_left} more minutes: {self.pause_reason}", details
            else:
                # Pause expired
                self.is_paused = False
                self.pause_reason = ""
                self.pause_until = None
                self._save_state()

        # Check 2: Kill switch - Maximum drawdown
        drawdown = self.get_drawdown_pct()
        if drawdown >= self.max_drawdown_pct:
            details["checks_failed"].append(f"KILL SWITCH: Drawdown {drawdown:.1f}%")
            return False, f"KILL SWITCH: Drawdown {drawdown:.1f}% >= {self.max_drawdown_pct}%", details
        details["checks_passed"].append(f"Drawdown OK: {drawdown:.1f}%")

        # Check 3: Daily loss limit
        daily_pnl = self.get_daily_pnl_pct()
        if daily_pnl <= -self.max_daily_loss_pct:
            self._pause_trading(f"Daily loss limit: {daily_pnl:.1f}%", 24)
            details["checks_failed"].append(f"Daily loss: {daily_pnl:.1f}%")
            return False, f"Daily loss limit: {daily_pnl:.1f}%", details
        details["checks_passed"].append(f"Daily P&L OK: {daily_pnl:.1f}%")

        # Check 4: Session loss limit
        session_pnl = self.get_session_pnl_pct()
        if session_pnl <= -self.max_session_loss_pct:
            self._pause_trading(f"Session loss: {session_pnl:.1f}%")
            details["checks_failed"].append(f"Session loss: {session_pnl:.1f}%")
            return False, f"Session loss limit: {session_pnl:.1f}%", details
        details["checks_passed"].append(f"Session P&L OK: {session_pnl:.1f}%")

        # Check 5: Consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            self._pause_trading(f"{self.consecutive_losses} consecutive losses")
            details["checks_failed"].append(f"Consecutive losses: {self.consecutive_losses}")
            return False, f"Too many consecutive losses: {self.consecutive_losses}", details
        details["checks_passed"].append(f"Consecutive losses OK: {self.consecutive_losses}")

        # Check 6: Daily trade count
        if len(self.daily_trades) >= self.max_daily_trades:
            details["checks_failed"].append(f"Daily trades: {len(self.daily_trades)}")
            return False, f"Daily trade limit: {len(self.daily_trades)}/{self.max_daily_trades}", details
        details["checks_passed"].append(f"Daily trades OK: {len(self.daily_trades)}")

        # Check 7: Minimum trade interval
        if self.last_trade_time:
            elapsed = (datetime.datetime.utcnow() - self.last_trade_time).total_seconds() / 60
            if elapsed < self.min_trade_interval_minutes:
                wait_time = int(self.min_trade_interval_minutes - elapsed)
                details["checks_failed"].append(f"Trade interval: {elapsed:.0f} min")
                return False, f"Wait {wait_time} more minutes (min interval: {self.min_trade_interval_minutes})", details
        details["checks_passed"].append("Trade interval OK")

        # Add warnings
        if drawdown >= self.max_drawdown_pct * 0.7:
            details["warnings"].append(f"Drawdown warning: {drawdown:.1f}%")
        if self.consecutive_losses >= self.max_consecutive_losses - 1:
            details["warnings"].append(f"Near consecutive loss limit: {self.consecutive_losses}")

        return True, "All checks passed", details

    def record_trade(self, trade: TradeRecord):
        """
        Record a completed trade.

        Args:
            trade: Completed trade record
        """
        # Check if new day
        if datetime.datetime.utcnow().date() != self.daily_date:
            self.daily_trades = []
            self.daily_pnl = 0.0
            self.daily_date = datetime.datetime.utcnow().date()

        # Update session
        self.session_trades.append(trade)
        self.session_pnl += trade.pnl_absolute

        # Update daily
        self.daily_trades.append(trade)
        self.daily_pnl += trade.pnl_absolute

        # Update capital
        self.current_capital += trade.pnl_absolute
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital

        # Update consecutive losses
        if trade.pnl_percent < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        # Update last trade time
        self.last_trade_time = trade.timestamp

        # Save state
        self._save_state()

        # Print update
        win_loss = "WIN" if trade.pnl_percent >= 0 else "LOSS"
        print(f"\n  [SESSION] Trade recorded: {trade.symbol} {win_loss}")
        print(f"    P&L: {trade.pnl_percent:+.2f}% (${trade.pnl_absolute:+.2f})")
        print(f"    Session P&L: {self.get_session_pnl_pct():+.2f}%")
        print(f"    Consecutive losses: {self.consecutive_losses}")
        print(f"    Daily trades: {len(self.daily_trades)}")

    def get_status(self) -> Dict:
        """Get current session status."""
        return {
            "initial_capital": self.initial_capital,
            "current_capital": self.current_capital,
            "peak_capital": self.peak_capital,
            "drawdown_pct": self.get_drawdown_pct(),
            "session_pnl": self.session_pnl,
            "session_pnl_pct": self.get_session_pnl_pct(),
            "session_trades": len(self.session_trades),
            "daily_pnl": self.daily_pnl,
            "daily_pnl_pct": self.get_daily_pnl_pct(),
            "daily_trades": len(self.daily_trades),
            "consecutive_losses": self.consecutive_losses,
            "is_paused": self.is_paused,
            "pause_reason": self.pause_reason,
            "pause_until": self.pause_until.isoformat() if self.pause_until else None,
        }

    def print_status(self):
        """Print current session status."""
        status = self.get_status()

        print("\n" + "=" * 50)
        print("SESSION STATUS")
        print("=" * 50)

        # Capital
        print(f"\n[CAPITAL]")
        print(f"  Initial:  ${status['initial_capital']:,.2f}")
        print(f"  Current:  ${status['current_capital']:,.2f}")
        print(f"  Peak:     ${status['peak_capital']:,.2f}")
        print(f"  Drawdown: {status['drawdown_pct']:.2f}%", end="")
        if status['drawdown_pct'] >= self.max_drawdown_pct * 0.7:
            print(" WARNING", end="")
        print()

        # Session
        print(f"\n[SESSION]")
        print(f"  P&L: ${status['session_pnl']:,.2f} ({status['session_pnl_pct']:+.2f}%)")
        print(f"  Trades: {status['session_trades']}")
        print(f"  Consecutive Losses: {status['consecutive_losses']}", end="")
        if status['consecutive_losses'] >= self.max_consecutive_losses - 1:
            print(" WARNING", end="")
        print()

        # Daily
        print(f"\n[DAILY]")
        print(f"  P&L: ${status['daily_pnl']:,.2f} ({status['daily_pnl_pct']:+.2f}%)")
        print(f"  Trades: {status['daily_trades']}/{self.max_daily_trades}")

        # Status
        print(f"\n[STATUS]")
        if status['is_paused']:
            print(f"  TRADING PAUSED: {status['pause_reason']}")
            if status['pause_until']:
                print(f"  Resume at: {status['pause_until']}")
        else:
            can_trade, reason, _ = self.can_trade()
            if can_trade:
                print(f"  Trading: ACTIVE")
            else:
                print(f"  Trading: BLOCKED - {reason}")

        # Limits
        print(f"\n[LIMITS]")
        print(f"  Max Drawdown: {self.max_drawdown_pct}%")
        print(f"  Max Session Loss: {self.max_session_loss_pct}%")
        print(f"  Max Daily Loss: {self.max_daily_loss_pct}%")
        print(f"  Max Consecutive Losses: {self.max_consecutive_losses}")
        print(f"  Max Daily Trades: {self.max_daily_trades}")
        print(f"  Min Trade Interval: {self.min_trade_interval_minutes} min")

        print("=" * 50)

    def reset_session(self):
        """Reset session tracking."""
        self.session_start = datetime.datetime.utcnow()
        self.session_trades = []
        self.session_pnl = 0.0
        # Keep consecutive losses and capital
        self._save_state()
        print("  [SESSION] Session reset")

    def reset_pause(self):
        """Manually reset pause state."""
        self.is_paused = False
        self.pause_reason = ""
        self.pause_until = None
        self._save_state()
        print("  [SESSION] Pause reset - trading enabled")


def create_trade_record(
    symbol: str,
    side: str,
    entry_price: float,
    exit_price: float,
    position_size: float,
    signals: List[str],
    exit_reason: str
) -> TradeRecord:
    """Helper to create a trade record."""
    pnl_percent = ((exit_price - entry_price) / entry_price) * 100
    if side.upper() == "SELL":
        pnl_percent = -pnl_percent
    pnl_absolute = position_size * (pnl_percent / 100)

    return TradeRecord(
        timestamp=datetime.datetime.utcnow(),
        symbol=symbol,
        side=side.upper(),
        entry_price=entry_price,
        exit_price=exit_price,
        pnl_percent=pnl_percent,
        pnl_absolute=pnl_absolute,
        position_size=position_size,
        signals=signals,
        exit_reason=exit_reason
    )


# =============================================================================
# MAIN - Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SESSION TRACKER TEST")
    print("=" * 60)

    # Create tracker
    tracker = SessionTracker(initial_capital=10000.0)
    tracker.print_status()

    # Test trade recording
    print("\n[SIMULATING TRADES]")
    print("-" * 40)

    # Simulate some wins
    for i in range(2):
        trade = create_trade_record(
            symbol="ATOMUSDT",
            side="BUY",
            entry_price=10.0,
            exit_price=10.3,  # 3% win
            position_size=200,
            signals=["BB", "MACD"],
            exit_reason="TP"
        )
        tracker.record_trade(trade)

        can_trade, reason, details = tracker.can_trade()
        print(f"  Can trade: {can_trade} - {reason}")

    # Simulate losses to trigger pause
    print("\n[SIMULATING CONSECUTIVE LOSSES]")
    print("-" * 40)

    for i in range(4):
        trade = create_trade_record(
            symbol="ATOMUSDT",
            side="BUY",
            entry_price=10.0,
            exit_price=9.8,  # 2% loss
            position_size=200,
            signals=["BB", "MACD"],
            exit_reason="SL"
        )
        tracker.record_trade(trade)

        can_trade, reason, details = tracker.can_trade()
        print(f"  Can trade: {can_trade}")
        if not can_trade:
            print(f"    Reason: {reason}")
            break

    print("\n[FINAL STATUS]")
    tracker.print_status()
