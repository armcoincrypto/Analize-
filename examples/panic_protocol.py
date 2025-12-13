#!/usr/bin/env python3
"""
Panic Protocol - Emergency Response for Live Trading

When things go wrong in live trading, you need IMMEDIATE action.
This protocol defines exactly what to do in each emergency scenario.

RULE: When in doubt, CLOSE EVERYTHING and STOP TRADING.

"In a crisis, the cost of inaction always exceeds the cost of action."
"""

import os
import sys
import json
import time
import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path
from enum import Enum


class PanicLevel(Enum):
    """Emergency severity levels."""
    WATCH = "watch"           # Monitor closely
    WARNING = "warning"       # Reduce exposure
    CRITICAL = "critical"     # Close new positions
    EMERGENCY = "emergency"   # Close ALL positions
    SHUTDOWN = "shutdown"     # Full system shutdown


@dataclass
class PanicTrigger:
    """Definition of a panic trigger."""
    name: str
    level: PanicLevel
    condition: str
    action: str
    automated: bool  # Can be automated or requires human


class PanicProtocol:
    """
    Emergency response protocol for live trading.

    Defines:
    1. Trigger conditions
    2. Response actions
    3. Recovery procedures

    CRITICAL: This runs INDEPENDENTLY of the main trading system.
    If main system fails, panic protocol must still work.
    """

    def __init__(self, state_file: str = None):
        if state_file is None:
            state_file = str(Path(__file__).parent / "panic_state.json")
        self.state_file = state_file

        # Current state
        self.current_level = PanicLevel.WATCH
        self.active_triggers: List[str] = []
        self.shutdown_time: Optional[datetime.datetime] = None

        # Define all panic triggers
        self.triggers = self._define_triggers()

        # Load persisted state
        self._load_state()

    def _define_triggers(self) -> Dict[str, PanicTrigger]:
        """Define all panic triggers and responses."""
        return {
            # WATCH level - monitor closely
            "drawdown_5pct": PanicTrigger(
                name="5% Drawdown",
                level=PanicLevel.WATCH,
                condition="Account drawdown reaches 5%",
                action="Log alert, reduce position sizes by 25%",
                automated=True
            ),
            "consecutive_losses_3": PanicTrigger(
                name="3 Consecutive Losses",
                level=PanicLevel.WATCH,
                condition="3 trades in a row lose",
                action="Pause for 30 minutes, review recent trades",
                automated=True
            ),

            # WARNING level - reduce exposure
            "drawdown_10pct": PanicTrigger(
                name="10% Drawdown",
                level=PanicLevel.WARNING,
                condition="Account drawdown reaches 10%",
                action="Reduce all position sizes by 50%, no new positions",
                automated=True
            ),
            "consecutive_losses_5": PanicTrigger(
                name="5 Consecutive Losses",
                level=PanicLevel.WARNING,
                condition="5 trades in a row lose",
                action="Pause trading for 4 hours",
                automated=True
            ),
            "btc_crash_5pct": PanicTrigger(
                name="BTC -5% Flash",
                level=PanicLevel.WARNING,
                condition="BTC drops 5% in 1 hour",
                action="Tighten all stops, no new positions",
                automated=True
            ),

            # CRITICAL level - close new positions
            "drawdown_15pct": PanicTrigger(
                name="15% Drawdown (KILL SWITCH)",
                level=PanicLevel.CRITICAL,
                condition="Account drawdown reaches 15%",
                action="Close all new positions, keep existing with tight stops",
                automated=True
            ),
            "consecutive_losses_7": PanicTrigger(
                name="7 Consecutive Losses",
                level=PanicLevel.CRITICAL,
                condition="7 trades in a row lose",
                action="Stop all trading for 24 hours",
                automated=True
            ),
            "btc_crash_10pct": PanicTrigger(
                name="BTC -10% Crash",
                level=PanicLevel.CRITICAL,
                condition="BTC drops 10% in 24 hours",
                action="Circuit breaker - no trading for 24 hours",
                automated=True
            ),
            "api_errors": PanicTrigger(
                name="API Errors",
                level=PanicLevel.CRITICAL,
                condition="3+ failed API calls in 5 minutes",
                action="Stop trading, switch to manual mode",
                automated=True
            ),

            # EMERGENCY level - close ALL positions
            "drawdown_20pct": PanicTrigger(
                name="20% Drawdown (EMERGENCY)",
                level=PanicLevel.EMERGENCY,
                condition="Account drawdown reaches 20%",
                action="CLOSE ALL POSITIONS IMMEDIATELY",
                automated=True
            ),
            "flash_crash": PanicTrigger(
                name="Flash Crash Detected",
                level=PanicLevel.EMERGENCY,
                condition="Price moves 15%+ in 5 minutes",
                action="CLOSE ALL POSITIONS, pause 24 hours",
                automated=True
            ),
            "exchange_issues": PanicTrigger(
                name="Exchange Problems",
                level=PanicLevel.EMERGENCY,
                condition="Exchange maintenance, unusual behavior",
                action="CLOSE ALL POSITIONS while possible",
                automated=False  # Requires human judgment
            ),

            # SHUTDOWN level - full system stop
            "drawdown_25pct": PanicTrigger(
                name="25% Drawdown (SHUTDOWN)",
                level=PanicLevel.SHUTDOWN,
                condition="Account drawdown reaches 25%",
                action="SHUTDOWN: Close all, disable system, require manual restart",
                automated=True
            ),
            "system_failure": PanicTrigger(
                name="System Failure",
                level=PanicLevel.SHUTDOWN,
                condition="Multiple system components fail",
                action="SHUTDOWN: Full system stop, manual intervention required",
                automated=True
            ),
            "security_breach": PanicTrigger(
                name="Security Concern",
                level=PanicLevel.SHUTDOWN,
                condition="Unauthorized access suspected",
                action="SHUTDOWN: Disable API keys, close all, investigate",
                automated=False
            ),
        }

    def _load_state(self):
        """Load persisted panic state."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    state = json.load(f)
                self.current_level = PanicLevel(state.get("level", "watch"))
                self.active_triggers = state.get("active_triggers", [])
                if state.get("shutdown_time"):
                    self.shutdown_time = datetime.datetime.fromisoformat(state["shutdown_time"])
            except Exception as e:
                print(f"Warning: Could not load panic state: {e}")

    def _save_state(self):
        """Persist panic state."""
        try:
            state = {
                "level": self.current_level.value,
                "active_triggers": self.active_triggers,
                "shutdown_time": self.shutdown_time.isoformat() if self.shutdown_time else None,
                "updated": datetime.datetime.utcnow().isoformat()
            }
            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save panic state: {e}")

    def trigger(self, trigger_name: str) -> Dict:
        """
        Trigger a panic response.

        Returns:
            Dict with action details
        """
        if trigger_name not in self.triggers:
            return {"error": f"Unknown trigger: {trigger_name}"}

        trigger = self.triggers[trigger_name]

        # Add to active triggers
        if trigger_name not in self.active_triggers:
            self.active_triggers.append(trigger_name)

        # Update level if higher
        level_priority = {
            PanicLevel.WATCH: 1,
            PanicLevel.WARNING: 2,
            PanicLevel.CRITICAL: 3,
            PanicLevel.EMERGENCY: 4,
            PanicLevel.SHUTDOWN: 5
        }

        if level_priority[trigger.level] > level_priority[self.current_level]:
            self.current_level = trigger.level

        # Record shutdown time
        if trigger.level == PanicLevel.SHUTDOWN:
            self.shutdown_time = datetime.datetime.utcnow()

        self._save_state()

        # Print alert
        self._print_alert(trigger)

        return {
            "trigger": trigger_name,
            "level": trigger.level.value,
            "action": trigger.action,
            "automated": trigger.automated,
            "current_level": self.current_level.value
        }

    def _print_alert(self, trigger: PanicTrigger):
        """Print formatted alert."""
        level_symbols = {
            PanicLevel.WATCH: "👀",
            PanicLevel.WARNING: "⚠️",
            PanicLevel.CRITICAL: "🔴",
            PanicLevel.EMERGENCY: "🚨",
            PanicLevel.SHUTDOWN: "💀"
        }

        symbol = level_symbols.get(trigger.level, "!")
        border = "!" * 60 if trigger.level in [PanicLevel.EMERGENCY, PanicLevel.SHUTDOWN] else "=" * 60

        print(f"\n{border}")
        print(f"  {symbol} PANIC ALERT: {trigger.name}")
        print(f"  Level: {trigger.level.value.upper()}")
        print(f"  Condition: {trigger.condition}")
        print(f"  Action: {trigger.action}")
        print(f"  Automated: {'YES' if trigger.automated else 'REQUIRES HUMAN'}")
        print(f"{border}\n")

    def clear_trigger(self, trigger_name: str):
        """Clear a specific trigger."""
        if trigger_name in self.active_triggers:
            self.active_triggers.remove(trigger_name)
            self._recalculate_level()
            self._save_state()

    def _recalculate_level(self):
        """Recalculate current level from active triggers."""
        level_priority = {
            PanicLevel.WATCH: 1,
            PanicLevel.WARNING: 2,
            PanicLevel.CRITICAL: 3,
            PanicLevel.EMERGENCY: 4,
            PanicLevel.SHUTDOWN: 5
        }

        max_level = PanicLevel.WATCH
        for trigger_name in self.active_triggers:
            if trigger_name in self.triggers:
                trigger = self.triggers[trigger_name]
                if level_priority[trigger.level] > level_priority[max_level]:
                    max_level = trigger.level

        self.current_level = max_level

    def reset(self, confirm: bool = False):
        """
        Reset panic protocol to normal.

        Requires confirmation for safety.
        """
        if not confirm:
            print("WARNING: Reset requires confirm=True")
            return False

        self.current_level = PanicLevel.WATCH
        self.active_triggers = []
        self.shutdown_time = None
        self._save_state()

        print("\n[PANIC PROTOCOL RESET]")
        print("  Level: WATCH (normal)")
        print("  Active Triggers: None")

        return True

    def get_actions(self) -> List[str]:
        """Get list of required actions based on current state."""
        actions = []

        if self.current_level == PanicLevel.WATCH:
            actions.append("Continue normal operations")
            actions.append("Monitor metrics closely")

        elif self.current_level == PanicLevel.WARNING:
            actions.append("Reduce position sizes by 50%")
            actions.append("Do not open new positions")
            actions.append("Tighten all stop losses")
            actions.append("Review recent trade performance")

        elif self.current_level == PanicLevel.CRITICAL:
            actions.append("STOP all new trading")
            actions.append("Close positions in profit")
            actions.append("Tighten stops on remaining positions")
            actions.append("Prepare for possible full close")
            actions.append("Manual review required before resuming")

        elif self.current_level == PanicLevel.EMERGENCY:
            actions.append("CLOSE ALL POSITIONS NOW")
            actions.append("Cancel all pending orders")
            actions.append("Disable automated trading")
            actions.append("Move funds to safety if needed")
            actions.append("Do not resume without investigation")

        elif self.current_level == PanicLevel.SHUTDOWN:
            actions.append("SYSTEM SHUTDOWN - ALL TRADING STOPPED")
            actions.append("Revoke API keys if security concern")
            actions.append("Full system audit required")
            actions.append("Manual restart only after review")
            actions.append("Consider permanent strategy retirement")

        return actions

    def can_trade(self) -> tuple[bool, str]:
        """Check if trading is allowed under current panic level."""
        if self.current_level == PanicLevel.WATCH:
            return True, "Normal operations"

        if self.current_level == PanicLevel.WARNING:
            return True, "Trading allowed with reduced size"

        if self.current_level == PanicLevel.CRITICAL:
            return False, "No new trades - close only"

        if self.current_level == PanicLevel.EMERGENCY:
            return False, "EMERGENCY - close all positions"

        if self.current_level == PanicLevel.SHUTDOWN:
            return False, "SHUTDOWN - system disabled"

        return False, "Unknown state"

    def print_status(self):
        """Print current panic protocol status."""
        level_colors = {
            PanicLevel.WATCH: "GREEN",
            PanicLevel.WARNING: "YELLOW",
            PanicLevel.CRITICAL: "ORANGE",
            PanicLevel.EMERGENCY: "RED",
            PanicLevel.SHUTDOWN: "BLACK"
        }

        print("\n" + "=" * 60)
        print("PANIC PROTOCOL STATUS")
        print("=" * 60)

        print(f"\n[CURRENT LEVEL]")
        print(f"  Level: {self.current_level.value.upper()} ({level_colors.get(self.current_level)})")

        can_trade, reason = self.can_trade()
        print(f"  Trading: {'ALLOWED' if can_trade else 'BLOCKED'}")
        print(f"  Reason: {reason}")

        if self.active_triggers:
            print(f"\n[ACTIVE TRIGGERS]")
            for trigger_name in self.active_triggers:
                if trigger_name in self.triggers:
                    t = self.triggers[trigger_name]
                    print(f"  - {t.name} ({t.level.value})")

        print(f"\n[REQUIRED ACTIONS]")
        for action in self.get_actions():
            print(f"  • {action}")

        if self.shutdown_time:
            print(f"\n[SHUTDOWN INFO]")
            print(f"  Shutdown Time: {self.shutdown_time}")
            duration = datetime.datetime.utcnow() - self.shutdown_time
            print(f"  Duration: {duration}")

        print("=" * 60)

    def print_all_triggers(self):
        """Print all defined panic triggers."""
        print("\n" + "=" * 60)
        print("PANIC PROTOCOL - ALL TRIGGERS")
        print("=" * 60)

        current_level = None
        for name, trigger in sorted(self.triggers.items(), key=lambda x: x[1].level.value):
            if trigger.level != current_level:
                current_level = trigger.level
                print(f"\n[{current_level.value.upper()}]")
                print("-" * 40)

            auto = "AUTO" if trigger.automated else "MANUAL"
            print(f"\n  {trigger.name} [{auto}]")
            print(f"    Trigger: {trigger.condition}")
            print(f"    Action: {trigger.action}")

        print("\n" + "=" * 60)


# =============================================================================
# QUICK FUNCTIONS
# =============================================================================

def panic_close_all():
    """Emergency function to close all positions."""
    protocol = PanicProtocol()
    protocol.trigger("flash_crash")
    print("\n[EMERGENCY CLOSE INITIATED]")
    print("All positions should be closed immediately.")
    print("Trading is HALTED.")

def panic_shutdown():
    """Full system shutdown."""
    protocol = PanicProtocol()
    protocol.trigger("system_failure")
    print("\n[SYSTEM SHUTDOWN INITIATED]")
    print("Full manual review required before restart.")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys

    protocol = PanicProtocol()

    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()

        if cmd == "status":
            protocol.print_status()

        elif cmd == "triggers":
            protocol.print_all_triggers()

        elif cmd == "reset":
            protocol.reset(confirm=True)

        elif cmd == "test":
            # Test triggering
            print("[TEST MODE]")
            protocol.trigger("consecutive_losses_3")
            protocol.print_status()

        elif cmd == "emergency":
            panic_close_all()

        elif cmd == "shutdown":
            panic_shutdown()

        else:
            print(f"Unknown command: {cmd}")
            print("Commands: status, triggers, reset, test, emergency, shutdown")

    else:
        protocol.print_status()
        print("\nUsage: python panic_protocol.py [status|triggers|reset|test|emergency|shutdown]")
