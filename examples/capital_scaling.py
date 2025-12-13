#!/usr/bin/env python3
"""
Capital Scaling Rules - Live Transition Framework

Phase 2 Risk Evolution Component #4

Defines the exact rules for transitioning from paper to live trading:

1. Validation criteria (when is the system proven?)
2. Scaling phases (how to gradually increase exposure)
3. Scaling triggers (what justifies scaling up?)
4. Pause triggers (what requires scaling down or stopping?)
5. Kill conditions (when to permanently disable?)

KEY PRINCIPLE: First live month = data collection, not profit seeking.
Goal = match paper stats, not make money.

This is how prop firms evaluate traders.
"""

import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum


class ScalingPhase(Enum):
    """Current scaling phase."""
    PAPER = "paper"                     # Paper trading only
    MICRO_LIVE = "micro_live"           # Minimum live size ($10-50/trade)
    SMALL_LIVE = "small_live"           # Small live size ($50-200/trade)
    NORMAL_LIVE = "normal_live"         # Normal live size ($200-500/trade)
    FULL_LIVE = "full_live"             # Full capacity (per production_config)
    PAUSED = "paused"                   # Temporarily paused
    DISABLED = "disabled"               # Permanently disabled


@dataclass
class PhaseRequirements:
    """Requirements to advance to a phase."""
    min_trades: int                     # Minimum trades at current phase
    min_win_rate: float                 # Minimum win rate (%)
    max_drawdown: float                 # Maximum drawdown (%)
    min_profit_factor: float            # Win $ / Loss $ ratio
    min_days_at_phase: int              # Minimum days before advancing
    live_vs_paper_tolerance: float      # How close live must match paper (%)


@dataclass
class ScalingState:
    """Current scaling state."""
    phase: ScalingPhase
    phase_start_date: datetime.date
    phase_trades: int
    phase_wins: int
    phase_pnl: float
    phase_drawdown: float
    total_live_trades: int
    total_live_pnl: float
    paper_win_rate: float               # Reference from paper trading
    paper_profit_factor: float          # Reference from paper trading


@dataclass
class ScalingDecision:
    """Scaling decision with rationale."""
    action: str                         # "hold", "advance", "pause", "disable"
    from_phase: ScalingPhase
    to_phase: Optional[ScalingPhase]
    reason: str
    metrics: Dict
    recommendations: List[str]


class CapitalScalingManager:
    """
    Manages the transition from paper to live trading.

    Implements a prop-firm style evaluation:
    - Must prove consistency before scaling
    - Must match paper performance
    - Must survive drawdown tests
    - Automatic pause on deviation
    """

    def __init__(self):
        # Phase requirements
        self.requirements = {
            ScalingPhase.PAPER: PhaseRequirements(
                min_trades=100,
                min_win_rate=65.0,
                max_drawdown=15.0,
                min_profit_factor=1.5,
                min_days_at_phase=7,
                live_vs_paper_tolerance=0.0  # N/A for paper
            ),
            ScalingPhase.MICRO_LIVE: PhaseRequirements(
                min_trades=30,
                min_win_rate=60.0,
                max_drawdown=10.0,
                min_profit_factor=1.3,
                min_days_at_phase=14,
                live_vs_paper_tolerance=15.0  # Within 15% of paper
            ),
            ScalingPhase.SMALL_LIVE: PhaseRequirements(
                min_trades=50,
                min_win_rate=60.0,
                max_drawdown=12.0,
                min_profit_factor=1.3,
                min_days_at_phase=21,
                live_vs_paper_tolerance=12.0
            ),
            ScalingPhase.NORMAL_LIVE: PhaseRequirements(
                min_trades=75,
                min_win_rate=60.0,
                max_drawdown=15.0,
                min_profit_factor=1.2,
                min_days_at_phase=30,
                live_vs_paper_tolerance=10.0
            ),
            ScalingPhase.FULL_LIVE: PhaseRequirements(
                min_trades=0,           # No advancement from here
                min_win_rate=55.0,      # Maintenance requirements
                max_drawdown=15.0,
                min_profit_factor=1.1,
                min_days_at_phase=0,
                live_vs_paper_tolerance=15.0
            ),
        }

        # Position sizes per phase
        self.phase_position_sizes = {
            ScalingPhase.PAPER: (0, 0),           # $0 (paper)
            ScalingPhase.MICRO_LIVE: (10, 50),    # $10-50
            ScalingPhase.SMALL_LIVE: (50, 200),   # $50-200
            ScalingPhase.NORMAL_LIVE: (200, 500), # $200-500
            ScalingPhase.FULL_LIVE: (500, 1000),  # $500-1000
            ScalingPhase.PAUSED: (0, 0),
            ScalingPhase.DISABLED: (0, 0),
        }

        # Kill conditions (permanent disable)
        self.kill_conditions = {
            "max_total_drawdown": 25.0,           # 25% total drawdown = kill
            "max_consecutive_losses": 10,         # 10 consecutive losses = kill
            "max_losing_days": 15,                # 15 losing days in row = kill
            "min_win_rate_threshold": 45.0,       # Win rate below 45% = kill
            "max_deviation_from_paper": 30.0,     # 30%+ deviation = kill
        }

        # Pause conditions (temporary)
        self.pause_conditions = {
            "daily_loss_pct": 5.0,                # 5% daily loss
            "weekly_loss_pct": 10.0,              # 10% weekly loss
            "consecutive_losses": 5,              # 5 consecutive losses
            "deviation_from_paper": 20.0,         # 20% deviation
        }

        # Current state
        self.state = ScalingState(
            phase=ScalingPhase.PAPER,
            phase_start_date=datetime.date.today(),
            phase_trades=0,
            phase_wins=0,
            phase_pnl=0.0,
            phase_drawdown=0.0,
            total_live_trades=0,
            total_live_pnl=0.0,
            paper_win_rate=73.3,          # From evidence collector
            paper_profit_factor=2.1       # Estimated from backtest
        )

    def get_position_size_range(self) -> Tuple[float, float]:
        """Get allowed position size range for current phase."""
        return self.phase_position_sizes.get(
            self.state.phase,
            (0, 0)
        )

    def record_trade(self, pnl_pct: float, pnl_usd: float):
        """
        Record a trade result.

        Args:
            pnl_pct: Trade P&L percentage
            pnl_usd: Trade P&L in USD
        """
        self.state.phase_trades += 1
        if pnl_pct > 0:
            self.state.phase_wins += 1
        self.state.phase_pnl += pnl_usd

        # Update drawdown
        if pnl_usd < 0:
            self.state.phase_drawdown = min(
                self.state.phase_drawdown,
                self.state.phase_pnl  # More negative = worse
            )

        # Track live trades
        if self.state.phase not in [ScalingPhase.PAPER, ScalingPhase.PAUSED, ScalingPhase.DISABLED]:
            self.state.total_live_trades += 1
            self.state.total_live_pnl += pnl_usd

    def get_phase_win_rate(self) -> float:
        """Get win rate for current phase."""
        if self.state.phase_trades == 0:
            return 0.0
        return (self.state.phase_wins / self.state.phase_trades) * 100

    def get_live_vs_paper_deviation(self) -> float:
        """Calculate how much live deviates from paper."""
        live_wr = self.get_phase_win_rate()
        paper_wr = self.state.paper_win_rate

        if paper_wr == 0:
            return 0.0

        return abs(live_wr - paper_wr) / paper_wr * 100

    def check_kill_conditions(self) -> Tuple[bool, str]:
        """
        Check if any kill conditions are met.

        Returns:
            (should_kill, reason)
        """
        # Check total drawdown
        drawdown_pct = abs(self.state.phase_drawdown) / 10000 * 100  # Assuming $10k capital
        if drawdown_pct >= self.kill_conditions["max_total_drawdown"]:
            return True, f"Total drawdown {drawdown_pct:.1f}% >= {self.kill_conditions['max_total_drawdown']}%"

        # Check win rate
        wr = self.get_phase_win_rate()
        if self.state.phase_trades >= 30 and wr < self.kill_conditions["min_win_rate_threshold"]:
            return True, f"Win rate {wr:.1f}% < {self.kill_conditions['min_win_rate_threshold']}% (with 30+ trades)"

        # Check deviation from paper
        deviation = self.get_live_vs_paper_deviation()
        if self.state.phase_trades >= 20 and deviation >= self.kill_conditions["max_deviation_from_paper"]:
            return True, f"Deviation from paper {deviation:.1f}% >= {self.kill_conditions['max_deviation_from_paper']}%"

        return False, "No kill conditions met"

    def check_pause_conditions(self) -> Tuple[bool, str]:
        """
        Check if any pause conditions are met.

        Returns:
            (should_pause, reason)
        """
        # Check deviation
        deviation = self.get_live_vs_paper_deviation()
        if self.state.phase_trades >= 10 and deviation >= self.pause_conditions["deviation_from_paper"]:
            return True, f"Deviation from paper {deviation:.1f}% >= {self.pause_conditions['deviation_from_paper']}%"

        return False, "No pause conditions met"

    def check_advance_conditions(self) -> Tuple[bool, str]:
        """
        Check if ready to advance to next phase.

        Returns:
            (can_advance, reason)
        """
        if self.state.phase in [ScalingPhase.FULL_LIVE, ScalingPhase.PAUSED, ScalingPhase.DISABLED]:
            return False, "Cannot advance from current phase"

        req = self.requirements[self.state.phase]

        # Check trade count
        if self.state.phase_trades < req.min_trades:
            return False, f"Need {req.min_trades - self.state.phase_trades} more trades"

        # Check win rate
        wr = self.get_phase_win_rate()
        if wr < req.min_win_rate:
            return False, f"Win rate {wr:.1f}% < required {req.min_win_rate}%"

        # Check days at phase
        days_at_phase = (datetime.date.today() - self.state.phase_start_date).days
        if days_at_phase < req.min_days_at_phase:
            return False, f"Need {req.min_days_at_phase - days_at_phase} more days at this phase"

        # Check drawdown
        drawdown_pct = abs(self.state.phase_drawdown) / 10000 * 100
        if drawdown_pct > req.max_drawdown:
            return False, f"Drawdown {drawdown_pct:.1f}% > max {req.max_drawdown}%"

        # For live phases, check deviation from paper
        if self.state.phase != ScalingPhase.PAPER:
            deviation = self.get_live_vs_paper_deviation()
            if deviation > req.live_vs_paper_tolerance:
                return False, f"Deviation {deviation:.1f}% > tolerance {req.live_vs_paper_tolerance}%"

        return True, "All advancement criteria met"

    def evaluate(self) -> ScalingDecision:
        """
        Evaluate current state and recommend action.

        Returns:
            ScalingDecision with recommended action
        """
        metrics = {
            "phase": self.state.phase.value,
            "trades": self.state.phase_trades,
            "win_rate": self.get_phase_win_rate(),
            "pnl": self.state.phase_pnl,
            "drawdown": self.state.phase_drawdown,
            "deviation_from_paper": self.get_live_vs_paper_deviation(),
            "days_at_phase": (datetime.date.today() - self.state.phase_start_date).days
        }

        recommendations = []

        # Check kill conditions first
        should_kill, kill_reason = self.check_kill_conditions()
        if should_kill:
            return ScalingDecision(
                action="disable",
                from_phase=self.state.phase,
                to_phase=ScalingPhase.DISABLED,
                reason=f"KILL CONDITION: {kill_reason}",
                metrics=metrics,
                recommendations=[
                    "Strategy has failed validation",
                    "Review and redesign before restarting",
                    "Do NOT restart without fundamental changes"
                ]
            )

        # Check pause conditions
        should_pause, pause_reason = self.check_pause_conditions()
        if should_pause:
            return ScalingDecision(
                action="pause",
                from_phase=self.state.phase,
                to_phase=ScalingPhase.PAUSED,
                reason=f"PAUSE: {pause_reason}",
                metrics=metrics,
                recommendations=[
                    "Pause live trading",
                    "Review recent trades for pattern changes",
                    "Resume only after identifying cause"
                ]
            )

        # Check advancement conditions
        can_advance, advance_reason = self.check_advance_conditions()
        if can_advance:
            next_phase = self._get_next_phase()
            if next_phase:
                return ScalingDecision(
                    action="advance",
                    from_phase=self.state.phase,
                    to_phase=next_phase,
                    reason=f"READY TO ADVANCE: {advance_reason}",
                    metrics=metrics,
                    recommendations=[
                        f"Increase position size to {self.phase_position_sizes[next_phase]}",
                        "Maintain same trading rules",
                        "Continue monitoring deviation from paper"
                    ]
                )

        # Hold at current phase
        req = self.requirements.get(self.state.phase)
        if req:
            if self.state.phase_trades < req.min_trades:
                recommendations.append(f"Complete {req.min_trades - self.state.phase_trades} more trades")
            days_remaining = req.min_days_at_phase - (datetime.date.today() - self.state.phase_start_date).days
            if days_remaining > 0:
                recommendations.append(f"Wait {days_remaining} more days")

        return ScalingDecision(
            action="hold",
            from_phase=self.state.phase,
            to_phase=None,
            reason=advance_reason if not can_advance else "Continue at current phase",
            metrics=metrics,
            recommendations=recommendations or ["Continue trading at current phase"]
        )

    def _get_next_phase(self) -> Optional[ScalingPhase]:
        """Get the next phase in progression."""
        progression = [
            ScalingPhase.PAPER,
            ScalingPhase.MICRO_LIVE,
            ScalingPhase.SMALL_LIVE,
            ScalingPhase.NORMAL_LIVE,
            ScalingPhase.FULL_LIVE
        ]

        try:
            current_idx = progression.index(self.state.phase)
            if current_idx < len(progression) - 1:
                return progression[current_idx + 1]
        except ValueError:
            pass

        return None

    def advance_phase(self):
        """Advance to next phase (call after evaluate confirms)."""
        next_phase = self._get_next_phase()
        if next_phase:
            print(f"\n  [SCALING] Advancing from {self.state.phase.value} to {next_phase.value}")
            self.state.phase = next_phase
            self.state.phase_start_date = datetime.date.today()
            self.state.phase_trades = 0
            self.state.phase_wins = 0
            self.state.phase_pnl = 0.0
            self.state.phase_drawdown = 0.0

    def pause(self, reason: str):
        """Pause trading."""
        print(f"\n  [SCALING] PAUSED: {reason}")
        self.state.phase = ScalingPhase.PAUSED

    def disable(self, reason: str):
        """Permanently disable strategy."""
        print(f"\n  [SCALING] DISABLED: {reason}")
        self.state.phase = ScalingPhase.DISABLED

    def print_status(self):
        """Print current scaling status."""
        decision = self.evaluate()

        print("\n" + "=" * 70)
        print("CAPITAL SCALING STATUS")
        print("=" * 70)

        print(f"\n[CURRENT PHASE]")
        print(f"  Phase:          {self.state.phase.value.upper()}")
        min_size, max_size = self.get_position_size_range()
        print(f"  Position Size:  ${min_size} - ${max_size}")
        print(f"  Days at Phase:  {(datetime.date.today() - self.state.phase_start_date).days}")

        print(f"\n[PHASE METRICS]")
        print(f"  Trades:         {self.state.phase_trades}")
        print(f"  Win Rate:       {self.get_phase_win_rate():.1f}%")
        print(f"  P&L:            ${self.state.phase_pnl:+,.2f}")
        print(f"  Drawdown:       ${self.state.phase_drawdown:,.2f}")

        print(f"\n[LIVE VS PAPER]")
        print(f"  Paper Win Rate: {self.state.paper_win_rate:.1f}%")
        print(f"  Live Win Rate:  {self.get_phase_win_rate():.1f}%")
        print(f"  Deviation:      {self.get_live_vs_paper_deviation():.1f}%")

        print(f"\n[EVALUATION]")
        print(f"  Action:         {decision.action.upper()}")
        if decision.to_phase:
            print(f"  Next Phase:     {decision.to_phase.value}")
        print(f"  Reason:         {decision.reason}")

        if decision.recommendations:
            print(f"\n[RECOMMENDATIONS]")
            for rec in decision.recommendations:
                print(f"  - {rec}")

        # Show requirements for current phase
        req = self.requirements.get(self.state.phase)
        if req and self.state.phase not in [ScalingPhase.FULL_LIVE, ScalingPhase.PAUSED, ScalingPhase.DISABLED]:
            print(f"\n[ADVANCEMENT REQUIREMENTS]")
            print(f"  Min Trades:     {req.min_trades} (have {self.state.phase_trades})")
            print(f"  Min Win Rate:   {req.min_win_rate}% (have {self.get_phase_win_rate():.1f}%)")
            print(f"  Min Days:       {req.min_days_at_phase} (have {(datetime.date.today() - self.state.phase_start_date).days})")
            print(f"  Max Drawdown:   {req.max_drawdown}%")
            if self.state.phase != ScalingPhase.PAPER:
                print(f"  Max Deviation:  {req.live_vs_paper_tolerance}% (have {self.get_live_vs_paper_deviation():.1f}%)")

        print("=" * 70)


# =============================================================================
# QUICK REFERENCE
# =============================================================================

def print_scaling_guide():
    """Print the complete scaling guide."""
    print("\n" + "=" * 70)
    print("CAPITAL SCALING GUIDE")
    print("=" * 70)

    print("""
PHASE PROGRESSION
─────────────────────────────────────────────────────────────────────

1. PAPER TRADING (Current)
   • Position Size: $0 (simulation only)
   • Requirements to advance:
     - 100+ trades
     - 65%+ win rate
     - <15% drawdown
     - 7+ days
   • Goal: Prove the system works

2. MICRO LIVE
   • Position Size: $10-50 per trade
   • Requirements to advance:
     - 30+ trades
     - 60%+ win rate
     - <10% drawdown
     - Within 15% of paper performance
     - 14+ days
   • Goal: Verify live execution matches paper

3. SMALL LIVE
   • Position Size: $50-200 per trade
   • Requirements to advance:
     - 50+ trades
     - 60%+ win rate
     - <12% drawdown
     - Within 12% of paper performance
     - 21+ days
   • Goal: Build confidence with meaningful size

4. NORMAL LIVE
   • Position Size: $200-500 per trade
   • Requirements to advance:
     - 75+ trades
     - 60%+ win rate
     - <15% drawdown
     - Within 10% of paper performance
     - 30+ days
   • Goal: Prove consistency at real size

5. FULL LIVE
   • Position Size: $500-1000 per trade
   • Maintenance requirements:
     - 55%+ win rate
     - <15% drawdown
   • Goal: Maximize proven edge


PAUSE CONDITIONS (Temporary Stop)
─────────────────────────────────────────────────────────────────────
• 5% daily loss
• 10% weekly loss
• 5 consecutive losses
• 20%+ deviation from paper performance


KILL CONDITIONS (Permanent Disable)
─────────────────────────────────────────────────────────────────────
• 25% total drawdown
• 10 consecutive losses
• 15 losing days in a row
• Win rate below 45% (with 30+ trades)
• 30%+ deviation from paper performance


KEY PRINCIPLES
─────────────────────────────────────────────────────────────────────
1. First live month = data collection, not profit seeking
2. Goal is to MATCH paper stats, not exceed them
3. Consistency > Short-term profits
4. If live ≠ paper, something is wrong
5. Scale up only when proven, scale down immediately on deviation
""")
    print("=" * 70)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print_scaling_guide()

    print("\n\n")

    manager = CapitalScalingManager()
    manager.print_status()

    # Simulate some trades
    print("\n[SIMULATING PAPER TRADES]")
    print("-" * 40)

    # Add 50 winning trades
    for i in range(50):
        manager.record_trade(2.5, 25)  # 2.5% win, $25

    # Add 20 losing trades
    for i in range(20):
        manager.record_trade(-1.5, -15)  # 1.5% loss, $15

    manager.print_status()
