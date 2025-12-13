#!/usr/bin/env python3
"""
Strategy Retirement Checker - When to Stop Trading a Strategy

Defines clear, objective criteria for when a strategy should be:
- Paused (temporary, fixable)
- Retired (permanent, strategy is dead)
- Investigated (concerning but not fatal)

"Knowing when to quit is more important than knowing when to trade."

This prevents:
- Trading a dead edge
- Losing money on hope
- Emotional attachment to failing strategies
"""

import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum


class StrategyStatus(Enum):
    """Current strategy status."""
    HEALTHY = "healthy"           # All systems normal
    INVESTIGATE = "investigate"   # Concerning trends
    PAUSE = "pause"               # Temporary stop
    RETIRE = "retire"             # Permanent stop


@dataclass
class HealthCheck:
    """Result of a single health check."""
    name: str
    status: StrategyStatus
    current_value: float
    threshold: float
    message: str


@dataclass
class RetirementAssessment:
    """Complete retirement assessment."""
    overall_status: StrategyStatus
    health_checks: List[HealthCheck]
    action_required: str
    investigation_items: List[str]
    reason: str


class StrategyRetirementChecker:
    """
    Evaluates whether a strategy should be retired.

    Retirement is PERMANENT. Before retiring:
    1. Confirm with multiple criteria
    2. Investigate possible causes
    3. Consider if fixable

    RETIRE only when edge is PROVEN DEAD, not just underperforming.
    """

    def __init__(self, reference_win_rate: float = 73.3, reference_pf: float = 2.1):
        """
        Initialize checker with reference performance.

        Args:
            reference_win_rate: Expected win rate from paper/backtest
            reference_pf: Expected profit factor from paper/backtest
        """
        self.reference_win_rate = reference_win_rate
        self.reference_pf = reference_pf

        # Retirement thresholds (PERMANENT)
        self.retirement_thresholds = {
            "max_drawdown_pct": 30.0,           # 30% drawdown = retire
            "min_win_rate": 45.0,               # Below 45% = retire
            "min_profit_factor": 0.8,           # Below 0.8 = retire
            "max_consecutive_losses": 15,       # 15 in a row = retire
            "max_deviation_from_paper": 40.0,   # 40% worse than paper = retire
            "min_trades_before_retire": 50,     # Need 50 trades before retirement valid
            "max_losing_months": 3,             # 3 consecutive losing months = retire
        }

        # Pause thresholds (TEMPORARY)
        self.pause_thresholds = {
            "max_drawdown_pct": 15.0,
            "min_win_rate": 55.0,
            "min_profit_factor": 1.0,
            "max_consecutive_losses": 7,
            "max_deviation_from_paper": 25.0,
        }

        # Investigation thresholds (WARNING)
        self.investigation_thresholds = {
            "max_drawdown_pct": 10.0,
            "min_win_rate": 60.0,
            "min_profit_factor": 1.3,
            "max_consecutive_losses": 5,
            "max_deviation_from_paper": 15.0,
        }

    def check_win_rate(self, current_wr: float, trade_count: int) -> HealthCheck:
        """Check win rate health."""
        if trade_count < 30:
            return HealthCheck(
                name="Win Rate",
                status=StrategyStatus.HEALTHY,
                current_value=current_wr,
                threshold=0,
                message=f"Need {30 - trade_count} more trades for valid assessment"
            )

        if current_wr < self.retirement_thresholds["min_win_rate"]:
            return HealthCheck(
                name="Win Rate",
                status=StrategyStatus.RETIRE,
                current_value=current_wr,
                threshold=self.retirement_thresholds["min_win_rate"],
                message=f"Win rate {current_wr:.1f}% below retirement threshold {self.retirement_thresholds['min_win_rate']}%"
            )

        if current_wr < self.pause_thresholds["min_win_rate"]:
            return HealthCheck(
                name="Win Rate",
                status=StrategyStatus.PAUSE,
                current_value=current_wr,
                threshold=self.pause_thresholds["min_win_rate"],
                message=f"Win rate {current_wr:.1f}% below pause threshold {self.pause_thresholds['min_win_rate']}%"
            )

        if current_wr < self.investigation_thresholds["min_win_rate"]:
            return HealthCheck(
                name="Win Rate",
                status=StrategyStatus.INVESTIGATE,
                current_value=current_wr,
                threshold=self.investigation_thresholds["min_win_rate"],
                message=f"Win rate {current_wr:.1f}% below expected {self.reference_win_rate:.1f}%"
            )

        return HealthCheck(
            name="Win Rate",
            status=StrategyStatus.HEALTHY,
            current_value=current_wr,
            threshold=self.investigation_thresholds["min_win_rate"],
            message=f"Win rate {current_wr:.1f}% is healthy"
        )

    def check_profit_factor(self, current_pf: float, trade_count: int) -> HealthCheck:
        """Check profit factor health."""
        if trade_count < 30:
            return HealthCheck(
                name="Profit Factor",
                status=StrategyStatus.HEALTHY,
                current_value=current_pf,
                threshold=0,
                message=f"Need {30 - trade_count} more trades for valid assessment"
            )

        if current_pf < self.retirement_thresholds["min_profit_factor"]:
            return HealthCheck(
                name="Profit Factor",
                status=StrategyStatus.RETIRE,
                current_value=current_pf,
                threshold=self.retirement_thresholds["min_profit_factor"],
                message=f"PF {current_pf:.2f} below retirement threshold {self.retirement_thresholds['min_profit_factor']}"
            )

        if current_pf < self.pause_thresholds["min_profit_factor"]:
            return HealthCheck(
                name="Profit Factor",
                status=StrategyStatus.PAUSE,
                current_value=current_pf,
                threshold=self.pause_thresholds["min_profit_factor"],
                message=f"PF {current_pf:.2f} below pause threshold {self.pause_thresholds['min_profit_factor']}"
            )

        if current_pf < self.investigation_thresholds["min_profit_factor"]:
            return HealthCheck(
                name="Profit Factor",
                status=StrategyStatus.INVESTIGATE,
                current_value=current_pf,
                threshold=self.investigation_thresholds["min_profit_factor"],
                message=f"PF {current_pf:.2f} below expected {self.reference_pf:.2f}"
            )

        return HealthCheck(
            name="Profit Factor",
            status=StrategyStatus.HEALTHY,
            current_value=current_pf,
            threshold=self.investigation_thresholds["min_profit_factor"],
            message=f"Profit factor {current_pf:.2f} is healthy"
        )

    def check_drawdown(self, current_dd: float) -> HealthCheck:
        """Check drawdown health."""
        if current_dd >= self.retirement_thresholds["max_drawdown_pct"]:
            return HealthCheck(
                name="Drawdown",
                status=StrategyStatus.RETIRE,
                current_value=current_dd,
                threshold=self.retirement_thresholds["max_drawdown_pct"],
                message=f"Drawdown {current_dd:.1f}% exceeds retirement limit {self.retirement_thresholds['max_drawdown_pct']}%"
            )

        if current_dd >= self.pause_thresholds["max_drawdown_pct"]:
            return HealthCheck(
                name="Drawdown",
                status=StrategyStatus.PAUSE,
                current_value=current_dd,
                threshold=self.pause_thresholds["max_drawdown_pct"],
                message=f"Drawdown {current_dd:.1f}% exceeds pause limit {self.pause_thresholds['max_drawdown_pct']}%"
            )

        if current_dd >= self.investigation_thresholds["max_drawdown_pct"]:
            return HealthCheck(
                name="Drawdown",
                status=StrategyStatus.INVESTIGATE,
                current_value=current_dd,
                threshold=self.investigation_thresholds["max_drawdown_pct"],
                message=f"Drawdown {current_dd:.1f}% approaching concerning levels"
            )

        return HealthCheck(
            name="Drawdown",
            status=StrategyStatus.HEALTHY,
            current_value=current_dd,
            threshold=self.investigation_thresholds["max_drawdown_pct"],
            message=f"Drawdown {current_dd:.1f}% is acceptable"
        )

    def check_consecutive_losses(self, streak: int) -> HealthCheck:
        """Check consecutive loss streak."""
        if streak >= self.retirement_thresholds["max_consecutive_losses"]:
            return HealthCheck(
                name="Loss Streak",
                status=StrategyStatus.RETIRE,
                current_value=streak,
                threshold=self.retirement_thresholds["max_consecutive_losses"],
                message=f"{streak} consecutive losses exceeds retirement limit"
            )

        if streak >= self.pause_thresholds["max_consecutive_losses"]:
            return HealthCheck(
                name="Loss Streak",
                status=StrategyStatus.PAUSE,
                current_value=streak,
                threshold=self.pause_thresholds["max_consecutive_losses"],
                message=f"{streak} consecutive losses exceeds pause limit"
            )

        if streak >= self.investigation_thresholds["max_consecutive_losses"]:
            return HealthCheck(
                name="Loss Streak",
                status=StrategyStatus.INVESTIGATE,
                current_value=streak,
                threshold=self.investigation_thresholds["max_consecutive_losses"],
                message=f"{streak} consecutive losses is concerning"
            )

        return HealthCheck(
            name="Loss Streak",
            status=StrategyStatus.HEALTHY,
            current_value=streak,
            threshold=self.investigation_thresholds["max_consecutive_losses"],
            message=f"Loss streak {streak} is acceptable"
        )

    def check_paper_deviation(self, live_wr: float, trade_count: int) -> HealthCheck:
        """Check deviation from paper trading performance."""
        if trade_count < 30:
            return HealthCheck(
                name="Paper Deviation",
                status=StrategyStatus.HEALTHY,
                current_value=0,
                threshold=0,
                message=f"Need {30 - trade_count} more trades for valid comparison"
            )

        deviation = abs(live_wr - self.reference_win_rate) / self.reference_win_rate * 100

        if deviation >= self.retirement_thresholds["max_deviation_from_paper"]:
            return HealthCheck(
                name="Paper Deviation",
                status=StrategyStatus.RETIRE,
                current_value=deviation,
                threshold=self.retirement_thresholds["max_deviation_from_paper"],
                message=f"Live deviates {deviation:.1f}% from paper - edge may be dead"
            )

        if deviation >= self.pause_thresholds["max_deviation_from_paper"]:
            return HealthCheck(
                name="Paper Deviation",
                status=StrategyStatus.PAUSE,
                current_value=deviation,
                threshold=self.pause_thresholds["max_deviation_from_paper"],
                message=f"Live deviates {deviation:.1f}% from paper - needs investigation"
            )

        if deviation >= self.investigation_thresholds["max_deviation_from_paper"]:
            return HealthCheck(
                name="Paper Deviation",
                status=StrategyStatus.INVESTIGATE,
                current_value=deviation,
                threshold=self.investigation_thresholds["max_deviation_from_paper"],
                message=f"Live deviates {deviation:.1f}% from paper"
            )

        return HealthCheck(
            name="Paper Deviation",
            status=StrategyStatus.HEALTHY,
            current_value=deviation,
            threshold=self.investigation_thresholds["max_deviation_from_paper"],
            message=f"Live matches paper within {deviation:.1f}%"
        )

    def assess(
        self,
        win_rate: float,
        profit_factor: float,
        drawdown_pct: float,
        consecutive_losses: int,
        trade_count: int
    ) -> RetirementAssessment:
        """
        Perform complete retirement assessment.

        Args:
            win_rate: Current win rate (%)
            profit_factor: Current profit factor
            drawdown_pct: Current drawdown (%)
            consecutive_losses: Current loss streak
            trade_count: Total trades

        Returns:
            Complete RetirementAssessment
        """
        checks = [
            self.check_win_rate(win_rate, trade_count),
            self.check_profit_factor(profit_factor, trade_count),
            self.check_drawdown(drawdown_pct),
            self.check_consecutive_losses(consecutive_losses),
            self.check_paper_deviation(win_rate, trade_count),
        ]

        # Determine overall status (worst of all checks)
        status_priority = {
            StrategyStatus.RETIRE: 4,
            StrategyStatus.PAUSE: 3,
            StrategyStatus.INVESTIGATE: 2,
            StrategyStatus.HEALTHY: 1
        }

        worst_check = max(checks, key=lambda c: status_priority.get(c.status, 0))
        overall_status = worst_check.status

        # Count statuses
        retire_count = sum(1 for c in checks if c.status == StrategyStatus.RETIRE)
        pause_count = sum(1 for c in checks if c.status == StrategyStatus.PAUSE)
        investigate_count = sum(1 for c in checks if c.status == StrategyStatus.INVESTIGATE)

        # Determine action
        if retire_count >= 2:
            action = "RETIRE IMMEDIATELY - Multiple fatal indicators"
            overall_status = StrategyStatus.RETIRE
        elif retire_count == 1:
            action = "PAUSE AND INVESTIGATE - One retirement indicator triggered"
            overall_status = StrategyStatus.PAUSE
        elif pause_count >= 2:
            action = "PAUSE TRADING - Multiple pause indicators"
            overall_status = StrategyStatus.PAUSE
        elif pause_count == 1:
            action = "REDUCE SIZE and monitor closely"
            overall_status = StrategyStatus.INVESTIGATE
        elif investigate_count >= 2:
            action = "INVESTIGATE - Multiple warning signs"
            overall_status = StrategyStatus.INVESTIGATE
        elif investigate_count == 1:
            action = "MONITOR - One warning sign"
            overall_status = StrategyStatus.INVESTIGATE
        else:
            action = "CONTINUE - All systems healthy"

        # Build investigation items
        investigation = []
        if overall_status in [StrategyStatus.INVESTIGATE, StrategyStatus.PAUSE, StrategyStatus.RETIRE]:
            investigation.append("Review recent losing trades for pattern changes")
            investigation.append("Check if market regime has shifted")
            investigation.append("Verify execution quality (slippage, fills)")
            investigation.append("Compare signal quality to paper trading period")

            if drawdown_pct > 10:
                investigation.append("Analyze position sizing - may be too aggressive")

            if consecutive_losses > 3:
                investigation.append("Check for regime change or correlation shock")

            if win_rate < 60:
                investigation.append("Review which signals are failing")

        # Build reason
        failing_checks = [c for c in checks if c.status != StrategyStatus.HEALTHY]
        if failing_checks:
            reasons = [c.message for c in failing_checks]
            reason = "; ".join(reasons)
        else:
            reason = "All health checks passed"

        return RetirementAssessment(
            overall_status=overall_status,
            health_checks=checks,
            action_required=action,
            investigation_items=investigation,
            reason=reason
        )

    def print_assessment(
        self,
        win_rate: float,
        profit_factor: float,
        drawdown_pct: float,
        consecutive_losses: int,
        trade_count: int
    ):
        """Print formatted assessment."""
        assessment = self.assess(win_rate, profit_factor, drawdown_pct, consecutive_losses, trade_count)

        print("\n" + "=" * 70)
        print("STRATEGY RETIREMENT ASSESSMENT")
        print("=" * 70)

        # Overall status with color coding
        status_display = {
            StrategyStatus.HEALTHY: "HEALTHY (green)",
            StrategyStatus.INVESTIGATE: "INVESTIGATE (yellow)",
            StrategyStatus.PAUSE: "PAUSE (orange)",
            StrategyStatus.RETIRE: "RETIRE (red)"
        }

        print(f"\n[OVERALL STATUS]")
        print(f"  Status: {status_display.get(assessment.overall_status)}")
        print(f"  Action: {assessment.action_required}")

        print(f"\n[HEALTH CHECKS]")
        print("-" * 50)
        for check in assessment.health_checks:
            status_icon = {
                StrategyStatus.HEALTHY: "+",
                StrategyStatus.INVESTIGATE: "?",
                StrategyStatus.PAUSE: "!",
                StrategyStatus.RETIRE: "X"
            }.get(check.status, " ")

            print(f"  {status_icon} {check.name}: {check.message}")

        if assessment.investigation_items:
            print(f"\n[INVESTIGATION ITEMS]")
            print("-" * 50)
            for item in assessment.investigation_items:
                print(f"  • {item}")

        print(f"\n[REFERENCE]")
        print("-" * 50)
        print(f"  Expected Win Rate: {self.reference_win_rate:.1f}%")
        print(f"  Expected Profit Factor: {self.reference_pf:.2f}")

        print(f"\n[THRESHOLDS]")
        print("-" * 50)
        print(f"  Retirement triggers:")
        print(f"    - Win Rate < {self.retirement_thresholds['min_win_rate']}%")
        print(f"    - Profit Factor < {self.retirement_thresholds['min_profit_factor']}")
        print(f"    - Drawdown > {self.retirement_thresholds['max_drawdown_pct']}%")
        print(f"    - Loss Streak > {self.retirement_thresholds['max_consecutive_losses']}")
        print(f"    - Paper Deviation > {self.retirement_thresholds['max_deviation_from_paper']}%")

        print("=" * 70)

        return assessment


# =============================================================================
# DECISION FLOWCHART
# =============================================================================

def print_retirement_flowchart():
    """Print the retirement decision flowchart."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                 STRATEGY RETIREMENT DECISION FLOWCHART               ║
╚══════════════════════════════════════════════════════════════════════╝

    [Is Drawdown > 30%?]
           │
           ├── YES ──► RETIRE IMMEDIATELY
           │
           NO
           │
    [Is Win Rate < 45%?] (with 50+ trades)
           │
           ├── YES ──► RETIRE
           │
           NO
           │
    [Is Profit Factor < 0.8?]
           │
           ├── YES ──► RETIRE
           │
           NO
           │
    [15+ Consecutive Losses?]
           │
           ├── YES ──► RETIRE
           │
           NO
           │
    [40%+ Deviation from Paper?]
           │
           ├── YES ──► RETIRE
           │
           NO
           │
    [Any PAUSE threshold hit?]
           │
           ├── YES ──► PAUSE and INVESTIGATE
           │              │
           │              ├── Can fix? ──► FIX and RESUME
           │              │
           │              └── Cannot fix? ──► RETIRE
           │
           NO
           │
    [Any INVESTIGATE threshold hit?]
           │
           ├── YES ──► REDUCE SIZE, MONITOR
           │              │
           │              ├── Improves? ──► RESUME NORMAL
           │              │
           │              └── Worsens? ──► PAUSE
           │
           NO
           │
    [All Healthy]
           │
           └── CONTINUE TRADING


═══════════════════════════════════════════════════════════════════════

REMEMBER:
• Retirement is PERMANENT - no restarts without fundamental redesign
• Don't retire during temporary market conditions
• Need 50+ trades before retirement decision is valid
• Check for regime changes before retiring

═══════════════════════════════════════════════════════════════════════
""")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print_retirement_flowchart()

    print("\n" + "=" * 70)
    print("RETIREMENT CHECK EXAMPLES")
    print("=" * 70)

    checker = StrategyRetirementChecker(
        reference_win_rate=73.3,  # From backtest
        reference_pf=2.1
    )

    # Example 1: Healthy strategy
    print("\n[EXAMPLE 1: HEALTHY STRATEGY]")
    checker.print_assessment(
        win_rate=71.5,
        profit_factor=1.85,
        drawdown_pct=8.0,
        consecutive_losses=2,
        trade_count=75
    )

    # Example 2: Concerning trends
    print("\n[EXAMPLE 2: CONCERNING TRENDS]")
    checker.print_assessment(
        win_rate=62.0,
        profit_factor=1.25,
        drawdown_pct=12.0,
        consecutive_losses=4,
        trade_count=100
    )

    # Example 3: Should retire
    print("\n[EXAMPLE 3: RETIREMENT CANDIDATE]")
    checker.print_assessment(
        win_rate=48.0,
        profit_factor=0.75,
        drawdown_pct=25.0,
        consecutive_losses=8,
        trade_count=150
    )
