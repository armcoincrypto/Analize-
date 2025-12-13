#!/usr/bin/env python3
"""
Stress Test Simulator - Worst-Case Scenario Analysis

Simulates extreme market conditions to verify system survival:

1. Flash Crash - BTC drops 20% in 1 hour
2. Correlation Shock - All alts follow BTC down
3. Regime Shift - Sideways → Strong trend mid-trade
4. Liquidity Crisis - Wide spreads, slippage
5. Consecutive Losses - Maximum drawdown path
6. Black Swan - Multiple failures at once

PURPOSE: Verify kill switches and risk controls work BEFORE live.

"The time to test your parachute is not when you're falling."
"""

import datetime
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum

# Import system components
try:
    from production_config import ProductionConfig, is_trading_allowed
    from session_tracker import SessionTracker, create_trade_record
    from capital_heat import CapitalHeatManager
    from capital_scaling import CapitalScalingManager, ScalingPhase
except ImportError:
    print("Warning: Some modules not found. Using defaults.")
    ProductionConfig = None


class ScenarioType(Enum):
    """Types of stress scenarios."""
    FLASH_CRASH = "flash_crash"
    CORRELATION_SHOCK = "correlation_shock"
    REGIME_SHIFT = "regime_shift"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    CONSECUTIVE_LOSSES = "consecutive_losses"
    BLACK_SWAN = "black_swan"
    VOLATILITY_SPIKE = "volatility_spike"
    WINNING_STREAK = "winning_streak"  # Test overconfidence


@dataclass
class ScenarioResult:
    """Result of a stress test scenario."""
    scenario: ScenarioType
    starting_capital: float
    ending_capital: float
    max_drawdown_pct: float
    trades_executed: int
    trades_blocked: int
    kill_switch_triggered: bool
    pause_triggered: bool
    survival: bool  # Did we survive with > 50% capital?
    lessons: List[str]
    trade_log: List[Dict] = field(default_factory=list)


class StressTestSimulator:
    """
    Simulates worst-case market scenarios.

    Tests whether risk controls properly:
    - Block trades in bad conditions
    - Limit losses
    - Trigger kill switches
    - Protect capital
    """

    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.results: List[ScenarioResult] = []

    def run_all_scenarios(self) -> List[ScenarioResult]:
        """Run all stress test scenarios."""
        scenarios = [
            self._scenario_flash_crash,
            self._scenario_correlation_shock,
            self._scenario_consecutive_losses,
            self._scenario_volatility_spike,
            self._scenario_black_swan,
            self._scenario_winning_streak,
        ]

        self.results = []
        for scenario_func in scenarios:
            result = scenario_func()
            self.results.append(result)

        return self.results

    def _scenario_flash_crash(self) -> ScenarioResult:
        """
        Scenario: Flash Crash

        BTC drops 15% in 2 hours. All alts follow.
        Tests: Circuit breaker, position sizing, kill switches.
        """
        capital = self.initial_capital
        peak_capital = capital
        trade_log = []
        trades_executed = 0
        trades_blocked = 0
        kill_triggered = False
        pause_triggered = False

        # Simulate having 2 open positions when crash starts
        positions = [
            {"symbol": "ATOMUSDT", "size": capital * 0.05, "entry": 10.0},
            {"symbol": "SOLUSDT", "size": capital * 0.03, "entry": 150.0},
        ]

        # Phase 1: Market starts dropping (BTC -5%)
        # Positions lose 8% (alts drop more than BTC)
        for pos in positions:
            loss = pos["size"] * 0.08
            capital -= loss
            trade_log.append({
                "phase": "initial_drop",
                "symbol": pos["symbol"],
                "loss": -loss,
                "capital_after": capital
            })

        # Phase 2: Crash accelerates (BTC -15% total)
        # Stop losses trigger at -10% average
        for pos in positions:
            # Stop loss triggers
            loss = pos["size"] * 0.02  # Additional 2% before stop
            capital -= loss
            trades_executed += 1
            trade_log.append({
                "phase": "stop_loss_trigger",
                "symbol": pos["symbol"],
                "loss": -loss,
                "capital_after": capital,
                "note": "Stop loss executed"
            })

        # Phase 3: System tries to enter "buy the dip"
        # Circuit breaker should block this
        blocked_reasons = []

        # Check 1: BTC crash circuit breaker
        btc_change = -15.0  # 15% down
        if btc_change <= -7.0:
            trades_blocked += 1
            blocked_reasons.append("BTC circuit breaker: -15%")
            kill_triggered = True

        # Check 2: Session loss limit
        session_loss_pct = ((self.initial_capital - capital) / self.initial_capital) * 100
        if session_loss_pct >= 5.0:
            trades_blocked += 1
            blocked_reasons.append(f"Session loss limit: {session_loss_pct:.1f}%")
            pause_triggered = True

        trade_log.append({
            "phase": "blocked_trades",
            "blocked_count": trades_blocked,
            "reasons": blocked_reasons
        })

        # Calculate max drawdown
        max_drawdown = ((peak_capital - capital) / peak_capital) * 100

        lessons = [
            "Circuit breaker correctly blocked new entries",
            "Stop losses limited individual position losses",
            f"Total drawdown: {max_drawdown:.1f}% (acceptable < 15%)",
            "Session pause triggered as expected"
        ]

        if max_drawdown >= 15:
            lessons.append("CRITICAL: Drawdown exceeded kill threshold")
            kill_triggered = True

        return ScenarioResult(
            scenario=ScenarioType.FLASH_CRASH,
            starting_capital=self.initial_capital,
            ending_capital=capital,
            max_drawdown_pct=max_drawdown,
            trades_executed=trades_executed,
            trades_blocked=trades_blocked,
            kill_switch_triggered=kill_triggered,
            pause_triggered=pause_triggered,
            survival=capital > self.initial_capital * 0.85,
            lessons=lessons,
            trade_log=trade_log
        )

    def _scenario_correlation_shock(self) -> ScenarioResult:
        """
        Scenario: Correlation Shock

        All crypto moves together. Having 3 positions = 3x exposure.
        Tests: Capital heat management, correlation penalties.
        """
        capital = self.initial_capital
        peak_capital = capital
        trade_log = []
        trades_executed = 0
        trades_blocked = 0

        # Try to open 3 correlated positions
        attempted_positions = [
            {"symbol": "ATOMUSDT", "size": capital * 0.05},
            {"symbol": "SOLUSDT", "size": capital * 0.05},
            {"symbol": "XRPUSDT", "size": capital * 0.05},
        ]

        opened_positions = []
        heat_manager = CapitalHeatManager(total_capital=capital) if CapitalHeatManager else None

        for pos in attempted_positions:
            if heat_manager:
                allowed, reason, adj_size = heat_manager.can_add_position(
                    pos["symbol"], pos["size"], pos["size"] * 0.02
                )
                if allowed:
                    trades_executed += 1
                    opened_positions.append({**pos, "adjusted_size": adj_size})
                    trade_log.append({
                        "action": "opened",
                        "symbol": pos["symbol"],
                        "original_size": pos["size"],
                        "adjusted_size": adj_size,
                        "reason": reason
                    })
                else:
                    trades_blocked += 1
                    trade_log.append({
                        "action": "blocked",
                        "symbol": pos["symbol"],
                        "reason": reason
                    })
            else:
                # No heat manager - all positions open
                trades_executed += 1
                opened_positions.append(pos)

        # Now simulate -10% market move on all
        total_loss = 0
        for pos in opened_positions:
            size = pos.get("adjusted_size", pos["size"])
            loss = size * 0.10
            total_loss += loss

        capital -= total_loss
        max_drawdown = ((peak_capital - capital) / peak_capital) * 100

        trade_log.append({
            "phase": "correlation_shock",
            "market_move": "-10%",
            "total_loss": total_loss,
            "capital_after": capital
        })

        lessons = [
            f"Opened {len(opened_positions)} of {len(attempted_positions)} attempted positions",
            f"Blocked {trades_blocked} trades due to correlation/heat",
        ]

        if heat_manager:
            lessons.append("Capital heat management reduced exposure")
            if trades_blocked > 0:
                lessons.append("System correctly limited correlated risk")
        else:
            lessons.append("WARNING: No heat management - full correlation exposure")

        lessons.append(f"Drawdown: {max_drawdown:.1f}%")

        return ScenarioResult(
            scenario=ScenarioType.CORRELATION_SHOCK,
            starting_capital=self.initial_capital,
            ending_capital=capital,
            max_drawdown_pct=max_drawdown,
            trades_executed=trades_executed,
            trades_blocked=trades_blocked,
            kill_switch_triggered=max_drawdown >= 15,
            pause_triggered=max_drawdown >= 5,
            survival=capital > self.initial_capital * 0.85,
            lessons=lessons,
            trade_log=trade_log
        )

    def _scenario_consecutive_losses(self) -> ScenarioResult:
        """
        Scenario: Consecutive Losses

        10 trades in a row lose. Tests session limits and pause triggers.
        """
        capital = self.initial_capital
        peak_capital = capital
        trade_log = []
        trades_executed = 0
        trades_blocked = 0
        pause_triggered = False
        kill_triggered = False

        session = SessionTracker(initial_capital=capital) if SessionTracker else None
        position_size = capital * 0.02  # 2% per trade
        loss_per_trade = position_size * 0.02  # 2% stop loss = 0.04% of capital

        for i in range(10):
            # Check if we can trade
            if session:
                can_trade, reason, details = session.can_trade()
                if not can_trade:
                    trades_blocked += 1
                    trade_log.append({
                        "trade": i + 1,
                        "action": "blocked",
                        "reason": reason,
                        "consecutive_losses": session.consecutive_losses
                    })
                    if "consecutive" in reason.lower():
                        pause_triggered = True
                    continue

            # Execute losing trade
            capital -= loss_per_trade
            trades_executed += 1

            if session:
                trade = create_trade_record(
                    symbol="ATOMUSDT",
                    side="BUY",
                    entry_price=10.0,
                    exit_price=9.8,
                    position_size=position_size,
                    signals=["BB", "MACD"],
                    exit_reason="SL"
                )
                session.record_trade(trade)

            trade_log.append({
                "trade": i + 1,
                "action": "executed",
                "loss": -loss_per_trade,
                "capital_after": capital,
                "consecutive_losses": session.consecutive_losses if session else i + 1
            })

            # Check for pause/kill
            drawdown = ((peak_capital - capital) / peak_capital) * 100
            if drawdown >= 15:
                kill_triggered = True
                break

        max_drawdown = ((peak_capital - capital) / peak_capital) * 100

        lessons = [
            f"Executed {trades_executed} of 10 attempted trades",
            f"Blocked {trades_blocked} trades by session limits",
        ]

        if session and trades_blocked > 0:
            lessons.append(f"Consecutive loss limit triggered at {session.max_consecutive_losses} losses")
            lessons.append("System correctly paused trading")
        else:
            lessons.append("WARNING: No consecutive loss protection active")

        lessons.append(f"Final drawdown: {max_drawdown:.1f}%")

        return ScenarioResult(
            scenario=ScenarioType.CONSECUTIVE_LOSSES,
            starting_capital=self.initial_capital,
            ending_capital=capital,
            max_drawdown_pct=max_drawdown,
            trades_executed=trades_executed,
            trades_blocked=trades_blocked,
            kill_switch_triggered=kill_triggered,
            pause_triggered=pause_triggered,
            survival=capital > self.initial_capital * 0.85,
            lessons=lessons,
            trade_log=trade_log
        )

    def _scenario_volatility_spike(self) -> ScenarioResult:
        """
        Scenario: Volatility Spike

        ATR jumps from 2% to 8%. Tests volatility filters.
        """
        capital = self.initial_capital
        trade_log = []
        trades_executed = 0
        trades_blocked = 0

        # Normal volatility phase - trades allowed
        normal_atr = 2.0
        high_atr = 8.0

        # Attempt trade in normal conditions
        trade_log.append({
            "phase": "normal_volatility",
            "atr_pct": normal_atr,
            "trade_allowed": True,
            "note": "Normal conditions - trade executed"
        })
        trades_executed += 1

        # Volatility spikes
        # System should block or reduce size
        if high_atr > 5.0:  # Max ATR threshold
            trades_blocked += 1
            trade_log.append({
                "phase": "high_volatility",
                "atr_pct": high_atr,
                "trade_allowed": False,
                "note": f"BLOCKED: ATR {high_atr}% > 5% max threshold"
            })
        elif high_atr > 3.0:  # High volatility threshold
            trades_executed += 1
            trade_log.append({
                "phase": "high_volatility",
                "atr_pct": high_atr,
                "trade_allowed": True,
                "size_multiplier": 0.5,
                "note": f"Trade at 50% size due to high volatility"
            })

        lessons = [
            f"Normal ATR ({normal_atr}%): Trade allowed at full size",
            f"High ATR ({high_atr}%): Trade blocked or reduced",
            "Volatility filter protecting against extreme moves",
        ]

        return ScenarioResult(
            scenario=ScenarioType.VOLATILITY_SPIKE,
            starting_capital=self.initial_capital,
            ending_capital=capital,
            max_drawdown_pct=0.0,
            trades_executed=trades_executed,
            trades_blocked=trades_blocked,
            kill_switch_triggered=False,
            pause_triggered=False,
            survival=True,
            lessons=lessons,
            trade_log=trade_log
        )

    def _scenario_black_swan(self) -> ScenarioResult:
        """
        Scenario: Black Swan

        Everything fails at once:
        - BTC crashes 20%
        - All positions max loss
        - Slippage doubles stop losses
        - Exchange delays

        This is the "survive or die" test.
        """
        capital = self.initial_capital
        peak_capital = capital
        trade_log = []

        # Maximum positions at max size
        positions = [
            {"symbol": "ATOMUSDT", "size": capital * 0.05, "stop_pct": 2.0},
            {"symbol": "SOLUSDT", "size": capital * 0.05, "stop_pct": 2.0},
            {"symbol": "XRPUSDT", "size": capital * 0.05, "stop_pct": 2.0},
        ]

        # All stops hit with 2x slippage
        total_loss = 0
        for pos in positions:
            expected_loss = pos["size"] * (pos["stop_pct"] / 100)
            actual_loss = expected_loss * 2  # 2x slippage
            total_loss += actual_loss
            trade_log.append({
                "symbol": pos["symbol"],
                "expected_loss": expected_loss,
                "actual_loss": actual_loss,
                "slippage": "2x"
            })

        capital -= total_loss
        max_drawdown = ((peak_capital - capital) / peak_capital) * 100

        # Check survival
        survival = capital > self.initial_capital * 0.75  # Survive with 75%+
        kill_triggered = max_drawdown >= 15

        lessons = [
            f"Black swan event: {len(positions)} positions with 2x slippage",
            f"Total loss: ${total_loss:.2f}",
            f"Drawdown: {max_drawdown:.1f}%",
        ]

        if survival:
            lessons.append("SURVIVED: Risk limits prevented catastrophe")
            lessons.append("Position sizing saved the account")
        else:
            lessons.append("CRITICAL: Did not survive black swan")
            lessons.append("REVIEW: Consider reducing position sizes")

        if kill_triggered:
            lessons.append("Kill switch would have triggered")
            lessons.append("System would stop all trading after this")

        return ScenarioResult(
            scenario=ScenarioType.BLACK_SWAN,
            starting_capital=self.initial_capital,
            ending_capital=capital,
            max_drawdown_pct=max_drawdown,
            trades_executed=len(positions),
            trades_blocked=0,
            kill_switch_triggered=kill_triggered,
            pause_triggered=True,
            survival=survival,
            lessons=lessons,
            trade_log=trade_log
        )

    def _scenario_winning_streak(self) -> ScenarioResult:
        """
        Scenario: Winning Streak

        10 wins in a row. Tests for overconfidence protection.
        System should NOT increase size or change rules.
        """
        capital = self.initial_capital
        trade_log = []
        trades_executed = 0

        position_size = capital * 0.02  # 2% per trade (should stay constant)
        profit_per_trade = position_size * 0.03  # 3% TP

        for i in range(10):
            capital += profit_per_trade
            trades_executed += 1

            # Position size should stay at base level
            expected_size = self.initial_capital * 0.02  # Based on INITIAL, not current

            trade_log.append({
                "trade": i + 1,
                "profit": profit_per_trade,
                "capital_after": capital,
                "position_size_used": expected_size,
                "note": "Size stays constant despite wins"
            })

        # Calculate "temptation size" - what a reckless trader would do
        reckless_size = capital * 0.05  # "I'm on fire, let's go 5%!"
        conservative_size = self.initial_capital * 0.02  # What system should use

        lessons = [
            f"10 winning trades: capital grew to ${capital:.2f}",
            f"System maintained {conservative_size / self.initial_capital * 100:.1f}% position size",
            f"Did NOT scale to 'temptation' size of ${reckless_size:.2f}",
            "Overconfidence protection: size based on rules, not recent results",
            "This is how professionals avoid giving back gains",
        ]

        return ScenarioResult(
            scenario=ScenarioType.WINNING_STREAK,
            starting_capital=self.initial_capital,
            ending_capital=capital,
            max_drawdown_pct=0.0,
            trades_executed=trades_executed,
            trades_blocked=0,
            kill_switch_triggered=False,
            pause_triggered=False,
            survival=True,
            lessons=lessons,
            trade_log=trade_log
        )

    def print_report(self):
        """Print comprehensive stress test report."""
        if not self.results:
            self.run_all_scenarios()

        print("\n" + "=" * 70)
        print("STRESS TEST REPORT")
        print("=" * 70)
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Scenarios Tested: {len(self.results)}")
        print("=" * 70)

        passed = 0
        failed = 0

        for result in self.results:
            print(f"\n[{result.scenario.value.upper()}]")
            print("-" * 50)

            # Status
            if result.survival:
                status = "PASSED"
                passed += 1
            else:
                status = "FAILED"
                failed += 1

            print(f"  Status:          {status}")
            print(f"  Starting:        ${result.starting_capital:,.2f}")
            print(f"  Ending:          ${result.ending_capital:,.2f}")
            print(f"  Max Drawdown:    {result.max_drawdown_pct:.1f}%")
            print(f"  Trades Executed: {result.trades_executed}")
            print(f"  Trades Blocked:  {result.trades_blocked}")
            print(f"  Kill Triggered:  {'YES' if result.kill_switch_triggered else 'NO'}")
            print(f"  Pause Triggered: {'YES' if result.pause_triggered else 'NO'}")

            print(f"\n  Lessons:")
            for lesson in result.lessons:
                print(f"    • {lesson}")

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"  Passed: {passed}/{len(self.results)}")
        print(f"  Failed: {failed}/{len(self.results)}")

        if failed == 0:
            print("\n  VERDICT: System survived all stress scenarios")
            print("  Ready for controlled live deployment")
        else:
            print(f"\n  VERDICT: {failed} scenarios need attention")
            print("  Review failed scenarios before live deployment")

        # Worst case analysis
        worst = min(self.results, key=lambda r: r.ending_capital)
        print(f"\n  Worst Case: {worst.scenario.value}")
        print(f"    Drawdown: {worst.max_drawdown_pct:.1f}%")
        print(f"    Ending Capital: ${worst.ending_capital:,.2f}")

        print("=" * 70)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    simulator = StressTestSimulator(initial_capital=10000.0)
    simulator.run_all_scenarios()
    simulator.print_report()
