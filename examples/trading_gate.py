#!/usr/bin/env python3
"""
Trading Gate - Daily "Should I Trade Today?" Decision

The single most important question before every trading session:

    "Are conditions favorable for MY strategy?"

This gate combines all system checks into one decision:
- Time/Day validation
- Regime assessment
- Capital/Risk status
- Session limits
- Market conditions

If gate says NO, you don't trade. Period.

"The best trade is often no trade."
"""

import ssl_bypass  # Must be first
import os
import json
import requests
import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from enum import Enum

# Import system components (with fallbacks)
try:
    from production_config import ProductionConfig, is_trading_allowed
except ImportError:
    ProductionConfig = None
    is_trading_allowed = None

try:
    from regime_scorer import RegimeConfidenceScorer
except ImportError:
    RegimeConfidenceScorer = None

try:
    from session_tracker import SessionTracker
except ImportError:
    SessionTracker = None


class GateDecision(Enum):
    """Gate decision levels."""
    OPEN = "open"           # Full trading allowed
    PARTIAL = "partial"     # Reduced size allowed
    CLOSED = "closed"       # No trading allowed


@dataclass
class GateCheck:
    """Result of a single gate check."""
    name: str
    passed: bool
    decision: GateDecision
    message: str
    size_multiplier: float = 1.0


@dataclass
class GateStatus:
    """Complete gate status."""
    decision: GateDecision
    trade_allowed: bool
    size_multiplier: float
    checks: List[GateCheck]
    summary: str
    recommendations: List[str]


class TradingGate:
    """
    The daily trading decision gate.

    Asks: "Should I trade today?"

    Combines:
    1. Time/Day checks (evidence-based)
    2. Market regime (current conditions)
    3. Volatility (ATR check)
    4. BTC health (correlation risk)
    5. Session status (limits, pauses)
    6. Capital status (drawdown, heat)

    Returns: OPEN, PARTIAL, or CLOSED
    """

    def __init__(self, symbol: str = "ATOMUSDT"):
        self.symbol = symbol

        # Load configs
        self.config = ProductionConfig() if ProductionConfig else None

        # Evidence-based optimal conditions
        self.optimal_hours = [15, 16, 17, 18, 19, 20, 21, 22]
        self.blocked_hours = [3, 4, 5, 6]
        self.optimal_days = [0, 1]  # Monday, Tuesday
        self.blocked_days = [3]     # Thursday

        # Thresholds
        self.min_atr_pct = 0.5
        self.max_atr_pct = 5.0
        self.btc_crash_threshold = -7.0

    def check_time_day(self) -> GateCheck:
        """Check if current time/day is favorable."""
        now = datetime.datetime.utcnow()
        hour = now.hour
        day = now.weekday()
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        # Blocked day
        if day in self.blocked_days:
            return GateCheck(
                name="Time/Day",
                passed=False,
                decision=GateDecision.CLOSED,
                message=f"BLOCKED: {day_names[day]} has 51.2% win rate (near coin flip)",
                size_multiplier=0.0
            )

        # Blocked hour
        if hour in self.blocked_hours:
            return GateCheck(
                name="Time/Day",
                passed=False,
                decision=GateDecision.CLOSED,
                message=f"BLOCKED: {hour}:00 UTC in worst-performing hours",
                size_multiplier=0.0
            )

        # Optimal conditions
        if hour in self.optimal_hours and day in self.optimal_days:
            return GateCheck(
                name="Time/Day",
                passed=True,
                decision=GateDecision.OPEN,
                message=f"OPTIMAL: {day_names[day]} {hour}:00 UTC",
                size_multiplier=1.0
            )

        # Good hour, any day
        if hour in self.optimal_hours:
            return GateCheck(
                name="Time/Day",
                passed=True,
                decision=GateDecision.OPEN,
                message=f"GOOD: {hour}:00 UTC in optimal hours",
                size_multiplier=0.9
            )

        # Suboptimal but allowed
        return GateCheck(
            name="Time/Day",
            passed=True,
            decision=GateDecision.PARTIAL,
            message=f"CAUTION: {day_names[day]} {hour}:00 UTC - suboptimal window",
            size_multiplier=0.7
        )

    def check_regime(self) -> GateCheck:
        """Check current market regime."""
        if not RegimeConfidenceScorer:
            return GateCheck(
                name="Regime",
                passed=True,
                decision=GateDecision.PARTIAL,
                message="Regime scorer not available - using caution",
                size_multiplier=0.75
            )

        try:
            scorer = RegimeConfidenceScorer()
            score = scorer.score_regime(self.symbol)

            if score.regime_type.value == "crisis":
                return GateCheck(
                    name="Regime",
                    passed=False,
                    decision=GateDecision.CLOSED,
                    message=f"CRISIS: {score.reason}",
                    size_multiplier=0.0
                )

            if score.regime_type.value == "sideways":
                return GateCheck(
                    name="Regime",
                    passed=True,
                    decision=GateDecision.OPEN,
                    message=f"OPTIMAL: Sideways regime (best for mean reversion)",
                    size_multiplier=score.size_multiplier
                )

            if score.regime_type.value in ["volatile", "strong_bear"]:
                return GateCheck(
                    name="Regime",
                    passed=True,
                    decision=GateDecision.PARTIAL,
                    message=f"CAUTION: {score.regime_type.value} - reduced size",
                    size_multiplier=score.size_multiplier
                )

            return GateCheck(
                name="Regime",
                passed=True,
                decision=GateDecision.OPEN if score.confidence >= 60 else GateDecision.PARTIAL,
                message=f"OK: {score.regime_type.value} ({score.confidence:.0f}% confidence)",
                size_multiplier=score.size_multiplier
            )

        except Exception as e:
            return GateCheck(
                name="Regime",
                passed=True,
                decision=GateDecision.PARTIAL,
                message=f"Could not check regime: {str(e)[:30]}",
                size_multiplier=0.75
            )

    def check_volatility(self) -> GateCheck:
        """Check current volatility (ATR)."""
        try:
            urls = [
                f"https://api.binance.us/api/v3/klines?symbol={self.symbol}&interval=1h&limit=14",
                f"https://api.binance.com/api/v3/klines?symbol={self.symbol}&interval=1h&limit=14",
            ]

            for url in urls:
                try:
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        if len(data) >= 2:
                            trs = []
                            for i in range(1, len(data)):
                                high = float(data[i][2])
                                low = float(data[i][3])
                                prev_close = float(data[i-1][4])
                                tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
                                trs.append(tr)

                            atr = sum(trs) / len(trs)
                            price = float(data[-1][4])
                            atr_pct = (atr / price) * 100

                            if atr_pct < self.min_atr_pct:
                                return GateCheck(
                                    name="Volatility",
                                    passed=False,
                                    decision=GateDecision.CLOSED,
                                    message=f"TOO QUIET: ATR {atr_pct:.2f}% < {self.min_atr_pct}%",
                                    size_multiplier=0.0
                                )

                            if atr_pct > self.max_atr_pct:
                                return GateCheck(
                                    name="Volatility",
                                    passed=False,
                                    decision=GateDecision.CLOSED,
                                    message=f"TOO VOLATILE: ATR {atr_pct:.2f}% > {self.max_atr_pct}%",
                                    size_multiplier=0.0
                                )

                            if atr_pct > 3.0:
                                return GateCheck(
                                    name="Volatility",
                                    passed=True,
                                    decision=GateDecision.PARTIAL,
                                    message=f"HIGH: ATR {atr_pct:.2f}% - reduce size 50%",
                                    size_multiplier=0.5
                                )

                            return GateCheck(
                                name="Volatility",
                                passed=True,
                                decision=GateDecision.OPEN,
                                message=f"NORMAL: ATR {atr_pct:.2f}%",
                                size_multiplier=1.0
                            )
                except:
                    continue

            return GateCheck(
                name="Volatility",
                passed=True,
                decision=GateDecision.PARTIAL,
                message="Could not check volatility - using caution",
                size_multiplier=0.75
            )

        except Exception as e:
            return GateCheck(
                name="Volatility",
                passed=True,
                decision=GateDecision.PARTIAL,
                message=f"Volatility check error: {str(e)[:30]}",
                size_multiplier=0.75
            )

    def check_btc_health(self) -> GateCheck:
        """Check BTC market health (circuit breaker)."""
        try:
            url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd&include_24hr_change=true"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                change_24h = data.get("bitcoin", {}).get("usd_24h_change", 0)

                if change_24h <= self.btc_crash_threshold:
                    return GateCheck(
                        name="BTC Health",
                        passed=False,
                        decision=GateDecision.CLOSED,
                        message=f"CIRCUIT BREAKER: BTC {change_24h:.1f}% in 24h",
                        size_multiplier=0.0
                    )

                if change_24h <= -5:
                    return GateCheck(
                        name="BTC Health",
                        passed=True,
                        decision=GateDecision.PARTIAL,
                        message=f"BTC WEAK: {change_24h:.1f}% - reduce size",
                        size_multiplier=0.5
                    )

                return GateCheck(
                    name="BTC Health",
                    passed=True,
                    decision=GateDecision.OPEN,
                    message=f"BTC OK: {change_24h:+.1f}%",
                    size_multiplier=1.0
                )

            return GateCheck(
                name="BTC Health",
                passed=True,
                decision=GateDecision.PARTIAL,
                message="Could not check BTC - using caution",
                size_multiplier=0.75
            )

        except Exception as e:
            return GateCheck(
                name="BTC Health",
                passed=True,
                decision=GateDecision.PARTIAL,
                message=f"BTC check error: {str(e)[:30]}",
                size_multiplier=0.75
            )

    def check_session(self) -> GateCheck:
        """Check session status (limits, pauses)."""
        if not SessionTracker:
            return GateCheck(
                name="Session",
                passed=True,
                decision=GateDecision.OPEN,
                message="Session tracker not loaded",
                size_multiplier=1.0
            )

        try:
            tracker = SessionTracker()
            can_trade, reason, details = tracker.can_trade()

            if not can_trade:
                return GateCheck(
                    name="Session",
                    passed=False,
                    decision=GateDecision.CLOSED,
                    message=f"SESSION BLOCKED: {reason}",
                    size_multiplier=0.0
                )

            # Check for warnings
            dd = details.get("drawdown_pct", 0)
            if dd > 10:
                return GateCheck(
                    name="Session",
                    passed=True,
                    decision=GateDecision.PARTIAL,
                    message=f"Drawdown warning: {dd:.1f}% - reduce size",
                    size_multiplier=0.5
                )

            return GateCheck(
                name="Session",
                passed=True,
                decision=GateDecision.OPEN,
                message="Session OK",
                size_multiplier=1.0
            )

        except Exception as e:
            return GateCheck(
                name="Session",
                passed=True,
                decision=GateDecision.OPEN,
                message=f"Session check skipped: {str(e)[:30]}",
                size_multiplier=1.0
            )

    def evaluate(self) -> GateStatus:
        """
        Run all gate checks and return decision.

        Returns:
            Complete GateStatus with decision and recommendations
        """
        checks = [
            self.check_time_day(),
            self.check_regime(),
            self.check_volatility(),
            self.check_btc_health(),
            self.check_session(),
        ]

        # Any CLOSED check = gate CLOSED
        closed_checks = [c for c in checks if c.decision == GateDecision.CLOSED]
        if closed_checks:
            return GateStatus(
                decision=GateDecision.CLOSED,
                trade_allowed=False,
                size_multiplier=0.0,
                checks=checks,
                summary=f"GATE CLOSED: {closed_checks[0].message}",
                recommendations=[
                    "Do NOT trade right now",
                    f"Reason: {closed_checks[0].message}",
                    "Wait for conditions to improve"
                ]
            )

        # Calculate combined size multiplier
        multipliers = [c.size_multiplier for c in checks]
        combined_multiplier = min(multipliers)  # Use most restrictive

        # Any PARTIAL = gate PARTIAL
        partial_checks = [c for c in checks if c.decision == GateDecision.PARTIAL]
        if partial_checks:
            reasons = [c.message for c in partial_checks]
            return GateStatus(
                decision=GateDecision.PARTIAL,
                trade_allowed=True,
                size_multiplier=combined_multiplier,
                checks=checks,
                summary=f"GATE PARTIAL: Trade with {combined_multiplier:.0%} size",
                recommendations=[
                    f"Trade allowed at {combined_multiplier:.0%} of normal size",
                    f"Cautions: {', '.join(reasons)}"
                ]
            )

        # All OPEN = gate OPEN
        return GateStatus(
            decision=GateDecision.OPEN,
            trade_allowed=True,
            size_multiplier=combined_multiplier,
            checks=checks,
            summary="GATE OPEN: Full trading allowed",
            recommendations=[
                "All conditions favorable",
                "Trade at normal size"
            ]
        )

    def print_status(self):
        """Print gate status."""
        status = self.evaluate()

        print("\n" + "=" * 60)
        print(f"TRADING GATE - {self.symbol}")
        print("=" * 60)
        print(f"Time: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
        print("-" * 60)

        # Decision
        decision_display = {
            GateDecision.OPEN: "OPEN (green) - Full trading",
            GateDecision.PARTIAL: "PARTIAL (yellow) - Reduced size",
            GateDecision.CLOSED: "CLOSED (red) - No trading"
        }
        print(f"\n[DECISION]")
        print(f"  Gate: {decision_display.get(status.decision)}")
        print(f"  Trade Allowed: {'YES' if status.trade_allowed else 'NO'}")
        print(f"  Size Multiplier: {status.size_multiplier:.0%}")

        # Checks
        print(f"\n[CHECKS]")
        for check in status.checks:
            icon = "+" if check.passed else "X"
            print(f"  {icon} {check.name}: {check.message}")

        # Recommendations
        print(f"\n[ACTION]")
        for rec in status.recommendations:
            print(f"  • {rec}")

        print("=" * 60)

        return status


def should_trade_now(symbol: str = "ATOMUSDT") -> Tuple[bool, str, float]:
    """
    Quick function: Should I trade right now?

    Returns:
        Tuple of (allowed, reason, size_multiplier)
    """
    gate = TradingGate(symbol)
    status = gate.evaluate()

    return status.trade_allowed, status.summary, status.size_multiplier


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TRADING GATE")
    print("=" * 60)
    print()
    print("The gate answers: 'Should I trade right now?'")
    print()
    print("If gate is CLOSED, you do NOT trade. Period.")
    print("This is how professionals avoid bad conditions.")
    print()

    # Check ATOMUSDT
    gate = TradingGate("ATOMUSDT")
    gate.print_status()

    # Quick check
    print("\n[QUICK CHECK]")
    print("-" * 40)
    allowed, reason, multiplier = should_trade_now("ATOMUSDT")
    print(f"  ATOMUSDT: {'TRADE' if allowed else 'NO TRADE'}")
    print(f"  Reason: {reason}")
    if allowed:
        print(f"  Size: {multiplier:.0%}")
