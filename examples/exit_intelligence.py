#!/usr/bin/env python3
"""
Exit Intelligence - Dynamic Post-Entry Management

Phase 2 Risk Evolution Component #2

Uses MFE (Maximum Favorable Excursion) and MAE (Maximum Adverse Excursion)
data to make intelligent exit decisions:

1. If trade reaches profit fast → protect aggressively (trailing stop)
2. If trade stagnates → exit early (time-based exit)
3. If trade approaches historical MAE → cut loss before typical stop

This improves PROFIT QUALITY, not just win rate.

CONCEPT:
- Traditional: Fixed TP at +3%, Fixed SL at -2%
- Intelligent: Dynamic based on trade behavior
"""

import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class ExitReason(Enum):
    """Reason for exit decision."""
    HOLD = "hold"                       # Keep position
    TRAILING_STOP = "trailing_stop"     # Trailing stop triggered
    TAKE_PROFIT = "take_profit"         # Take profit target hit
    STOP_LOSS = "stop_loss"             # Stop loss hit
    TIME_EXIT = "time_exit"             # Stagnation timeout
    MAE_PROTECTION = "mae_protection"   # MAE-based early cut
    MOMENTUM_FADE = "momentum_fade"     # Momentum dying
    BREAK_EVEN = "break_even"           # Move to break even


@dataclass
class TradeState:
    """Current state of an open trade."""
    symbol: str
    side: str                           # "BUY" or "SELL"
    entry_price: float
    entry_time: datetime.datetime
    current_price: float
    highest_price: float                # Highest since entry (for longs)
    lowest_price: float                 # Lowest since entry (for shorts)
    position_size: float

    @property
    def pnl_pct(self) -> float:
        """Current P&L percentage."""
        if self.side == "BUY":
            return ((self.current_price - self.entry_price) / self.entry_price) * 100
        else:
            return ((self.entry_price - self.current_price) / self.entry_price) * 100

    @property
    def mfe_pct(self) -> float:
        """Maximum Favorable Excursion percentage."""
        if self.side == "BUY":
            return ((self.highest_price - self.entry_price) / self.entry_price) * 100
        else:
            return ((self.entry_price - self.lowest_price) / self.entry_price) * 100

    @property
    def mae_pct(self) -> float:
        """Maximum Adverse Excursion percentage (as positive number)."""
        if self.side == "BUY":
            return ((self.entry_price - self.lowest_price) / self.entry_price) * 100
        else:
            return ((self.highest_price - self.entry_price) / self.entry_price) * 100

    @property
    def holding_time_minutes(self) -> float:
        """Time held in minutes."""
        return (datetime.datetime.utcnow() - self.entry_time).total_seconds() / 60


@dataclass
class ExitDecision:
    """Exit decision with rationale."""
    action: ExitReason
    exit_now: bool
    adjusted_stop: Optional[float]
    adjusted_target: Optional[float]
    urgency: float                      # 0-100 how urgent is this exit
    reason: str
    details: Dict = field(default_factory=dict)


@dataclass
class ExitProfile:
    """Historical exit profile from evidence data."""
    avg_mfe: float = 4.5               # Average MFE from winning trades
    avg_mae: float = 1.8               # Average MAE from winning trades
    avg_winner_mfe: float = 5.2        # MFE of winners specifically
    avg_loser_mae: float = 3.5         # MAE of losers specifically
    avg_hold_time_winners: float = 180  # Minutes
    avg_hold_time_losers: float = 120   # Minutes
    fast_profit_threshold: float = 2.0  # MFE reached in < 30 min
    stagnation_threshold: float = 120   # Minutes without progress


class ExitIntelligence:
    """
    Intelligent exit management system.

    Uses trade behavior and historical patterns to make
    dynamic exit decisions that improve profit quality.

    Key strategies:
    1. Fast profit protection - trail aggressively when MFE reached quickly
    2. Stagnation exit - cut trades that aren't moving
    3. MAE-based protection - exit before typical losing trade MAE
    4. Momentum fade detection - exit when momentum dies
    """

    def __init__(self, profile: ExitProfile = None):
        """
        Initialize exit intelligence.

        Args:
            profile: Historical exit profile (from evidence collector)
        """
        self.profile = profile or ExitProfile()

        # Default parameters
        self.base_stop_pct = 2.0
        self.base_target_pct = 3.0
        self.trailing_activation_pct = 2.0
        self.trailing_distance_pct = 1.0
        self.break_even_threshold_pct = 1.5
        self.max_hold_time_minutes = 360  # 6 hours max

    def analyze_trade(self, trade: TradeState) -> ExitDecision:
        """
        Analyze open trade and provide exit decision.

        Args:
            trade: Current trade state

        Returns:
            ExitDecision with action and rationale
        """
        decisions = []

        # Check each exit condition
        decisions.append(self._check_stop_loss(trade))
        decisions.append(self._check_take_profit(trade))
        decisions.append(self._check_trailing_stop(trade))
        decisions.append(self._check_stagnation(trade))
        decisions.append(self._check_mae_protection(trade))
        decisions.append(self._check_momentum_fade(trade))
        decisions.append(self._check_break_even(trade))

        # Find highest urgency exit decision
        exit_decisions = [d for d in decisions if d.exit_now]
        if exit_decisions:
            # Return most urgent exit
            return max(exit_decisions, key=lambda d: d.urgency)

        # No exit needed - return hold with any adjustments
        hold_decisions = [d for d in decisions if not d.exit_now]
        if hold_decisions:
            # Merge stop/target adjustments
            best = max(hold_decisions, key=lambda d: d.urgency)

            # Get tightest stop and target adjustments
            adjusted_stops = [d.adjusted_stop for d in hold_decisions if d.adjusted_stop]
            adjusted_targets = [d.adjusted_target for d in hold_decisions if d.adjusted_target]

            if adjusted_stops:
                if trade.side == "BUY":
                    best.adjusted_stop = max(adjusted_stops)  # Highest stop for longs
                else:
                    best.adjusted_stop = min(adjusted_stops)  # Lowest stop for shorts

            if adjusted_targets:
                if trade.side == "BUY":
                    best.adjusted_target = min(adjusted_targets)  # Closest target
                else:
                    best.adjusted_target = max(adjusted_targets)

            return best

        return ExitDecision(
            action=ExitReason.HOLD,
            exit_now=False,
            adjusted_stop=None,
            adjusted_target=None,
            urgency=0,
            reason="No exit conditions met"
        )

    def _check_stop_loss(self, trade: TradeState) -> ExitDecision:
        """Check if stop loss is hit."""
        stop_price = trade.entry_price * (1 - self.base_stop_pct / 100) if trade.side == "BUY" \
            else trade.entry_price * (1 + self.base_stop_pct / 100)

        is_stopped = (trade.side == "BUY" and trade.current_price <= stop_price) or \
                     (trade.side == "SELL" and trade.current_price >= stop_price)

        if is_stopped:
            return ExitDecision(
                action=ExitReason.STOP_LOSS,
                exit_now=True,
                adjusted_stop=None,
                adjusted_target=None,
                urgency=100,
                reason=f"Stop loss hit at {self.base_stop_pct}%",
                details={"stop_price": stop_price, "current_pnl": trade.pnl_pct}
            )

        return ExitDecision(
            action=ExitReason.HOLD,
            exit_now=False,
            adjusted_stop=stop_price,
            adjusted_target=None,
            urgency=0,
            reason="Stop loss not hit"
        )

    def _check_take_profit(self, trade: TradeState) -> ExitDecision:
        """Check if take profit is hit."""
        target_price = trade.entry_price * (1 + self.base_target_pct / 100) if trade.side == "BUY" \
            else trade.entry_price * (1 - self.base_target_pct / 100)

        is_target_hit = (trade.side == "BUY" and trade.current_price >= target_price) or \
                        (trade.side == "SELL" and trade.current_price <= target_price)

        if is_target_hit:
            return ExitDecision(
                action=ExitReason.TAKE_PROFIT,
                exit_now=True,
                adjusted_stop=None,
                adjusted_target=None,
                urgency=90,
                reason=f"Take profit hit at {self.base_target_pct}%",
                details={"target_price": target_price, "current_pnl": trade.pnl_pct}
            )

        return ExitDecision(
            action=ExitReason.HOLD,
            exit_now=False,
            adjusted_stop=None,
            adjusted_target=target_price,
            urgency=0,
            reason="Take profit not hit"
        )

    def _check_trailing_stop(self, trade: TradeState) -> ExitDecision:
        """
        Check trailing stop - activates when MFE exceeds threshold.

        Key insight: If trade reached +3% but now at +1.5%, protect the gain.
        """
        if trade.mfe_pct < self.trailing_activation_pct:
            return ExitDecision(
                action=ExitReason.HOLD,
                exit_now=False,
                adjusted_stop=None,
                adjusted_target=None,
                urgency=0,
                reason="Trailing stop not activated yet"
            )

        # Calculate trailing stop level
        if trade.side == "BUY":
            trail_stop = trade.highest_price * (1 - self.trailing_distance_pct / 100)
            is_trailing_hit = trade.current_price <= trail_stop
        else:
            trail_stop = trade.lowest_price * (1 + self.trailing_distance_pct / 100)
            is_trailing_hit = trade.current_price >= trail_stop

        # Check how much profit we're giving back
        profit_giveback = trade.mfe_pct - trade.pnl_pct

        if is_trailing_hit:
            return ExitDecision(
                action=ExitReason.TRAILING_STOP,
                exit_now=True,
                adjusted_stop=None,
                adjusted_target=None,
                urgency=85,
                reason=f"Trailing stop hit. MFE was {trade.mfe_pct:.1f}%, locking {trade.pnl_pct:.1f}%",
                details={
                    "mfe": trade.mfe_pct,
                    "current_pnl": trade.pnl_pct,
                    "profit_protected": trade.pnl_pct,
                    "profit_giveback": profit_giveback
                }
            )

        # Adjust stop to trailing level
        return ExitDecision(
            action=ExitReason.HOLD,
            exit_now=False,
            adjusted_stop=trail_stop,
            adjusted_target=None,
            urgency=40,
            reason=f"Trailing stop active at ${trail_stop:.4f} (protecting {trade.pnl_pct:.1f}% of {trade.mfe_pct:.1f}% MFE)",
            details={
                "mfe": trade.mfe_pct,
                "trailing_stop": trail_stop,
                "profit_at_risk": profit_giveback
            }
        )

    def _check_stagnation(self, trade: TradeState) -> ExitDecision:
        """
        Check for trade stagnation - exit if no progress.

        Key insight: Winning trades typically move fast.
        If a trade hasn't moved in X minutes, it's likely a loser.
        """
        hold_time = trade.holding_time_minutes

        # Check if trade has been flat for too long
        if hold_time < self.profile.stagnation_threshold:
            return ExitDecision(
                action=ExitReason.HOLD,
                exit_now=False,
                adjusted_stop=None,
                adjusted_target=None,
                urgency=0,
                reason=f"Trade held {hold_time:.0f} min (stagnation check at {self.profile.stagnation_threshold})"
            )

        # Trade is old - check if it's made progress
        if trade.mfe_pct < 1.0 and abs(trade.pnl_pct) < 0.5:
            return ExitDecision(
                action=ExitReason.TIME_EXIT,
                exit_now=True,
                adjusted_stop=None,
                adjusted_target=None,
                urgency=60,
                reason=f"Stagnation exit: {hold_time:.0f} min with only {trade.mfe_pct:.1f}% MFE",
                details={
                    "hold_time": hold_time,
                    "mfe": trade.mfe_pct,
                    "current_pnl": trade.pnl_pct
                }
            )

        # Max hold time check
        if hold_time > self.max_hold_time_minutes:
            return ExitDecision(
                action=ExitReason.TIME_EXIT,
                exit_now=True,
                adjusted_stop=None,
                adjusted_target=None,
                urgency=70,
                reason=f"Max hold time exceeded: {hold_time:.0f} min > {self.max_hold_time_minutes}",
                details={
                    "hold_time": hold_time,
                    "current_pnl": trade.pnl_pct
                }
            )

        return ExitDecision(
            action=ExitReason.HOLD,
            exit_now=False,
            adjusted_stop=None,
            adjusted_target=None,
            urgency=0,
            reason="No stagnation detected"
        )

    def _check_mae_protection(self, trade: TradeState) -> ExitDecision:
        """
        Check MAE-based protection - exit before typical loser MAE.

        Key insight: Losing trades have a characteristic MAE pattern.
        If current MAE approaches typical loser MAE, cut early.
        """
        typical_loser_mae = self.profile.avg_loser_mae
        warning_threshold = typical_loser_mae * 0.7  # 70% of typical loser MAE

        if trade.mae_pct >= typical_loser_mae:
            return ExitDecision(
                action=ExitReason.MAE_PROTECTION,
                exit_now=True,
                adjusted_stop=None,
                adjusted_target=None,
                urgency=75,
                reason=f"MAE protection: {trade.mae_pct:.1f}% MAE >= typical loser ({typical_loser_mae:.1f}%)",
                details={
                    "current_mae": trade.mae_pct,
                    "typical_loser_mae": typical_loser_mae,
                    "current_pnl": trade.pnl_pct
                }
            )

        if trade.mae_pct >= warning_threshold:
            # Tighten stop
            if trade.side == "BUY":
                tight_stop = trade.entry_price * (1 - typical_loser_mae * 0.8 / 100)
            else:
                tight_stop = trade.entry_price * (1 + typical_loser_mae * 0.8 / 100)

            return ExitDecision(
                action=ExitReason.HOLD,
                exit_now=False,
                adjusted_stop=tight_stop,
                adjusted_target=None,
                urgency=50,
                reason=f"MAE warning: {trade.mae_pct:.1f}% approaching typical loser ({typical_loser_mae:.1f}%)",
                details={
                    "current_mae": trade.mae_pct,
                    "tight_stop": tight_stop
                }
            )

        return ExitDecision(
            action=ExitReason.HOLD,
            exit_now=False,
            adjusted_stop=None,
            adjusted_target=None,
            urgency=0,
            reason="MAE within normal range"
        )

    def _check_momentum_fade(self, trade: TradeState) -> ExitDecision:
        """
        Check for momentum fade - profit giving back too much.

        Key insight: If trade was +3% but now +0.5%, momentum has faded.
        Better to exit with small profit than risk it becoming a loser.
        """
        if trade.mfe_pct < 2.0:
            return ExitDecision(
                action=ExitReason.HOLD,
                exit_now=False,
                adjusted_stop=None,
                adjusted_target=None,
                urgency=0,
                reason="Not enough MFE for momentum fade check"
            )

        # Calculate giveback percentage
        giveback_pct = trade.mfe_pct - trade.pnl_pct
        giveback_ratio = giveback_pct / trade.mfe_pct if trade.mfe_pct > 0 else 0

        # If given back more than 60% of profits, momentum is fading
        if giveback_ratio > 0.6 and trade.pnl_pct > 0:
            return ExitDecision(
                action=ExitReason.MOMENTUM_FADE,
                exit_now=True,
                adjusted_stop=None,
                adjusted_target=None,
                urgency=65,
                reason=f"Momentum fade: MFE {trade.mfe_pct:.1f}% → now {trade.pnl_pct:.1f}% ({giveback_ratio*100:.0f}% giveback)",
                details={
                    "mfe": trade.mfe_pct,
                    "current_pnl": trade.pnl_pct,
                    "giveback_pct": giveback_pct,
                    "giveback_ratio": giveback_ratio
                }
            )

        if giveback_ratio > 0.4:
            return ExitDecision(
                action=ExitReason.HOLD,
                exit_now=False,
                adjusted_stop=None,
                adjusted_target=None,
                urgency=30,
                reason=f"Momentum weakening: {giveback_ratio*100:.0f}% profit giveback",
                details={"giveback_ratio": giveback_ratio}
            )

        return ExitDecision(
            action=ExitReason.HOLD,
            exit_now=False,
            adjusted_stop=None,
            adjusted_target=None,
            urgency=0,
            reason="Momentum intact"
        )

    def _check_break_even(self, trade: TradeState) -> ExitDecision:
        """
        Check if stop should be moved to break-even.

        Once trade reaches threshold profit, protect entry.
        """
        if trade.mfe_pct >= self.break_even_threshold_pct:
            # Calculate break-even stop (with small buffer)
            buffer = 0.1  # 0.1% buffer above entry
            if trade.side == "BUY":
                be_stop = trade.entry_price * (1 + buffer / 100)
            else:
                be_stop = trade.entry_price * (1 - buffer / 100)

            # Only suggest if current stop is worse
            current_stop = trade.entry_price * (1 - self.base_stop_pct / 100) if trade.side == "BUY" \
                else trade.entry_price * (1 + self.base_stop_pct / 100)

            should_adjust = (trade.side == "BUY" and be_stop > current_stop) or \
                           (trade.side == "SELL" and be_stop < current_stop)

            if should_adjust:
                return ExitDecision(
                    action=ExitReason.BREAK_EVEN,
                    exit_now=False,
                    adjusted_stop=be_stop,
                    adjusted_target=None,
                    urgency=35,
                    reason=f"Move stop to break-even: MFE reached {trade.mfe_pct:.1f}%",
                    details={
                        "break_even_stop": be_stop,
                        "mfe": trade.mfe_pct
                    }
                )

        return ExitDecision(
            action=ExitReason.HOLD,
            exit_now=False,
            adjusted_stop=None,
            adjusted_target=None,
            urgency=0,
            reason="Break-even not triggered yet"
        )

    def print_analysis(self, trade: TradeState):
        """Print detailed exit analysis."""
        decision = self.analyze_trade(trade)

        print("\n" + "=" * 60)
        print(f"EXIT INTELLIGENCE ANALYSIS")
        print("=" * 60)

        print(f"\n[TRADE STATUS]")
        print(f"  Symbol:      {trade.symbol}")
        print(f"  Side:        {trade.side}")
        print(f"  Entry:       ${trade.entry_price:.4f}")
        print(f"  Current:     ${trade.current_price:.4f}")
        print(f"  P&L:         {trade.pnl_pct:+.2f}%")
        print(f"  MFE:         {trade.mfe_pct:.2f}%")
        print(f"  MAE:         {trade.mae_pct:.2f}%")
        print(f"  Hold Time:   {trade.holding_time_minutes:.0f} min")

        print(f"\n[DECISION]")
        print(f"  Action:      {decision.action.value.upper()}")
        print(f"  Exit Now:    {'YES' if decision.exit_now else 'NO'}")
        print(f"  Urgency:     {decision.urgency}/100")
        print(f"  Reason:      {decision.reason}")

        if decision.adjusted_stop:
            print(f"  Adj. Stop:   ${decision.adjusted_stop:.4f}")
        if decision.adjusted_target:
            print(f"  Adj. Target: ${decision.adjusted_target:.4f}")

        if decision.details:
            print(f"\n[DETAILS]")
            for key, value in decision.details.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")

        print("=" * 60)

        return decision


# =============================================================================
# MAIN - Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("EXIT INTELLIGENCE TEST")
    print("=" * 60)

    intelligence = ExitIntelligence()

    # Test scenarios
    scenarios = [
        # Scenario 1: Profitable trade with good MFE
        TradeState(
            symbol="ATOMUSDT",
            side="BUY",
            entry_price=10.0,
            entry_time=datetime.datetime.utcnow() - datetime.timedelta(hours=1),
            current_price=10.25,
            highest_price=10.35,
            lowest_price=9.95,
            position_size=200
        ),
        # Scenario 2: Trade giving back profits
        TradeState(
            symbol="ATOMUSDT",
            side="BUY",
            entry_price=10.0,
            entry_time=datetime.datetime.utcnow() - datetime.timedelta(hours=2),
            current_price=10.08,
            highest_price=10.40,
            lowest_price=9.98,
            position_size=200
        ),
        # Scenario 3: Stagnant trade
        TradeState(
            symbol="ATOMUSDT",
            side="BUY",
            entry_price=10.0,
            entry_time=datetime.datetime.utcnow() - datetime.timedelta(hours=3),
            current_price=10.02,
            highest_price=10.08,
            lowest_price=9.97,
            position_size=200
        ),
        # Scenario 4: High MAE approaching loser pattern
        TradeState(
            symbol="ATOMUSDT",
            side="BUY",
            entry_price=10.0,
            entry_time=datetime.datetime.utcnow() - datetime.timedelta(minutes=45),
            current_price=9.72,
            highest_price=10.05,
            lowest_price=9.65,
            position_size=200
        ),
    ]

    for i, trade in enumerate(scenarios, 1):
        print(f"\n{'='*60}")
        print(f"SCENARIO {i}")
        intelligence.print_analysis(trade)
