#!/usr/bin/env python3
"""
Capital Heat Management - Portfolio-Level Exposure Control

Phase 2 Risk Evolution Component #3

Manages total portfolio exposure to prevent correlated disasters:

1. Track total capital at risk across all positions
2. Detect when multiple symbols might move together (correlation)
3. Limit new positions when heat is high
4. Reduce size when approaching limits

KEY INSIGHT: Even with per-trade limits, if ATOM, SOL, and XRP
all align signals at once, you could have 3x normal exposure.

This module ensures portfolio-level safety.
"""

import ssl_bypass  # Must be first!
import os
import json
import requests
import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from pathlib import Path


@dataclass
class Position:
    """An open trading position."""
    symbol: str
    side: str                   # "LONG" or "SHORT"
    entry_price: float
    current_price: float
    size_usd: float
    entry_time: datetime.datetime
    stop_loss: float
    risk_usd: float             # $ at risk (size * stop distance %)

    @property
    def pnl_usd(self) -> float:
        """Current P&L in USD."""
        if self.side == "LONG":
            return self.size_usd * ((self.current_price - self.entry_price) / self.entry_price)
        else:
            return self.size_usd * ((self.entry_price - self.current_price) / self.entry_price)

    @property
    def pnl_pct(self) -> float:
        """Current P&L percentage."""
        if self.side == "LONG":
            return ((self.current_price - self.entry_price) / self.entry_price) * 100
        else:
            return ((self.entry_price - self.current_price) / self.entry_price) * 100


@dataclass
class HeatStatus:
    """Portfolio heat assessment."""
    total_exposure_usd: float
    total_exposure_pct: float
    total_risk_usd: float       # Total $ at risk if all stops hit
    total_risk_pct: float       # Risk as % of capital
    position_count: int
    correlation_risk: float     # 0-100 how correlated positions are
    heat_level: str             # "low", "medium", "high", "critical"
    new_trade_allowed: bool
    max_new_position_pct: float # Maximum size for new position
    reason: str
    positions: List[Position] = field(default_factory=list)


class CapitalHeatManager:
    """
    Portfolio-level capital and exposure management.

    Tracks:
    - Total exposure across all positions
    - Total risk ($ that could be lost if stops hit)
    - Correlation between positions
    - Heat level (how much capacity remains)

    Prevents:
    - Overexposure when multiple signals align
    - Correlated disasters (all crypto down together)
    - Exceeding portfolio risk limits
    """

    def __init__(
        self,
        total_capital: float = 10000.0,
        max_total_exposure_pct: float = 20.0,
        max_total_risk_pct: float = 6.0,
        max_positions: int = 3,
        max_single_position_pct: float = 5.0,
        correlation_penalty: float = 0.5,
        state_file: str = None
    ):
        """
        Initialize capital heat manager.

        Args:
            total_capital: Total trading capital
            max_total_exposure_pct: Maximum total exposure as % of capital
            max_total_risk_pct: Maximum total risk as % of capital
            max_positions: Maximum concurrent positions
            max_single_position_pct: Maximum single position size as % of capital
            correlation_penalty: How much to reduce size for correlated assets
            state_file: Path to persist state
        """
        self.total_capital = total_capital
        self.max_total_exposure_pct = max_total_exposure_pct
        self.max_total_risk_pct = max_total_risk_pct
        self.max_positions = max_positions
        self.max_single_position_pct = max_single_position_pct
        self.correlation_penalty = correlation_penalty

        # State file
        if state_file is None:
            state_file = str(Path(__file__).parent / "capital_heat_state.json")
        self.state_file = state_file

        # Open positions
        self.positions: Dict[str, Position] = {}

        # Correlation matrix (simplified - all crypto is correlated)
        self.correlation_groups = {
            "crypto": ["ATOMUSDT", "SOLUSDT", "XRPUSDT", "BTCUSDT", "ETHUSDT"],
        }

        # Load state
        self._load_state()

    def _load_state(self):
        """Load persisted state."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    state = json.load(f)

                self.total_capital = state.get("total_capital", self.total_capital)

                # Reconstruct positions
                for sym, pos_data in state.get("positions", {}).items():
                    self.positions[sym] = Position(
                        symbol=pos_data["symbol"],
                        side=pos_data["side"],
                        entry_price=pos_data["entry_price"],
                        current_price=pos_data["current_price"],
                        size_usd=pos_data["size_usd"],
                        entry_time=datetime.datetime.fromisoformat(pos_data["entry_time"]),
                        stop_loss=pos_data["stop_loss"],
                        risk_usd=pos_data["risk_usd"]
                    )

            except Exception as e:
                print(f"Warning: Could not load heat state: {e}")

    def _save_state(self):
        """Persist state."""
        try:
            positions_data = {}
            for sym, pos in self.positions.items():
                positions_data[sym] = {
                    "symbol": pos.symbol,
                    "side": pos.side,
                    "entry_price": pos.entry_price,
                    "current_price": pos.current_price,
                    "size_usd": pos.size_usd,
                    "entry_time": pos.entry_time.isoformat(),
                    "stop_loss": pos.stop_loss,
                    "risk_usd": pos.risk_usd
                }

            state = {
                "total_capital": self.total_capital,
                "positions": positions_data,
                "updated": datetime.datetime.utcnow().isoformat()
            }

            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=2)

        except Exception as e:
            print(f"Warning: Could not save heat state: {e}")

    def _calculate_correlation_risk(self) -> float:
        """
        Calculate correlation risk score.

        If all positions are in the same correlation group,
        risk is higher because they'll move together.

        Returns:
            Correlation risk score 0-100
        """
        if len(self.positions) <= 1:
            return 0.0

        # Check how many positions are in same group
        symbols = list(self.positions.keys())
        group_counts = {}

        for group_name, group_symbols in self.correlation_groups.items():
            count = sum(1 for sym in symbols if sym in group_symbols)
            if count > 0:
                group_counts[group_name] = count

        if not group_counts:
            return 0.0

        # If most positions are in same group, high correlation
        max_in_group = max(group_counts.values())
        correlation_ratio = max_in_group / len(self.positions)

        return correlation_ratio * 100

    def _determine_heat_level(
        self,
        exposure_pct: float,
        risk_pct: float,
        position_count: int,
        correlation_risk: float
    ) -> Tuple[str, str]:
        """
        Determine heat level and reason.

        Returns:
            Tuple of (heat_level, reason)
        """
        # Critical checks
        if risk_pct >= self.max_total_risk_pct:
            return "critical", f"Risk at limit: {risk_pct:.1f}% >= {self.max_total_risk_pct}%"

        if exposure_pct >= self.max_total_exposure_pct:
            return "critical", f"Exposure at limit: {exposure_pct:.1f}% >= {self.max_total_exposure_pct}%"

        if position_count >= self.max_positions:
            return "critical", f"Position count at limit: {position_count} >= {self.max_positions}"

        # High checks (80%+ of limits)
        if risk_pct >= self.max_total_risk_pct * 0.8:
            return "high", f"Risk approaching limit: {risk_pct:.1f}%"

        if exposure_pct >= self.max_total_exposure_pct * 0.8:
            return "high", f"Exposure approaching limit: {exposure_pct:.1f}%"

        if correlation_risk >= 80:
            return "high", f"High correlation risk: {correlation_risk:.0f}%"

        # Medium checks (50-80% of limits)
        if risk_pct >= self.max_total_risk_pct * 0.5:
            return "medium", f"Moderate risk: {risk_pct:.1f}%"

        if exposure_pct >= self.max_total_exposure_pct * 0.5:
            return "medium", f"Moderate exposure: {exposure_pct:.1f}%"

        if correlation_risk >= 50:
            return "medium", f"Moderate correlation: {correlation_risk:.0f}%"

        return "low", "Heat is low - full capacity available"

    def get_heat_status(self) -> HeatStatus:
        """
        Get current portfolio heat status.

        Returns:
            Complete HeatStatus assessment
        """
        # Calculate totals
        total_exposure = sum(pos.size_usd for pos in self.positions.values())
        total_risk = sum(pos.risk_usd for pos in self.positions.values())

        exposure_pct = (total_exposure / self.total_capital * 100) if self.total_capital > 0 else 0
        risk_pct = (total_risk / self.total_capital * 100) if self.total_capital > 0 else 0

        correlation_risk = self._calculate_correlation_risk()

        heat_level, reason = self._determine_heat_level(
            exposure_pct, risk_pct, len(self.positions), correlation_risk
        )

        # Determine if new trade allowed
        new_trade_allowed = heat_level not in ["critical"]

        # Calculate max new position size
        remaining_exposure = max(0, self.max_total_exposure_pct - exposure_pct)
        remaining_risk = max(0, self.max_total_risk_pct - risk_pct)

        # Use the more restrictive limit
        max_new_pct = min(remaining_exposure, self.max_single_position_pct)

        # Apply correlation penalty if positions exist
        if len(self.positions) > 0 and correlation_risk > 50:
            max_new_pct *= (1 - self.correlation_penalty)

        # Apply heat level reduction
        heat_multipliers = {
            "low": 1.0,
            "medium": 0.75,
            "high": 0.5,
            "critical": 0.0
        }
        max_new_pct *= heat_multipliers.get(heat_level, 1.0)

        return HeatStatus(
            total_exposure_usd=total_exposure,
            total_exposure_pct=exposure_pct,
            total_risk_usd=total_risk,
            total_risk_pct=risk_pct,
            position_count=len(self.positions),
            correlation_risk=correlation_risk,
            heat_level=heat_level,
            new_trade_allowed=new_trade_allowed,
            max_new_position_pct=max_new_pct,
            reason=reason,
            positions=list(self.positions.values())
        )

    def can_add_position(
        self,
        symbol: str,
        size_usd: float,
        risk_usd: float
    ) -> Tuple[bool, str, float]:
        """
        Check if a new position can be added.

        Args:
            symbol: Trading symbol
            size_usd: Position size in USD
            risk_usd: Amount at risk in USD

        Returns:
            Tuple of (allowed, reason, adjusted_size)
        """
        status = self.get_heat_status()

        # Check if trading allowed
        if not status.new_trade_allowed:
            return False, status.reason, 0.0

        # Check if symbol already has position
        if symbol in self.positions:
            return False, f"Already have position in {symbol}", 0.0

        # Check position count
        if status.position_count >= self.max_positions:
            return False, f"Max positions reached: {status.position_count}/{self.max_positions}", 0.0

        # Calculate what this would add
        new_exposure_pct = (status.total_exposure_usd + size_usd) / self.total_capital * 100
        new_risk_pct = (status.total_risk_usd + risk_usd) / self.total_capital * 100

        # Check exposure limit
        if new_exposure_pct > self.max_total_exposure_pct:
            max_allowed = (self.max_total_exposure_pct / 100 * self.total_capital) - status.total_exposure_usd
            if max_allowed <= 0:
                return False, "Exposure limit reached", 0.0
            return True, f"Size reduced to fit exposure limit", max_allowed

        # Check risk limit
        if new_risk_pct > self.max_total_risk_pct:
            return False, f"Would exceed risk limit: {new_risk_pct:.1f}% > {self.max_total_risk_pct}%", 0.0

        # Check single position limit
        size_pct = size_usd / self.total_capital * 100
        if size_pct > self.max_single_position_pct:
            max_allowed = self.max_single_position_pct / 100 * self.total_capital
            return True, f"Size reduced to single position limit", max_allowed

        # Apply heat-based reduction
        if status.heat_level == "high":
            reduced = size_usd * 0.5
            return True, f"Size reduced 50% due to high heat", reduced

        if status.heat_level == "medium" and status.correlation_risk > 50:
            reduced = size_usd * 0.75
            return True, f"Size reduced 25% due to correlation risk", reduced

        return True, "Position approved at full size", size_usd

    def add_position(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        size_usd: float,
        stop_loss: float
    ):
        """
        Add a new position to tracking.

        Args:
            symbol: Trading symbol
            side: "LONG" or "SHORT"
            entry_price: Entry price
            size_usd: Position size in USD
            stop_loss: Stop loss price
        """
        # Calculate risk
        if side == "LONG":
            risk_pct = (entry_price - stop_loss) / entry_price
        else:
            risk_pct = (stop_loss - entry_price) / entry_price

        risk_usd = size_usd * abs(risk_pct)

        position = Position(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            current_price=entry_price,
            size_usd=size_usd,
            entry_time=datetime.datetime.utcnow(),
            stop_loss=stop_loss,
            risk_usd=risk_usd
        )

        self.positions[symbol] = position
        self._save_state()

        print(f"  [HEAT] Position added: {symbol} {side} ${size_usd:.2f} (risk: ${risk_usd:.2f})")

    def update_position_price(self, symbol: str, current_price: float):
        """Update current price for a position."""
        if symbol in self.positions:
            self.positions[symbol].current_price = current_price
            self._save_state()

    def close_position(self, symbol: str) -> Optional[Position]:
        """
        Close and remove a position.

        Returns:
            The closed position, or None if not found
        """
        if symbol in self.positions:
            position = self.positions.pop(symbol)
            self._save_state()
            print(f"  [HEAT] Position closed: {symbol} P&L: {position.pnl_pct:+.2f}%")
            return position
        return None

    def update_capital(self, new_capital: float):
        """Update total capital (after P&L realization)."""
        self.total_capital = new_capital
        self._save_state()

    def print_status(self):
        """Print current heat status."""
        status = self.get_heat_status()

        print("\n" + "=" * 60)
        print("CAPITAL HEAT STATUS")
        print("=" * 60)

        print(f"\n[CAPITAL]")
        print(f"  Total Capital: ${self.total_capital:,.2f}")

        print(f"\n[EXPOSURE]")
        print(f"  Total Exposure: ${status.total_exposure_usd:,.2f} ({status.total_exposure_pct:.1f}%)")
        print(f"  Max Allowed:    ${self.max_total_exposure_pct / 100 * self.total_capital:,.2f} ({self.max_total_exposure_pct}%)")
        print(f"  Remaining:      ${(self.max_total_exposure_pct - status.total_exposure_pct) / 100 * self.total_capital:,.2f}")

        print(f"\n[RISK]")
        print(f"  Total Risk:     ${status.total_risk_usd:,.2f} ({status.total_risk_pct:.1f}%)")
        print(f"  Max Allowed:    ${self.max_total_risk_pct / 100 * self.total_capital:,.2f} ({self.max_total_risk_pct}%)")

        print(f"\n[POSITIONS]")
        print(f"  Count:          {status.position_count}/{self.max_positions}")
        print(f"  Correlation:    {status.correlation_risk:.0f}%")

        if status.positions:
            print(f"\n  Open Positions:")
            for pos in status.positions:
                print(f"    - {pos.symbol} {pos.side}: ${pos.size_usd:.2f} ({pos.pnl_pct:+.2f}%)")

        print(f"\n[HEAT LEVEL]")
        heat_colors = {
            "low": "LOW (green)",
            "medium": "MEDIUM (yellow)",
            "high": "HIGH (orange)",
            "critical": "CRITICAL (red)"
        }
        print(f"  Level:          {heat_colors.get(status.heat_level, status.heat_level)}")
        print(f"  Reason:         {status.reason}")

        print(f"\n[NEW POSITIONS]")
        print(f"  Allowed:        {'YES' if status.new_trade_allowed else 'NO'}")
        print(f"  Max Size:       {status.max_new_position_pct:.1f}% (${status.max_new_position_pct / 100 * self.total_capital:,.2f})")

        print("=" * 60)


# =============================================================================
# MAIN - Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CAPITAL HEAT MANAGER TEST")
    print("=" * 60)

    manager = CapitalHeatManager(total_capital=10000.0)
    manager.print_status()

    # Test adding positions
    print("\n[ADDING POSITIONS]")
    print("-" * 40)

    # Position 1
    allowed, reason, size = manager.can_add_position("ATOMUSDT", 500, 25)
    print(f"  ATOMUSDT $500: {reason}")
    if allowed:
        manager.add_position("ATOMUSDT", "LONG", 10.0, size, 9.5)

    manager.print_status()

    # Position 2
    allowed, reason, size = manager.can_add_position("SOLUSDT", 500, 25)
    print(f"\n  SOLUSDT $500: {reason}")
    if allowed:
        manager.add_position("SOLUSDT", "LONG", 150.0, size, 142.5)

    manager.print_status()

    # Position 3 - should hit limits
    allowed, reason, size = manager.can_add_position("XRPUSDT", 500, 25)
    print(f"\n  XRPUSDT $500: {reason}")
    if allowed:
        manager.add_position("XRPUSDT", "LONG", 2.0, size, 1.9)

    manager.print_status()

    # Try to add 4th position - should be blocked
    allowed, reason, size = manager.can_add_position("BTCUSDT", 500, 25)
    print(f"\n  BTCUSDT $500: {reason}")

    # Close a position
    print("\n[CLOSING POSITION]")
    print("-" * 40)
    manager.close_position("SOLUSDT")

    manager.print_status()
