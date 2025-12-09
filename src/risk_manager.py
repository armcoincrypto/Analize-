"""
Risk Manager for LongTrader

Enforces risk limits:
- Position sizing (5% per trade)
- Max positions (3)
- Min balance check
- Max drawdown check
"""

from typing import Dict, Any, Tuple, List
from dataclasses import dataclass

from .database import Position


@dataclass
class RiskCheck:
    """Result of risk check."""
    allowed: bool
    reason: str
    suggested_qty: float = 0.0
    suggested_usdt: float = 0.0


class RiskManager:
    """
    Risk management for trading.

    Enforces:
    - Position size limits
    - Max concurrent positions
    - Minimum balance requirements
    - Drawdown limits
    """

    def __init__(self, config: Dict[str, Any]):
        risk_config = config.get("risk", {})

        self.position_size_pct = risk_config.get("position_size_pct", 5.0)
        self.max_positions = risk_config.get("max_positions", 3)
        self.max_drawdown_pct = risk_config.get("max_drawdown_pct", 15.0)
        self.min_balance_usdt = risk_config.get("min_balance_usdt", 50.0)

    def can_open_position(
        self,
        balance_usdt: float,
        open_positions: List[Position],
        symbol: str,
    ) -> RiskCheck:
        """
        Check if a new position can be opened.

        Args:
            balance_usdt: Available USDT balance
            open_positions: List of current open positions
            symbol: Symbol to trade

        Returns:
            RiskCheck with allowed status and reason
        """
        # Check if already in position
        for pos in open_positions:
            if pos.symbol == symbol:
                return RiskCheck(
                    allowed=False,
                    reason=f"Already have position in {symbol}",
                )

        # Check max positions
        if len(open_positions) >= self.max_positions:
            return RiskCheck(
                allowed=False,
                reason=f"Max positions reached ({self.max_positions})",
            )

        # Check minimum balance
        if balance_usdt < self.min_balance_usdt:
            return RiskCheck(
                allowed=False,
                reason=f"Balance ${balance_usdt:.2f} below minimum ${self.min_balance_usdt}",
            )

        # Calculate position size
        position_usdt = balance_usdt * (self.position_size_pct / 100)

        if position_usdt < 10:
            return RiskCheck(
                allowed=False,
                reason=f"Position size ${position_usdt:.2f} too small",
            )

        return RiskCheck(
            allowed=True,
            reason="Risk checks passed",
            suggested_usdt=position_usdt,
        )

    def calculate_position_size(
        self,
        balance_usdt: float,
        current_price: float,
    ) -> Tuple[float, float]:
        """
        Calculate position size based on balance.

        Args:
            balance_usdt: Available USDT balance
            current_price: Current asset price

        Returns:
            (quantity, usdt_amount)
        """
        usdt_amount = balance_usdt * (self.position_size_pct / 100)
        quantity = usdt_amount / current_price

        return round(quantity, 4), round(usdt_amount, 2)

    def check_drawdown(
        self,
        initial_balance: float,
        current_balance: float,
    ) -> RiskCheck:
        """
        Check if drawdown limit exceeded.

        Args:
            initial_balance: Starting balance
            current_balance: Current balance

        Returns:
            RiskCheck with status
        """
        if initial_balance <= 0:
            return RiskCheck(allowed=True, reason="No initial balance set")

        drawdown_pct = ((initial_balance - current_balance) / initial_balance) * 100

        if drawdown_pct >= self.max_drawdown_pct:
            return RiskCheck(
                allowed=False,
                reason=f"Drawdown {drawdown_pct:.1f}% exceeds limit {self.max_drawdown_pct}%",
            )

        return RiskCheck(
            allowed=True,
            reason=f"Drawdown {drawdown_pct:.1f}% within limits",
        )

    def get_exposure_summary(
        self,
        balance_usdt: float,
        open_positions: List[Position],
        current_prices: Dict[str, float] = None,
    ) -> Dict[str, Any]:
        """
        Get summary of current exposure.

        Args:
            balance_usdt: Available balance
            open_positions: List of open positions
            current_prices: Dict of symbol -> current price

        Returns:
            Exposure summary dict
        """
        total_invested = sum(
            p.entry_price * p.entry_qty for p in open_positions
        )

        # Calculate unrealized P&L if prices provided
        unrealized_pnl = 0.0
        if current_prices:
            for pos in open_positions:
                if pos.symbol in current_prices:
                    current = current_prices[pos.symbol]
                    unrealized_pnl += (current - pos.entry_price) * pos.entry_qty

        total_equity = balance_usdt + total_invested + unrealized_pnl

        return {
            "available_balance": round(balance_usdt, 2),
            "invested": round(total_invested, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "total_equity": round(total_equity, 2),
            "open_positions": len(open_positions),
            "max_positions": self.max_positions,
            "position_slots_free": self.max_positions - len(open_positions),
        }

    def validate_order(
        self,
        order_usdt: float,
        balance_usdt: float,
    ) -> RiskCheck:
        """
        Validate an order amount.

        Args:
            order_usdt: Order size in USDT
            balance_usdt: Available balance

        Returns:
            RiskCheck with status
        """
        if order_usdt > balance_usdt:
            return RiskCheck(
                allowed=False,
                reason=f"Order ${order_usdt:.2f} exceeds balance ${balance_usdt:.2f}",
            )

        max_allowed = balance_usdt * (self.position_size_pct / 100) * 1.1
        if order_usdt > max_allowed:
            return RiskCheck(
                allowed=False,
                reason=f"Order ${order_usdt:.2f} exceeds position limit ${max_allowed:.2f}",
            )

        return RiskCheck(allowed=True, reason="Order validated")


# Test function
def test_risk_manager():
    """Test the risk manager."""
    print("Testing Risk Manager...")
    print("=" * 50)

    config = {
        "risk": {
            "position_size_pct": 5.0,
            "max_positions": 3,
            "max_drawdown_pct": 15.0,
            "min_balance_usdt": 50.0,
        }
    }

    rm = RiskManager(config)

    # Test position check
    check = rm.can_open_position(
        balance_usdt=1000.0,
        open_positions=[],
        symbol="XRPUSDT",
    )
    print(f"Can open position: {check.allowed}")
    print(f"Suggested USDT: ${check.suggested_usdt}")

    # Test position sizing
    qty, usdt = rm.calculate_position_size(1000.0, 2.50)
    print(f"Position size: {qty} XRP (${usdt})")

    # Test drawdown
    dd_check = rm.check_drawdown(1000.0, 900.0)
    print(f"Drawdown check: {dd_check.reason}")

    # Test exposure
    exposure = rm.get_exposure_summary(1000.0, [])
    print(f"Exposure: {exposure}")

    print("\nRisk Manager working!")


if __name__ == "__main__":
    test_risk_manager()
