"""
Position Manager for LongTrader

Manages position lifecycle:
- Open new positions
- Track open positions
- Close positions with P&L calculation
- Position queries and statistics
"""

import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any

from .database import Database, Position, OrderRecord, SignalLog


class PositionManager:
    """
    High-level position management.

    Wraps Database class with business logic for:
    - Position opening with validation
    - Position closing with P&L
    - Position queries
    """

    def __init__(self, db: Database):
        self.db = db

    def open_position(
        self,
        symbol: str,
        entry_price: float,
        entry_qty: float,
        order_id: str = "",
    ) -> Position:
        """
        Open a new position.

        Args:
            symbol: Trading pair (e.g., "XRPUSDT")
            entry_price: Entry price
            entry_qty: Quantity bought
            order_id: Bybit order ID

        Returns:
            Created Position object
        """
        # Check if position already exists for symbol
        existing = self.db.get_position_by_symbol(symbol)
        if existing:
            raise ValueError(f"Position already open for {symbol}")

        position = Position(
            id=str(uuid.uuid4()),
            symbol=symbol,
            side="LONG",
            entry_price=entry_price,
            entry_qty=entry_qty,
            entry_date=datetime.now(),
            entry_order_id=order_id,
            status="OPEN",
        )

        self.db.create_position(position)
        return position

    def close_position(
        self,
        position_id: str,
        exit_price: float,
        reason: str,
        order_id: str = "",
    ) -> Optional[Position]:
        """
        Close a position.

        Args:
            position_id: Position UUID
            exit_price: Exit price
            reason: Exit reason (TARGET, TIME_EXIT, MANUAL, etc.)
            order_id: Bybit order ID

        Returns:
            Closed Position with P&L calculated
        """
        return self.db.close_position(
            position_id=position_id,
            exit_price=exit_price,
            exit_reason=reason,
            exit_order_id=order_id,
        )

    def close_position_by_symbol(
        self,
        symbol: str,
        exit_price: float,
        reason: str,
        order_id: str = "",
    ) -> Optional[Position]:
        """
        Close position by symbol.

        Args:
            symbol: Trading pair
            exit_price: Exit price
            reason: Exit reason
            order_id: Bybit order ID

        Returns:
            Closed Position or None if no open position
        """
        position = self.db.get_position_by_symbol(symbol)
        if not position:
            return None

        return self.close_position(
            position_id=position.id,
            exit_price=exit_price,
            reason=reason,
            order_id=order_id,
        )

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get open position for symbol."""
        return self.db.get_position_by_symbol(symbol)

    def get_open_positions(self) -> List[Position]:
        """Get all open positions."""
        return self.db.get_open_positions()

    def get_open_position_count(self) -> int:
        """Get number of open positions."""
        return len(self.db.get_open_positions())

    def has_position(self, symbol: str) -> bool:
        """Check if position exists for symbol."""
        return self.db.get_position_by_symbol(symbol) is not None

    def get_position_value(self, position: Position, current_price: float) -> float:
        """Calculate current position value."""
        return position.entry_qty * current_price

    def get_position_pnl(self, position: Position, current_price: float) -> Dict[str, float]:
        """
        Calculate unrealized P&L for a position.

        Returns:
            Dict with pnl_usdt and pnl_pct
        """
        pnl_usdt = (current_price - position.entry_price) * position.entry_qty
        pnl_pct = ((current_price - position.entry_price) / position.entry_price) * 100

        return {
            "pnl_usdt": round(pnl_usdt, 2),
            "pnl_pct": round(pnl_pct, 2),
        }

    def get_total_exposure(self, positions: List[Position] = None) -> float:
        """Calculate total USDT exposure across positions."""
        if positions is None:
            positions = self.get_open_positions()

        return sum(p.entry_price * p.entry_qty for p in positions)

    def log_signal(
        self,
        symbol: str,
        signal_type: str,
        price: float,
        lower_band: float,
        middle_band: float,
        protection_passed: int,
        protection_details: str,
        action_taken: str,
    ):
        """Log a signal for audit."""
        signal = SignalLog(
            symbol=symbol,
            signal_type=signal_type,
            price=price,
            lower_band=lower_band,
            middle_band=middle_band,
            protection_passed=protection_passed,
            protection_details=protection_details,
            action_taken=action_taken,
            created_at=datetime.now(),
        )
        self.db.log_signal(signal)

    def log_order(
        self,
        position_id: Optional[str],
        symbol: str,
        side: str,
        order_type: str,
        qty: float,
        price: Optional[float],
        filled_qty: float,
        filled_price: float,
        status: str,
        bybit_order_id: str,
    ) -> str:
        """Log an order."""
        order = OrderRecord(
            id=str(uuid.uuid4()),
            position_id=position_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            qty=qty,
            price=price,
            filled_qty=filled_qty,
            filled_price=filled_price,
            status=status,
            bybit_order_id=bybit_order_id,
            created_at=datetime.now(),
        )
        return self.db.create_order(order)

    def get_stats(self) -> Dict[str, Any]:
        """Get trading statistics."""
        return self.db.get_stats()

    def get_recent_positions(self, limit: int = 10) -> List[Position]:
        """Get recent positions."""
        return self.db.get_all_positions(limit=limit)


# Test function
def test_position_manager():
    """Test the position manager."""
    print("Testing Position Manager...")
    print("=" * 50)

    db = Database("data/test_longtrader.db")
    pm = PositionManager(db)

    # Open a position
    try:
        pos = pm.open_position(
            symbol="TESTUSDT",
            entry_price=100.0,
            entry_qty=10.0,
            order_id="test-order-1",
        )
        print(f"Opened position: {pos.id[:8]}...")
    except ValueError as e:
        print(f"Position already exists: {e}")
        pos = pm.get_position("TESTUSDT")

    # Get open positions
    positions = pm.get_open_positions()
    print(f"Open positions: {len(positions)}")

    # Calculate P&L
    if pos:
        pnl = pm.get_position_pnl(pos, 110.0)
        print(f"Unrealized P&L: ${pnl['pnl_usdt']} ({pnl['pnl_pct']}%)")

    # Get stats
    stats = pm.get_stats()
    print(f"Stats: {stats}")

    print("\nPosition Manager working!")


if __name__ == "__main__":
    test_position_manager()
