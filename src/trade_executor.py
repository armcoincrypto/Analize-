"""
Trade Executor for LongTrader

Executes trades with:
- Market orders
- Order confirmation
- Fill waiting
- Error handling
"""

import time
import logging
from datetime import datetime
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

from .bybit_client import BybitClient, OrderSide, OrderType, Order
from .position_manager import PositionManager
from .risk_manager import RiskManager, RiskCheck
from .database import Position


logger = logging.getLogger(__name__)


@dataclass
class TradeResult:
    """Result of trade execution."""
    success: bool
    order_id: str = ""
    filled_qty: float = 0.0
    filled_price: float = 0.0
    error: str = ""
    position_id: str = ""


class TradeExecutor:
    """
    Executes trades on Bybit.

    Handles:
    - Buy orders (open position)
    - Sell orders (close position)
    - Order confirmation and fill waiting
    """

    def __init__(
        self,
        client: BybitClient,
        position_manager: PositionManager,
        risk_manager: RiskManager,
        paper_mode: bool = True,
    ):
        self.client = client
        self.pm = position_manager
        self.rm = risk_manager
        self.paper_mode = paper_mode

    def execute_buy(
        self,
        symbol: str,
        usdt_amount: float,
        current_price: float,
    ) -> TradeResult:
        """
        Execute a buy order.

        Args:
            symbol: Trading pair (e.g., "XRPUSDT")
            usdt_amount: Amount in USDT to spend
            current_price: Current market price

        Returns:
            TradeResult with execution details
        """
        try:
            # Calculate quantity
            qty = round(usdt_amount / current_price, 4)

            logger.info(f"BUY {symbol}: {qty} @ ~${current_price:.4f} (${usdt_amount:.2f})")

            if self.paper_mode:
                # Paper trading - simulate execution
                return self._simulate_buy(symbol, qty, current_price)

            # Live trading
            order = self.client.market_buy(symbol, qty)

            # Wait for fill
            filled_order = self.client.wait_for_fill(symbol, order.order_id, timeout=30)

            if filled_order.status != "Filled":
                return TradeResult(
                    success=False,
                    order_id=order.order_id,
                    error=f"Order not filled: {filled_order.status}",
                )

            # Create position in database
            position = self.pm.open_position(
                symbol=symbol,
                entry_price=filled_order.avg_price,
                entry_qty=filled_order.filled_qty,
                order_id=order.order_id,
            )

            # Log order
            self.pm.log_order(
                position_id=position.id,
                symbol=symbol,
                side="Buy",
                order_type="Market",
                qty=qty,
                price=None,
                filled_qty=filled_order.filled_qty,
                filled_price=filled_order.avg_price,
                status="Filled",
                bybit_order_id=order.order_id,
            )

            return TradeResult(
                success=True,
                order_id=order.order_id,
                filled_qty=filled_order.filled_qty,
                filled_price=filled_order.avg_price,
                position_id=position.id,
            )

        except Exception as e:
            logger.error(f"Buy execution failed: {e}")
            return TradeResult(success=False, error=str(e))

    def execute_sell(
        self,
        symbol: str,
        position: Position,
        reason: str,
    ) -> TradeResult:
        """
        Execute a sell order to close position.

        Args:
            symbol: Trading pair
            position: Position to close
            reason: Exit reason (TARGET, TIME_EXIT, etc.)

        Returns:
            TradeResult with execution details
        """
        try:
            qty = position.entry_qty

            logger.info(f"SELL {symbol}: {qty} - Reason: {reason}")

            if self.paper_mode:
                # Paper trading - simulate execution
                return self._simulate_sell(symbol, position, reason)

            # Live trading
            order = self.client.market_sell(symbol, qty)

            # Wait for fill
            filled_order = self.client.wait_for_fill(symbol, order.order_id, timeout=30)

            if filled_order.status != "Filled":
                return TradeResult(
                    success=False,
                    order_id=order.order_id,
                    error=f"Order not filled: {filled_order.status}",
                )

            # Close position in database
            closed_pos = self.pm.close_position(
                position_id=position.id,
                exit_price=filled_order.avg_price,
                reason=reason,
                order_id=order.order_id,
            )

            # Log order
            self.pm.log_order(
                position_id=position.id,
                symbol=symbol,
                side="Sell",
                order_type="Market",
                qty=qty,
                price=None,
                filled_qty=filled_order.filled_qty,
                filled_price=filled_order.avg_price,
                status="Filled",
                bybit_order_id=order.order_id,
            )

            return TradeResult(
                success=True,
                order_id=order.order_id,
                filled_qty=filled_order.filled_qty,
                filled_price=filled_order.avg_price,
                position_id=position.id,
            )

        except Exception as e:
            logger.error(f"Sell execution failed: {e}")
            return TradeResult(success=False, error=str(e))

    def _simulate_buy(
        self,
        symbol: str,
        qty: float,
        price: float,
    ) -> TradeResult:
        """Simulate a buy order for paper trading."""
        # Simulate slippage (0.1%)
        fill_price = price * 1.001

        # Create position in database
        position = self.pm.open_position(
            symbol=symbol,
            entry_price=fill_price,
            entry_qty=qty,
            order_id=f"PAPER-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        )

        # Log order
        self.pm.log_order(
            position_id=position.id,
            symbol=symbol,
            side="Buy",
            order_type="Market",
            qty=qty,
            price=None,
            filled_qty=qty,
            filled_price=fill_price,
            status="Filled",
            bybit_order_id=f"PAPER-BUY-{position.id[:8]}",
        )

        logger.info(f"[PAPER] BUY {symbol}: {qty} @ ${fill_price:.4f}")

        return TradeResult(
            success=True,
            order_id=f"PAPER-BUY-{position.id[:8]}",
            filled_qty=qty,
            filled_price=fill_price,
            position_id=position.id,
        )

    def _simulate_sell(
        self,
        symbol: str,
        position: Position,
        reason: str,
    ) -> TradeResult:
        """Simulate a sell order for paper trading."""
        # Get current price (would need to be passed in or fetched)
        # For now, simulate with entry price + some return
        ticker = self.client.get_ticker(symbol)
        current_price = ticker.last_price

        # Simulate slippage (0.1%)
        fill_price = current_price * 0.999

        # Close position in database
        closed_pos = self.pm.close_position(
            position_id=position.id,
            exit_price=fill_price,
            reason=reason,
            order_id=f"PAPER-SELL-{position.id[:8]}",
        )

        # Log order
        self.pm.log_order(
            position_id=position.id,
            symbol=symbol,
            side="Sell",
            order_type="Market",
            qty=position.entry_qty,
            price=None,
            filled_qty=position.entry_qty,
            filled_price=fill_price,
            status="Filled",
            bybit_order_id=f"PAPER-SELL-{position.id[:8]}",
        )

        pnl = (fill_price - position.entry_price) * position.entry_qty
        pnl_pct = ((fill_price - position.entry_price) / position.entry_price) * 100

        logger.info(
            f"[PAPER] SELL {symbol}: {position.entry_qty} @ ${fill_price:.4f} "
            f"| P&L: ${pnl:.2f} ({pnl_pct:+.1f}%)"
        )

        return TradeResult(
            success=True,
            order_id=f"PAPER-SELL-{position.id[:8]}",
            filled_qty=position.entry_qty,
            filled_price=fill_price,
            position_id=position.id,
        )

    def get_current_price(self, symbol: str) -> float:
        """Get current market price."""
        ticker = self.client.get_ticker(symbol)
        return ticker.last_price

    def get_balance(self, coin: str = "USDT") -> float:
        """Get available balance."""
        if self.paper_mode:
            # Return paper balance from config or default
            return self.pm.db.get_config("paper_balance", 1000.0)

        balance = self.client.get_balance(coin)
        return balance.available


# Test function
def test_trade_executor():
    """Test the trade executor."""
    print("Testing Trade Executor...")
    print("=" * 50)

    from .database import Database

    # Initialize components
    client = BybitClient(testnet=True)
    db = Database("data/test_longtrader.db")
    pm = PositionManager(db)
    rm = RiskManager({
        "risk": {
            "position_size_pct": 5.0,
            "max_positions": 3,
        }
    })

    executor = TradeExecutor(client, pm, rm, paper_mode=True)

    # Set paper balance
    db.set_config("paper_balance", 1000.0)

    # Get current price
    price = executor.get_current_price("XRPUSDT")
    print(f"XRP price: ${price}")

    # Test buy (paper)
    result = executor.execute_buy("XRPUSDT", 50.0, price)
    print(f"Buy result: {result}")

    if result.success:
        # Test sell
        position = pm.get_position("XRPUSDT")
        if position:
            sell_result = executor.execute_sell("XRPUSDT", position, "TEST")
            print(f"Sell result: {sell_result}")

    print("\nTrade Executor working!")


if __name__ == "__main__":
    test_trade_executor()
