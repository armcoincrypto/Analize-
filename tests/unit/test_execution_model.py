"""Tests for realistic execution model."""

import numpy as np
import pytest

from analize.optimizer.execution_model import (
    ExchangeRules,
    ExecutionResult,
    OrderbookDepth,
    OrderSide,
    OrderType,
    RealisticExecutionModel,
)
from analize.utils.time import utcnow


class TestExchangeRules:
    """Tests for ExchangeRules dataclass."""

    def test_default_rules(self) -> None:
        """Test default exchange rules."""
        rules = ExchangeRules(symbol="BTCUSDT")
        assert rules.tick_size == 0.01
        assert rules.lot_size == 0.001
        assert rules.min_notional == 10.0
        assert rules.maker_fee == 0.02
        assert rules.taker_fee == 0.04

    def test_custom_rules(self) -> None:
        """Test custom exchange rules."""
        rules = ExchangeRules(
            symbol="ETHUSDT",
            tick_size=0.001,
            lot_size=0.0001,
            min_notional=5.0,
            maker_fee=0.01,
            taker_fee=0.02,
        )
        assert rules.tick_size == 0.001
        assert rules.min_notional == 5.0

    def test_round_price(self) -> None:
        """Test price rounding to tick size."""
        rules = ExchangeRules(symbol="TEST", tick_size=0.01)
        assert rules.round_price(100.123) == 100.12
        assert rules.round_price(100.126) == 100.13
        assert rules.round_price(100.00) == 100.00

    def test_round_quantity(self) -> None:
        """Test quantity rounding to lot size."""
        rules = ExchangeRules(symbol="TEST", lot_size=0.001)
        assert rules.round_qty(1.2345) == 1.234
        assert rules.round_qty(1.2349) == 1.235
        assert rules.round_qty(0.0001) == 0.0

    def test_validate_order_min_notional(self) -> None:
        """Test min notional validation."""
        rules = ExchangeRules(symbol="TEST", min_notional=10.0)

        # Should fail - too small
        is_valid, violations = rules.validate_order(
            price=100.0,
            qty=0.05,  # $5 notional
            side=OrderSide.BUY,
        )
        assert not is_valid
        assert any("notional" in v.lower() for v in violations)

        # Should pass
        is_valid, violations = rules.validate_order(
            price=100.0,
            qty=0.2,  # $20 notional
            side=OrderSide.BUY,
        )
        assert is_valid


class TestOrderbookDepth:
    """Tests for OrderbookDepth dataclass."""

    @pytest.fixture
    def sample_orderbook(self) -> OrderbookDepth:
        """Create sample orderbook for testing."""
        return OrderbookDepth(
            bids=[
                (100.00, 10.0),
                (99.99, 20.0),
                (99.98, 30.0),
                (99.95, 50.0),
                (99.90, 100.0),
            ],
            asks=[
                (100.01, 10.0),
                (100.02, 20.0),
                (100.03, 30.0),
                (100.05, 50.0),
                (100.10, 100.0),
            ],
        )

    def test_best_bid_ask(self, sample_orderbook: OrderbookDepth) -> None:
        """Test best bid/ask prices."""
        assert sample_orderbook.best_bid == 100.00
        assert sample_orderbook.best_ask == 100.01

    def test_mid_price(self, sample_orderbook: OrderbookDepth) -> None:
        """Test mid price calculation."""
        # Mid = (100.00 + 100.01) / 2 = 100.005
        assert sample_orderbook.mid_price == pytest.approx(100.005, rel=0.001)

    def test_spread(self, sample_orderbook: OrderbookDepth) -> None:
        """Test spread calculation."""
        # Spread = 100.01 - 100.00 = 0.01
        assert sample_orderbook.spread == pytest.approx(0.01, rel=0.01)

    def test_spread_bps(self, sample_orderbook: OrderbookDepth) -> None:
        """Test spread in basis points."""
        # Spread bps = (0.01 / 100.005) * 10000 ≈ 1 bp
        assert sample_orderbook.spread_bps == pytest.approx(1.0, rel=0.1)

    def test_calculate_market_impact_small_order(self, sample_orderbook: OrderbookDepth) -> None:
        """Test market impact for small orders."""
        # Small order should have minimal impact
        avg_price, slippage = sample_orderbook.calculate_market_impact(
            qty=5.0,  # Less than first level (10)
            side=OrderSide.BUY,
        )

        # Should execute at best ask
        assert avg_price == pytest.approx(100.01, rel=0.001)
        assert slippage < 0.1  # Less than 0.1% slippage

    def test_calculate_market_impact_large_order(self, sample_orderbook: OrderbookDepth) -> None:
        """Test market impact for large orders."""
        # Large order should walk the book
        avg_price, slippage = sample_orderbook.calculate_market_impact(
            qty=50.0,  # More than first 3 levels combined (10+20+30=60)
            side=OrderSide.BUY,
        )

        # Should have higher average price
        assert avg_price > 100.01
        assert slippage > 0.01  # Some slippage


class TestRealisticExecutionModel:
    """Tests for RealisticExecutionModel class."""

    @pytest.fixture
    def model(self) -> RealisticExecutionModel:
        """Create execution model instance."""
        return RealisticExecutionModel()

    @pytest.fixture
    def sample_orderbook(self) -> OrderbookDepth:
        """Create sample orderbook for testing."""
        return OrderbookDepth(
            bids=[
                (100.00, 10.0),
                (99.99, 20.0),
                (99.98, 30.0),
            ],
            asks=[
                (100.01, 10.0),
                (100.02, 20.0),
                (100.03, 30.0),
            ],
        )

    def test_get_rules_known_symbol(self, model: RealisticExecutionModel) -> None:
        """Test getting rules for known symbol."""
        rules = model.get_rules("BTCUSDT")
        assert rules.symbol == "BTCUSDT"
        assert rules.tick_size == 0.01

    def test_get_rules_unknown_symbol(self, model: RealisticExecutionModel) -> None:
        """Test getting default rules for unknown symbol."""
        rules = model.get_rules("UNKNOWN")
        assert rules.symbol == "UNKNOWN"

    def test_simulate_latency(self, model: RealisticExecutionModel) -> None:
        """Test latency simulation."""
        latencies = [model.simulate_latency() for _ in range(100)]

        # All latencies should be positive
        assert all(lat > 0 for lat in latencies)

        # Mean should be around configured mean
        assert np.mean(latencies) == pytest.approx(50.0, rel=0.5)

    def test_estimate_slippage_from_volatility(self, model: RealisticExecutionModel) -> None:
        """Test volatility-based slippage estimation."""
        # Low volatility, small order
        low_slippage = model.estimate_slippage_from_volatility(
            price=100.0,
            qty=0.1,
            position_size_usd=10.0,
            atr_pct=0.5,
        )

        # High volatility, large order
        high_slippage = model.estimate_slippage_from_volatility(
            price=100.0,
            qty=10.0,
            position_size_usd=1000.0,
            atr_pct=2.0,
        )

        assert high_slippage > low_slippage
        assert low_slippage >= 0
        assert high_slippage <= 5.0  # Capped


class TestExecutionResult:
    """Tests for ExecutionResult dataclass."""

    def test_creation(self) -> None:
        """Test ExecutionResult creation."""
        from datetime import datetime

        result = ExecutionResult(
            order_id="ORD-001",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            requested_qty=10.0,
            filled_qty=10.0,
            requested_price=None,
            avg_fill_price=45000.0,
            slippage_pct=0.05,
            commission=4.5,
            commission_asset="USDT",
            execution_time=utcnow(),
            latency_ms=50.0,
            is_partial=False,
            is_rejected=False,
        )
        assert result.filled_qty == 10.0
        assert result.is_partial is False

    def test_fill_ratio(self) -> None:
        """Test fill ratio calculation."""
        from datetime import datetime

        result = ExecutionResult(
            order_id="ORD-001",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            requested_qty=10.0,
            filled_qty=5.0,  # 50% filled
            requested_price=None,
            avg_fill_price=45000.0,
            slippage_pct=0.05,
            commission=4.5,
            commission_asset="USDT",
            execution_time=utcnow(),
            latency_ms=50.0,
            is_partial=True,
            is_rejected=False,
        )
        assert result.fill_ratio == 0.5

    def test_total_cost(self) -> None:
        """Test total cost calculation."""
        from datetime import datetime

        result = ExecutionResult(
            order_id="ORD-001",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            requested_qty=0.1,
            filled_qty=0.1,
            requested_price=None,
            avg_fill_price=45000.0,  # 0.1 * 45000 = 4500
            slippage_pct=0.05,
            commission=4.5,  # Commission
            commission_asset="USDT",
            execution_time=utcnow(),
            latency_ms=50.0,
            is_partial=False,
            is_rejected=False,
        )
        # Total cost = 0.1 * 45000 + 4.5 = 4504.5
        assert result.total_cost == pytest.approx(4504.5, rel=0.01)
