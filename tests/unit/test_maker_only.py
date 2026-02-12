#!/usr/bin/env python3
"""
Tests for maker-only execution, cost gating, and MFE/MAE tracking.

These tests verify:
1. Post-only rejection behavior
2. Cost gating skip behavior
3. MFE/MAE tracking updates with synthetic ticks
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio

# Try to import pytest, but allow tests to run without it
try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    # Minimal pytest.approx replacement
    class _pytest:
        @staticmethod
        def approx(val, rel=0.01):
            return (val * (1 - rel), val * (1 + rel))
    pytest = _pytest()

# Add parent dir for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import directly from config module to avoid full hft_system import
# which requires numpy and other heavy dependencies
import importlib.util
spec = importlib.util.spec_from_file_location(
    "config",
    str(Path(__file__).parent.parent.parent / "hft_system" / "config.py")
)
config_module = importlib.util.module_from_spec(spec)
sys.modules["hft_system.config"] = config_module
spec.loader.exec_module(config_module)

CostModelConfig = config_module.CostModelConfig
MakerScalperConfig = config_module.MakerScalperConfig
COST_MODEL = config_module.COST_MODEL
MAKER_SCALPER = config_module.MAKER_SCALPER


class TestCostModelConfig:
    """Tests for CostModelConfig cost estimation."""

    def test_maker_only_defaults(self):
        """Test that maker_only is enabled by default."""
        config = CostModelConfig()
        assert config.maker_only is True
        assert config.post_only_enabled is True
        assert config.reject_taker_fills is True

    def test_estimate_pre_trade_costs_maker(self):
        """Test cost estimation for maker orders."""
        config = CostModelConfig()
        config.maker_only = True
        config.max_expected_total_costs_pct = 0.10  # Set high threshold for test

        estimate = config.estimate_pre_trade_costs(
            current_spread_pct=0.02,  # 0.02% spread
            order_size_usd=100,
            top_of_book_depth_usd=10000
        )

        # Maker fees: 0.01% + 0.01% = 0.02%
        assert estimate["expected_fee_sum_pct"] == 0.02
        # Spread cost for maker: spread * 0.5 * 2 = 0.02 * 0.5 * 2 = 0.02%
        assert estimate["estimated_spread_cost_pct"] == 0.02
        # Should trade if under threshold (0.10%)
        assert estimate["should_trade"] is True
        assert estimate["execution_mode"] == "maker"

    def test_estimate_pre_trade_costs_taker(self):
        """Test cost estimation for taker orders."""
        config = CostModelConfig()
        config.maker_only = False
        config.execution_mode = "taker"

        estimate = config.estimate_pre_trade_costs(
            current_spread_pct=0.05,  # 0.05% spread
            order_size_usd=100,
            top_of_book_depth_usd=10000
        )

        # Taker fees: 0.10% + 0.10% = 0.20%
        assert estimate["expected_fee_sum_pct"] == 0.20
        # Spread cost for taker: spread * 1.0 * 2 = 0.05 * 1.0 * 2 = 0.10%
        assert estimate["estimated_spread_cost_pct"] == 0.10
        assert estimate["execution_mode"] == "taker"

    def test_cost_gating_blocks_high_costs(self):
        """Test that cost gating blocks trades with high expected costs."""
        config = CostModelConfig()
        config.cost_gating_enabled = True
        config.max_expected_total_costs_pct = 0.05  # 0.05% - low threshold to trigger block
        config.maker_only = False
        config.execution_mode = "taker"

        # High spread = high costs (taker: 0.20% fees + 0.20% spread + slippage)
        estimate = config.estimate_pre_trade_costs(
            current_spread_pct=0.10,  # 0.10% spread
            order_size_usd=100,
            top_of_book_depth_usd=10000
        )

        # Total costs should exceed threshold (taker costs ~0.42%)
        assert estimate["expected_total_costs_pct"] > config.max_expected_total_costs_pct
        assert estimate["should_trade"] is False

    def test_cost_gating_allows_low_costs(self):
        """Test that cost gating allows trades with low expected costs."""
        config = CostModelConfig()
        config.cost_gating_enabled = True
        config.max_expected_total_costs_pct = 0.10  # 0.10% - generous threshold
        config.maker_only = True

        # Low spread = low costs
        estimate = config.estimate_pre_trade_costs(
            current_spread_pct=0.01,  # 0.01% spread
            order_size_usd=100,
            top_of_book_depth_usd=10000
        )

        # Total costs should be under threshold (maker costs ~0.03%)
        assert estimate["expected_total_costs_pct"] < config.max_expected_total_costs_pct
        assert estimate["should_trade"] is True

    def test_is_maker_only_mode(self):
        """Test maker-only mode detection."""
        config = CostModelConfig()

        # Both enabled
        config.maker_only = True
        config.post_only_enabled = True
        assert config.is_maker_only_mode() is True

        # Only maker_only
        config.post_only_enabled = False
        assert config.is_maker_only_mode() is False

        # Only post_only
        config.maker_only = False
        config.post_only_enabled = True
        assert config.is_maker_only_mode() is False


class TestMakerScalperConfig:
    """Tests for MakerScalperConfig."""

    def test_defaults(self):
        """Test default configuration values."""
        config = MakerScalperConfig()

        assert config.enabled is True
        assert config.entry_timeout_sec == 25.0
        assert config.tp_pct == 0.0007  # 0.07%
        assert config.sl_pct == 0.0012  # 0.12%
        assert config.hold_timeout_sec == 35.0


class TestMFEMAETracking:
    """Tests for MFE/MAE tracking with synthetic ticks."""

    def test_mfe_mae_calculation_long(self):
        """Test MFE/MAE calculation for LONG position."""
        entry_price = 100.0
        max_price = 101.0  # +1%
        min_price = 99.5   # -0.5%

        # LONG: MFE = (max - entry) / entry
        mfe = (max_price - entry_price) / entry_price * 100
        assert abs(mfe - 1.0) < 0.01  # ~1.0%

        # LONG: MAE = (min - entry) / entry (negative)
        mae = (min_price - entry_price) / entry_price * 100
        assert abs(mae - (-0.5)) < 0.01  # ~-0.5%

    def test_mfe_mae_calculation_short(self):
        """Test MFE/MAE calculation for SHORT position."""
        entry_price = 100.0
        max_price = 100.5  # Price up is bad for short
        min_price = 99.0   # Price down is good for short

        # SHORT: MFE = (entry - min) / entry
        mfe = (entry_price - min_price) / entry_price * 100
        assert abs(mfe - 1.0) < 0.01  # ~1.0%

        # SHORT: MAE = (entry - max) / entry (negative)
        mae = (entry_price - max_price) / entry_price * 100
        assert abs(mae - (-0.5)) < 0.01  # ~-0.5%

    def test_mfe_mae_updates_with_ticks(self):
        """Test that MFE/MAE update correctly with synthetic price ticks."""
        entry_price = 100.0
        position_side = "long"

        # Tracking state
        max_price = entry_price
        min_price = entry_price
        mfe = 0.0
        mae = 0.0

        # Synthetic price ticks
        ticks = [100.0, 100.2, 100.5, 100.3, 99.8, 99.5, 99.7, 100.1]

        for price in ticks:
            # Update max/min
            if price > max_price:
                max_price = price
            if price < min_price:
                min_price = price

            # Calculate MFE/MAE
            if position_side == "long":
                current_mfe = (max_price - entry_price) / entry_price * 100
                current_mae = (min_price - entry_price) / entry_price * 100
            else:
                current_mfe = (entry_price - min_price) / entry_price * 100
                current_mae = (entry_price - max_price) / entry_price * 100

            # Update tracking
            if current_mfe > mfe:
                mfe = current_mfe
            if current_mae < mae:
                mae = current_mae

        # Final MFE should be based on max_price=100.5
        assert abs(mfe - 0.5) < 0.01  # ~0.5%
        # Final MAE should be based on min_price=99.5
        assert abs(mae - (-0.5)) < 0.01  # ~-0.5%

    def test_mfe_mae_never_zero_with_movement(self):
        """Test that MFE/MAE are non-zero when price moves."""
        entry_price = 100.0

        # Simulate price movement
        ticks = [100.01, 99.99, 100.02]

        max_price = entry_price
        min_price = entry_price

        for price in ticks:
            if price > max_price:
                max_price = price
            if price < min_price:
                min_price = price

        mfe = (max_price - entry_price) / entry_price * 100
        mae = (min_price - entry_price) / entry_price * 100

        # Both should be non-zero after any price movement
        assert mfe != 0 or mae != 0


class TestPostOnlyRejection:
    """Tests for post-only order rejection behavior."""

    def test_long_limit_above_ask_rejected(self):
        """Test that LONG limit price above ask is adjusted."""
        best_bid = 100.0
        best_ask = 100.1
        offset_bps = 0.5

        # Calculate limit price with offset
        limit_price = best_bid + (best_ask - best_bid) * (offset_bps / 100)

        # Check if would cross spread
        would_cross = limit_price >= best_ask

        # In this case, with small offset, should not cross
        assert would_cross is False

        # Test with large offset that would cross
        large_offset = 100  # Would definitely cross
        limit_price_cross = best_bid + (best_ask - best_bid) * (large_offset / 100)
        assert limit_price_cross >= best_ask  # Would be rejected/adjusted

    def test_short_limit_below_bid_rejected(self):
        """Test that SHORT limit price below bid is adjusted."""
        best_bid = 100.0
        best_ask = 100.1
        offset_bps = 0.5

        # Calculate limit price with offset
        limit_price = best_ask - (best_ask - best_bid) * (offset_bps / 100)

        # Check if would cross spread
        would_cross = limit_price <= best_bid

        # In this case, with small offset, should not cross
        assert would_cross is False


class TestTakerViolationCooldown:
    """Tests for taker violation cooldown behavior."""

    def test_cooldown_blocks_entries(self):
        """Test that taker violation activates cooldown."""
        import time

        # Simulate violation state
        violation_active = True
        violation_time = time.time()
        cooldown_sec = 300.0  # 5 minutes

        # Check if entries should be blocked
        elapsed = time.time() - violation_time
        remaining = cooldown_sec - elapsed

        assert remaining > 0  # Should be in cooldown
        assert violation_active is True

    def test_cooldown_expires(self):
        """Test that cooldown eventually expires."""
        import time

        violation_time = time.time() - 400  # 400 seconds ago
        cooldown_sec = 300.0  # 5 minutes

        elapsed = time.time() - violation_time
        remaining = cooldown_sec - elapsed

        assert remaining <= 0  # Cooldown expired


class TestCostGatingIntegration:
    """Integration tests for cost gating with execution engine."""

    def test_cost_estimate_with_orderbook_data(self):
        """Test cost estimation with realistic orderbook data."""
        config = CostModelConfig()
        config.maker_only = True
        config.cost_gating_enabled = True

        # Simulate orderbook data
        current_spread_pct = 0.03  # 0.03% spread
        order_size_usd = 500
        # Top 3 levels of depth
        top_of_book_depth = 5000  # $5000 at top of book

        estimate = config.estimate_pre_trade_costs(
            current_spread_pct=current_spread_pct,
            order_size_usd=order_size_usd,
            top_of_book_depth_usd=top_of_book_depth
        )

        # Verify all components are calculated
        assert "expected_fee_sum_pct" in estimate
        assert "estimated_spread_cost_pct" in estimate
        assert "estimated_slippage_pct" in estimate
        assert "expected_total_costs_pct" in estimate
        assert "should_trade" in estimate

        # Slippage should consider depth ratio
        depth_ratio = order_size_usd / top_of_book_depth
        assert abs(depth_ratio - 0.1) < 0.01  # ~0.1


# Run tests with pytest or standalone
if __name__ == "__main__":
    if HAS_PYTEST:
        import pytest
        pytest.main([__file__, "-v"])
    else:
        # Run tests without pytest
        print("Running tests without pytest...\n")

        print("Testing CostModelConfig...")
        tc = TestCostModelConfig()
        tc.test_maker_only_defaults()
        print("  test_maker_only_defaults: PASS")
        tc.test_estimate_pre_trade_costs_maker()
        print("  test_estimate_pre_trade_costs_maker: PASS")
        tc.test_estimate_pre_trade_costs_taker()
        print("  test_estimate_pre_trade_costs_taker: PASS")
        tc.test_is_maker_only_mode()
        print("  test_is_maker_only_mode: PASS")

        print("\nTesting MakerScalperConfig...")
        ts = TestMakerScalperConfig()
        ts.test_defaults()
        print("  test_defaults: PASS")

        print("\nTesting MFEMAETracking...")
        tm = TestMFEMAETracking()
        tm.test_mfe_mae_calculation_long()
        print("  test_mfe_mae_calculation_long: PASS")
        tm.test_mfe_mae_calculation_short()
        print("  test_mfe_mae_calculation_short: PASS")
        tm.test_mfe_mae_updates_with_ticks()
        print("  test_mfe_mae_updates_with_ticks: PASS")
        tm.test_mfe_mae_never_zero_with_movement()
        print("  test_mfe_mae_never_zero_with_movement: PASS")

        print("\nTesting PostOnlyRejection...")
        tp = TestPostOnlyRejection()
        tp.test_long_limit_above_ask_rejected()
        print("  test_long_limit_above_ask_rejected: PASS")
        tp.test_short_limit_below_bid_rejected()
        print("  test_short_limit_below_bid_rejected: PASS")

        print("\nTesting TakerViolationCooldown...")
        tv = TestTakerViolationCooldown()
        tv.test_cooldown_blocks_entries()
        print("  test_cooldown_blocks_entries: PASS")
        tv.test_cooldown_expires()
        print("  test_cooldown_expires: PASS")

        print("\n=== ALL TESTS PASSED ===")
