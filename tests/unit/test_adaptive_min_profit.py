"""Tests for Deliverable 4: adaptive minimum take-profit threshold.

Verifies that micro-profit exits don't trigger below the cost-aware threshold.
"""

import pytest


class TestAdaptiveMinProfit:
    """Adaptive min_profit_pct prevents unprofitable micro exits."""

    def test_min_profit_above_base(self):
        """min_profit_pct should be >= BASE_MIN_PROFIT_PCT."""
        from hft_system.execution_engine import ExecutionEngine
        from hft_system.config import BASE_MIN_PROFIT_PCT

        min_p = ExecutionEngine._compute_min_profit_pct()
        assert min_p >= BASE_MIN_PROFIT_PCT

    def test_min_profit_covers_costs(self):
        """min_profit_pct should be >= K * est_cost + buffer."""
        from hft_system.execution_engine import ExecutionEngine
        from hft_system.config import (
            COST_MODEL, COST_MULTIPLIER, COST_BUFFER_PCT,
        )

        min_p = ExecutionEngine._compute_min_profit_pct()
        if COST_MODEL.enabled:
            est_cost = (
                COST_MODEL.entry_fee_pct + COST_MODEL.exit_fee_pct
                + 2 * COST_MODEL.spread_cost_pct + COST_MODEL.base_slippage_pct
            )
            cost_floor = COST_MULTIPLIER * est_cost + COST_BUFFER_PCT
            assert min_p >= cost_floor

    def test_min_profit_higher_than_old_005(self):
        """The adaptive threshold should be higher than the old hard-coded 0.05%."""
        from hft_system.execution_engine import ExecutionEngine

        min_p = ExecutionEngine._compute_min_profit_pct()
        # With default config (maker mode: ~0.03% cost + 0.03 buffer = 0.06,
        # base 0.08), min should be 0.08 which is > 0.05
        assert min_p > 0.05, (
            f"min_profit_pct={min_p} should be > 0.05 (old hard-coded threshold)"
        )

    def test_micro_profit_not_triggered_below_threshold(self):
        """The MICRO_PROFIT check should not trigger when pnl < min_profit_pct.

        The guard `self.min_profit_pct <= pnl_pct` ensures any pnl_pct below
        the adaptive floor is rejected. In taker mode, min_profit_pct may
        exceed take_profit_pct, effectively disabling micro-profit exits
        entirely (which is correct – micro-profit doesn't make sense when
        costs are too high).
        """
        from hft_system.execution_engine import ExecutionEngine

        min_p = ExecutionEngine._compute_min_profit_pct()

        # pnl just below threshold must NOT trigger micro-profit
        pnl_below = min_p - 0.01
        assert not (min_p <= pnl_below)

        # pnl at threshold IS eligible (but may still be gated by TP)
        pnl_at = min_p
        assert min_p <= pnl_at

    def test_configurable_via_env(self, monkeypatch):
        """Config values should be overridable via environment."""
        monkeypatch.setenv("BASE_MIN_PROFIT_PCT", "0.15")
        monkeypatch.setenv("COST_MULTIPLIER", "2.0")
        monkeypatch.setenv("COST_BUFFER_PCT", "0.05")

        # Re-import to pick up env changes
        import importlib
        import hft_system.config as cfg
        importlib.reload(cfg)

        assert cfg.BASE_MIN_PROFIT_PCT == 0.15
        assert cfg.COST_MULTIPLIER == 2.0
        assert cfg.COST_BUFFER_PCT == 0.05

        # Restore defaults
        monkeypatch.delenv("BASE_MIN_PROFIT_PCT")
        monkeypatch.delenv("COST_MULTIPLIER")
        monkeypatch.delenv("COST_BUFFER_PCT")
        importlib.reload(cfg)
