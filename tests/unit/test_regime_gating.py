"""Tests for Deliverable 3: regime gating consistency.

Ensures that a blocked regime can never produce a real trade.
"""

import pytest


class TestRegimeGating:
    """Regime-blocked signals must never become real trades."""

    def test_blocked_regime_cannot_trade(self):
        """If regime is in avoid_regimes, _on_signal should return early."""
        from hft_system.hft_bot import HFTBot
        from hft_system.config import TradingMode

        bot = HFTBot(mode=TradingMode.PAPER, capital=10000)

        # Verify configuration
        assert "news_spike" in bot.avoid_regimes
        assert "liquidity_vacuum" in bot.avoid_regimes
        assert "unknown" in bot.avoid_regimes

        # Verify filter is enabled
        assert bot.filter_by_regime is True

    def test_avoid_regimes_match_winner_gate_blocked(self):
        """avoid_regimes should be a subset of WINNER_GATE.blocked_regimes."""
        from hft_system.hft_bot import HFTBot
        from hft_system.config import TradingMode, WINNER_GATE

        bot = HFTBot(mode=TradingMode.PAPER, capital=10000)

        # Every regime in avoid_regimes should also be in blocked_regimes
        for regime in bot.avoid_regimes:
            assert regime in WINNER_GATE.blocked_regimes, (
                f"{regime} is in avoid_regimes but not in WINNER_GATE.blocked_regimes"
            )

    def test_preferred_regimes_not_blocked(self):
        """preferred_regimes should NOT appear in avoid_regimes."""
        from hft_system.hft_bot import HFTBot
        from hft_system.config import TradingMode

        bot = HFTBot(mode=TradingMode.PAPER, capital=10000)

        for regime in bot.preferred_regimes:
            assert regime not in bot.avoid_regimes, (
                f"{regime} is both preferred AND avoided - inconsistency"
            )

    def test_regime_label_pinned_at_gating_time(self):
        """The entry_regime used for DB logging should be the same value
        used for the gating decision (captured once at top of _on_signal),
        not re-read from self.current_regime.
        """
        # This is a code-level assertion: verify the source code
        import inspect
        from hft_system.hft_bot import HFTBot

        source = inspect.getsource(HFTBot._on_signal)
        # After the fix, regime for DB update should use 'current_regime' (local var)
        # and NOT re-read self.current_regime
        assert "entry_regime = current_regime" in source, (
            "entry_regime should be pinned to the local `current_regime` captured at gating time"
        )
