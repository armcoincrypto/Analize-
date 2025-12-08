"""Tests for trading metrics."""

import numpy as np
import pandas as pd
import pytest

from analize.stats.metrics import TradingMetrics


class TestTradingMetrics:
    """Tests for TradingMetrics class."""

    def test_win_rate(self) -> None:
        """Test win rate calculation."""
        assert TradingMetrics.win_rate(7, 10) == 70.0
        assert TradingMetrics.win_rate(0, 10) == 0.0
        assert TradingMetrics.win_rate(10, 10) == 100.0
        assert TradingMetrics.win_rate(0, 0) == 0.0

    def test_profit_factor(self) -> None:
        """Test profit factor calculation."""
        assert TradingMetrics.profit_factor(100, 50) == 2.0
        assert TradingMetrics.profit_factor(50, 100) == 0.5
        assert TradingMetrics.profit_factor(100, 0) == float("inf")
        assert TradingMetrics.profit_factor(0, 100) == 0.0

    def test_expectancy(self) -> None:
        """Test expectancy calculation."""
        # 60% win rate, avg win 2%, avg loss 1%
        exp = TradingMetrics.expectancy(60, 2.0, 1.0)
        # E = 0.6 * 2 - 0.4 * 1 = 1.2 - 0.4 = 0.8
        assert abs(exp - 0.8) < 0.001

    def test_risk_reward_ratio(self) -> None:
        """Test risk:reward ratio calculation."""
        assert TradingMetrics.risk_reward_ratio(2.0, 1.0) == 2.0
        assert TradingMetrics.risk_reward_ratio(1.0, 2.0) == 0.5

    def test_sharpe_ratio(self) -> None:
        """Test Sharpe ratio calculation."""
        # Positive returns with low variance = high Sharpe
        returns = pd.Series([0.01, 0.02, 0.01, 0.015, 0.01])
        sharpe = TradingMetrics.sharpe_ratio(returns)
        assert sharpe > 0

        # Negative returns = negative Sharpe
        returns = pd.Series([-0.01, -0.02, -0.01, -0.015, -0.01])
        sharpe = TradingMetrics.sharpe_ratio(returns)
        assert sharpe < 0

    def test_sortino_ratio(self) -> None:
        """Test Sortino ratio calculation."""
        # Positive returns with no downside = high Sortino
        returns = pd.Series([0.01, 0.02, 0.01, 0.015, 0.01])
        sortino = TradingMetrics.sortino_ratio(returns)
        assert sortino > 0

    def test_max_drawdown(self) -> None:
        """Test max drawdown calculation."""
        # Equity curve: up, up, down, down, up
        equity = pd.Series([100, 110, 105, 95, 100])
        max_dd, peak_idx, trough_idx = TradingMetrics.max_drawdown(equity)

        # Max drawdown should be negative
        assert max_dd < 0

    def test_kelly_criterion(self) -> None:
        """Test Kelly criterion calculation."""
        # 60% win rate, 2:1 R:R
        kelly = TradingMetrics.kelly_criterion(60, 2.0, 1.0)
        # K = 0.6 - 0.4/2 = 0.6 - 0.2 = 0.4
        assert abs(kelly - 0.4) < 0.001

        # Negative Kelly = don't bet
        kelly = TradingMetrics.kelly_criterion(30, 1.0, 2.0)
        assert kelly == 0.0

    def test_consecutive_stats(self) -> None:
        """Test consecutive win/loss stats."""
        # W, W, W, L, L, W
        results = pd.Series([1, 1, 1, -1, -1, 1])
        stats = TradingMetrics.consecutive_stats(results)

        assert stats["max_consecutive_wins"] == 3
        assert stats["max_consecutive_losses"] == 2

    def test_calculate_all(self) -> None:
        """Test comprehensive metrics calculation."""
        pnl = pd.Series([10, -5, 15, -3, 8, -2, 12, -4, 7, -1])
        metrics = TradingMetrics.calculate_all(pnl)

        assert "total_trades" in metrics
        assert metrics["total_trades"] == 10
        assert "wins" in metrics
        assert metrics["wins"] == 5
        assert "win_rate" in metrics
        assert metrics["win_rate"] == 50.0
        assert "profit_factor" in metrics
        assert "expectancy" in metrics
        assert "sharpe_ratio" in metrics
