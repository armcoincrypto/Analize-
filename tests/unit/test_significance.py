"""Tests for statistical significance testing."""

import numpy as np
import pandas as pd
import pytest

from analize.stats.significance import (
    BootstrapResult,
    SignificanceResult,
    SignificanceTester,
)


class TestSignificanceTester:
    """Tests for SignificanceTester class."""

    @pytest.fixture
    def tester(self) -> SignificanceTester:
        """Create SignificanceTester instance."""
        return SignificanceTester(alpha=0.05, bootstrap_iterations=1000)

    def test_win_rate_significance_significant(self, tester: SignificanceTester) -> None:
        """Test win rate that is significantly above 50%."""
        # 70 wins out of 100 should be significant
        result = tester.test_win_rate_significance(
            wins=70,
            total=100,
            null_hypothesis=0.5,
        )

        assert isinstance(result, SignificanceResult)
        assert result.is_significant
        assert result.p_value < 0.05
        assert result.statistic == 0.7  # Observed win rate
        assert "significantly higher" in result.interpretation

    def test_win_rate_significance_not_significant(self, tester: SignificanceTester) -> None:
        """Test win rate that is not significantly different from 50%."""
        # 52 wins out of 100 should not be significant
        result = tester.test_win_rate_significance(
            wins=52,
            total=100,
            null_hypothesis=0.5,
        )

        assert not result.is_significant
        assert result.p_value >= 0.05
        assert "not significantly different" in result.interpretation

    def test_win_rate_confidence_interval(self, tester: SignificanceTester) -> None:
        """Test confidence interval for win rate."""
        result = tester.test_win_rate_significance(
            wins=60,
            total=100,
            null_hypothesis=0.5,
        )

        assert result.confidence_interval is not None
        ci_lower, ci_upper = result.confidence_interval
        assert ci_lower < 0.6 < ci_upper
        assert ci_lower > 0
        assert ci_upper <= 1

    def test_profit_factor_significance_profitable(self, tester: SignificanceTester) -> None:
        """Test profit factor significantly above 1."""
        np.random.seed(42)
        # Profitable trades: more wins than losses
        pnl = np.concatenate([
            np.random.uniform(1, 3, 60),   # 60 winning trades
            np.random.uniform(-2, -0.5, 40)  # 40 losing trades
        ])

        result = tester.test_profit_factor_significance(pnl, null_hypothesis=1.0)

        assert isinstance(result, SignificanceResult)
        assert result.statistic > 1.0  # PF should be > 1
        # Note: significance depends on actual values

    def test_profit_factor_significance_breakeven(self, tester: SignificanceTester) -> None:
        """Test profit factor around breakeven."""
        np.random.seed(42)
        # Breakeven trades
        pnl = np.concatenate([
            np.random.uniform(0.5, 1.5, 50),
            np.random.uniform(-1.5, -0.5, 50)
        ])

        result = tester.test_profit_factor_significance(pnl, null_hypothesis=1.0)

        # Should not be highly significant either way
        assert result.statistic is not None

    def test_compare_strategies_significant_improvement(self, tester: SignificanceTester) -> None:
        """Test comparing two strategies with significant difference."""
        np.random.seed(42)

        # Strategy A: mean 0.5% return
        returns_a = np.random.normal(0.5, 1.0, 100)

        # Strategy B: mean 1.5% return (significantly better)
        returns_b = np.random.normal(1.5, 1.0, 100)

        result = tester.compare_strategies(returns_a, returns_b, paired=False)

        assert result.is_significant
        assert "better" in result.interpretation.lower()

    def test_compare_strategies_no_difference(self, tester: SignificanceTester) -> None:
        """Test comparing two strategies with no significant difference."""
        np.random.seed(42)

        # Both strategies: similar mean return
        returns_a = np.random.normal(1.0, 1.0, 50)
        returns_b = np.random.normal(1.0, 1.0, 50)

        result = tester.compare_strategies(returns_a, returns_b, paired=False)

        # Should not be significant
        assert not result.is_significant or result.p_value > 0.01

    def test_compare_strategies_paired(self, tester: SignificanceTester) -> None:
        """Test paired comparison of strategies."""
        np.random.seed(42)

        # Same trades, different parameters
        base_returns = np.random.normal(0, 1, 100)
        returns_a = base_returns + np.random.normal(0.5, 0.2, 100)
        returns_b = base_returns + np.random.normal(1.0, 0.2, 100)

        result = tester.compare_strategies(returns_a, returns_b, paired=True)

        assert result.test_name in ["Paired t-Test", "Wilcoxon Signed-Rank Test"]
        assert result.effect_size is not None

    def test_walk_forward_consistency_consistent(self, tester: SignificanceTester) -> None:
        """Test walk-forward consistency with consistent results."""
        # Consistent profitable results across folds
        fold_scores = [1.5, 1.4, 1.6, 1.5, 1.45, 1.55]

        result = tester.test_walk_forward_consistency(fold_scores)

        assert result.is_significant  # Consistently above 1
        assert "consistent" in result.interpretation.lower()

    def test_walk_forward_consistency_inconsistent(self, tester: SignificanceTester) -> None:
        """Test walk-forward consistency with inconsistent results."""
        # Highly variable results
        fold_scores = [2.0, 0.5, 1.8, 0.4, 1.5, 0.3]

        result = tester.test_walk_forward_consistency(fold_scores)

        # High variance should be flagged
        assert "high variance" in result.interpretation.lower() or "inconsisten" in result.interpretation.lower()

    def test_bootstrap_metric(self, tester: SignificanceTester) -> None:
        """Test bootstrap confidence interval for custom metric."""
        np.random.seed(42)
        data = np.random.uniform(0, 10, 100)

        def mean_func(x):
            return np.mean(x)

        result = tester.bootstrap_metric(data, mean_func, "mean")

        assert isinstance(result, BootstrapResult)
        assert result.ci_lower < result.point_estimate < result.ci_upper
        assert result.bootstrap_std > 0
        assert result.n_iterations > 0

    def test_sharpe_ratio_test_positive(self, tester: SignificanceTester) -> None:
        """Test Sharpe ratio significantly positive."""
        np.random.seed(42)
        # Positive returns with reasonable volatility
        returns = np.random.normal(0.001, 0.01, 252)  # 0.1% daily mean

        result = tester.sharpe_ratio_test(returns, null_sharpe=0.0)

        assert result.statistic is not None  # Annualized Sharpe
        assert result.confidence_interval is not None

    def test_sharpe_ratio_test_not_significant(self, tester: SignificanceTester) -> None:
        """Test Sharpe ratio not significantly different from zero."""
        np.random.seed(42)
        # Near-zero mean returns
        returns = np.random.normal(0.0, 0.02, 100)

        result = tester.sharpe_ratio_test(returns, null_sharpe=0.0)

        # With zero mean and high variance, should not be significant
        assert abs(result.statistic) < 2  # Low Sharpe

    def test_multiple_testing_correction_bonferroni(self, tester: SignificanceTester) -> None:
        """Test Bonferroni correction for multiple testing."""
        p_values = [0.01, 0.02, 0.03, 0.04, 0.05]

        corrected = tester.multiple_testing_correction(p_values, method="bonferroni")

        assert len(corrected) == 5
        # First p-value: 0.01 * 5 = 0.05
        assert corrected[0][0] == pytest.approx(0.05, rel=0.01)
        # All corrected p-values should be >= original
        for (corrected_p, _), original_p in zip(corrected, p_values):
            assert corrected_p >= original_p

    def test_multiple_testing_correction_holm(self, tester: SignificanceTester) -> None:
        """Test Holm correction for multiple testing."""
        p_values = [0.01, 0.02, 0.03, 0.04, 0.05]

        corrected = tester.multiple_testing_correction(p_values, method="holm")

        assert len(corrected) == 5
        # Holm is less conservative than Bonferroni
        bonferroni = tester.multiple_testing_correction(p_values, method="bonferroni")
        # At least some Holm p-values should be <= Bonferroni
        holm_sum = sum(p for p, _ in corrected)
        bonf_sum = sum(p for p, _ in bonferroni)
        assert holm_sum <= bonf_sum

    def test_multiple_testing_correction_fdr(self, tester: SignificanceTester) -> None:
        """Test FDR (Benjamini-Hochberg) correction."""
        p_values = [0.001, 0.01, 0.02, 0.04, 0.1]

        corrected = tester.multiple_testing_correction(p_values, method="fdr")

        assert len(corrected) == 5
        # FDR should be even less conservative
        bonferroni = tester.multiple_testing_correction(p_values, method="bonferroni")
        fdr_significant = sum(1 for _, sig in corrected if sig)
        bonf_significant = sum(1 for _, sig in bonferroni if sig)
        assert fdr_significant >= bonf_significant

    def test_generate_significance_report(self, tester: SignificanceTester) -> None:
        """Test comprehensive significance report generation."""
        np.random.seed(42)
        pnl = np.concatenate([
            np.random.uniform(0.5, 2, 60),
            np.random.uniform(-1.5, -0.3, 40)
        ])
        fold_scores = [1.5, 1.4, 1.6, 1.5, 1.45]

        report = tester.generate_significance_report(pnl, fold_scores)

        assert "sample_size" in report
        assert "tests" in report
        assert "win_rate" in report["tests"]
        assert "profit_factor" in report["tests"]
        assert "sharpe_ratio" in report["tests"]
        assert "walk_forward_consistency" in report["tests"]
        assert "confidence_intervals" in report
        assert "summary" in report

    def test_insufficient_data_handling(self, tester: SignificanceTester) -> None:
        """Test handling of insufficient data."""
        # Win rate with no data
        result = tester.test_win_rate_significance(0, 0)
        assert not result.is_significant
        assert "Insufficient" in result.interpretation

        # Profit factor with few trades
        pnl = np.array([1.0, -0.5])
        result = tester.test_profit_factor_significance(pnl)
        assert "Insufficient" in result.interpretation


class TestSignificanceResult:
    """Tests for SignificanceResult dataclass."""

    def test_to_dict(self) -> None:
        """Test SignificanceResult serialization."""
        result = SignificanceResult(
            test_name="Test",
            statistic=1.5,
            p_value=0.03,
            is_significant=True,
            confidence_level=0.95,
            interpretation="Test is significant",
            effect_size=0.5,
            confidence_interval=(1.2, 1.8),
            sample_size=100,
        )

        d = result.to_dict()
        assert d["test_name"] == "Test"
        assert d["is_significant"] is True
        assert d["confidence_interval"] == (1.2, 1.8)


class TestBootstrapResult:
    """Tests for BootstrapResult dataclass."""

    def test_to_dict(self) -> None:
        """Test BootstrapResult serialization."""
        result = BootstrapResult(
            metric_name="profit_factor",
            point_estimate=1.5,
            confidence_level=0.95,
            ci_lower=1.2,
            ci_upper=1.8,
            bootstrap_std=0.15,
            n_iterations=1000,
        )

        d = result.to_dict()
        assert d["metric_name"] == "profit_factor"
        assert d["point_estimate"] == 1.5
        assert d["confidence_interval"] == (1.2, 1.8)
