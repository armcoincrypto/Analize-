"""
Statistical significance testing for trading analysis.

Provides rigorous hypothesis testing to ensure optimization results
are statistically significant and not due to random chance.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class SignificanceResult:
    """Results from statistical significance testing."""

    test_name: str
    statistic: float
    p_value: float
    is_significant: bool
    confidence_level: float
    interpretation: str

    # Additional details
    effect_size: float | None = None
    confidence_interval: tuple[float, float] | None = None
    sample_size: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_name": self.test_name,
            "statistic": round(self.statistic, 4),
            "p_value": round(self.p_value, 6),
            "is_significant": self.is_significant,
            "confidence_level": self.confidence_level,
            "interpretation": self.interpretation,
            "effect_size": round(self.effect_size, 4) if self.effect_size else None,
            "confidence_interval": (
                (round(self.confidence_interval[0], 4), round(self.confidence_interval[1], 4))
                if self.confidence_interval
                else None
            ),
            "sample_size": self.sample_size,
        }


@dataclass
class BootstrapResult:
    """Results from bootstrap confidence interval estimation."""

    metric_name: str
    point_estimate: float
    confidence_level: float
    ci_lower: float
    ci_upper: float
    bootstrap_std: float
    n_iterations: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric_name": self.metric_name,
            "point_estimate": round(self.point_estimate, 4),
            "confidence_level": self.confidence_level,
            "confidence_interval": (round(self.ci_lower, 4), round(self.ci_upper, 4)),
            "bootstrap_std": round(self.bootstrap_std, 4),
            "n_iterations": self.n_iterations,
        }


class SignificanceTester:
    """
    Performs statistical significance tests for trading analysis.

    Tests include:
    - Win rate significance (binomial test)
    - Profit factor significance (bootstrap)
    - Strategy comparison (paired t-test, Wilcoxon)
    - Walk-forward consistency (ANOVA)
    """

    def __init__(
        self,
        alpha: float = 0.05,
        bootstrap_iterations: int = 10000,
    ):
        """
        Initialize the significance tester.

        Args:
            alpha: Significance level (default 0.05 = 95% confidence)
            bootstrap_iterations: Number of bootstrap samples
        """
        self.alpha = alpha
        self.confidence_level = 1 - alpha
        self.bootstrap_iterations = bootstrap_iterations

    def test_win_rate_significance(
        self,
        wins: int,
        total: int,
        null_hypothesis: float = 0.5,
    ) -> SignificanceResult:
        """
        Test if win rate is significantly different from null hypothesis.

        Uses exact binomial test.

        Args:
            wins: Number of winning trades
            total: Total number of trades
            null_hypothesis: Expected win rate under null (default 50%)

        Returns:
            SignificanceResult with test outcome
        """
        if total == 0:
            return SignificanceResult(
                test_name="Binomial Test (Win Rate)",
                statistic=0,
                p_value=1.0,
                is_significant=False,
                confidence_level=self.confidence_level,
                interpretation="Insufficient data for analysis",
                sample_size=0,
            )

        # Observed win rate
        observed_rate = wins / total

        # Two-sided binomial test
        result = stats.binomtest(wins, total, null_hypothesis, alternative="two-sided")
        p_value = result.pvalue

        # Confidence interval for win rate
        ci = result.proportion_ci(confidence_level=self.confidence_level)

        # Effect size (Cohen's h)
        # h = 2 * (arcsin(sqrt(p1)) - arcsin(sqrt(p2)))
        effect_size = 2 * (np.arcsin(np.sqrt(observed_rate)) - np.arcsin(np.sqrt(null_hypothesis)))

        is_significant = p_value < self.alpha

        if is_significant:
            direction = "higher" if observed_rate > null_hypothesis else "lower"
            interpretation = (
                f"Win rate ({observed_rate*100:.1f}%) is significantly {direction} "
                f"than {null_hypothesis*100:.0f}% (p={p_value:.4f})"
            )
        else:
            interpretation = (
                f"Win rate ({observed_rate*100:.1f}%) is not significantly different "
                f"from {null_hypothesis*100:.0f}% (p={p_value:.4f})"
            )

        return SignificanceResult(
            test_name="Binomial Test (Win Rate)",
            statistic=observed_rate,
            p_value=p_value,
            is_significant=is_significant,
            confidence_level=self.confidence_level,
            interpretation=interpretation,
            effect_size=effect_size,
            confidence_interval=(ci.low, ci.high),
            sample_size=total,
        )

    def test_profit_factor_significance(
        self,
        pnl_series: pd.Series | np.ndarray,
        null_hypothesis: float = 1.0,
    ) -> SignificanceResult:
        """
        Test if profit factor is significantly different from null hypothesis.

        Uses bootstrap test since profit factor doesn't have a known distribution.

        Args:
            pnl_series: Series of trade PnLs
            null_hypothesis: Expected PF under null (default 1.0 = breakeven)

        Returns:
            SignificanceResult with test outcome
        """
        pnl = np.array(pnl_series)
        pnl = pnl[~np.isnan(pnl)]

        if len(pnl) < 10:
            return SignificanceResult(
                test_name="Bootstrap Test (Profit Factor)",
                statistic=0,
                p_value=1.0,
                is_significant=False,
                confidence_level=self.confidence_level,
                interpretation="Insufficient data for bootstrap analysis (need >= 10 trades)",
                sample_size=len(pnl),
            )

        # Calculate observed profit factor
        gross_profit = np.sum(pnl[pnl > 0])
        gross_loss = abs(np.sum(pnl[pnl <= 0]))
        observed_pf = gross_profit / gross_loss if gross_loss > 0 else 10.0

        # Bootstrap
        bootstrap_pfs = self._bootstrap_profit_factor(pnl)

        # P-value: proportion of bootstrap samples <= null hypothesis
        if observed_pf > null_hypothesis:
            p_value = np.mean(bootstrap_pfs <= null_hypothesis) * 2  # Two-sided
        else:
            p_value = np.mean(bootstrap_pfs >= null_hypothesis) * 2

        p_value = min(p_value, 1.0)

        # Confidence interval
        ci_lower = np.percentile(bootstrap_pfs, (self.alpha / 2) * 100)
        ci_upper = np.percentile(bootstrap_pfs, (1 - self.alpha / 2) * 100)

        # Effect size (log ratio)
        effect_size = np.log(observed_pf / null_hypothesis) if null_hypothesis > 0 else 0

        is_significant = p_value < self.alpha

        if is_significant:
            direction = "higher" if observed_pf > null_hypothesis else "lower"
            interpretation = (
                f"Profit Factor ({observed_pf:.2f}) is significantly {direction} "
                f"than {null_hypothesis:.1f} (p={p_value:.4f})"
            )
        else:
            interpretation = (
                f"Profit Factor ({observed_pf:.2f}) is not significantly different "
                f"from {null_hypothesis:.1f} (p={p_value:.4f})"
            )

        return SignificanceResult(
            test_name="Bootstrap Test (Profit Factor)",
            statistic=observed_pf,
            p_value=p_value,
            is_significant=is_significant,
            confidence_level=self.confidence_level,
            interpretation=interpretation,
            effect_size=effect_size,
            confidence_interval=(ci_lower, ci_upper),
            sample_size=len(pnl),
        )

    def _bootstrap_profit_factor(self, pnl: np.ndarray) -> np.ndarray:
        """Generate bootstrap distribution of profit factor."""
        bootstrap_pfs = []
        n = len(pnl)

        for _ in range(self.bootstrap_iterations):
            sample = np.random.choice(pnl, size=n, replace=True)
            gross_profit = np.sum(sample[sample > 0])
            gross_loss = abs(np.sum(sample[sample <= 0]))

            if gross_loss > 0:
                pf = gross_profit / gross_loss
            else:
                pf = 10.0  # Cap

            bootstrap_pfs.append(min(pf, 10.0))

        return np.array(bootstrap_pfs)

    def compare_strategies(
        self,
        returns_a: pd.Series | np.ndarray,
        returns_b: pd.Series | np.ndarray,
        paired: bool = True,
    ) -> SignificanceResult:
        """
        Compare two strategy returns for significant difference.

        Args:
            returns_a: Returns from strategy A (e.g., current params)
            returns_b: Returns from strategy B (e.g., proposed params)
            paired: Whether returns are paired (same trades, different params)

        Returns:
            SignificanceResult with comparison outcome
        """
        a = np.array(returns_a)
        b = np.array(returns_b)

        # Remove NaN
        if paired:
            mask = ~(np.isnan(a) | np.isnan(b))
            a, b = a[mask], b[mask]
        else:
            a = a[~np.isnan(a)]
            b = b[~np.isnan(b)]

        if len(a) < 10 or len(b) < 10:
            return SignificanceResult(
                test_name="Strategy Comparison",
                statistic=0,
                p_value=1.0,
                is_significant=False,
                confidence_level=self.confidence_level,
                interpretation="Insufficient data for comparison",
                sample_size=min(len(a), len(b)),
            )

        mean_a, mean_b = np.mean(a), np.mean(b)
        diff = mean_b - mean_a

        if paired:
            # Paired t-test (parametric)
            t_stat, t_pvalue = stats.ttest_rel(a, b)

            # Wilcoxon signed-rank test (non-parametric)
            try:
                w_stat, w_pvalue = stats.wilcoxon(a, b)
            except ValueError:
                w_stat, w_pvalue = 0, 1.0

            # Use Wilcoxon for non-normal data, t-test for normal
            # Simple normality check
            _, normality_p = stats.shapiro(b - a) if len(a) < 5000 else (0, 0.05)

            if normality_p < 0.05:
                # Non-normal, use Wilcoxon
                statistic, p_value = w_stat, w_pvalue
                test_name = "Wilcoxon Signed-Rank Test"
            else:
                statistic, p_value = t_stat, t_pvalue
                test_name = "Paired t-Test"

            # Cohen's d for paired data
            diff_scores = b - a
            effect_size = np.mean(diff_scores) / np.std(diff_scores) if np.std(diff_scores) > 0 else 0

            # Confidence interval for mean difference
            se = stats.sem(diff_scores)
            ci = stats.t.interval(self.confidence_level, len(diff_scores) - 1, loc=np.mean(diff_scores), scale=se)

        else:
            # Independent samples t-test
            t_stat, t_pvalue = stats.ttest_ind(a, b)

            # Mann-Whitney U test (non-parametric)
            u_stat, u_pvalue = stats.mannwhitneyu(a, b, alternative="two-sided")

            # Use Mann-Whitney for non-normal data
            _, norm_a_p = stats.shapiro(a) if len(a) < 5000 else (0, 0.05)
            _, norm_b_p = stats.shapiro(b) if len(b) < 5000 else (0, 0.05)

            if norm_a_p < 0.05 or norm_b_p < 0.05:
                statistic, p_value = u_stat, u_pvalue
                test_name = "Mann-Whitney U Test"
            else:
                statistic, p_value = t_stat, t_pvalue
                test_name = "Independent t-Test"

            # Cohen's d for independent samples
            pooled_std = np.sqrt(((len(a) - 1) * np.var(a) + (len(b) - 1) * np.var(b)) / (len(a) + len(b) - 2))
            effect_size = (mean_b - mean_a) / pooled_std if pooled_std > 0 else 0

            # Confidence interval for difference of means
            se = np.sqrt(np.var(a) / len(a) + np.var(b) / len(b))
            ci = (diff - 1.96 * se, diff + 1.96 * se)

        is_significant = p_value < self.alpha

        if is_significant:
            direction = "better" if diff > 0 else "worse"
            interpretation = (
                f"Strategy B is significantly {direction} than Strategy A "
                f"(mean diff: {diff:.4f}, p={p_value:.4f}, d={effect_size:.2f})"
            )
        else:
            interpretation = (
                f"No significant difference between strategies "
                f"(mean diff: {diff:.4f}, p={p_value:.4f})"
            )

        return SignificanceResult(
            test_name=test_name,
            statistic=statistic,
            p_value=p_value,
            is_significant=is_significant,
            confidence_level=self.confidence_level,
            interpretation=interpretation,
            effect_size=effect_size,
            confidence_interval=ci,
            sample_size=min(len(a), len(b)),
        )

    def test_walk_forward_consistency(
        self,
        fold_scores: list[float],
    ) -> SignificanceResult:
        """
        Test if walk-forward scores are consistent across folds.

        Uses chi-square test for variance and Shapiro-Wilk for normality.

        Args:
            fold_scores: List of test scores from each fold

        Returns:
            SignificanceResult indicating consistency
        """
        scores = np.array(fold_scores)
        scores = scores[~np.isnan(scores)]

        if len(scores) < 3:
            return SignificanceResult(
                test_name="Walk-Forward Consistency",
                statistic=0,
                p_value=1.0,
                is_significant=False,
                confidence_level=self.confidence_level,
                interpretation="Insufficient folds for consistency analysis",
                sample_size=len(scores),
            )

        # Test if scores are normally distributed
        _, normality_p = stats.shapiro(scores)

        # Test if variance is acceptable (coefficient of variation < 50%)
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        cv = std_score / mean_score if mean_score > 0 else float("inf")

        # Levene's test for equal variances (compare to theoretical constant)
        # We'll use a simpler coefficient of variation threshold

        # One-sample t-test: are scores significantly > 1 (profitable)?
        t_stat, t_pvalue = stats.ttest_1samp(scores, 1.0)

        is_consistent = cv < 0.5 and t_pvalue < self.alpha

        if is_consistent:
            interpretation = (
                f"Walk-forward results are consistent "
                f"(CV={cv:.2f}, mean={mean_score:.2f}, p={t_pvalue:.4f})"
            )
        else:
            issues = []
            if cv >= 0.5:
                issues.append(f"high variance (CV={cv:.2f})")
            if t_pvalue >= self.alpha:
                issues.append(f"not significantly profitable (p={t_pvalue:.4f})")
            interpretation = (
                f"Walk-forward results show inconsistency: {', '.join(issues)}"
            )

        # Confidence interval for mean score
        se = stats.sem(scores)
        ci = stats.t.interval(self.confidence_level, len(scores) - 1, loc=mean_score, scale=se)

        return SignificanceResult(
            test_name="Walk-Forward Consistency",
            statistic=cv,
            p_value=t_pvalue,
            is_significant=is_consistent,
            confidence_level=self.confidence_level,
            interpretation=interpretation,
            effect_size=mean_score - 1.0,  # Distance from breakeven
            confidence_interval=ci,
            sample_size=len(scores),
        )

    def bootstrap_metric(
        self,
        data: pd.Series | np.ndarray,
        metric_func: callable,
        metric_name: str = "metric",
    ) -> BootstrapResult:
        """
        Bootstrap confidence interval for any metric.

        Args:
            data: Raw data to bootstrap
            metric_func: Function to compute metric from data
            metric_name: Name of the metric

        Returns:
            BootstrapResult with confidence interval
        """
        data = np.array(data)
        data = data[~np.isnan(data)]

        if len(data) < 10:
            return BootstrapResult(
                metric_name=metric_name,
                point_estimate=0,
                confidence_level=self.confidence_level,
                ci_lower=0,
                ci_upper=0,
                bootstrap_std=0,
                n_iterations=0,
            )

        # Point estimate
        point_estimate = metric_func(data)

        # Bootstrap
        bootstrap_values = []
        n = len(data)

        for _ in range(self.bootstrap_iterations):
            sample = np.random.choice(data, size=n, replace=True)
            try:
                value = metric_func(sample)
                bootstrap_values.append(value)
            except Exception:
                continue

        if not bootstrap_values:
            return BootstrapResult(
                metric_name=metric_name,
                point_estimate=point_estimate,
                confidence_level=self.confidence_level,
                ci_lower=point_estimate,
                ci_upper=point_estimate,
                bootstrap_std=0,
                n_iterations=0,
            )

        bootstrap_values = np.array(bootstrap_values)

        ci_lower = np.percentile(bootstrap_values, (self.alpha / 2) * 100)
        ci_upper = np.percentile(bootstrap_values, (1 - self.alpha / 2) * 100)
        bootstrap_std = np.std(bootstrap_values)

        return BootstrapResult(
            metric_name=metric_name,
            point_estimate=point_estimate,
            confidence_level=self.confidence_level,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            bootstrap_std=bootstrap_std,
            n_iterations=len(bootstrap_values),
        )

    def sharpe_ratio_test(
        self,
        returns: pd.Series | np.ndarray,
        null_sharpe: float = 0.0,
        periods_per_year: int = 252,
    ) -> SignificanceResult:
        """
        Test if Sharpe ratio is significantly different from null.

        Uses the method from Lo (2002) for Sharpe ratio standard error.

        Args:
            returns: Series of returns
            null_sharpe: Sharpe ratio under null hypothesis
            periods_per_year: Annualization factor

        Returns:
            SignificanceResult with test outcome
        """
        returns = np.array(returns)
        returns = returns[~np.isnan(returns)]
        n = len(returns)

        if n < 30:
            return SignificanceResult(
                test_name="Sharpe Ratio Test",
                statistic=0,
                p_value=1.0,
                is_significant=False,
                confidence_level=self.confidence_level,
                interpretation="Insufficient data for Sharpe ratio test (need >= 30 observations)",
                sample_size=n,
            )

        # Calculate Sharpe ratio
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)

        if std_return == 0:
            return SignificanceResult(
                test_name="Sharpe Ratio Test",
                statistic=0,
                p_value=1.0,
                is_significant=False,
                confidence_level=self.confidence_level,
                interpretation="Zero volatility, cannot compute Sharpe ratio",
                sample_size=n,
            )

        # Non-annualized Sharpe
        sharpe = mean_return / std_return

        # Annualized Sharpe
        annualized_sharpe = sharpe * np.sqrt(periods_per_year)

        # Standard error of Sharpe ratio (Lo, 2002)
        # SE(SR) = sqrt((1 + 0.5 * SR^2) / n)
        se_sharpe = np.sqrt((1 + 0.5 * sharpe**2) / n)
        se_annualized = se_sharpe * np.sqrt(periods_per_year)

        # Z-test
        z_stat = (annualized_sharpe - null_sharpe) / se_annualized
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        # Confidence interval
        ci_lower = annualized_sharpe - 1.96 * se_annualized
        ci_upper = annualized_sharpe + 1.96 * se_annualized

        is_significant = p_value < self.alpha

        if is_significant:
            direction = "higher" if annualized_sharpe > null_sharpe else "lower"
            interpretation = (
                f"Sharpe Ratio ({annualized_sharpe:.2f}) is significantly {direction} "
                f"than {null_sharpe:.1f} (p={p_value:.4f})"
            )
        else:
            interpretation = (
                f"Sharpe Ratio ({annualized_sharpe:.2f}) is not significantly different "
                f"from {null_sharpe:.1f} (p={p_value:.4f})"
            )

        return SignificanceResult(
            test_name="Sharpe Ratio Test (Lo, 2002)",
            statistic=annualized_sharpe,
            p_value=p_value,
            is_significant=is_significant,
            confidence_level=self.confidence_level,
            interpretation=interpretation,
            effect_size=annualized_sharpe - null_sharpe,
            confidence_interval=(ci_lower, ci_upper),
            sample_size=n,
        )

    def multiple_testing_correction(
        self,
        p_values: list[float],
        method: str = "bonferroni",
    ) -> list[tuple[float, bool]]:
        """
        Correct for multiple testing.

        Args:
            p_values: List of p-values
            method: Correction method ("bonferroni", "holm", "fdr")

        Returns:
            List of (corrected_p_value, is_significant) tuples
        """
        n = len(p_values)
        if n == 0:
            return []

        p_array = np.array(p_values)

        if method == "bonferroni":
            # Most conservative
            corrected = np.minimum(p_array * n, 1.0)
            significant = corrected < self.alpha

        elif method == "holm":
            # Step-down Bonferroni
            sorted_indices = np.argsort(p_array)
            corrected = np.zeros(n)
            for i, idx in enumerate(sorted_indices):
                corrected[idx] = min(p_array[idx] * (n - i), 1.0)
            # Ensure monotonicity
            for i in range(1, n):
                idx = sorted_indices[i]
                prev_idx = sorted_indices[i - 1]
                corrected[idx] = max(corrected[idx], corrected[prev_idx])
            significant = corrected < self.alpha

        elif method == "fdr":
            # Benjamini-Hochberg FDR
            sorted_indices = np.argsort(p_array)
            corrected = np.zeros(n)
            for i, idx in enumerate(sorted_indices):
                corrected[idx] = p_array[idx] * n / (i + 1)
            # Ensure monotonicity (reverse direction)
            for i in range(n - 2, -1, -1):
                idx = sorted_indices[i]
                next_idx = sorted_indices[i + 1]
                corrected[idx] = min(corrected[idx], corrected[next_idx])
            corrected = np.minimum(corrected, 1.0)
            significant = corrected < self.alpha

        else:
            raise ValueError(f"Unknown method: {method}")

        return list(zip(corrected.tolist(), significant.tolist()))

    def generate_significance_report(
        self,
        pnl_series: pd.Series | np.ndarray,
        fold_scores: list[float] | None = None,
    ) -> dict[str, Any]:
        """
        Generate a comprehensive significance report.

        Args:
            pnl_series: Trade PnLs
            fold_scores: Optional walk-forward fold scores

        Returns:
            Dictionary with all significance tests
        """
        pnl = np.array(pnl_series)
        pnl = pnl[~np.isnan(pnl)]

        wins = np.sum(pnl > 0)
        total = len(pnl)

        report = {
            "sample_size": total,
            "tests": {},
        }

        # Win rate test
        wr_test = self.test_win_rate_significance(wins, total)
        report["tests"]["win_rate"] = wr_test.to_dict()

        # Profit factor test
        pf_test = self.test_profit_factor_significance(pnl)
        report["tests"]["profit_factor"] = pf_test.to_dict()

        # Sharpe ratio test
        sr_test = self.sharpe_ratio_test(pnl)
        report["tests"]["sharpe_ratio"] = sr_test.to_dict()

        # Bootstrap confidence intervals for key metrics
        def calc_win_rate(x):
            return np.mean(x > 0) * 100

        def calc_pf(x):
            gp = np.sum(x[x > 0])
            gl = abs(np.sum(x[x <= 0]))
            return gp / gl if gl > 0 else 10.0

        wr_bootstrap = self.bootstrap_metric(pnl, calc_win_rate, "win_rate")
        pf_bootstrap = self.bootstrap_metric(pnl, calc_pf, "profit_factor")

        report["confidence_intervals"] = {
            "win_rate": wr_bootstrap.to_dict(),
            "profit_factor": pf_bootstrap.to_dict(),
        }

        # Walk-forward consistency if provided
        if fold_scores:
            wf_test = self.test_walk_forward_consistency(fold_scores)
            report["tests"]["walk_forward_consistency"] = wf_test.to_dict()

        # Overall assessment
        significant_count = sum(
            1 for t in report["tests"].values() if t["is_significant"]
        )
        total_tests = len(report["tests"])

        report["summary"] = {
            "significant_tests": significant_count,
            "total_tests": total_tests,
            "overall_confidence": (
                "High" if significant_count == total_tests
                else "Medium" if significant_count > total_tests / 2
                else "Low"
            ),
            "recommendation": (
                "Results appear statistically robust"
                if significant_count >= total_tests - 1
                else "Results may be due to chance - more data needed"
            ),
        }

        return report
