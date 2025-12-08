"""
CloudAIAnalyzer - Main entry point for the Analize platform.

Provides a unified interface for trading bot analysis, optimization,
and monitoring capabilities.
"""

from typing import Any

import pandas as pd

from analize.config import Settings, get_settings
from analize.stats.engine import StatsEngine
from analize.stats.metrics import TradingMetrics
from analize.stats.significance import SignificanceTester


class CloudAIAnalyzer:
    """
    Main analyzer class for Cloud AI trading analysis.

    Provides unified access to:
    - Statistical analysis of trading signals
    - Performance metrics calculation
    - Statistical significance testing
    - Drift detection and monitoring

    Example:
        >>> from analize import CloudAIAnalyzer
        >>> analyzer = CloudAIAnalyzer()
        >>> metrics = analyzer.calculate_metrics(pnl_series)
        >>> report = analyzer.generate_report(signals)
    """

    def __init__(self, settings: Settings | None = None):
        """
        Initialize the CloudAIAnalyzer.

        Args:
            settings: Optional settings override. Uses default settings if not provided.
        """
        self.settings = settings or get_settings()
        self.stats_engine = StatsEngine()
        self.significance_tester = SignificanceTester()

    def calculate_metrics(self, pnl: pd.Series) -> dict[str, Any]:
        """
        Calculate comprehensive trading metrics from P&L series.

        Args:
            pnl: Series of profit/loss values per trade

        Returns:
            Dictionary containing all calculated metrics including:
            - win_rate: Percentage of winning trades
            - profit_factor: Ratio of gross profit to gross loss
            - sharpe_ratio: Risk-adjusted return metric
            - max_drawdown: Maximum peak-to-trough decline
            - expectancy: Expected value per trade
        """
        return TradingMetrics.calculate_all(pnl)

    def analyze_signals(self, signals: list) -> dict[str, Any]:
        """
        Perform comprehensive analysis on trading signals.

        Args:
            signals: List of SignalRecord objects

        Returns:
            Dictionary with analysis results by symbol, filter, time, and volatility
        """
        return self.stats_engine.analyze_signals(signals)

    def test_significance(
        self,
        wins: int,
        total: int,
        null_hypothesis: float = 0.5,
        alpha: float = 0.05,
    ) -> dict[str, Any]:
        """
        Test if win rate is statistically significant.

        Args:
            wins: Number of winning trades
            total: Total number of trades
            null_hypothesis: Expected win rate under null hypothesis
            alpha: Significance level

        Returns:
            Dictionary with p-value, significance flag, and confidence interval
        """
        result = self.significance_tester.test_win_rate_significance(
            wins=wins,
            total=total,
            null_hypothesis=null_hypothesis,
            alpha=alpha,
        )
        return result.to_dict()

    def compare_strategies(
        self,
        returns_a: list[float],
        returns_b: list[float],
        paired: bool = False,
        alpha: float = 0.05,
    ) -> dict[str, Any]:
        """
        Compare two trading strategies statistically.

        Args:
            returns_a: Returns from strategy A
            returns_b: Returns from strategy B
            paired: Whether to use paired test (same time periods)
            alpha: Significance level

        Returns:
            Dictionary with comparison results
        """
        result = self.significance_tester.compare_strategies(
            returns_a=returns_a,
            returns_b=returns_b,
            paired=paired,
            alpha=alpha,
        )
        return result.to_dict()

    def bootstrap_confidence_interval(
        self,
        data: list[float],
        metric_func: callable = None,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
    ) -> dict[str, Any]:
        """
        Calculate bootstrap confidence interval for a metric.

        Args:
            data: Sample data
            metric_func: Function to calculate metric (default: mean)
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level

        Returns:
            Dictionary with point estimate and confidence interval
        """
        if metric_func is None:
            metric_func = lambda x: sum(x) / len(x) if x else 0

        result = self.significance_tester.bootstrap_metric(
            data=data,
            metric_func=metric_func,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
        )
        return result.to_dict()

    def generate_significance_report(
        self,
        wins: int,
        total: int,
        gross_profit: float,
        gross_loss: float,
        returns: list[float],
        alpha: float = 0.05,
    ) -> dict[str, Any]:
        """
        Generate comprehensive significance report.

        Args:
            wins: Number of winning trades
            total: Total number of trades
            gross_profit: Total profit from winning trades
            gross_loss: Total loss from losing trades (positive value)
            returns: List of individual trade returns
            alpha: Significance level

        Returns:
            Complete significance analysis report
        """
        return self.significance_tester.generate_significance_report(
            wins=wins,
            total=total,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            returns=returns,
            alpha=alpha,
        )

    @property
    def version(self) -> str:
        """Return the analyzer version."""
        from analize import __version__
        return __version__

    def __repr__(self) -> str:
        return f"CloudAIAnalyzer(version={self.version})"
