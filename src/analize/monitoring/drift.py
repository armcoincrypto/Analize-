"""
Performance drift detection.

Monitors for material divergence between historical and live performance,
triggering alerts when metrics deviate beyond acceptable thresholds.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from analize.config import get_settings
from analize.stats.metrics import TradingMetrics
from analize.utils.time import utcnow, utcnow_iso


class DriftSeverity(str, Enum):
    """Drift alert severity levels."""

    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class DriftMetric(str, Enum):
    """Metrics monitored for drift."""

    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    TRADE_FREQUENCY = "trade_frequency"
    AVG_PNL = "avg_pnl"
    MAX_DRAWDOWN = "max_drawdown"
    SHARPE_RATIO = "sharpe_ratio"
    AVG_SLIPPAGE = "avg_slippage"


@dataclass
class DriftThreshold:
    """Threshold configuration for drift detection."""

    metric: DriftMetric
    warning_threshold_pct: float  # % deviation for warning
    critical_threshold_pct: float  # % deviation for critical
    min_sample_size: int = 30  # Minimum trades to evaluate
    lookback_days: int = 7  # Days of recent data to compare


@dataclass
class DriftAlert:
    """Alert generated when drift is detected."""

    alert_id: str
    metric: DriftMetric
    severity: DriftSeverity
    created_at: datetime
    symbol: str | None

    # Values
    baseline_value: float
    current_value: float
    deviation_pct: float

    # Context
    baseline_period: tuple[datetime, datetime]
    current_period: tuple[datetime, datetime]
    sample_size: int

    message: str
    recommendation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "metric": self.metric.value,
            "severity": self.severity.value,
            "created_at": self.created_at.isoformat(),
            "symbol": self.symbol,
            "baseline_value": self.baseline_value,
            "current_value": self.current_value,
            "deviation_pct": round(self.deviation_pct, 2),
            "baseline_period": [
                self.baseline_period[0].isoformat(),
                self.baseline_period[1].isoformat(),
            ],
            "current_period": [
                self.current_period[0].isoformat(),
                self.current_period[1].isoformat(),
            ],
            "sample_size": self.sample_size,
            "message": self.message,
            "recommendation": self.recommendation,
        }


class DriftDetector:
    """
    Detects performance drift between historical and recent results.

    Monitors key metrics and alerts when recent performance materially
    diverges from the historical baseline.
    """

    DEFAULT_THRESHOLDS = [
        DriftThreshold(
            metric=DriftMetric.WIN_RATE,
            warning_threshold_pct=15.0,  # 15% relative deviation
            critical_threshold_pct=25.0,
            min_sample_size=30,
            lookback_days=7,
        ),
        DriftThreshold(
            metric=DriftMetric.PROFIT_FACTOR,
            warning_threshold_pct=20.0,
            critical_threshold_pct=40.0,
            min_sample_size=30,
            lookback_days=7,
        ),
        DriftThreshold(
            metric=DriftMetric.TRADE_FREQUENCY,
            warning_threshold_pct=30.0,
            critical_threshold_pct=50.0,
            min_sample_size=10,
            lookback_days=3,
        ),
        DriftThreshold(
            metric=DriftMetric.AVG_PNL,
            warning_threshold_pct=25.0,
            critical_threshold_pct=50.0,
            min_sample_size=30,
            lookback_days=7,
        ),
        DriftThreshold(
            metric=DriftMetric.MAX_DRAWDOWN,
            warning_threshold_pct=50.0,  # Drawdown getting worse
            critical_threshold_pct=100.0,
            min_sample_size=30,
            lookback_days=14,
        ),
        DriftThreshold(
            metric=DriftMetric.AVG_SLIPPAGE,
            warning_threshold_pct=50.0,
            critical_threshold_pct=100.0,
            min_sample_size=30,
            lookback_days=3,
        ),
    ]

    def __init__(
        self,
        thresholds: list[DriftThreshold] | None = None,
        baseline_days: int = 30,
    ):
        """
        Initialize drift detector.

        Args:
            thresholds: List of threshold configurations
            baseline_days: Days of historical data to use as baseline
        """
        self.thresholds = {t.metric: t for t in (thresholds or self.DEFAULT_THRESHOLDS)}
        self.baseline_days = baseline_days
        self._alert_counter = 0

    def calculate_metrics(
        self,
        df: pd.DataFrame,
        start_date: datetime,
        end_date: datetime,
    ) -> dict[DriftMetric, float]:
        """
        Calculate all monitored metrics for a period.

        Args:
            df: DataFrame with trade data (must have timestamp, pnl_pct, etc.)
            start_date: Period start
            end_date: Period end

        Returns:
            Dictionary of metric values
        """
        # Filter to period
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        # Convert timezone-aware datetime to naive for comparison with pandas datetime64
        start_date_naive = start_date.replace(tzinfo=None) if hasattr(start_date, 'tzinfo') and start_date.tzinfo else start_date
        end_date_naive = end_date.replace(tzinfo=None) if hasattr(end_date, 'tzinfo') and end_date.tzinfo else end_date
        mask = (df["timestamp"] >= start_date_naive) & (df["timestamp"] <= end_date_naive)
        period_df = df[mask]

        if period_df.empty:
            return {}

        metrics = {}

        # Win rate
        if "pnl_pct" in period_df.columns:
            pnl = period_df["pnl_pct"].dropna()
            if len(pnl) > 0:
                wins = (pnl > 0).sum()
                metrics[DriftMetric.WIN_RATE] = wins / len(pnl) * 100

                # Profit factor
                gross_profit = pnl[pnl > 0].sum()
                gross_loss = abs(pnl[pnl <= 0].sum())
                if gross_loss > 0:
                    metrics[DriftMetric.PROFIT_FACTOR] = gross_profit / gross_loss
                else:
                    metrics[DriftMetric.PROFIT_FACTOR] = 10.0  # Cap at 10

                # Average PnL
                metrics[DriftMetric.AVG_PNL] = pnl.mean()

                # Max drawdown
                equity = pnl.cumsum()
                running_max = equity.cummax()
                drawdown = equity - running_max
                metrics[DriftMetric.MAX_DRAWDOWN] = abs(drawdown.min())

                # Sharpe ratio
                if pnl.std() > 0:
                    metrics[DriftMetric.SHARPE_RATIO] = pnl.mean() / pnl.std() * np.sqrt(252)

        # Trade frequency (trades per day)
        days = (end_date - start_date).days or 1
        metrics[DriftMetric.TRADE_FREQUENCY] = len(period_df) / days

        # Average slippage
        if "slippage_pct" in period_df.columns:
            slippage = period_df["slippage_pct"].dropna()
            if len(slippage) > 0:
                metrics[DriftMetric.AVG_SLIPPAGE] = slippage.mean()

        return metrics

    def detect_drift(
        self,
        df: pd.DataFrame,
        reference_date: datetime | None = None,
        symbol: str | None = None,
    ) -> list[DriftAlert]:
        """
        Detect drift between baseline and recent performance.

        Args:
            df: DataFrame with all trade data
            reference_date: Date to use as "now" (defaults to actual now)
            symbol: Optional symbol to filter

        Returns:
            List of drift alerts
        """
        reference_date = reference_date or utcnow()
        alerts = []

        # Filter by symbol if provided
        if symbol and "symbol" in df.columns:
            df = df[df["symbol"] == symbol]

        if df.empty:
            return alerts

        for metric, threshold in self.thresholds.items():
            alert = self._check_metric_drift(
                df=df,
                metric=metric,
                threshold=threshold,
                reference_date=reference_date,
                symbol=symbol,
            )
            if alert:
                alerts.append(alert)

        return alerts

    def _check_metric_drift(
        self,
        df: pd.DataFrame,
        metric: DriftMetric,
        threshold: DriftThreshold,
        reference_date: datetime,
        symbol: str | None,
    ) -> DriftAlert | None:
        """Check a single metric for drift."""
        # Define periods
        current_end = reference_date
        current_start = current_end - timedelta(days=threshold.lookback_days)

        baseline_end = current_start
        baseline_start = baseline_end - timedelta(days=self.baseline_days)

        # Calculate metrics for both periods
        baseline_metrics = self.calculate_metrics(df, baseline_start, baseline_end)
        current_metrics = self.calculate_metrics(df, current_start, current_end)

        if metric not in baseline_metrics or metric not in current_metrics:
            return None

        baseline_value = baseline_metrics[metric]
        current_value = current_metrics[metric]

        # Check sample size
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        # Convert timezone-aware datetime to naive for comparison with pandas datetime64
        current_start_naive = current_start.replace(tzinfo=None) if current_start.tzinfo else current_start
        current_end_naive = current_end.replace(tzinfo=None) if current_end.tzinfo else current_end
        current_mask = (df["timestamp"] >= current_start_naive) & (df["timestamp"] <= current_end_naive)
        current_count = current_mask.sum()

        if current_count < threshold.min_sample_size:
            return None

        # Calculate deviation
        if baseline_value == 0:
            if current_value == 0:
                return None
            deviation_pct = 100.0  # Changed from 0 to something
        else:
            deviation_pct = abs((current_value - baseline_value) / baseline_value) * 100

        # For metrics where lower is worse (e.g., win rate, PF), check if declining
        declining_metrics = {
            DriftMetric.WIN_RATE,
            DriftMetric.PROFIT_FACTOR,
            DriftMetric.AVG_PNL,
            DriftMetric.SHARPE_RATIO,
        }

        if metric in declining_metrics and current_value > baseline_value:
            # Performance improved, no alert needed
            return None

        # For metrics where higher is worse (e.g., drawdown, slippage)
        increasing_bad_metrics = {
            DriftMetric.MAX_DRAWDOWN,
            DriftMetric.AVG_SLIPPAGE,
        }

        if metric in increasing_bad_metrics and current_value < baseline_value:
            # Performance improved, no alert needed
            return None

        # Trade frequency can be concerning either way
        # (sudden drop = system issue, sudden spike = over-trading)

        # Determine severity
        if deviation_pct >= threshold.critical_threshold_pct:
            severity = DriftSeverity.CRITICAL
        elif deviation_pct >= threshold.warning_threshold_pct:
            severity = DriftSeverity.WARNING
        else:
            return None  # Within acceptable range

        # Generate alert
        self._alert_counter += 1
        alert_id = f"DRIFT-{self._alert_counter:06d}"

        # Generate message
        direction = "decreased" if current_value < baseline_value else "increased"
        message = (
            f"{metric.value} has {direction} by {deviation_pct:.1f}% "
            f"(baseline: {baseline_value:.2f}, current: {current_value:.2f})"
        )

        # Generate recommendation
        recommendation = self._get_recommendation(metric, baseline_value, current_value, deviation_pct)

        return DriftAlert(
            alert_id=alert_id,
            metric=metric,
            severity=severity,
            created_at=utcnow(),
            symbol=symbol,
            baseline_value=baseline_value,
            current_value=current_value,
            deviation_pct=deviation_pct,
            baseline_period=(baseline_start, baseline_end),
            current_period=(current_start, current_end),
            sample_size=current_count,
            message=message,
            recommendation=recommendation,
        )

    def _get_recommendation(
        self,
        metric: DriftMetric,
        baseline: float,
        current: float,
        deviation_pct: float,
    ) -> str:
        """Generate recommendation based on drift type."""
        recommendations = {
            DriftMetric.WIN_RATE: (
                "Review recent losing trades for patterns. "
                "Check if market regime has changed. "
                "Consider tightening entry filters or reducing position sizes."
            ),
            DriftMetric.PROFIT_FACTOR: (
                "Analyze if winners are getting smaller or losers larger. "
                "Review TP/SL settings. "
                "Check for increased slippage or adverse market conditions."
            ),
            DriftMetric.TRADE_FREQUENCY: (
                "If decreased: Check for system issues, filter changes, or market conditions. "
                "If increased: Verify no duplicate signals or over-trading bugs."
            ),
            DriftMetric.AVG_PNL: (
                "Review trade quality distribution. "
                "Check if market volatility has changed. "
                "Consider adjusting position sizing or TP targets."
            ),
            DriftMetric.MAX_DRAWDOWN: (
                "URGENT: Review risk management immediately. "
                "Consider reducing position sizes or pausing trading. "
                "Analyze if drawdown is from a few large losses or many small ones."
            ),
            DriftMetric.AVG_SLIPPAGE: (
                "Check exchange connectivity and execution times. "
                "Review if trading during low liquidity periods. "
                "Consider using limit orders or reducing order sizes."
            ),
        }

        return recommendations.get(
            metric,
            "Review recent performance and market conditions."
        )

    def run_daily_check(
        self,
        df: pd.DataFrame,
        symbols: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Run daily drift check across all symbols.

        Args:
            df: DataFrame with all trade data
            symbols: List of symbols to check (or None for all)

        Returns:
            Dictionary with check results
        """
        if symbols is None and "symbol" in df.columns:
            symbols = df["symbol"].unique().tolist()
        elif symbols is None:
            symbols = [None]  # Check all data together

        results = {
            "check_time": utcnow_iso(),
            "symbols_checked": len(symbols),
            "alerts": [],
            "alerts_by_severity": {
                "CRITICAL": 0,
                "WARNING": 0,
                "INFO": 0,
            },
        }

        for symbol in symbols:
            alerts = self.detect_drift(df, symbol=symbol)
            for alert in alerts:
                results["alerts"].append(alert.to_dict())
                results["alerts_by_severity"][alert.severity.value] += 1

        results["has_critical"] = results["alerts_by_severity"]["CRITICAL"] > 0
        results["has_warnings"] = results["alerts_by_severity"]["WARNING"] > 0

        return results

    def get_drift_summary(
        self,
        df: pd.DataFrame,
        reference_date: datetime | None = None,
    ) -> dict[str, Any]:
        """
        Get a summary of current vs baseline metrics.

        Useful for dashboards and quick health checks.
        """
        reference_date = reference_date or utcnow()

        # Calculate metrics for baseline and current periods
        current_end = reference_date
        current_start = current_end - timedelta(days=7)
        baseline_end = current_start
        baseline_start = baseline_end - timedelta(days=self.baseline_days)

        baseline_metrics = self.calculate_metrics(df, baseline_start, baseline_end)
        current_metrics = self.calculate_metrics(df, current_start, current_end)

        summary = {
            "reference_date": reference_date.isoformat(),
            "baseline_period": f"{baseline_start.date()} to {baseline_end.date()}",
            "current_period": f"{current_start.date()} to {current_end.date()}",
            "metrics": {},
        }

        for metric in DriftMetric:
            baseline = baseline_metrics.get(metric, 0)
            current = current_metrics.get(metric, 0)

            if baseline != 0:
                change_pct = (current - baseline) / baseline * 100
            else:
                change_pct = 0

            summary["metrics"][metric.value] = {
                "baseline": round(baseline, 4),
                "current": round(current, 4),
                "change_pct": round(change_pct, 2),
                "status": "ok" if abs(change_pct) < 15 else ("warning" if abs(change_pct) < 30 else "critical"),
            }

        return summary
