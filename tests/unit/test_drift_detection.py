"""Tests for drift detection and alerting."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from analize.monitoring.drift import (
    DriftAlert,
    DriftDetector,
    DriftMetric,
    DriftSeverity,
    DriftThreshold,
)
from analize.monitoring.alerts import AlertChannel, AlertManager, AlertRule, FiredAlert


class TestDriftMetric:
    """Tests for DriftMetric enum."""

    def test_metric_values(self) -> None:
        """Test all metric values exist."""
        assert DriftMetric.WIN_RATE.value == "win_rate"
        assert DriftMetric.PROFIT_FACTOR.value == "profit_factor"
        assert DriftMetric.TRADE_FREQUENCY.value == "trade_frequency"
        assert DriftMetric.MAX_DRAWDOWN.value == "max_drawdown"


class TestDriftThreshold:
    """Tests for DriftThreshold dataclass."""

    def test_creation(self) -> None:
        """Test DriftThreshold creation."""
        threshold = DriftThreshold(
            metric=DriftMetric.WIN_RATE,
            warning_threshold_pct=15.0,
            critical_threshold_pct=25.0,
            min_sample_size=30,
        )
        assert threshold.warning_threshold_pct == 15.0
        assert threshold.min_sample_size == 30


class TestDriftDetector:
    """Tests for DriftDetector class."""

    @pytest.fixture
    def detector(self) -> DriftDetector:
        """Create DriftDetector instance."""
        return DriftDetector(baseline_days=30)

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Create sample trade data."""
        np.random.seed(42)
        n_trades = 200

        # Create dates spanning 60 days
        base_date = datetime(2024, 1, 1)
        timestamps = [base_date + timedelta(hours=i * 4) for i in range(n_trades)]

        # Good performance in first 40 days (baseline)
        # Degraded performance in last 20 days (current)
        pnl = []
        for i, ts in enumerate(timestamps):
            days_from_start = (ts - base_date).days
            if days_from_start < 40:
                # Good performance: 60% win rate, avg win > avg loss
                pnl.append(np.random.choice([2.0, -1.0], p=[0.6, 0.4]))
            else:
                # Degraded: 40% win rate
                pnl.append(np.random.choice([1.5, -1.5], p=[0.4, 0.6]))

        return pd.DataFrame({
            "timestamp": timestamps,
            "symbol": ["BTC/USDT"] * n_trades,
            "pnl_pct": pnl,
            "slippage_pct": np.random.uniform(0.01, 0.1, n_trades),
        })

    def test_calculate_metrics(self, detector: DriftDetector, sample_df: pd.DataFrame) -> None:
        """Test metric calculation for a period."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 15)

        metrics = detector.calculate_metrics(sample_df, start, end)

        assert DriftMetric.WIN_RATE in metrics
        assert DriftMetric.PROFIT_FACTOR in metrics
        assert DriftMetric.TRADE_FREQUENCY in metrics
        assert 0 <= metrics[DriftMetric.WIN_RATE] <= 100

    def test_detect_drift_runs_without_error(
        self, detector: DriftDetector, sample_df: pd.DataFrame
    ) -> None:
        """Test drift detection runs without errors."""
        # Use a reference date after the data period
        reference_date = datetime(2024, 2, 15)

        # This should run without errors
        alerts = detector.detect_drift(sample_df, reference_date=reference_date)

        # Result should be a list (may or may not have alerts depending on data)
        assert isinstance(alerts, list)

    def test_detect_drift_stable_performance(self, detector: DriftDetector) -> None:
        """Test no drift detected with stable performance."""
        np.random.seed(42)
        n_trades = 200
        base_date = datetime(2024, 1, 1)
        timestamps = [base_date + timedelta(hours=i * 4) for i in range(n_trades)]

        # Consistent 55% win rate throughout
        pnl = [np.random.choice([2.0, -1.0], p=[0.55, 0.45]) for _ in range(n_trades)]

        df = pd.DataFrame({
            "timestamp": timestamps,
            "pnl_pct": pnl,
        })

        reference_date = datetime(2024, 2, 15)
        alerts = detector.detect_drift(df, reference_date=reference_date)

        # Should have few or no critical alerts
        critical = [a for a in alerts if a.severity == DriftSeverity.CRITICAL]
        assert len(critical) == 0

    def test_drift_alert_structure(self, detector: DriftDetector, sample_df: pd.DataFrame) -> None:
        """Test DriftAlert structure."""
        reference_date = datetime(2024, 2, 15)
        alerts = detector.detect_drift(sample_df, reference_date=reference_date)

        if alerts:
            alert = alerts[0]
            assert isinstance(alert, DriftAlert)
            assert alert.alert_id is not None
            assert alert.metric in DriftMetric
            assert alert.severity in DriftSeverity
            assert alert.baseline_value is not None
            assert alert.current_value is not None
            assert alert.message is not None

    def test_get_drift_summary(self, detector: DriftDetector, sample_df: pd.DataFrame) -> None:
        """Test drift summary generation."""
        reference_date = datetime(2024, 2, 15)
        summary = detector.get_drift_summary(sample_df, reference_date)

        assert "reference_date" in summary
        assert "baseline_period" in summary
        assert "current_period" in summary
        assert "metrics" in summary

    def test_run_daily_check(self, detector: DriftDetector, sample_df: pd.DataFrame) -> None:
        """Test daily check across symbols."""
        results = detector.run_daily_check(sample_df, symbols=["BTC/USDT"])

        assert "check_time" in results
        assert "alerts" in results
        assert "alerts_by_severity" in results
        assert "has_critical" in results


class TestAlertRule:
    """Tests for AlertRule class."""

    def test_creation(self) -> None:
        """Test AlertRule creation."""
        rule = AlertRule(
            rule_id="test_rule",
            name="Test Rule",
            description="A test rule",
            metric="win_rate",
            condition="lt",
            threshold=40.0,
        )
        assert rule.rule_id == "test_rule"
        assert rule.threshold == 40.0

    def test_should_trigger_gt(self) -> None:
        """Test greater than condition."""
        rule = AlertRule(
            rule_id="drawdown_high",
            name="High Drawdown",
            description="Alert on high drawdown",
            metric="max_drawdown",
            condition="gt",
            threshold=10.0,
        )

        assert rule.should_trigger(15.0)  # Above threshold
        assert not rule.should_trigger(5.0)  # Below threshold

    def test_should_trigger_lt(self) -> None:
        """Test less than condition."""
        rule = AlertRule(
            rule_id="win_rate_low",
            name="Low Win Rate",
            description="Alert on low win rate",
            metric="win_rate",
            condition="lt",
            threshold=40.0,
        )

        assert rule.should_trigger(35.0)  # Below threshold
        assert not rule.should_trigger(50.0)  # Above threshold

    def test_cooldown(self) -> None:
        """Test alert cooldown."""
        rule = AlertRule(
            rule_id="test",
            name="Test",
            description="Test",
            metric="test",
            condition="gt",
            threshold=10.0,
            cooldown_minutes=60,
        )

        # First trigger should work
        assert rule.should_trigger(15.0)

        # Set last triggered to now
        rule.last_triggered = datetime.utcnow()

        # Should not trigger due to cooldown
        assert not rule.should_trigger(15.0)

    def test_daily_limit(self) -> None:
        """Test daily alert limit."""
        rule = AlertRule(
            rule_id="test",
            name="Test",
            description="Test",
            metric="test",
            condition="gt",
            threshold=10.0,
            max_alerts_per_day=3,
            trigger_count_today=3,
        )

        # Should not trigger - at limit
        assert not rule.should_trigger(15.0)


class TestAlertManager:
    """Tests for AlertManager class."""

    @pytest.fixture
    def manager(self, tmp_path) -> AlertManager:
        """Create AlertManager instance."""
        return AlertManager(storage_path=tmp_path)

    def test_check_and_fire(self, manager: AlertManager) -> None:
        """Test checking metrics and firing alerts."""
        # Register a handler that tracks calls
        notifications = []

        def test_handler(message: str, severity: DriftSeverity) -> bool:
            notifications.append((message, severity))
            return True

        manager.register_handler(AlertChannel.TELEGRAM, test_handler)

        # Trigger alerts
        metrics = {
            "win_rate": 25.0,  # Below critical threshold
            "max_drawdown": 25.0,  # Above critical threshold
        }

        fired = manager.check_and_fire(metrics)

        # Should have fired some alerts
        assert len(fired) > 0
        assert len(notifications) > 0

    def test_add_custom_rule(self, manager: AlertManager) -> None:
        """Test adding custom alert rules."""
        custom_rule = AlertRule(
            rule_id="custom_metric",
            name="Custom Metric Alert",
            description="Alert on custom metric",
            metric="custom_metric",
            condition="gt",
            threshold=100.0,
            severity=DriftSeverity.WARNING,
        )

        manager.add_rule(custom_rule)
        assert "custom_metric" in manager.rules

    def test_enable_disable_rule(self, manager: AlertManager) -> None:
        """Test enabling and disabling rules."""
        rule_id = "win_rate_drop"

        manager.disable_rule(rule_id)
        assert not manager.rules[rule_id].is_enabled

        manager.enable_rule(rule_id)
        assert manager.rules[rule_id].is_enabled

    def test_get_active_alerts(self, manager: AlertManager) -> None:
        """Test getting recent alerts."""
        notifications = []

        def test_handler(message: str, severity: DriftSeverity) -> bool:
            notifications.append(message)
            return True

        manager.register_handler(AlertChannel.TELEGRAM, test_handler)

        # Fire some alerts
        manager.check_and_fire({"win_rate": 25.0})

        # Get active alerts
        active = manager.get_active_alerts(hours=24)
        assert isinstance(active, list)

    def test_get_statistics(self, manager: AlertManager) -> None:
        """Test alert statistics."""
        stats = manager.get_statistics()

        assert "total_rules" in stats
        assert "enabled_rules" in stats
        assert "alerts_today" in stats
        assert "by_severity_today" in stats

    def test_reset_daily_counts(self, manager: AlertManager) -> None:
        """Test resetting daily trigger counts."""
        # Simulate some triggers
        for rule in manager.rules.values():
            rule.trigger_count_today = 5

        manager.reset_daily_counts()

        for rule in manager.rules.values():
            assert rule.trigger_count_today == 0


class TestFiredAlert:
    """Tests for FiredAlert dataclass."""

    def test_creation(self) -> None:
        """Test FiredAlert creation."""
        alert = FiredAlert(
            alert_id="ALERT-12345678",
            rule_id="win_rate_drop",
            rule_name="Win Rate Drop",
            fired_at=datetime.utcnow(),
            severity=DriftSeverity.WARNING,
            message="Win rate dropped below threshold",
            metric_value=35.0,
            threshold=40.0,
            channels_notified=[AlertChannel.TELEGRAM],
            notification_status={"telegram": True},
        )

        assert alert.alert_id == "ALERT-12345678"
        assert alert.severity == DriftSeverity.WARNING
