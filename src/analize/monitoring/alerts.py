"""
Alert management for Analize.

Configures and manages alert rules for:
- Performance drift
- System health
- Job failures
- Ingestion issues
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from analize.config import get_settings
from analize.monitoring.drift import DriftAlert, DriftSeverity
from analize.utils.time import utcnow, utcnow_iso


class AlertChannel(str, Enum):
    """Alert notification channels."""

    TELEGRAM = "telegram"
    SLACK = "slack"
    EMAIL = "email"
    WEBHOOK = "webhook"


@dataclass
class AlertRule:
    """Configuration for an alert rule."""

    rule_id: str
    name: str
    description: str

    # What triggers the alert
    metric: str  # Metric name to monitor
    condition: str  # "gt", "lt", "eq", "change_pct"
    threshold: float
    window_minutes: int = 60  # Time window to evaluate

    # Alert configuration
    severity: DriftSeverity = DriftSeverity.WARNING
    channels: list[AlertChannel] = field(default_factory=lambda: [AlertChannel.TELEGRAM])

    # Rate limiting
    cooldown_minutes: int = 60  # Minimum time between alerts
    max_alerts_per_day: int = 10

    # State
    is_enabled: bool = True
    last_triggered: datetime | None = None
    trigger_count_today: int = 0

    def should_trigger(self, value: float) -> bool:
        """Check if the rule should trigger."""
        if not self.is_enabled:
            return False

        # Check cooldown
        if self.last_triggered:
            elapsed = (utcnow() - self.last_triggered).total_seconds() / 60
            if elapsed < self.cooldown_minutes:
                return False

        # Check daily limit
        if self.trigger_count_today >= self.max_alerts_per_day:
            return False

        # Evaluate condition
        if self.condition == "gt":
            return value > self.threshold
        elif self.condition == "lt":
            return value < self.threshold
        elif self.condition == "eq":
            return abs(value - self.threshold) < 0.001
        elif self.condition == "change_pct":
            return abs(value) > self.threshold

        return False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "description": self.description,
            "metric": self.metric,
            "condition": self.condition,
            "threshold": self.threshold,
            "window_minutes": self.window_minutes,
            "severity": self.severity.value,
            "channels": [c.value for c in self.channels],
            "cooldown_minutes": self.cooldown_minutes,
            "max_alerts_per_day": self.max_alerts_per_day,
            "is_enabled": self.is_enabled,
            "last_triggered": self.last_triggered.isoformat() if self.last_triggered else None,
            "trigger_count_today": self.trigger_count_today,
        }


@dataclass
class FiredAlert:
    """Record of a fired alert."""

    alert_id: str
    rule_id: str
    rule_name: str
    fired_at: datetime
    severity: DriftSeverity
    message: str
    metric_value: float
    threshold: float
    channels_notified: list[AlertChannel]
    notification_status: dict[str, bool]  # channel -> success


class AlertManager:
    """
    Manages alert rules and notifications.

    Coordinates with drift detection and external notification services.
    """

    # Default alert rules
    DEFAULT_RULES = [
        AlertRule(
            rule_id="win_rate_drop",
            name="Win Rate Drop",
            description="Alert when win rate drops significantly",
            metric="win_rate",
            condition="lt",
            threshold=40.0,  # Below 40% win rate
            severity=DriftSeverity.WARNING,
        ),
        AlertRule(
            rule_id="win_rate_critical",
            name="Critical Win Rate",
            description="Alert when win rate drops to critical levels",
            metric="win_rate",
            condition="lt",
            threshold=30.0,
            severity=DriftSeverity.CRITICAL,
        ),
        AlertRule(
            rule_id="profit_factor_low",
            name="Low Profit Factor",
            description="Alert when profit factor drops below 1",
            metric="profit_factor",
            condition="lt",
            threshold=1.0,
            severity=DriftSeverity.CRITICAL,
        ),
        AlertRule(
            rule_id="drawdown_high",
            name="High Drawdown",
            description="Alert when drawdown exceeds threshold",
            metric="max_drawdown",
            condition="gt",
            threshold=10.0,  # 10% drawdown
            severity=DriftSeverity.WARNING,
        ),
        AlertRule(
            rule_id="drawdown_critical",
            name="Critical Drawdown",
            description="Alert when drawdown exceeds critical threshold",
            metric="max_drawdown",
            condition="gt",
            threshold=20.0,  # 20% drawdown
            severity=DriftSeverity.CRITICAL,
        ),
        AlertRule(
            rule_id="no_trades",
            name="No Recent Trades",
            description="Alert when trade frequency drops to zero",
            metric="trade_frequency",
            condition="lt",
            threshold=0.5,  # Less than 0.5 trades per day
            window_minutes=1440,  # 24 hours
            severity=DriftSeverity.WARNING,
        ),
        AlertRule(
            rule_id="slippage_high",
            name="High Slippage",
            description="Alert when average slippage is too high",
            metric="avg_slippage",
            condition="gt",
            threshold=0.2,  # 0.2% slippage
            severity=DriftSeverity.WARNING,
        ),
        AlertRule(
            rule_id="ingestion_lag",
            name="Ingestion Lag",
            description="Alert when data ingestion is delayed",
            metric="ingestion_lag_seconds",
            condition="gt",
            threshold=300,  # 5 minutes
            severity=DriftSeverity.WARNING,
        ),
        AlertRule(
            rule_id="job_failure",
            name="Job Failure",
            description="Alert when a job fails",
            metric="job_failures",
            condition="gt",
            threshold=0,
            severity=DriftSeverity.CRITICAL,
        ),
    ]

    def __init__(
        self,
        rules: list[AlertRule] | None = None,
        storage_path: Path | None = None,
    ):
        settings = get_settings()
        self.storage_path = storage_path or (settings.storage.local_data_path / "alerts")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Initialize rules
        self.rules = {r.rule_id: r for r in (rules or self.DEFAULT_RULES)}

        # Notification handlers
        self._handlers: dict[AlertChannel, Callable] = {}

        # Alert history
        self._history: list[FiredAlert] = []
        self._load_state()

    def _load_state(self) -> None:
        """Load alert state from storage."""
        state_file = self.storage_path / "alert_state.json"
        if state_file.exists():
            try:
                with open(state_file) as f:
                    state = json.load(f)

                for rule_id, rule_state in state.get("rules", {}).items():
                    if rule_id in self.rules:
                        rule = self.rules[rule_id]
                        if rule_state.get("last_triggered"):
                            rule.last_triggered = datetime.fromisoformat(rule_state["last_triggered"])
                        rule.trigger_count_today = rule_state.get("trigger_count_today", 0)
            except Exception:
                pass

    def _save_state(self) -> None:
        """Save alert state to storage."""
        state = {
            "rules": {
                rule_id: {
                    "last_triggered": rule.last_triggered.isoformat() if rule.last_triggered else None,
                    "trigger_count_today": rule.trigger_count_today,
                }
                for rule_id, rule in self.rules.items()
            },
            "updated_at": utcnow_iso(),
        }

        state_file = self.storage_path / "alert_state.json"
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)

    def register_handler(self, channel: AlertChannel, handler: Callable) -> None:
        """Register a notification handler for a channel."""
        self._handlers[channel] = handler

    def add_rule(self, rule: AlertRule) -> None:
        """Add a new alert rule."""
        self.rules[rule.rule_id] = rule

    def remove_rule(self, rule_id: str) -> bool:
        """Remove an alert rule."""
        if rule_id in self.rules:
            del self.rules[rule_id]
            return True
        return False

    def enable_rule(self, rule_id: str) -> bool:
        """Enable an alert rule."""
        if rule_id in self.rules:
            self.rules[rule_id].is_enabled = True
            return True
        return False

    def disable_rule(self, rule_id: str) -> bool:
        """Disable an alert rule."""
        if rule_id in self.rules:
            self.rules[rule_id].is_enabled = False
            return True
        return False

    def check_and_fire(self, metrics: dict[str, float]) -> list[FiredAlert]:
        """
        Check all rules against current metrics and fire alerts.

        Args:
            metrics: Dictionary of metric name -> current value

        Returns:
            List of fired alerts
        """
        fired = []

        for rule in self.rules.values():
            if rule.metric not in metrics:
                continue

            value = metrics[rule.metric]

            if rule.should_trigger(value):
                alert = self._fire_alert(rule, value)
                if alert:
                    fired.append(alert)

        self._save_state()
        return fired

    def _fire_alert(self, rule: AlertRule, value: float) -> FiredAlert | None:
        """Fire an alert and send notifications."""
        alert_id = f"ALERT-{uuid4().hex[:8].upper()}"

        message = self._format_message(rule, value)

        # Send notifications
        notification_status = {}
        for channel in rule.channels:
            if channel in self._handlers:
                try:
                    success = self._handlers[channel](message, rule.severity)
                    notification_status[channel.value] = success
                except Exception:
                    notification_status[channel.value] = False
            else:
                notification_status[channel.value] = False

        # Update rule state
        rule.last_triggered = utcnow()
        rule.trigger_count_today += 1

        # Create alert record
        alert = FiredAlert(
            alert_id=alert_id,
            rule_id=rule.rule_id,
            rule_name=rule.name,
            fired_at=utcnow(),
            severity=rule.severity,
            message=message,
            metric_value=value,
            threshold=rule.threshold,
            channels_notified=rule.channels,
            notification_status=notification_status,
        )

        self._history.append(alert)
        self._save_alert_history(alert)

        return alert

    def _format_message(self, rule: AlertRule, value: float) -> str:
        """Format alert message."""
        severity_emoji = {
            DriftSeverity.INFO: "ℹ️",
            DriftSeverity.WARNING: "⚠️",
            DriftSeverity.CRITICAL: "🚨",
        }

        emoji = severity_emoji.get(rule.severity, "📢")
        condition_text = {
            "gt": "exceeded",
            "lt": "fell below",
            "eq": "equals",
            "change_pct": "changed by",
        }

        return (
            f"{emoji} [{rule.severity.value}] {rule.name}\n\n"
            f"{rule.description}\n\n"
            f"Metric: {rule.metric}\n"
            f"Current Value: {value:.4f}\n"
            f"Threshold: {rule.threshold:.4f} ({condition_text.get(rule.condition, rule.condition)})\n"
            f"Time: {utcnow_iso()}"
        )

    def _save_alert_history(self, alert: FiredAlert) -> None:
        """Save alert to history file."""
        history_file = self.storage_path / f"alerts_{utcnow().strftime('%Y%m%d')}.json"

        history = []
        if history_file.exists():
            with open(history_file) as f:
                history = json.load(f)

        history.append({
            "alert_id": alert.alert_id,
            "rule_id": alert.rule_id,
            "rule_name": alert.rule_name,
            "fired_at": alert.fired_at.isoformat(),
            "severity": alert.severity.value,
            "message": alert.message,
            "metric_value": alert.metric_value,
            "threshold": alert.threshold,
            "notification_status": alert.notification_status,
        })

        with open(history_file, "w") as f:
            json.dump(history, f, indent=2)

    def process_drift_alerts(self, drift_alerts: list[DriftAlert]) -> list[FiredAlert]:
        """
        Process drift alerts from the drift detector.

        Converts drift alerts to fired alerts and sends notifications.
        """
        fired = []

        for drift_alert in drift_alerts:
            # Create a temporary rule for this drift alert
            rule = AlertRule(
                rule_id=f"drift_{drift_alert.metric.value}",
                name=f"Drift: {drift_alert.metric.value}",
                description=drift_alert.message,
                metric=drift_alert.metric.value,
                condition="change_pct",
                threshold=drift_alert.deviation_pct,
                severity=drift_alert.severity,
                channels=[AlertChannel.TELEGRAM, AlertChannel.SLACK],
            )

            alert = self._fire_alert(rule, drift_alert.current_value)
            if alert:
                fired.append(alert)

        return fired

    def get_active_alerts(self, hours: int = 24) -> list[dict[str, Any]]:
        """Get alerts from the last N hours."""
        cutoff = utcnow() - timedelta(hours=hours)
        return [
            {
                "alert_id": a.alert_id,
                "rule_name": a.rule_name,
                "severity": a.severity.value,
                "fired_at": a.fired_at.isoformat(),
                "message": a.message,
            }
            for a in self._history
            if a.fired_at >= cutoff
        ]

    def get_statistics(self) -> dict[str, Any]:
        """Get alert statistics."""
        now = utcnow()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_start = today_start - timedelta(days=7)

        today_alerts = [a for a in self._history if a.fired_at >= today_start]
        week_alerts = [a for a in self._history if a.fired_at >= week_start]

        return {
            "total_rules": len(self.rules),
            "enabled_rules": sum(1 for r in self.rules.values() if r.is_enabled),
            "alerts_today": len(today_alerts),
            "alerts_this_week": len(week_alerts),
            "by_severity_today": {
                severity.value: sum(1 for a in today_alerts if a.severity == severity)
                for severity in DriftSeverity
            },
            "most_triggered_rules": self._get_most_triggered(week_alerts),
        }

    def _get_most_triggered(self, alerts: list[FiredAlert], top_n: int = 5) -> list[dict[str, Any]]:
        """Get the most frequently triggered rules."""
        from collections import Counter
        rule_counts = Counter(a.rule_id for a in alerts)
        return [
            {"rule_id": rule_id, "count": count}
            for rule_id, count in rule_counts.most_common(top_n)
        ]

    def reset_daily_counts(self) -> None:
        """Reset daily trigger counts (call at midnight)."""
        for rule in self.rules.values():
            rule.trigger_count_today = 0
        self._save_state()
