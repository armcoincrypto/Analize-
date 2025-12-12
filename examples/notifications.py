#!/usr/bin/env python3
"""
Notification System - Multi-Channel Alert Management

Supports:
1. Telegram notifications (ops alerts + trading signals)
2. Discord webhooks
3. Email notifications
4. Slack webhooks
5. Desktop notifications
6. Log file alerts

Separates trading signals from operational alerts.

Author: Cloud AI Analyzer
"""

import os
import json
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
from pathlib import Path
import logging


# =============================================================================
# CONFIGURATION
# =============================================================================

class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    TRADE = "TRADE"


class AlertChannel(Enum):
    """Notification channels."""
    TELEGRAM = "telegram"
    DISCORD = "discord"
    EMAIL = "email"
    SLACK = "slack"
    LOG = "log"


@dataclass
class NotificationConfig:
    """Notification system configuration."""

    # Telegram
    telegram_bot_token: str = ""
    telegram_ops_chat_id: str = ""      # Operational alerts
    telegram_signals_chat_id: str = ""   # Trading signals

    # Discord
    discord_ops_webhook: str = ""
    discord_signals_webhook: str = ""

    # Email
    smtp_server: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    email_from: str = ""
    email_to: List[str] = field(default_factory=list)

    # Slack
    slack_ops_webhook: str = ""
    slack_signals_webhook: str = ""

    # General
    enabled_channels: List[AlertChannel] = field(default_factory=lambda: [AlertChannel.LOG])
    min_alert_level: AlertLevel = AlertLevel.INFO
    log_file: str = "alerts.log"

    @classmethod
    def from_env(cls) -> 'NotificationConfig':
        """Load configuration from environment variables."""
        return cls(
            telegram_bot_token=os.environ.get('TELEGRAM_BOT_TOKEN', ''),
            telegram_ops_chat_id=os.environ.get('TELEGRAM_OPS_CHAT_ID', ''),
            telegram_signals_chat_id=os.environ.get('TELEGRAM_SIGNALS_CHAT_ID', ''),
            discord_ops_webhook=os.environ.get('DISCORD_OPS_WEBHOOK', ''),
            discord_signals_webhook=os.environ.get('DISCORD_SIGNALS_WEBHOOK', ''),
            smtp_server=os.environ.get('SMTP_SERVER', ''),
            smtp_port=int(os.environ.get('SMTP_PORT', '587')),
            smtp_user=os.environ.get('SMTP_USER', ''),
            smtp_password=os.environ.get('SMTP_PASSWORD', ''),
            email_from=os.environ.get('EMAIL_FROM', ''),
            email_to=os.environ.get('EMAIL_TO', '').split(',') if os.environ.get('EMAIL_TO') else [],
            slack_ops_webhook=os.environ.get('SLACK_OPS_WEBHOOK', ''),
            slack_signals_webhook=os.environ.get('SLACK_SIGNALS_WEBHOOK', ''),
        )


@dataclass
class Alert:
    """Alert message structure."""
    level: AlertLevel
    title: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    symbol: str = ""
    data: Dict = field(default_factory=dict)
    channel_type: str = "ops"  # "ops" or "signals"


# =============================================================================
# NOTIFICATION MANAGER
# =============================================================================

class NotificationManager:
    """
    Multi-channel notification system.

    Features:
    - Send alerts to multiple channels
    - Separate ops alerts from trading signals
    - Rate limiting to prevent spam
    - Alert history tracking
    - Configurable severity filtering
    """

    def __init__(self, config: NotificationConfig = None):
        """Initialize notification manager."""
        self.config = config or NotificationConfig.from_env()
        self.alert_history: List[Alert] = []
        self._last_alert_time: Dict[str, datetime] = {}
        self._rate_limit_seconds = 60  # Min seconds between same alerts

        # Setup logging
        self.logger = logging.getLogger('notifications')
        self.logger.setLevel(logging.INFO)

        log_path = Path(__file__).parent / self.config.log_file
        handler = logging.FileHandler(log_path)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)

        self._check_configuration()

    def _check_configuration(self):
        """Check and report configuration status."""
        print("\n" + "=" * 60)
        print("NOTIFICATION SYSTEM STATUS")
        print("=" * 60)

        channels_status = []

        # Telegram
        if self.config.telegram_bot_token and self.config.telegram_ops_chat_id:
            channels_status.append("✓ Telegram (OPS)")
            self.config.enabled_channels.append(AlertChannel.TELEGRAM)
        else:
            channels_status.append("✗ Telegram (not configured)")

        # Discord
        if self.config.discord_ops_webhook:
            channels_status.append("✓ Discord (OPS)")
            self.config.enabled_channels.append(AlertChannel.DISCORD)
        else:
            channels_status.append("✗ Discord (not configured)")

        # Email
        if self.config.smtp_server and self.config.email_to:
            channels_status.append("✓ Email")
            self.config.enabled_channels.append(AlertChannel.EMAIL)
        else:
            channels_status.append("✗ Email (not configured)")

        # Slack
        if self.config.slack_ops_webhook:
            channels_status.append("✓ Slack (OPS)")
            self.config.enabled_channels.append(AlertChannel.SLACK)
        else:
            channels_status.append("✗ Slack (not configured)")

        # Always enable logging
        channels_status.append("✓ Log File")

        for status in channels_status:
            print(f"  {status}")

        print("=" * 60)

    # =========================================================================
    # CORE SEND METHODS
    # =========================================================================

    def send_alert(self, alert: Alert) -> bool:
        """
        Send alert to all configured channels.

        Args:
            alert: Alert to send

        Returns:
            True if at least one channel succeeded
        """
        # Check rate limiting
        alert_key = f"{alert.level.value}:{alert.title}"
        if alert_key in self._last_alert_time:
            elapsed = (datetime.now() - self._last_alert_time[alert_key]).seconds
            if elapsed < self._rate_limit_seconds:
                return False  # Rate limited

        self._last_alert_time[alert_key] = datetime.now()
        self.alert_history.append(alert)

        success = False

        # Always log
        self._send_to_log(alert)
        success = True

        # Telegram
        if AlertChannel.TELEGRAM in self.config.enabled_channels:
            if self._send_to_telegram(alert):
                success = True

        # Discord
        if AlertChannel.DISCORD in self.config.enabled_channels:
            if self._send_to_discord(alert):
                success = True

        # Email (only for CRITICAL)
        if AlertChannel.EMAIL in self.config.enabled_channels:
            if alert.level == AlertLevel.CRITICAL:
                if self._send_to_email(alert):
                    success = True

        # Slack
        if AlertChannel.SLACK in self.config.enabled_channels:
            if self._send_to_slack(alert):
                success = True

        return success

    # =========================================================================
    # CHANNEL-SPECIFIC METHODS
    # =========================================================================

    def _send_to_telegram(self, alert: Alert) -> bool:
        """Send alert to Telegram."""
        try:
            # Select chat based on alert type
            if alert.channel_type == "signals" and self.config.telegram_signals_chat_id:
                chat_id = self.config.telegram_signals_chat_id
            else:
                chat_id = self.config.telegram_ops_chat_id

            if not chat_id:
                return False

            # Format message
            emoji = {
                AlertLevel.INFO: "ℹ️",
                AlertLevel.WARNING: "⚠️",
                AlertLevel.CRITICAL: "🚨",
                AlertLevel.TRADE: "📊",
            }.get(alert.level, "📢")

            message = f"{emoji} *{alert.title}*\n\n{alert.message}"

            if alert.symbol:
                message += f"\n\n📈 Symbol: `{alert.symbol}`"

            if alert.data:
                message += f"\n\n```\n{json.dumps(alert.data, indent=2)}\n```"

            message += f"\n\n🕐 {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"

            # Send via Telegram API
            url = f"https://api.telegram.org/bot{self.config.telegram_bot_token}/sendMessage"
            response = requests.post(url, json={
                'chat_id': chat_id,
                'text': message,
                'parse_mode': 'Markdown',
                'disable_web_page_preview': True,
            }, timeout=10)

            return response.status_code == 200

        except Exception as e:
            self.logger.error(f"Telegram error: {e}")
            return False

    def _send_to_discord(self, alert: Alert) -> bool:
        """Send alert to Discord webhook."""
        try:
            # Select webhook based on alert type
            if alert.channel_type == "signals" and self.config.discord_signals_webhook:
                webhook_url = self.config.discord_signals_webhook
            else:
                webhook_url = self.config.discord_ops_webhook

            if not webhook_url:
                return False

            # Color based on level
            color = {
                AlertLevel.INFO: 0x3498db,      # Blue
                AlertLevel.WARNING: 0xf39c12,   # Orange
                AlertLevel.CRITICAL: 0xe74c3c,  # Red
                AlertLevel.TRADE: 0x2ecc71,     # Green
            }.get(alert.level, 0x95a5a6)

            embed = {
                'title': alert.title,
                'description': alert.message,
                'color': color,
                'timestamp': alert.timestamp.isoformat(),
                'fields': [],
            }

            if alert.symbol:
                embed['fields'].append({
                    'name': 'Symbol',
                    'value': alert.symbol,
                    'inline': True,
                })

            if alert.data:
                for key, value in list(alert.data.items())[:5]:
                    embed['fields'].append({
                        'name': str(key),
                        'value': str(value)[:100],
                        'inline': True,
                    })

            response = requests.post(webhook_url, json={
                'embeds': [embed],
            }, timeout=10)

            return response.status_code in [200, 204]

        except Exception as e:
            self.logger.error(f"Discord error: {e}")
            return False

    def _send_to_slack(self, alert: Alert) -> bool:
        """Send alert to Slack webhook."""
        try:
            # Select webhook based on alert type
            if alert.channel_type == "signals" and self.config.slack_signals_webhook:
                webhook_url = self.config.slack_signals_webhook
            else:
                webhook_url = self.config.slack_ops_webhook

            if not webhook_url:
                return False

            emoji = {
                AlertLevel.INFO: ":information_source:",
                AlertLevel.WARNING: ":warning:",
                AlertLevel.CRITICAL: ":rotating_light:",
                AlertLevel.TRADE: ":chart_with_upwards_trend:",
            }.get(alert.level, ":bell:")

            blocks = [
                {
                    'type': 'header',
                    'text': {
                        'type': 'plain_text',
                        'text': f"{emoji} {alert.title}",
                    }
                },
                {
                    'type': 'section',
                    'text': {
                        'type': 'mrkdwn',
                        'text': alert.message,
                    }
                },
            ]

            if alert.symbol or alert.data:
                fields = []
                if alert.symbol:
                    fields.append({
                        'type': 'mrkdwn',
                        'text': f"*Symbol:* {alert.symbol}",
                    })
                for key, value in list(alert.data.items())[:4]:
                    fields.append({
                        'type': 'mrkdwn',
                        'text': f"*{key}:* {value}",
                    })

                blocks.append({
                    'type': 'section',
                    'fields': fields[:10],  # Max 10 fields
                })

            response = requests.post(webhook_url, json={
                'blocks': blocks,
            }, timeout=10)

            return response.status_code == 200

        except Exception as e:
            self.logger.error(f"Slack error: {e}")
            return False

    def _send_to_email(self, alert: Alert) -> bool:
        """Send alert via email."""
        try:
            if not self.config.smtp_server or not self.config.email_to:
                return False

            msg = MIMEMultipart()
            msg['From'] = self.config.email_from
            msg['To'] = ', '.join(self.config.email_to)
            msg['Subject'] = f"[{alert.level.value}] {alert.title}"

            body = f"""
Cloud AI Analyzer Alert
=======================

Level: {alert.level.value}
Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
Symbol: {alert.symbol or 'N/A'}

Message:
{alert.message}

Data:
{json.dumps(alert.data, indent=2) if alert.data else 'N/A'}
            """

            msg.attach(MIMEText(body, 'plain'))

            with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
                server.starttls()
                if self.config.smtp_user and self.config.smtp_password:
                    server.login(self.config.smtp_user, self.config.smtp_password)
                server.send_message(msg)

            return True

        except Exception as e:
            self.logger.error(f"Email error: {e}")
            return False

    def _send_to_log(self, alert: Alert):
        """Log alert to file."""
        log_msg = f"[{alert.level.value}] {alert.title}: {alert.message}"
        if alert.symbol:
            log_msg += f" (Symbol: {alert.symbol})"

        if alert.level == AlertLevel.CRITICAL:
            self.logger.critical(log_msg)
        elif alert.level == AlertLevel.WARNING:
            self.logger.warning(log_msg)
        else:
            self.logger.info(log_msg)

    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================

    def info(self, title: str, message: str, symbol: str = "", data: Dict = None):
        """Send info alert."""
        self.send_alert(Alert(
            level=AlertLevel.INFO,
            title=title,
            message=message,
            symbol=symbol,
            data=data or {},
        ))

    def warning(self, title: str, message: str, symbol: str = "", data: Dict = None):
        """Send warning alert."""
        self.send_alert(Alert(
            level=AlertLevel.WARNING,
            title=title,
            message=message,
            symbol=symbol,
            data=data or {},
        ))

    def critical(self, title: str, message: str, symbol: str = "", data: Dict = None):
        """Send critical alert."""
        self.send_alert(Alert(
            level=AlertLevel.CRITICAL,
            title=title,
            message=message,
            symbol=symbol,
            data=data or {},
        ))

    def trade_signal(self, symbol: str, direction: str, confidence: float,
                     entry_price: float, data: Dict = None):
        """Send trade signal notification."""
        self.send_alert(Alert(
            level=AlertLevel.TRADE,
            title=f"Trade Signal: {direction} {symbol}",
            message=f"Direction: {direction}\nConfidence: {confidence:.1%}\nEntry: ${entry_price:.4f}",
            symbol=symbol,
            data=data or {},
            channel_type="signals",
        ))

    # =========================================================================
    # SPECIFIC ALERTS
    # =========================================================================

    def alert_circuit_breaker(self, reason: str, until: datetime):
        """Send circuit breaker alert."""
        self.critical(
            title="🚨 CIRCUIT BREAKER TRIGGERED",
            message=f"Trading has been paused.\n\nReason: {reason}\nResumes: {until.strftime('%Y-%m-%d %H:%M')}",
            data={'reason': reason, 'resumes': until.isoformat()},
        )

    def alert_data_stale(self, feed: str, last_update: datetime):
        """Send stale data alert."""
        age = (datetime.now() - last_update).seconds // 60
        self.warning(
            title="⚠️ Stale Data Feed",
            message=f"Feed '{feed}' hasn't updated in {age} minutes.\nLast update: {last_update.strftime('%Y-%m-%d %H:%M')}",
            data={'feed': feed, 'age_minutes': age},
        )

    def alert_position_opened(self, symbol: str, direction: str, size: float,
                              entry_price: float, stop_loss: float):
        """Send position opened alert."""
        self.trade_signal(
            symbol=symbol,
            direction=direction,
            confidence=0.0,
            entry_price=entry_price,
            data={
                'size_usd': f"${size:,.2f}",
                'stop_loss': f"${stop_loss:.4f}",
                'action': 'OPENED',
            },
        )

    def alert_position_closed(self, symbol: str, pnl: float, pnl_pct: float,
                              exit_price: float):
        """Send position closed alert."""
        result = "WIN ✓" if pnl > 0 else "LOSS ✗"
        self.send_alert(Alert(
            level=AlertLevel.TRADE,
            title=f"Position Closed: {symbol} - {result}",
            message=f"P&L: ${pnl:+,.2f} ({pnl_pct:+.2f}%)\nExit: ${exit_price:.4f}",
            symbol=symbol,
            data={'pnl': pnl, 'pnl_pct': pnl_pct, 'result': result},
            channel_type="signals",
        ))

    def alert_drift_detected(self, feature: str, drift_score: float):
        """Send feature drift alert."""
        self.warning(
            title="⚠️ Feature Drift Detected",
            message=f"Feature '{feature}' shows significant drift.\nDrift score: {drift_score:.4f}",
            data={'feature': feature, 'drift_score': drift_score},
        )

    def alert_model_degradation(self, model: str, accuracy: float, baseline: float):
        """Send model degradation alert."""
        self.warning(
            title="⚠️ Model Accuracy Degradation",
            message=f"Model '{model}' accuracy dropped.\nCurrent: {accuracy:.1%}\nBaseline: {baseline:.1%}",
            data={'model': model, 'accuracy': accuracy, 'baseline': baseline},
        )

    def alert_daily_summary(self, trades: int, pnl: float, win_rate: float):
        """Send daily trading summary."""
        emoji = "📈" if pnl >= 0 else "📉"
        self.info(
            title=f"{emoji} Daily Trading Summary",
            message=f"Total Trades: {trades}\nNet P&L: ${pnl:+,.2f}\nWin Rate: {win_rate:.1%}",
            data={'trades': trades, 'pnl': pnl, 'win_rate': win_rate},
        )


# =============================================================================
# DEMO / TEST
# =============================================================================

def demo_notifications():
    """Demonstrate notification system."""
    print("\n" + "=" * 70)
    print("NOTIFICATION SYSTEM DEMO")
    print("=" * 70)

    notifier = NotificationManager()

    # Test info alert
    print("\n[1] Sending INFO alert...")
    notifier.info(
        title="System Started",
        message="Cloud AI Analyzer is now running.",
        data={'version': '1.0', 'mode': 'paper'}
    )

    # Test warning alert
    print("\n[2] Sending WARNING alert...")
    notifier.warning(
        title="High Volatility",
        message="BTC volatility is above normal levels.",
        symbol="BTCUSDT",
        data={'volatility': '5.2%'}
    )

    # Test trade signal
    print("\n[3] Sending TRADE signal...")
    notifier.trade_signal(
        symbol="XRPUSDT",
        direction="LONG",
        confidence=0.75,
        entry_price=2.03,
        data={'signals': ['RSI', 'MACD', 'Whale']}
    )

    # Test circuit breaker
    print("\n[4] Sending CIRCUIT BREAKER alert...")
    notifier.alert_circuit_breaker(
        reason="BTC dropped 12% in 7 days",
        until=datetime.now()
    )

    print("\n[5] Check alerts.log for logged messages")
    print("=" * 70)


if __name__ == "__main__":
    demo_notifications()
