"""
Notification sender for alerts and reports.
"""

import asyncio
from typing import Any

import httpx

from analize.config import get_settings
from analize.models.reports import Alert


class NotificationSender:
    """Sends notifications via various channels."""

    def __init__(self):
        self.settings = get_settings().notifications
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def send_telegram(self, message: str, parse_mode: str = "HTML") -> bool:
        """Send message via Telegram."""
        if not self.settings.telegram_bot_token or not self.settings.telegram_chat_id:
            return False

        client = await self._get_client()
        url = f"https://api.telegram.org/bot{self.settings.telegram_bot_token}/sendMessage"

        try:
            response = await client.post(
                url,
                json={
                    "chat_id": self.settings.telegram_chat_id,
                    "text": message,
                    "parse_mode": parse_mode,
                },
            )
            return response.status_code == 200
        except Exception:
            return False

    async def send_slack(self, message: str, blocks: list[dict] | None = None) -> bool:
        """Send message via Slack webhook."""
        if not self.settings.slack_webhook_url:
            return False

        client = await self._get_client()

        payload: dict[str, Any] = {"text": message}
        if blocks:
            payload["blocks"] = blocks

        try:
            response = await client.post(self.settings.slack_webhook_url, json=payload)
            return response.status_code == 200
        except Exception:
            return False

    async def send_email(
        self,
        subject: str,
        body: str,
        to: list[str] | None = None,
    ) -> bool:
        """Send email notification."""
        if not self.settings.smtp_host:
            return False

        recipients = to or self.settings.email_to
        if not recipients:
            return False

        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart

            msg = MIMEMultipart()
            msg["From"] = self.settings.email_from
            msg["To"] = ", ".join(recipients)
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "html"))

            with smtplib.SMTP(self.settings.smtp_host, self.settings.smtp_port) as server:
                server.starttls()
                if self.settings.smtp_user and self.settings.smtp_password:
                    server.login(self.settings.smtp_user, self.settings.smtp_password)
                server.sendmail(self.settings.email_from, recipients, msg.as_string())

            return True
        except Exception:
            return False

    async def send_alert(self, alert: Alert) -> dict[str, bool]:
        """Send alert via all configured channels."""
        results = {}

        # Format message
        message = self._format_alert_message(alert)

        # Send via each channel
        if self.settings.telegram_bot_token:
            results["telegram"] = await self.send_telegram(message)

        if self.settings.slack_webhook_url:
            results["slack"] = await self.send_slack(
                message,
                blocks=self._format_slack_blocks(alert),
            )

        if self.settings.smtp_host and self.settings.email_to:
            results["email"] = await self.send_email(
                subject=f"[Analize Alert] {alert.title}",
                body=self._format_email_body(alert),
            )

        return results

    def _format_alert_message(self, alert: Alert) -> str:
        """Format alert for text channels."""
        severity_emoji = {
            "INFO": "ℹ️",
            "WARNING": "⚠️",
            "CRITICAL": "🚨",
        }
        emoji = severity_emoji.get(alert.severity, "📢")

        return (
            f"{emoji} <b>{alert.title}</b>\n\n"
            f"{alert.message}\n\n"
            f"Metric: {alert.metric_name}\n"
            f"Value: {alert.metric_value:.4f}\n"
            f"Threshold: {alert.threshold:.4f}\n"
            f"Symbol: {alert.symbol or 'All'}\n"
            f"Time: {alert.triggered_at.isoformat()}"
        )

    def _format_slack_blocks(self, alert: Alert) -> list[dict]:
        """Format alert as Slack blocks."""
        return [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": f"🚨 {alert.title}"},
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": alert.message},
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Metric:* {alert.metric_name}"},
                    {"type": "mrkdwn", "text": f"*Value:* {alert.metric_value:.4f}"},
                    {"type": "mrkdwn", "text": f"*Threshold:* {alert.threshold:.4f}"},
                    {"type": "mrkdwn", "text": f"*Symbol:* {alert.symbol or 'All'}"},
                ],
            },
        ]

    def _format_email_body(self, alert: Alert) -> str:
        """Format alert as HTML email body."""
        severity_colors = {
            "INFO": "#17a2b8",
            "WARNING": "#ffc107",
            "CRITICAL": "#dc3545",
        }
        color = severity_colors.get(alert.severity, "#6c757d")

        return f"""
        <html>
        <body style="font-family: Arial, sans-serif;">
            <div style="background-color: {color}; color: white; padding: 10px 20px; border-radius: 5px;">
                <h2 style="margin: 0;">{alert.title}</h2>
            </div>
            <div style="padding: 20px;">
                <p>{alert.message}</p>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr>
                        <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Metric</strong></td>
                        <td style="padding: 8px; border-bottom: 1px solid #ddd;">{alert.metric_name}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Value</strong></td>
                        <td style="padding: 8px; border-bottom: 1px solid #ddd;">{alert.metric_value:.4f}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Threshold</strong></td>
                        <td style="padding: 8px; border-bottom: 1px solid #ddd;">{alert.threshold:.4f}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Symbol</strong></td>
                        <td style="padding: 8px; border-bottom: 1px solid #ddd;">{alert.symbol or 'All'}</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px;"><strong>Time</strong></td>
                        <td style="padding: 8px;">{alert.triggered_at.isoformat()}</td>
                    </tr>
                </table>
            </div>
            <div style="background-color: #f8f9fa; padding: 10px 20px; font-size: 12px; color: #6c757d;">
                Analize - Cloud AI Analyzer for ScalperBot
            </div>
        </body>
        </html>
        """


async def send_daily_report_notification(report_summary: str) -> None:
    """Send daily report notification."""
    sender = NotificationSender()
    try:
        await sender.send_telegram(f"📊 <b>Daily Report</b>\n\n{report_summary}")
        await sender.send_slack(f"📊 *Daily Report*\n\n{report_summary}")
    finally:
        await sender.close()
