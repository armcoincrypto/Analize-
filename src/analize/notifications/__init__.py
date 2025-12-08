"""
Notifications module for Analize.

Handles alerts and notifications via:
- Telegram
- Slack
- Email
"""

from analize.notifications.sender import NotificationSender

__all__ = ["NotificationSender"]
