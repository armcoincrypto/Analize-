"""
Telegram Notifier for LongTrader

Sends notifications for:
- Trade executions (buy/sell)
- Signals detected
- Daily summaries
- Errors and warnings
"""

import requests
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

from .database import Position


logger = logging.getLogger(__name__)


class TelegramNotifier:
    """
    Telegram notification sender.

    Uses Telegram Bot API to send trading alerts.
    """

    API_URL = "https://api.telegram.org/bot{token}/{method}"

    def __init__(self, bot_token: str, chat_id: str, enabled: bool = True):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = enabled

    def _send_request(self, method: str, params: Dict[str, Any]) -> bool:
        """Send request to Telegram API."""
        if not self.enabled:
            logger.debug("Telegram disabled, skipping notification")
            return True

        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram not configured (missing token or chat_id)")
            return False

        try:
            url = self.API_URL.format(token=self.bot_token, method=method)
            response = requests.post(url, json=params, timeout=10)
            data = response.json()

            if not data.get("ok"):
                logger.error(f"Telegram API error: {data.get('description')}")
                return False

            return True

        except Exception as e:
            logger.error(f"Telegram request failed: {e}")
            return False

    def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """
        Send a text message.

        Args:
            text: Message text (supports HTML formatting)
            parse_mode: HTML or Markdown

        Returns:
            True if sent successfully
        """
        params = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
        }
        return self._send_request("sendMessage", params)

    def notify_buy(
        self,
        symbol: str,
        qty: float,
        price: float,
        usdt_amount: float,
        protection_passed: int,
        paper_mode: bool = False,
    ) -> bool:
        """Notify of buy execution."""
        mode = "[PAPER] " if paper_mode else ""

        message = f"""
{mode}<b>BUY</b> {symbol}

<b>Quantity:</b> {qty:.4f}
<b>Price:</b> ${price:.4f}
<b>Amount:</b> ${usdt_amount:.2f}
<b>Protections:</b> {protection_passed}/4 passed

<i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>
"""
        return self.send_message(message.strip())

    def notify_sell(
        self,
        symbol: str,
        qty: float,
        entry_price: float,
        exit_price: float,
        pnl_usdt: float,
        pnl_pct: float,
        reason: str,
        paper_mode: bool = False,
    ) -> bool:
        """Notify of sell execution."""
        mode = "[PAPER] " if paper_mode else ""
        emoji = "" if pnl_usdt >= 0 else ""

        message = f"""
{mode}<b>SELL</b> {symbol} {emoji}

<b>Quantity:</b> {qty:.4f}
<b>Entry:</b> ${entry_price:.4f}
<b>Exit:</b> ${exit_price:.4f}
<b>P&L:</b> ${pnl_usdt:+.2f} ({pnl_pct:+.1f}%)
<b>Reason:</b> {reason}

<i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>
"""
        return self.send_message(message.strip())

    def notify_signal(
        self,
        symbol: str,
        signal_type: str,
        price: float,
        lower_band: float,
        middle_band: float,
        protection_passed: int,
        reason: str,
    ) -> bool:
        """Notify of signal detection."""
        if signal_type == "STRONG_BUY":
            emoji = ""
        elif signal_type == "WEAK_BUY":
            emoji = ""
        elif signal_type == "SELL":
            emoji = ""
        else:
            emoji = ""

        message = f"""
{emoji} <b>SIGNAL:</b> {signal_type}

<b>Symbol:</b> {symbol}
<b>Price:</b> ${price:.4f}
<b>Lower Band:</b> ${lower_band:.4f}
<b>Middle Band:</b> ${middle_band:.4f}
<b>Protections:</b> {protection_passed}/4
<b>Details:</b> {reason}

<i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>
"""
        return self.send_message(message.strip())

    def notify_daily_summary(
        self,
        stats: Dict[str, Any],
        open_positions: List[Position],
        balance: float,
    ) -> bool:
        """Send daily summary."""
        positions_text = ""
        if open_positions:
            positions_text = "\n<b>Open Positions:</b>\n"
            for pos in open_positions:
                days = (datetime.now() - pos.entry_date).days
                positions_text += f"  - {pos.symbol}: {pos.entry_qty} @ ${pos.entry_price:.4f} ({days}d)\n"
        else:
            positions_text = "\n<b>No open positions</b>\n"

        message = f"""
<b>DAILY SUMMARY</b>

<b>Balance:</b> ${balance:.2f}
<b>Total Positions:</b> {stats.get('total_positions', 0)}
<b>Open:</b> {stats.get('open_positions', 0)}
<b>Closed:</b> {stats.get('closed_positions', 0)}
<b>Win Rate:</b> {stats.get('win_rate', 0):.1f}%
<b>Total P&L:</b> ${stats.get('total_pnl_usdt', 0):.2f}
{positions_text}
<i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>
"""
        return self.send_message(message.strip())

    def notify_startup(self, config: Dict[str, Any]) -> bool:
        """Notify bot startup."""
        mode = config.get("bot", {}).get("mode", "paper")
        coins = list(config.get("coins", {}).keys())

        message = f"""
<b>LongTrader Started</b>

<b>Mode:</b> {mode.upper()}
<b>Exchange:</b> {config.get('exchange', {}).get('name', 'bybit')}
<b>Coins:</b> {', '.join(coins)}
<b>Check Interval:</b> {config.get('bot', {}).get('check_interval_seconds', 3600)}s

<i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>
"""
        return self.send_message(message.strip())

    def notify_error(self, error: str, context: str = "") -> bool:
        """Notify of error."""
        message = f"""
<b>ERROR</b>

<b>Context:</b> {context or 'Unknown'}
<b>Error:</b> {error}

<i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>
"""
        return self.send_message(message.strip())

    def notify_shutdown(self, reason: str = "Manual") -> bool:
        """Notify bot shutdown."""
        message = f"""
<b>LongTrader Stopped</b>

<b>Reason:</b> {reason}

<i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>
"""
        return self.send_message(message.strip())


# Test function
def test_notifier():
    """Test the notifier (requires valid credentials)."""
    print("Testing Telegram Notifier...")
    print("=" * 50)

    # Test with disabled mode
    notifier = TelegramNotifier(
        bot_token="test",
        chat_id="test",
        enabled=False,
    )

    # These won't actually send (disabled)
    result = notifier.send_message("Test message")
    print(f"Send message (disabled): {result}")

    result = notifier.notify_signal(
        symbol="XRPUSDT",
        signal_type="STRONG_BUY",
        price=2.50,
        lower_band=2.45,
        middle_band=2.60,
        protection_passed=4,
        reason="All protections passed",
    )
    print(f"Notify signal (disabled): {result}")

    print("\nNotifier initialized (disabled mode for testing)")
    print("To test live, provide valid bot_token and chat_id")


if __name__ == "__main__":
    test_notifier()
