#!/usr/bin/env python3
"""
Telegram Bot Setup Helper

This script helps you:
1. Get your chat_id for the alert bot
2. Test sending messages
3. Verify the bot is working
"""

import requests
import json
from pathlib import Path

# Load config
CONFIG_PATH = Path(__file__).parent / "alert_config.json"


def load_config():
    """Load config file."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    return {}


def save_config(config):
    """Save config file."""
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)


def get_bot_token():
    """Get bot token from config or user input."""
    config = load_config()
    token = config.get("notifications", {}).get("telegram", {}).get("bot_token", "")

    if token and token != "YOUR_BOT_TOKEN":
        return token

    print("\nNo bot token found in config.")
    print("To create a Telegram bot:")
    print("  1. Open Telegram and search for @BotFather")
    print("  2. Send /newbot and follow instructions")
    print("  3. Copy the token BotFather gives you")
    print()

    token = input("Enter your bot token: ").strip()
    return token


def get_updates(token: str):
    """Get recent messages sent to the bot."""
    url = f"https://api.telegram.org/bot{token}/getUpdates"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error: {e}")
    return None


def send_message(token: str, chat_id: str, message: str):
    """Send a test message."""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        response = requests.post(
            url,
            json={
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "HTML"
            },
            timeout=10
        )
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    print("=" * 60)
    print("TELEGRAM BOT SETUP")
    print("=" * 60)

    # Get bot token
    token = get_bot_token()

    if not token:
        print("\nNo token provided. Exiting.")
        return

    print(f"\nUsing bot token: {token[:10]}...{token[-5:]}")

    # Verify bot
    print("\nVerifying bot...")
    url = f"https://api.telegram.org/bot{token}/getMe"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            bot_info = response.json()["result"]
            print(f"  Bot name: @{bot_info['username']}")
            print(f"  Bot ID: {bot_info['id']}")
        else:
            print(f"  Error: Invalid token")
            return
    except Exception as e:
        print(f"  Error: {e}")
        return

    # Get chat_id
    print("\n" + "=" * 60)
    print("STEP 1: GET YOUR CHAT ID")
    print("=" * 60)
    print(f"""
To get your chat_id:
  1. Open Telegram
  2. Find your bot: @{bot_info['username']}
  3. Send any message to the bot (e.g., "hello")
  4. Press Enter here after sending the message
""")

    input("Press Enter after sending a message to your bot...")

    # Fetch updates
    print("\nFetching messages...")
    updates = get_updates(token)

    if not updates or not updates.get("ok"):
        print("  Could not fetch updates")
        return

    messages = updates.get("result", [])

    if not messages:
        print("  No messages found. Make sure you sent a message to the bot.")
        print("  Try sending another message and run this script again.")
        return

    # Find chat_id
    chat_ids = set()
    for msg in messages:
        if "message" in msg:
            chat = msg["message"]["chat"]
            chat_ids.add((chat["id"], chat.get("first_name", chat.get("title", "Unknown"))))

    print("\nFound chat(s):")
    for chat_id, name in chat_ids:
        print(f"  Chat ID: {chat_id} (Name: {name})")

    # Use the first/most recent one
    chat_id = list(chat_ids)[0][0]

    # Test message
    print("\n" + "=" * 60)
    print("STEP 2: TEST MESSAGE")
    print("=" * 60)

    test_msg = """
<b>Bollinger Alert System</b>

Setup successful!

Your alerts will appear here when:
- STRONG BUY signals occur
- Price approaches buy zone
- Exit targets are reached

<i>Strategy: Bollinger Mean Reversion with 5 Protection Filters</i>
"""

    print(f"\nSending test message to chat {chat_id}...")
    if send_message(token, str(chat_id), test_msg):
        print("  Test message sent successfully!")
    else:
        print("  Failed to send test message")
        return

    # Update config
    print("\n" + "=" * 60)
    print("STEP 3: SAVE CONFIGURATION")
    print("=" * 60)

    config = load_config()
    if "notifications" not in config:
        config["notifications"] = {}
    if "telegram" not in config["notifications"]:
        config["notifications"]["telegram"] = {}

    config["notifications"]["telegram"]["enabled"] = True
    config["notifications"]["telegram"]["bot_token"] = token
    config["notifications"]["telegram"]["chat_id"] = str(chat_id)

    save_config(config)
    print(f"\nConfiguration saved!")
    print(f"  Bot token: {token[:10]}...{token[-5:]}")
    print(f"  Chat ID: {chat_id}")

    print("\n" + "=" * 60)
    print("SETUP COMPLETE!")
    print("=" * 60)
    print("""
Your Telegram alerts are now configured.

To start monitoring:
  python examples/bollinger_alerts.py

You will receive alerts for:
  - STRONG BUY signals (all protections passed)
  - WEAK BUY warnings (some protections failed)
  - Price near buy zone alerts
  - Sell signals at middle band
""")


if __name__ == "__main__":
    main()
