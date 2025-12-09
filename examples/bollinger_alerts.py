"""
Production Bollinger Band Alert System

Monitors XRP, SOL, ATOM with optimal Bollinger settings and alerts on buy opportunities.
Supports console alerts, file logging, and webhook notifications.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
import os
from typing import Optional, List, Dict
from analize.features.indicators import TechnicalIndicators


# ============================================================================
# CONFIGURATION
# ============================================================================

# Optimal settings per coin (from multi-strategy analysis)
COIN_CONFIG = {
    "XRPUSDT": {
        "name": "XRP",
        "period": 20,
        "std_dev": 1.5,
        "expected_profit": 133.0,  # % from backtest
        "win_rate": 100.0,
        "avg_trade_days": 5,
    },
    "SOLUSDT": {
        "name": "SOL",
        "period": 20,
        "std_dev": 1.5,
        "expected_profit": 78.0,
        "win_rate": 86.0,
        "avg_trade_days": 6,
    },
    "ATOMUSDT": {
        "name": "ATOM",
        "period": 20,
        "std_dev": 1.5,
        "expected_profit": 65.0,
        "win_rate": 83.0,
        "avg_trade_days": 4,
    },
}

# Alert settings
ALERT_LOG_PATH = Path(__file__).parent.parent / "data" / "alerts.json"
CHECK_INTERVAL_SECONDS = 3600  # 1 hour (for daily strategy)

# Webhook URL (set via environment variable for security)
WEBHOOK_URL = os.environ.get("ALERT_WEBHOOK_URL", "")


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class CoinStatus:
    """Current status of a monitored coin."""
    symbol: str
    name: str
    current_price: float
    lower_band: float
    middle_band: float
    upper_band: float
    distance_to_buy: float  # % above lower band
    signal: str  # "BUY", "SELL", "HOLD"
    timestamp: datetime
    config: dict


@dataclass
class Alert:
    """Alert record."""
    alert_id: str
    symbol: str
    alert_type: str  # "BUY_SIGNAL", "SELL_SIGNAL", "NEAR_BUY"
    price: float
    lower_band: float
    middle_band: float
    distance: float
    message: str
    timestamp: datetime
    acknowledged: bool = False


# ============================================================================
# DATA FETCHING
# ============================================================================

def fetch_daily_data(symbol: str, days: int = 60) -> pd.DataFrame:
    """Fetch daily candles from Binance."""
    urls = [
        "https://api.binance.com/api/v3/klines",
        "https://api.binance.us/api/v3/klines",
    ]

    params = {
        "symbol": symbol,
        "interval": "1d",
        "startTime": int((datetime.now() - timedelta(days=days)).timestamp() * 1000),
        "endTime": int(datetime.now().timestamp() * 1000),
        "limit": 1000
    }

    for url in urls:
        try:
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    df = pd.DataFrame(data, columns=[
                        "timestamp", "open", "high", "low", "close", "volume",
                        "close_time", "quote_volume", "trades", "taker_buy_base",
                        "taker_buy_quote", "ignore"
                    ])
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                    for col in ["open", "high", "low", "close", "volume"]:
                        df[col] = df[col].astype(float)
                    return df[["timestamp", "open", "high", "low", "close", "volume"]]
        except Exception:
            continue

    return pd.DataFrame()


# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_coin(symbol: str, config: dict) -> Optional[CoinStatus]:
    """Analyze a coin and return its current status."""
    df = fetch_daily_data(symbol, days=60)

    if df.empty or len(df) < config["period"] + 5:
        return None

    # Calculate Bollinger Bands
    upper, middle, lower = TechnicalIndicators.bollinger_bands(
        df["close"],
        period=config["period"],
        std_dev=config["std_dev"]
    )

    current_price = df["close"].iloc[-1]
    current_lower = lower.iloc[-1]
    current_middle = middle.iloc[-1]
    current_upper = upper.iloc[-1]

    # Calculate distance to buy zone
    distance = ((current_price - current_lower) / current_lower) * 100

    # Determine signal
    if current_price <= current_lower:
        signal = "BUY"
    elif current_price >= current_middle:
        signal = "SELL"
    else:
        signal = "HOLD"

    return CoinStatus(
        symbol=symbol,
        name=config["name"],
        current_price=current_price,
        lower_band=current_lower,
        middle_band=current_middle,
        upper_band=current_upper,
        distance_to_buy=distance,
        signal=signal,
        timestamp=datetime.now(),
        config=config,
    )


# ============================================================================
# ALERTING
# ============================================================================

class AlertManager:
    """Manages alerts and notifications."""

    def __init__(self):
        self.alerts: List[Alert] = []
        self.sent_alerts: set = set()  # Track sent alerts to avoid duplicates
        self._load_alerts()

    def _load_alerts(self):
        """Load previous alerts from file."""
        if ALERT_LOG_PATH.exists():
            try:
                with open(ALERT_LOG_PATH, "r") as f:
                    data = json.load(f)
                    for a in data.get("alerts", []):
                        self.sent_alerts.add(a.get("alert_id", ""))
            except Exception:
                pass

    def _save_alerts(self):
        """Save alerts to file."""
        ALERT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "last_updated": datetime.now().isoformat(),
            "alerts": [
                {
                    "alert_id": a.alert_id,
                    "symbol": a.symbol,
                    "alert_type": a.alert_type,
                    "price": a.price,
                    "lower_band": a.lower_band,
                    "middle_band": a.middle_band,
                    "distance": a.distance,
                    "message": a.message,
                    "timestamp": a.timestamp.isoformat(),
                    "acknowledged": a.acknowledged,
                }
                for a in self.alerts[-100:]  # Keep last 100 alerts
            ]
        }

        with open(ALERT_LOG_PATH, "w") as f:
            json.dump(data, f, indent=2)

    def create_alert(self, status: CoinStatus) -> Optional[Alert]:
        """Create an alert based on coin status."""
        # Generate unique alert ID based on date and signal
        today = datetime.now().strftime("%Y-%m-%d")
        alert_id = f"{status.symbol}_{status.signal}_{today}"

        # Skip if already sent today
        if alert_id in self.sent_alerts:
            return None

        # Determine alert type and message
        if status.signal == "BUY":
            alert_type = "BUY_SIGNAL"
            message = (
                f"BUY SIGNAL: {status.name} at ${status.current_price:.4f}\n"
                f"Price has touched the lower Bollinger Band!\n"
                f"Expected profit: +{status.config['expected_profit']:.0f}%\n"
                f"Historical win rate: {status.config['win_rate']:.0f}%\n"
                f"Target: ${status.middle_band:.4f} (middle band)"
            )
        elif status.distance_to_buy <= 3.0:
            alert_type = "NEAR_BUY"
            message = (
                f"NEAR BUY ZONE: {status.name} at ${status.current_price:.4f}\n"
                f"Only {status.distance_to_buy:.1f}% above lower band\n"
                f"Lower band: ${status.lower_band:.4f}\n"
                f"Watch closely for entry opportunity!"
            )
        elif status.signal == "SELL":
            alert_type = "SELL_SIGNAL"
            message = (
                f"SELL SIGNAL: {status.name} at ${status.current_price:.4f}\n"
                f"Price has reached the middle band!\n"
                f"Consider taking profits."
            )
        else:
            return None

        alert = Alert(
            alert_id=alert_id,
            symbol=status.symbol,
            alert_type=alert_type,
            price=status.current_price,
            lower_band=status.lower_band,
            middle_band=status.middle_band,
            distance=status.distance_to_buy,
            message=message,
            timestamp=datetime.now(),
        )

        self.alerts.append(alert)
        self.sent_alerts.add(alert_id)
        self._save_alerts()

        return alert

    def send_console_alert(self, alert: Alert):
        """Print alert to console."""
        border = "=" * 60
        print(f"\n{border}")
        print(f"{'ALERT':^60}")
        print(border)
        print(f"\n{alert.message}\n")
        print(f"Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(border + "\n")

    def send_webhook_alert(self, alert: Alert):
        """Send alert via webhook (Telegram, Discord, Slack, etc.)."""
        if not WEBHOOK_URL:
            return

        try:
            payload = {
                "text": alert.message,
                "alert_type": alert.alert_type,
                "symbol": alert.symbol,
                "price": alert.price,
                "timestamp": alert.timestamp.isoformat(),
            }

            requests.post(
                WEBHOOK_URL,
                json=payload,
                timeout=10,
                headers={"Content-Type": "application/json"}
            )
        except Exception as e:
            print(f"  Warning: Failed to send webhook: {e}")

    def process_alert(self, alert: Alert):
        """Process and send alert through all channels."""
        self.send_console_alert(alert)
        self.send_webhook_alert(alert)


# ============================================================================
# MONITORING
# ============================================================================

def print_status_table(statuses: List[CoinStatus]):
    """Print a status table for all coins."""
    print("\n" + "=" * 80)
    print(f"{'BOLLINGER BAND MONITOR':^80}")
    print(f"{'Updated: ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'):^80}")
    print("=" * 80)

    print(f"\n{'Coin':<8} {'Price':<12} {'Lower':<12} {'Middle':<12} {'Distance':<12} {'Signal':<10}")
    print("-" * 80)

    for s in statuses:
        signal_display = s.signal
        if s.signal == "BUY":
            signal_display = "BUY NOW"
        elif s.distance_to_buy <= 3.0:
            signal_display = "WATCH"

        print(f"{s.name:<8} ${s.current_price:<10.4f} ${s.lower_band:<10.4f} "
              f"${s.middle_band:<10.4f} {s.distance_to_buy:>+10.1f}%  {signal_display:<10}")

    print("-" * 80)


def print_recommendations(statuses: List[CoinStatus]):
    """Print trading recommendations."""
    buy_signals = [s for s in statuses if s.signal == "BUY"]
    near_buy = [s for s in statuses if s.signal != "BUY" and s.distance_to_buy <= 3.0]
    sell_signals = [s for s in statuses if s.signal == "SELL"]

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    if buy_signals:
        print("\n  BUY NOW:")
        for s in buy_signals:
            print(f"    {s.name}: Price ${s.current_price:.4f} at lower band")
            print(f"            Target: ${s.middle_band:.4f} (+{((s.middle_band/s.current_price)-1)*100:.1f}%)")

    if near_buy:
        print("\n  WATCH CLOSELY (Near Buy Zone):")
        for s in near_buy:
            print(f"    {s.name}: {s.distance_to_buy:+.1f}% from buy zone")
            print(f"            Buy if drops to ${s.lower_band:.4f}")

    if sell_signals:
        print("\n  CONSIDER SELLING:")
        for s in sell_signals:
            print(f"    {s.name}: Price at middle band")

    if not buy_signals and not near_buy and not sell_signals:
        print("\n  No immediate action needed.")
        closest = min(statuses, key=lambda x: x.distance_to_buy)
        print(f"  Closest to buy: {closest.name} ({closest.distance_to_buy:+.1f}% from lower band)")

    print("\n" + "=" * 80)


def run_monitor(continuous: bool = False, interval_seconds: int = CHECK_INTERVAL_SECONDS):
    """Run the monitoring system."""
    print("=" * 80)
    print("BOLLINGER BAND ALERT SYSTEM")
    print("=" * 80)

    print(f"\nMonitoring {len(COIN_CONFIG)} coins:")
    for symbol, config in COIN_CONFIG.items():
        print(f"  - {config['name']}: Bollinger({config['period']}, {config['std_dev']})")

    print(f"\nStrategy: Buy at lower band, sell at middle band")
    print(f"Fee assumption: 0.2% per round trip")

    if continuous:
        print(f"Mode: Continuous (checking every {interval_seconds} seconds)")
    else:
        print(f"Mode: Single check")

    alert_manager = AlertManager()

    while True:
        try:
            # Analyze all coins
            statuses = []
            print(f"\n  Fetching data...", end=" ")

            for symbol, config in COIN_CONFIG.items():
                status = analyze_coin(symbol, config)
                if status:
                    statuses.append(status)

            print(f"Done ({len(statuses)}/{len(COIN_CONFIG)} coins)")

            if not statuses:
                print("  Warning: Could not fetch data for any coins!")
                if continuous:
                    time.sleep(60)
                    continue
                else:
                    break

            # Print status table
            print_status_table(statuses)

            # Check for alerts
            for status in statuses:
                alert = alert_manager.create_alert(status)
                if alert:
                    alert_manager.process_alert(alert)

            # Print recommendations
            print_recommendations(statuses)

            if not continuous:
                break

            # Wait for next check
            print(f"\nNext check in {interval_seconds} seconds...")
            print("(Press Ctrl+C to stop)")
            time.sleep(interval_seconds)

        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user.")
            break
        except Exception as e:
            print(f"\n  Error: {e}")
            if continuous:
                print("  Retrying in 60 seconds...")
                time.sleep(60)
            else:
                break


def check_once():
    """Run a single check and print results."""
    run_monitor(continuous=False)


def run_continuous():
    """Run continuous monitoring."""
    run_monitor(continuous=True, interval_seconds=CHECK_INTERVAL_SECONDS)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point."""
    print("\n" + "=" * 60)
    print("BOLLINGER ALERT SYSTEM")
    print("=" * 60)

    print("""
  Options:
  --------
  1. Single check (check now and exit)
  2. Continuous monitoring (check every hour)
  3. Quick monitor (check every 5 minutes)

  Webhook Setup:
  --------------
  Set ALERT_WEBHOOK_URL environment variable for notifications:
    export ALERT_WEBHOOK_URL="https://your-webhook-url"

  Supported: Discord, Slack, Telegram bots, custom webhooks
""")

    try:
        choice = input("Select option (1-3) [1]: ").strip() or "1"
    except:
        choice = "1"

    if choice == "1":
        check_once()
    elif choice == "2":
        run_continuous()
    elif choice == "3":
        run_monitor(continuous=True, interval_seconds=300)
    else:
        check_once()


if __name__ == "__main__":
    main()
