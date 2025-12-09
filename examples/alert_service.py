#!/usr/bin/env python3
"""
Production Alert Service

A robust alert service with Telegram integration, configurable settings,
and proper error handling for production deployment.

Usage:
    python alert_service.py                    # Interactive mode
    python alert_service.py --check            # Single check
    python alert_service.py --daemon           # Run as daemon
    python alert_service.py --status           # Show current status
"""

import argparse
import json
import os
import sys
import time
import signal
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from analize.features.indicators import TechnicalIndicators


# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent.parent
CONFIG_PATH = BASE_DIR / "examples" / "alert_config.json"
DATA_DIR = BASE_DIR / "data"
ALERT_LOG = DATA_DIR / "alerts.json"
PID_FILE = DATA_DIR / "alert_service.pid"


def load_config() -> dict:
    """Load configuration from JSON file."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    return {}


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class CoinAnalysis:
    symbol: str
    name: str
    price: float
    lower_band: float
    middle_band: float
    upper_band: float
    distance_pct: float
    signal: str
    timestamp: datetime


# ============================================================================
# API CLIENT
# ============================================================================

class BinanceClient:
    """Simple Binance API client."""

    ENDPOINTS = [
        "https://api.binance.com/api/v3",
        "https://api.binance.us/api/v3",
    ]

    def __init__(self, timeout: int = 15):
        self.timeout = timeout

    def get_klines(self, symbol: str, interval: str = "1d", limit: int = 60) -> pd.DataFrame:
        """Fetch candlestick data."""
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }

        for base_url in self.ENDPOINTS:
            try:
                response = requests.get(
                    f"{base_url}/klines",
                    params=params,
                    timeout=self.timeout
                )
                if response.status_code == 200:
                    data = response.json()
                    if data:
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
# NOTIFICATION HANDLERS
# ============================================================================

class TelegramNotifier:
    """Send alerts via Telegram."""

    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"

    def send(self, message: str) -> bool:
        """Send a message."""
        try:
            response = requests.post(
                f"{self.base_url}/sendMessage",
                json={
                    "chat_id": self.chat_id,
                    "text": message,
                    "parse_mode": "HTML",
                },
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            print(f"  Telegram error: {e}")
            return False


class WebhookNotifier:
    """Send alerts via generic webhook."""

    def __init__(self, url: str):
        self.url = url

    def send(self, message: str, data: dict = None) -> bool:
        """Send a webhook notification."""
        try:
            payload = {"text": message}
            if data:
                payload.update(data)

            response = requests.post(
                self.url,
                json=payload,
                timeout=10,
                headers={"Content-Type": "application/json"}
            )
            return response.status_code in [200, 201, 204]
        except Exception as e:
            print(f"  Webhook error: {e}")
            return False


# ============================================================================
# ANALYZER
# ============================================================================

class BollingerAnalyzer:
    """Analyze coins using Bollinger Bands."""

    def __init__(self, client: BinanceClient):
        self.client = client

    def analyze(self, symbol: str, config: dict) -> Optional[CoinAnalysis]:
        """Analyze a single coin."""
        df = self.client.get_klines(symbol, interval="1d", limit=60)

        if df.empty or len(df) < config["period"] + 5:
            return None

        upper, middle, lower = TechnicalIndicators.bollinger_bands(
            df["close"],
            period=config["period"],
            std_dev=config["std_dev"]
        )

        price = df["close"].iloc[-1]
        low = lower.iloc[-1]
        mid = middle.iloc[-1]
        high = upper.iloc[-1]

        distance = ((price - low) / low) * 100

        if price <= low:
            signal = "BUY"
        elif price >= mid:
            signal = "SELL"
        else:
            signal = "HOLD"

        return CoinAnalysis(
            symbol=symbol,
            name=config["name"],
            price=price,
            lower_band=low,
            middle_band=mid,
            upper_band=high,
            distance_pct=distance,
            signal=signal,
            timestamp=datetime.now(),
        )


# ============================================================================
# ALERT SERVICE
# ============================================================================

class AlertService:
    """Main alert service."""

    def __init__(self, config: dict):
        self.config = config
        self.client = BinanceClient()
        self.analyzer = BollingerAnalyzer(self.client)
        self.notifiers: List = []
        self.sent_alerts: set = set()
        self.running = False

        self._setup_notifiers()
        self._load_sent_alerts()

    def _setup_notifiers(self):
        """Initialize notification handlers."""
        notif_config = self.config.get("notifications", {})

        # Telegram
        tg = notif_config.get("telegram", {})
        if tg.get("enabled") and tg.get("bot_token") and tg.get("chat_id"):
            self.notifiers.append(TelegramNotifier(tg["bot_token"], tg["chat_id"]))
            print("  Telegram notifications enabled")

        # Webhook
        wh = notif_config.get("webhook", {})
        if wh.get("enabled") and wh.get("url"):
            self.notifiers.append(WebhookNotifier(wh["url"]))
            print("  Webhook notifications enabled")

    def _load_sent_alerts(self):
        """Load previously sent alerts."""
        if ALERT_LOG.exists():
            try:
                with open(ALERT_LOG, "r") as f:
                    data = json.load(f)
                    for alert in data.get("alerts", []):
                        self.sent_alerts.add(alert.get("id", ""))
            except Exception:
                pass

    def _save_alert(self, alert_id: str, analysis: CoinAnalysis, message: str):
        """Save alert to log file."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        alerts = []
        if ALERT_LOG.exists():
            try:
                with open(ALERT_LOG, "r") as f:
                    alerts = json.load(f).get("alerts", [])
            except Exception:
                pass

        alerts.append({
            "id": alert_id,
            "symbol": analysis.symbol,
            "signal": analysis.signal,
            "price": analysis.price,
            "lower_band": analysis.lower_band,
            "distance": analysis.distance_pct,
            "message": message,
            "timestamp": analysis.timestamp.isoformat(),
        })

        with open(ALERT_LOG, "w") as f:
            json.dump({"alerts": alerts[-200:]}, f, indent=2)

        self.sent_alerts.add(alert_id)

    def _send_alert(self, message: str):
        """Send alert through all configured notifiers."""
        # Console
        print("\n" + "=" * 60)
        print("ALERT")
        print("=" * 60)
        print(message)
        print("=" * 60 + "\n")

        # Notifiers
        for notifier in self.notifiers:
            notifier.send(message)

    def check_coin(self, symbol: str, coin_config: dict) -> Optional[CoinAnalysis]:
        """Check a single coin for alerts."""
        analysis = self.analyzer.analyze(symbol, coin_config)

        if not analysis:
            return None

        # Generate alert ID
        today = datetime.now().strftime("%Y-%m-%d")
        alert_id = f"{symbol}_{analysis.signal}_{today}"

        # Check for BUY signal
        if analysis.signal == "BUY" and alert_id not in self.sent_alerts:
            message = (
                f"BUY SIGNAL: {analysis.name}\n"
                f"Price: ${analysis.price:.4f}\n"
                f"Lower Band: ${analysis.lower_band:.4f}\n"
                f"Target: ${analysis.middle_band:.4f}\n"
                f"Potential: +{((analysis.middle_band/analysis.price)-1)*100:.1f}%\n"
                f"\n{coin_config.get('notes', '')}"
            )
            self._send_alert(message)
            self._save_alert(alert_id, analysis, message)

        # Check for near-buy
        threshold = self.config.get("settings", {}).get("near_buy_threshold_percent", 3.0)
        near_id = f"{symbol}_NEAR_{today}"

        if analysis.signal == "HOLD" and analysis.distance_pct <= threshold and near_id not in self.sent_alerts:
            message = (
                f"NEAR BUY: {analysis.name}\n"
                f"Price: ${analysis.price:.4f}\n"
                f"Distance to buy: {analysis.distance_pct:.1f}%\n"
                f"Buy zone: ${analysis.lower_band:.4f}\n"
                f"\nWatch closely!"
            )
            self._send_alert(message)
            self._save_alert(near_id, analysis, message)

        return analysis

    def check_all(self) -> List[CoinAnalysis]:
        """Check all configured coins."""
        results = []
        coins = self.config.get("coins", {})

        for symbol, coin_config in coins.items():
            if not coin_config.get("enabled", True):
                continue

            analysis = self.check_coin(symbol, coin_config)
            if analysis:
                results.append(analysis)

        return results

    def print_status(self, analyses: List[CoinAnalysis]):
        """Print current status."""
        print("\n" + "=" * 75)
        print(f"{'BOLLINGER MONITOR':^75}")
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S'):^75}")
        print("=" * 75)

        print(f"\n{'Coin':<8} {'Price':<12} {'Lower':<12} {'Middle':<12} {'Dist':<10} {'Signal':<8}")
        print("-" * 75)

        for a in analyses:
            print(f"{a.name:<8} ${a.price:<10.4f} ${a.lower_band:<10.4f} "
                  f"${a.middle_band:<10.4f} {a.distance_pct:>+8.1f}%  {a.signal:<8}")

        print("-" * 75)

        # Summary
        buy_signals = [a for a in analyses if a.signal == "BUY"]
        near_buy = [a for a in analyses if a.distance_pct <= 3.0 and a.signal != "BUY"]

        if buy_signals:
            print(f"\n  ACTION: BUY {', '.join(a.name for a in buy_signals)}")
        elif near_buy:
            print(f"\n  WATCH: {', '.join(a.name for a in near_buy)} approaching buy zone")
        else:
            closest = min(analyses, key=lambda x: x.distance_pct) if analyses else None
            if closest:
                print(f"\n  Closest to buy: {closest.name} ({closest.distance_pct:+.1f}%)")

    def run_once(self):
        """Run a single check."""
        print("\nChecking all coins...", end=" ")
        results = self.check_all()
        print(f"Done ({len(results)} coins)")
        self.print_status(results)

    def run_daemon(self):
        """Run as a continuous daemon."""
        interval = self.config.get("settings", {}).get("check_interval_seconds", 3600)

        print(f"\nStarting daemon (interval: {interval}s)")
        print("Press Ctrl+C to stop\n")

        # Write PID
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(PID_FILE, "w") as f:
            f.write(str(os.getpid()))

        self.running = True

        def handle_signal(sig, frame):
            self.running = False
            print("\nShutting down...")

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

        while self.running:
            try:
                self.run_once()
                print(f"\nNext check in {interval}s...")

                # Sleep in small intervals to allow signal handling
                for _ in range(interval):
                    if not self.running:
                        break
                    time.sleep(1)
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(60)

        # Cleanup
        if PID_FILE.exists():
            PID_FILE.unlink()

        print("Daemon stopped.")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Bollinger Alert Service")
    parser.add_argument("--check", action="store_true", help="Run single check")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon")
    parser.add_argument("--status", action="store_true", help="Show status only")
    parser.add_argument("--config", type=str, help="Path to config file")

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config) if args.config else CONFIG_PATH
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        print(f"Warning: Config not found at {config_path}, using defaults")
        config = {
            "coins": {
                "XRPUSDT": {"name": "XRP", "enabled": True, "period": 20, "std_dev": 1.5},
                "SOLUSDT": {"name": "SOL", "enabled": True, "period": 20, "std_dev": 1.5},
                "ATOMUSDT": {"name": "ATOM", "enabled": True, "period": 20, "std_dev": 1.5},
            },
            "settings": {"check_interval_seconds": 3600}
        }

    service = AlertService(config)

    if args.daemon:
        service.run_daemon()
    elif args.check or args.status:
        service.run_once()
    else:
        # Interactive menu
        print("\n" + "=" * 50)
        print("BOLLINGER ALERT SERVICE")
        print("=" * 50)
        print("\nOptions:")
        print("  1. Check now")
        print("  2. Run daemon")
        print("  3. Exit")

        try:
            choice = input("\nSelect (1-3) [1]: ").strip() or "1"
        except:
            choice = "1"

        if choice == "1":
            service.run_once()
        elif choice == "2":
            service.run_daemon()


if __name__ == "__main__":
    main()
