"""
Production Bollinger Band Alert System with Smart Protections

Monitors XRP, SOL, ATOM with optimal Bollinger settings and 5 protection filters:
1. Trend Filter - Price > 200-day MA
2. Volume Filter - Volume >= 80% of average
3. First-Touch Rejection - Skip first touch after crash
4. Band Penetration - Price must go 1%+ below band
5. Time-Based Exit - Exit after 22 days max

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
import sys
from typing import Optional, List, Dict

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from analize.features.indicators import TechnicalIndicators
from analize.exchanges import MultiExchangeClient, Exchange


# ============================================================================
# CONFIGURATION
# ============================================================================

# Optimal settings per coin (from multi-strategy analysis)
COIN_CONFIG = {
    "XRPUSDT": {
        "name": "XRP",
        "period": 20,
        "std_dev": 1.5,
        "expected_profit": 133.0,
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

# Protection parameters
PROTECTION_CONFIG = {
    "trend_ma_period": 200,          # 200-day MA for trend filter
    "volume_ma_period": 20,          # 20-day volume MA
    "volume_min_ratio": 0.8,         # Min 80% of average volume
    "crash_lookback_days": 10,       # Days to look back for crash
    "crash_threshold_pct": -15.0,    # -15% = crash
    "first_touch_cooldown": 5,       # Days to wait after crash
    "penetration_min_pct": 1.0,      # Must go 1%+ below band
    "max_hold_days": 22,             # Exit if no bounce after 22 days
}

# Alert settings
ALERT_LOG_PATH = Path(__file__).parent.parent / "data" / "alerts.json"
CHECK_INTERVAL_SECONDS = 3600

# Webhook URL (set via environment variable)
WEBHOOK_URL = os.environ.get("ALERT_WEBHOOK_URL", "")


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ProtectionStatus:
    """Status of all 5 protection filters."""
    trend_ok: bool = False
    volume_ok: bool = False
    not_first_touch: bool = False
    penetration_ok: bool = False

    # Details for display
    price: float = 0.0
    ma_200: float = 0.0
    volume_ratio: float = 0.0
    days_since_crash: int = 999
    penetration_pct: float = 0.0

    @property
    def all_passed(self) -> bool:
        return self.trend_ok and self.volume_ok and self.not_first_touch and self.penetration_ok

    @property
    def passed_count(self) -> int:
        return sum([self.trend_ok, self.volume_ok, self.not_first_touch, self.penetration_ok])

    def get_failed_filters(self) -> List[str]:
        failed = []
        if not self.trend_ok:
            failed.append("TREND")
        if not self.volume_ok:
            failed.append("VOLUME")
        if not self.not_first_touch:
            failed.append("FIRST-TOUCH")
        if not self.penetration_ok:
            failed.append("PENETRATION")
        return failed


@dataclass
class CoinStatus:
    """Current status of a monitored coin with protection info."""
    symbol: str
    name: str
    current_price: float
    lower_band: float
    middle_band: float
    upper_band: float
    distance_to_buy: float
    signal: str  # "STRONG_BUY", "WEAK_BUY", "SELL", "HOLD"
    protection: ProtectionStatus
    timestamp: datetime
    config: dict


@dataclass
class Alert:
    """Alert record."""
    alert_id: str
    symbol: str
    alert_type: str
    price: float
    lower_band: float
    middle_band: float
    distance: float
    message: str
    protection_status: str
    timestamp: datetime
    acknowledged: bool = False


# ============================================================================
# DATA FETCHING (BYBIT PRIMARY)
# ============================================================================

# Initialize exchange client (Bybit primary with fallbacks)
EXCHANGE_CLIENT = MultiExchangeClient(preferred_exchange=Exchange.BYBIT)


def fetch_daily_data(symbol: str, days: int = 250) -> pd.DataFrame:
    """Fetch daily candles from Bybit (with fallback to other exchanges)."""
    df, exchange_name = EXCHANGE_CLIENT.get_klines(symbol, "1d", days)
    if not df.empty:
        print(f"    [{exchange_name}]", end="")
    return df


# ============================================================================
# PROTECTION FILTERS
# ============================================================================

class ProtectionFilters:
    """All 5 protection filters."""

    @staticmethod
    def check_trend(df: pd.DataFrame, idx: int) -> tuple:
        """Filter 1: Trend - Price must be above 200-day MA."""
        ma_period = PROTECTION_CONFIG["trend_ma_period"]

        if idx < ma_period:
            return False, 0, 0

        ma_200 = df["close"].iloc[idx-ma_period+1:idx+1].mean()
        price = df["close"].iloc[idx]

        return price > ma_200, price, ma_200

    @staticmethod
    def check_volume(df: pd.DataFrame, idx: int) -> tuple:
        """Filter 2: Volume - Must be >= 80% of 20-day average."""
        ma_period = PROTECTION_CONFIG["volume_ma_period"]
        min_ratio = PROTECTION_CONFIG["volume_min_ratio"]

        if idx < ma_period:
            return False, 0

        vol_ma = df["volume"].iloc[idx-ma_period:idx].mean()
        current_vol = df["volume"].iloc[idx]

        ratio = current_vol / vol_ma if vol_ma > 0 else 0
        return ratio >= min_ratio, ratio

    @staticmethod
    def check_first_touch(df: pd.DataFrame, idx: int) -> tuple:
        """Filter 3: First-Touch - Skip first touch after crash."""
        lookback = PROTECTION_CONFIG["crash_lookback_days"]
        threshold = PROTECTION_CONFIG["crash_threshold_pct"]
        cooldown = PROTECTION_CONFIG["first_touch_cooldown"]

        if idx < lookback:
            return True, 999

        lookback_start = max(0, idx - lookback)
        lookback_slice = df["close"].iloc[lookback_start:idx]
        high_in_lookback = lookback_slice.max()
        current_price = df["close"].iloc[idx]

        drop_pct = ((current_price - high_in_lookback) / high_in_lookback) * 100

        if drop_pct <= threshold:
            # Use argmax() for position-based index (not DataFrame index)
            high_pos_in_slice = lookback_slice.values.argmax()
            # Days since high = distance from high position to current position
            days_since = (idx - lookback_start - 1) - high_pos_in_slice

            if days_since < cooldown:
                return False, max(0, days_since)

        return True, 999

    @staticmethod
    def check_penetration(price: float, lower_band: float) -> tuple:
        """Filter 4: Penetration - Price must go 1%+ below lower band."""
        min_pct = PROTECTION_CONFIG["penetration_min_pct"]

        if lower_band <= 0:
            return False, 0

        penetration = ((lower_band - price) / lower_band) * 100
        return penetration >= min_pct, penetration

    @staticmethod
    def check_all(df: pd.DataFrame, idx: int, lower_band: float) -> ProtectionStatus:
        """Check all 4 entry protection filters."""
        status = ProtectionStatus()

        status.trend_ok, status.price, status.ma_200 = ProtectionFilters.check_trend(df, idx)
        status.volume_ok, status.volume_ratio = ProtectionFilters.check_volume(df, idx)
        status.not_first_touch, status.days_since_crash = ProtectionFilters.check_first_touch(df, idx)
        status.penetration_ok, status.penetration_pct = ProtectionFilters.check_penetration(
            df["close"].iloc[idx], lower_band
        )

        return status


# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_coin(symbol: str, config: dict) -> Optional[CoinStatus]:
    """Analyze a coin with protection filters."""
    df = fetch_daily_data(symbol, days=250)

    if df.empty or len(df) < PROTECTION_CONFIG["trend_ma_period"] + 5:
        return None

    # Calculate Bollinger Bands
    upper, middle, lower = TechnicalIndicators.bollinger_bands(
        df["close"],
        period=config["period"],
        std_dev=config["std_dev"]
    )

    idx = len(df) - 1
    current_price = df["close"].iloc[idx]
    current_lower = lower.iloc[idx]
    current_middle = middle.iloc[idx]
    current_upper = upper.iloc[idx]

    # Calculate distance to buy zone
    distance = ((current_price - current_lower) / current_lower) * 100

    # Check protection filters
    protection = ProtectionFilters.check_all(df, idx, current_lower)

    # Determine signal with protections
    if current_price <= current_lower:
        if protection.all_passed:
            signal = "STRONG_BUY"
        else:
            signal = "WEAK_BUY"
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
        protection=protection,
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
        self.sent_alerts: set = set()
        self._load_alerts()

    def _load_alerts(self):
        if ALERT_LOG_PATH.exists():
            try:
                with open(ALERT_LOG_PATH, "r") as f:
                    data = json.load(f)
                    for a in data.get("alerts", []):
                        self.sent_alerts.add(a.get("alert_id", ""))
            except Exception:
                pass

    def _save_alerts(self):
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
                    "protection_status": a.protection_status,
                    "timestamp": a.timestamp.isoformat(),
                    "acknowledged": a.acknowledged,
                }
                for a in self.alerts[-100:]
            ]
        }

        with open(ALERT_LOG_PATH, "w") as f:
            json.dump(data, f, indent=2)

    def create_alert(self, status: CoinStatus) -> Optional[Alert]:
        """Create an alert based on coin status."""
        today = datetime.now().strftime("%Y-%m-%d")
        alert_id = f"{status.symbol}_{status.signal}_{today}"

        if alert_id in self.sent_alerts:
            return None

        p = status.protection
        protection_str = f"{p.passed_count}/4 filters passed"

        if status.signal == "STRONG_BUY":
            alert_type = "STRONG_BUY"
            message = (
                f"STRONG BUY: {status.name} at ${status.current_price:.4f}\n"
                f"ALL 4 PROTECTIONS PASSED!\n\n"
                f"  Trend (>200MA):  PASS\n"
                f"  Volume (>=80%):  PASS ({p.volume_ratio:.0%})\n"
                f"  First-Touch:     PASS\n"
                f"  Penetration:     PASS ({p.penetration_pct:.1f}%)\n\n"
                f"Target: ${status.middle_band:.4f} (+{((status.middle_band/status.current_price)-1)*100:.1f}%)\n"
                f"Expected profit: +{status.config['expected_profit']:.0f}%"
            )

        elif status.signal == "WEAK_BUY":
            alert_type = "WEAK_BUY"
            failed = status.protection.get_failed_filters()
            message = (
                f"WEAK BUY: {status.name} at ${status.current_price:.4f}\n"
                f"CAUTION: Only {p.passed_count}/4 protections passed!\n"
                f"Failed: {', '.join(failed)}\n\n"
                f"  Trend (>200MA):  {'PASS' if p.trend_ok else 'FAIL'}\n"
                f"  Volume (>=80%):  {'PASS' if p.volume_ok else 'FAIL'} ({p.volume_ratio:.0%})\n"
                f"  First-Touch:     {'PASS' if p.not_first_touch else 'FAIL'}\n"
                f"  Penetration:     {'PASS' if p.penetration_ok else 'FAIL'} ({p.penetration_pct:.1f}%)\n\n"
                f"RISKY ENTRY - Consider waiting!"
            )

        elif status.distance_to_buy <= 3.0 and status.signal == "HOLD":
            alert_type = "NEAR_BUY"
            message = (
                f"NEAR BUY: {status.name} at ${status.current_price:.4f}\n"
                f"Only {status.distance_to_buy:.1f}% above lower band\n"
                f"Lower band: ${status.lower_band:.4f}\n"
                f"Watch for entry opportunity!"
            )

        elif status.signal == "SELL":
            alert_type = "SELL_SIGNAL"
            message = (
                f"SELL SIGNAL: {status.name} at ${status.current_price:.4f}\n"
                f"Price at middle band - consider taking profits"
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
            protection_status=protection_str,
            timestamp=datetime.now(),
        )

        self.alerts.append(alert)
        self.sent_alerts.add(alert_id)
        self._save_alerts()

        return alert

    def send_console_alert(self, alert: Alert):
        border = "=" * 60
        print(f"\n{border}")
        print(f"{'ALERT - ' + alert.alert_type:^60}")
        print(border)
        print(f"\n{alert.message}\n")
        print(f"Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(border + "\n")

    def send_webhook_alert(self, alert: Alert):
        if not WEBHOOK_URL:
            return

        try:
            payload = {
                "text": alert.message,
                "alert_type": alert.alert_type,
                "symbol": alert.symbol,
                "price": alert.price,
                "protection_status": alert.protection_status,
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
        self.send_console_alert(alert)
        self.send_webhook_alert(alert)


# ============================================================================
# MONITORING
# ============================================================================

def print_status_table(statuses: List[CoinStatus]):
    """Print status table with protection info."""
    print("\n" + "=" * 95)
    print(f"{'PROTECTED BOLLINGER MONITOR':^95}")
    print(f"{'Updated: ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'):^95}")
    print("=" * 95)

    print(f"\n{'Coin':<6} {'Price':<11} {'Lower':<11} {'Middle':<11} {'Dist':<9} {'Signal':<12} {'Protections':<15}")
    print("-" * 95)

    for s in statuses:
        p = s.protection
        prot_str = f"{p.passed_count}/4"
        if p.all_passed:
            prot_str += " ALL OK"
        else:
            prot_str += " " + ",".join([f[0] for f in p.get_failed_filters()])

        signal_display = s.signal
        if s.signal == "STRONG_BUY":
            signal_display = "STRONG BUY"
        elif s.signal == "WEAK_BUY":
            signal_display = "WEAK BUY"

        print(f"{s.name:<6} ${s.current_price:<9.4f} ${s.lower_band:<9.4f} "
              f"${s.middle_band:<9.4f} {s.distance_to_buy:>+7.1f}%  {signal_display:<12} {prot_str:<15}")

    print("-" * 95)


def print_protection_details(statuses: List[CoinStatus]):
    """Print detailed protection status."""
    print("\n" + "=" * 95)
    print("PROTECTION FILTER DETAILS")
    print("=" * 95)

    for s in statuses:
        p = s.protection
        print(f"\n  {s.name}:")
        print(f"    1. Trend (Price > 200MA):    {'PASS' if p.trend_ok else 'FAIL':8} (${p.price:.2f} vs MA ${p.ma_200:.2f})")
        print(f"    2. Volume (>= 80% avg):      {'PASS' if p.volume_ok else 'FAIL':8} ({p.volume_ratio:.0%} of average)")
        print(f"    3. First-Touch (cooldown):   {'PASS' if p.not_first_touch else 'FAIL':8} ({p.days_since_crash} days since crash)")
        print(f"    4. Penetration (>= 1%):      {'PASS' if p.penetration_ok else 'FAIL':8} ({p.penetration_pct:.1f}% below band)")


def print_recommendations(statuses: List[CoinStatus]):
    """Print trading recommendations."""
    strong_buy = [s for s in statuses if s.signal == "STRONG_BUY"]
    weak_buy = [s for s in statuses if s.signal == "WEAK_BUY"]
    near_buy = [s for s in statuses if s.signal == "HOLD" and s.distance_to_buy <= 3.0]
    sell_signals = [s for s in statuses if s.signal == "SELL"]

    print("\n" + "=" * 95)
    print("RECOMMENDATIONS")
    print("=" * 95)

    if strong_buy:
        print("\n  STRONG BUY (All protections passed):")
        for s in strong_buy:
            print(f"    {s.name}: ${s.current_price:.4f} -> Target ${s.middle_band:.4f} (+{((s.middle_band/s.current_price)-1)*100:.1f}%)")

    if weak_buy:
        print("\n  WEAK BUY (Some protections failed - CAUTION):")
        for s in weak_buy:
            failed = s.protection.get_failed_filters()
            print(f"    {s.name}: ${s.current_price:.4f} - Failed: {', '.join(failed)}")

    if near_buy:
        print("\n  WATCH CLOSELY (Near buy zone):")
        for s in near_buy:
            print(f"    {s.name}: {s.distance_to_buy:+.1f}% from buy zone")

    if sell_signals:
        print("\n  CONSIDER SELLING:")
        for s in sell_signals:
            print(f"    {s.name}: Price at middle band")

    if not strong_buy and not weak_buy and not near_buy and not sell_signals:
        print("\n  No action needed.")
        closest = min(statuses, key=lambda x: x.distance_to_buy) if statuses else None
        if closest:
            print(f"  Closest to buy: {closest.name} ({closest.distance_to_buy:+.1f}% from lower band)")

    print("\n" + "=" * 95)


def run_monitor(continuous: bool = False, interval_seconds: int = CHECK_INTERVAL_SECONDS):
    """Run the monitoring system."""
    print("=" * 95)
    print("PROTECTED BOLLINGER ALERT SYSTEM")
    print("=" * 95)

    print(f"\nExchange: BYBIT (with fallback to MEXC, Binance, KuCoin)")
    print(f"Monitoring {len(COIN_CONFIG)} coins with 5 protection filters:")
    for symbol, config in COIN_CONFIG.items():
        print(f"  - {config['name']}: Bollinger({config['period']}, {config['std_dev']})")

    print(f"""
Strategy: Buy at lower band (with protections), sell at middle band

PROTECTION FILTERS:
  1. TREND:       Price must be > 200-day MA
  2. VOLUME:      Volume must be >= 80% of 20-day average
  3. FIRST-TOUCH: Skip first touch after >15% crash
  4. PENETRATION: Price must go 1%+ below lower band
  5. TIME-EXIT:   Exit after 22 days if no bounce
""")

    if continuous:
        print(f"Mode: Continuous (checking every {interval_seconds} seconds)")
    else:
        print(f"Mode: Single check")

    alert_manager = AlertManager()

    while True:
        try:
            statuses = []
            print(f"\n  Fetching data...", end=" ")

            for symbol, config in COIN_CONFIG.items():
                status = analyze_coin(symbol, config)
                if status:
                    statuses.append(status)

            print(f"Done ({len(statuses)}/{len(COIN_CONFIG)} coins)")

            if not statuses:
                print("  Warning: Could not fetch data!")
                if continuous:
                    time.sleep(60)
                    continue
                else:
                    break

            print_status_table(statuses)
            print_protection_details(statuses)

            for status in statuses:
                alert = alert_manager.create_alert(status)
                if alert:
                    alert_manager.process_alert(alert)

            print_recommendations(statuses)

            if not continuous:
                break

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


def main():
    """Main entry point."""
    print("\n" + "=" * 60)
    print("PROTECTED BOLLINGER ALERT SYSTEM")
    print("=" * 60)

    print("""
  Options:
  --------
  1. Single check (check now and exit)
  2. Continuous monitoring (check every hour)
  3. Quick monitor (check every 5 minutes)

  Protection Filters Active:
  --------------------------
  1. Trend Filter       - Price > 200-day MA
  2. Volume Filter      - Volume >= 80% average
  3. First-Touch        - Skip after crash
  4. Penetration        - 1%+ below band
  5. Time Exit          - Max 22 days hold
""")

    try:
        choice = input("Select option (1-3) [1]: ").strip() or "1"
    except:
        choice = "1"

    if choice == "1":
        run_monitor(continuous=False)
    elif choice == "2":
        run_monitor(continuous=True, interval_seconds=CHECK_INTERVAL_SECONDS)
    elif choice == "3":
        run_monitor(continuous=True, interval_seconds=300)
    else:
        run_monitor(continuous=False)


if __name__ == "__main__":
    main()
