"""
Event Detector - Track Specific Price-Moving Events

Monitors and detects events known to influence coin prices:
- XRP: SEC lawsuit, Ripple partnerships, ODL volume
- SOL: Network upgrades, outages, NFT/DeFi launches
- ATOM: IBC launches, airdrops, staking changes

Usage:
    python examples/event_detector.py
    python examples/event_detector.py --coin XRP
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import re
import json


# =============================================================================
# EVENT PATTERNS FOR EACH COIN
# =============================================================================

XRP_EVENTS = {
    "legal_positive": {
        "keywords": ["ripple wins", "sec loses", "xrp not security", "ruling favor",
                    "dismiss", "settlement", "cleared", "victory"],
        "impact": "+20-50%",
        "typical_duration_days": 3,
        "weight": 5,
    },
    "legal_negative": {
        "keywords": ["sec charges", "lawsuit", "penalty", "violation", "fraud"],
        "impact": "-20-40%",
        "typical_duration_days": 2,
        "weight": -5,
    },
    "partnership": {
        "keywords": ["bank partner", "financial institution", "remittance",
                    "cross-border", "santander", "sbi", "moneygram"],
        "impact": "+10-25%",
        "typical_duration_days": 2,
        "weight": 3,
    },
    "odl_volume": {
        "keywords": ["odl", "on-demand liquidity", "volume record", "corridor"],
        "impact": "+5-15%",
        "typical_duration_days": 1,
        "weight": 2,
    },
    "listing": {
        "keywords": ["listing", "coinbase", "kraken", "bitstamp", "available"],
        "impact": "+10-30%",
        "typical_duration_days": 1,
        "weight": 3,
    },
}

SOL_EVENTS = {
    "network_upgrade": {
        "keywords": ["upgrade", "mainnet", "release", "v1.", "v2.", "improvement"],
        "impact": "+10-25%",
        "typical_duration_days": 2,
        "weight": 3,
    },
    "network_outage": {
        "keywords": ["outage", "down", "offline", "congestion", "degraded", "halt"],
        "impact": "-10-30%",
        "typical_duration_days": 1,
        "weight": -4,
    },
    "nft_defi": {
        "keywords": ["nft launch", "defi", "magic eden", "raydium", "marinade",
                    "jupiter", "tvl", "volume record"],
        "impact": "+5-15%",
        "typical_duration_days": 1,
        "weight": 2,
    },
    "vc_investment": {
        "keywords": ["funding", "raised", "investment", "a]6z", "polychain",
                    "alameda", "multicoin"],
        "impact": "+10-20%",
        "typical_duration_days": 2,
        "weight": 3,
    },
    "eth_comparison": {
        "keywords": ["ethereum killer", "faster than eth", "cheaper than eth",
                    "visa transactions"],
        "impact": "+5-10%",
        "typical_duration_days": 1,
        "weight": 1,
    },
}

ATOM_EVENTS = {
    "ibc_launch": {
        "keywords": ["ibc", "inter-blockchain", "new chain", "connected",
                    "cosmos hub", "zone launch"],
        "impact": "+10-20%",
        "typical_duration_days": 2,
        "weight": 3,
    },
    "airdrop": {
        "keywords": ["airdrop", "staker reward", "snapshot", "claim", "distribution"],
        "impact": "+15-30%",
        "typical_duration_days": 3,
        "weight": 4,
    },
    "staking": {
        "keywords": ["staking", "validator", "apr", "yield", "delegation"],
        "impact": "+5-10%",
        "typical_duration_days": 1,
        "weight": 2,
    },
    "osmosis": {
        "keywords": ["osmosis", "dex", "liquidity pool", "osmo"],
        "impact": "+5-15%",
        "typical_duration_days": 1,
        "weight": 2,
    },
    "upgrade": {
        "keywords": ["upgrade", "gaia", "vega", "theta", "rho", "proposal passed"],
        "impact": "+5-15%",
        "typical_duration_days": 2,
        "weight": 2,
    },
}

COIN_EVENTS = {
    "XRP": XRP_EVENTS,
    "SOL": SOL_EVENTS,
    "ATOM": ATOM_EVENTS,
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class DetectedEvent:
    """A detected event."""
    coin: str
    event_type: str
    headline: str
    source: str
    timestamp: datetime
    impact_estimate: str
    weight: int
    url: str = ""

    def to_dict(self) -> Dict:
        return {
            "coin": self.coin,
            "event_type": self.event_type,
            "headline": self.headline,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "impact": self.impact_estimate,
            "weight": self.weight,
            "url": self.url,
        }


@dataclass
class EventSummary:
    """Summary of events for a coin."""
    coin: str
    events: List[DetectedEvent] = field(default_factory=list)
    net_weight: int = 0
    dominant_event: Optional[str] = None
    recommendation: str = "HOLD"

    def calculate(self):
        """Calculate summary metrics."""
        if not self.events:
            return

        self.net_weight = sum(e.weight for e in self.events)

        # Find dominant event
        if self.events:
            sorted_events = sorted(self.events, key=lambda x: abs(x.weight), reverse=True)
            self.dominant_event = sorted_events[0].event_type

        # Recommendation
        if self.net_weight >= 5:
            self.recommendation = "STRONG BUY"
        elif self.net_weight >= 3:
            self.recommendation = "BUY"
        elif self.net_weight <= -3:
            self.recommendation = "AVOID"
        else:
            self.recommendation = "HOLD"


# =============================================================================
# NEWS SOURCES
# =============================================================================

class NewsFetcher:
    """Fetch news from multiple sources."""

    def __init__(self):
        self.sources = {
            "cryptopanic": self._fetch_cryptopanic,
            "google_news": self._fetch_google_news,
        }

    def fetch_all(self, coin: str, days: int = 7) -> List[Dict]:
        """Fetch from all sources."""
        all_news = []

        for source, fetcher in self.sources.items():
            try:
                news = fetcher(coin, days)
                for item in news:
                    item["source"] = source
                all_news.extend(news)
            except Exception as e:
                print(f"  Warning: {source} fetch failed: {e}")

        return all_news

    def _fetch_cryptopanic(self, coin: str, days: int = 7) -> List[Dict]:
        """Fetch from CryptoPanic."""
        try:
            params = {
                "auth_token": "demo",
                "currencies": coin.lower(),
                "public": "true",
                "kind": "news",
            }

            response = requests.get(
                "https://cryptopanic.com/api/v1/posts/",
                params=params,
                timeout=10
            )

            if response.status_code == 200:
                results = response.json().get("results", [])
                return [{
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "published": r.get("published_at", ""),
                } for r in results]
        except:
            pass
        return []

    def _fetch_google_news(self, coin: str, days: int = 7) -> List[Dict]:
        """Fetch from Google News RSS (simplified)."""
        # Note: For production, use proper news API
        # This is a simplified version
        coin_names = {
            "XRP": "ripple+xrp",
            "SOL": "solana+sol",
            "ATOM": "cosmos+atom",
        }

        query = coin_names.get(coin, coin.lower())

        try:
            # Using Google News RSS
            url = f"https://news.google.com/rss/search?q={query}+crypto&hl=en-US&gl=US&ceid=US:en"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                # Parse RSS (simplified - would use feedparser in production)
                import xml.etree.ElementTree as ET
                root = ET.fromstring(response.content)

                items = []
                for item in root.findall(".//item")[:20]:
                    title = item.find("title")
                    link = item.find("link")
                    pub_date = item.find("pubDate")

                    if title is not None:
                        items.append({
                            "title": title.text or "",
                            "url": link.text if link is not None else "",
                            "published": pub_date.text if pub_date is not None else "",
                        })
                return items
        except:
            pass
        return []


# =============================================================================
# EVENT DETECTOR
# =============================================================================

class EventDetector:
    """Detect price-moving events from news."""

    def __init__(self):
        self.news_fetcher = NewsFetcher()

    def detect_events(self, coin: str, days: int = 7) -> List[DetectedEvent]:
        """Detect events for a coin."""
        events_config = COIN_EVENTS.get(coin, {})
        if not events_config:
            return []

        # Fetch news
        news_items = self.news_fetcher.fetch_all(coin, days)

        detected = []
        seen_headlines = set()  # Dedupe

        for item in news_items:
            title = item.get("title", "").lower()

            if title in seen_headlines:
                continue
            seen_headlines.add(title)

            # Check against event patterns
            for event_type, config in events_config.items():
                keywords = config.get("keywords", [])

                for keyword in keywords:
                    if keyword.lower() in title:
                        # Parse timestamp
                        try:
                            pub = item.get("published", "")
                            if pub:
                                timestamp = datetime.fromisoformat(pub.replace("Z", "+00:00"))
                            else:
                                timestamp = datetime.now()
                        except:
                            timestamp = datetime.now()

                        detected.append(DetectedEvent(
                            coin=coin,
                            event_type=event_type,
                            headline=item.get("title", ""),
                            source=item.get("source", "unknown"),
                            timestamp=timestamp,
                            impact_estimate=config.get("impact", "unknown"),
                            weight=config.get("weight", 0),
                            url=item.get("url", ""),
                        ))
                        break  # Only one event type per headline

        return detected

    def get_summary(self, coin: str, days: int = 7) -> EventSummary:
        """Get event summary for a coin."""
        events = self.detect_events(coin, days)
        summary = EventSummary(coin=coin, events=events)
        summary.calculate()
        return summary

    def analyze_all_coins(self, days: int = 7) -> Dict[str, EventSummary]:
        """Analyze all coins."""
        results = {}
        for coin in COIN_EVENTS.keys():
            print(f"  Scanning {coin} events...")
            results[coin] = self.get_summary(coin, days)
        return results


# =============================================================================
# HISTORICAL EVENT ANALYZER
# =============================================================================

class HistoricalEventAnalyzer:
    """Analyze historical correlation between events and price moves."""

    # Known historical events with outcomes
    KNOWN_EVENTS = {
        "XRP": [
            {"date": "2023-07-13", "event": "SEC ruling - XRP not security", "impact_pct": 75},
            {"date": "2023-10-03", "event": "SEC appeal denied", "impact_pct": 12},
            {"date": "2024-08-07", "event": "SEC $125M settlement", "impact_pct": -15},
        ],
        "SOL": [
            {"date": "2024-03-18", "event": "Jupiter airdrop", "impact_pct": 25},
            {"date": "2024-02-06", "event": "Network outage", "impact_pct": -8},
            {"date": "2023-12-25", "event": "NFT volume record", "impact_pct": 15},
        ],
        "ATOM": [
            {"date": "2024-01-15", "event": "Celestia airdrop snapshot", "impact_pct": 30},
            {"date": "2023-09-20", "event": "IBC v8 upgrade", "impact_pct": 12},
            {"date": "2024-02-01", "event": "dYdX migration", "impact_pct": 8},
        ],
    }

    def get_event_patterns(self, coin: str) -> Dict:
        """Get event patterns for a coin."""
        events = self.KNOWN_EVENTS.get(coin, [])

        if not events:
            return {}

        # Categorize
        positive = [e for e in events if e["impact_pct"] > 0]
        negative = [e for e in events if e["impact_pct"] < 0]

        return {
            "coin": coin,
            "total_events": len(events),
            "positive_events": len(positive),
            "negative_events": len(negative),
            "avg_positive_impact": np.mean([e["impact_pct"] for e in positive]) if positive else 0,
            "avg_negative_impact": np.mean([e["impact_pct"] for e in negative]) if negative else 0,
            "events": events,
        }

    def print_patterns(self):
        """Print historical event patterns."""
        print("\n" + "=" * 70)
        print("HISTORICAL EVENT ANALYSIS")
        print("=" * 70)

        for coin in COIN_EVENTS.keys():
            patterns = self.get_event_patterns(coin)
            if not patterns:
                continue

            print(f"\n{coin}:")
            print(f"  Total Known Events: {patterns['total_events']}")
            print(f"  Positive: {patterns['positive_events']} (avg +{patterns['avg_positive_impact']:.1f}%)")
            print(f"  Negative: {patterns['negative_events']} (avg {patterns['avg_negative_impact']:.1f}%)")
            print(f"\n  Key Events:")
            for e in patterns['events'][:5]:
                print(f"    {e['date']}: {e['event']} ({e['impact_pct']:+.0f}%)")


# =============================================================================
# EVENT ALERT SYSTEM
# =============================================================================

class EventAlertSystem:
    """Alert system for detected events."""

    def __init__(self, telegram_token: str = None, chat_id: str = None):
        self.telegram_token = telegram_token
        self.chat_id = chat_id

    def format_alert(self, summary: EventSummary) -> str:
        """Format event summary as alert."""
        lines = [
            f"🔔 EVENT ALERT: {summary.coin}",
            f"Recommendation: {summary.recommendation}",
            f"Net Weight: {summary.net_weight}",
            "",
        ]

        if summary.dominant_event:
            lines.append(f"Dominant Event: {summary.dominant_event}")

        if summary.events:
            lines.append("\nRecent Events:")
            for e in summary.events[:5]:
                emoji = "🟢" if e.weight > 0 else "🔴" if e.weight < 0 else "⚪"
                lines.append(f"{emoji} {e.event_type}: {e.headline[:60]}...")

        return "\n".join(lines)

    def send_telegram(self, message: str) -> bool:
        """Send Telegram alert."""
        if not self.telegram_token or not self.chat_id:
            return False

        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "HTML",
            }
            response = requests.post(url, json=data, timeout=10)
            return response.status_code == 200
        except:
            return False

    def check_and_alert(self, summaries: Dict[str, EventSummary]):
        """Check summaries and send alerts for significant events."""
        for coin, summary in summaries.items():
            if abs(summary.net_weight) >= 3:
                alert = self.format_alert(summary)
                print(f"\n{alert}")

                if self.telegram_token:
                    self.send_telegram(alert)


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Event Detector")
    parser.add_argument("--coin", type=str, help="Single coin to analyze")
    parser.add_argument("--days", type=int, default=7, help="Days to look back")
    parser.add_argument("--historical", action="store_true", help="Show historical patterns")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    print("=" * 70)
    print("EVENT DETECTOR")
    print("Scanning for price-moving events...")
    print("=" * 70)

    detector = EventDetector()

    if args.historical:
        analyzer = HistoricalEventAnalyzer()
        analyzer.print_patterns()
        return

    if args.coin:
        summaries = {args.coin.upper(): detector.get_summary(args.coin.upper(), args.days)}
    else:
        summaries = detector.analyze_all_coins(args.days)

    if args.json:
        output = {}
        for coin, summary in summaries.items():
            output[coin] = {
                "events": [e.to_dict() for e in summary.events],
                "net_weight": summary.net_weight,
                "recommendation": summary.recommendation,
            }
        print(json.dumps(output, indent=2))
        return

    # Print results
    for coin, summary in summaries.items():
        print(f"\n{'='*40}")
        print(f"{coin}")
        print(f"{'='*40}")
        print(f"Net Weight: {summary.net_weight}")
        print(f"Recommendation: {summary.recommendation}")

        if summary.events:
            print(f"\nDetected Events ({len(summary.events)}):")
            for e in summary.events[:10]:
                emoji = "🟢" if e.weight > 0 else "🔴" if e.weight < 0 else "⚪"
                print(f"  {emoji} [{e.event_type}] {e.headline[:60]}...")
                print(f"     Impact: {e.impact_estimate} | Weight: {e.weight}")
        else:
            print("\nNo significant events detected")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for coin, summary in summaries.items():
        emoji = "🟢" if summary.recommendation in ["BUY", "STRONG BUY"] else "🔴" if summary.recommendation == "AVOID" else "⚪"
        print(f"{emoji} {coin}: {summary.recommendation} (weight: {summary.net_weight})")

    # Known event patterns
    print("\n" + "-" * 70)
    print("KNOWN EVENT TRIGGERS:")
    print("-" * 70)
    print("""
XRP:
  🟢 Legal wins (+20-50%)
  🟢 Bank partnerships (+10-25%)
  🔴 SEC actions (-20-40%)

SOL:
  🟢 Network upgrades (+10-25%)
  🟢 NFT/DeFi launches (+5-15%)
  🔴 Outages (-10-30%)

ATOM:
  🟢 Airdrops announced (+15-30%)
  🟢 IBC chain launches (+10-20%)
  🟢 Staking increases (+5-10%)
""")


if __name__ == "__main__":
    main()
