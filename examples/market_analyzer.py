"""
Market Analyzer - News, Sentiment & Technical Signal Fusion

Analyzes XRP, SOL, ATOM for:
1. Technical indicators (BB, RSI, Volume)
2. News/Event detection
3. On-chain signals (whale alerts, exchange flows)
4. Scoring engine for buy signals
5. Backtesting validation

Usage:
    python examples/market_analyzer.py
    python examples/market_analyzer.py --coin XRP --days 365
    python examples/market_analyzer.py --backtest
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import json
import time


# =============================================================================
# CONFIGURATION
# =============================================================================

COINS = {
    "XRP": {
        "symbol": "XRPUSDT",
        "keywords": ["ripple", "xrp", "sec", "lawsuit", "garlinghouse", "odl", "xrpl"],
        "negative_keywords": ["hack", "crash", "delisting", "sec charges"],
        "whale_threshold_usd": 500_000,
    },
    "SOL": {
        "symbol": "SOLUSDT",
        "keywords": ["solana", "sol", "nft", "defi", "upgrade", "outage", "phantom"],
        "negative_keywords": ["outage", "down", "hack", "exploit", "congestion"],
        "whale_threshold_usd": 1_000_000,
    },
    "ATOM": {
        "symbol": "ATOMUSDT",
        "keywords": ["cosmos", "atom", "ibc", "airdrop", "tendermint", "osmosis"],
        "negative_keywords": ["hack", "exploit", "vulnerability"],
        "whale_threshold_usd": 300_000,
    },
}

# Signal weights
WEIGHTS = {
    "fundamental": 3,  # News/events
    "onchain": 2,      # Whale/exchange flows
    "technical": 1,    # BB, RSI, volume
}

# Scoring threshold for buy signal
BUY_THRESHOLD = 4


# =============================================================================
# DATA CLASSES
# =============================================================================

class SignalStrength(Enum):
    STRONG = 3
    MEDIUM = 2
    WEAK = 1
    NONE = 0
    NEGATIVE = -1


@dataclass
class TechnicalSignal:
    """Technical indicator signals."""
    bb_touch: bool = False
    bb_penetration_pct: float = 0.0
    rsi_oversold: bool = False
    rsi_value: float = 50.0
    volume_spike: bool = False
    volume_ratio: float = 1.0
    above_200ma: bool = False
    price: float = 0.0
    atr: float = 0.0
    atr_pct: float = 0.0

    @property
    def score(self) -> int:
        """Calculate technical score."""
        s = 0
        if self.bb_touch:
            s += 1
        if self.bb_penetration_pct >= 1.0:
            s += 1
        if self.rsi_oversold:
            s += 1
        if self.volume_spike:
            s += 1
        return s


@dataclass
class NewsSignal:
    """News/event signals."""
    positive_news: List[str] = field(default_factory=list)
    negative_news: List[str] = field(default_factory=list)
    news_score: int = 0
    latest_event: str = ""
    event_date: Optional[datetime] = None

    @property
    def score(self) -> int:
        """Calculate news score."""
        return max(0, len(self.positive_news) - len(self.negative_news) * 2)


@dataclass
class OnChainSignal:
    """On-chain signals."""
    whale_accumulation: bool = False
    whale_count: int = 0
    exchange_outflow: bool = False
    exchange_flow_pct: float = 0.0
    large_transfers: List[Dict] = field(default_factory=list)

    @property
    def score(self) -> int:
        """Calculate on-chain score."""
        s = 0
        if self.whale_accumulation:
            s += 2
        if self.exchange_outflow:
            s += 1
        return s


@dataclass
class MarketSignal:
    """Combined market signal."""
    coin: str
    timestamp: datetime
    technical: TechnicalSignal
    news: NewsSignal
    onchain: OnChainSignal
    btc_filter_ok: bool = True

    @property
    def total_score(self) -> int:
        """Calculate weighted total score."""
        tech_score = self.technical.score * WEIGHTS["technical"]
        news_score = self.news.score * WEIGHTS["fundamental"]
        chain_score = self.onchain.score * WEIGHTS["onchain"]
        return tech_score + news_score + chain_score

    @property
    def is_buy_signal(self) -> bool:
        """Check if meets buy threshold."""
        return self.total_score >= BUY_THRESHOLD and self.btc_filter_ok

    def to_dict(self) -> Dict:
        return {
            "coin": self.coin,
            "timestamp": self.timestamp.isoformat(),
            "price": self.technical.price,
            "total_score": self.total_score,
            "is_buy": self.is_buy_signal,
            "tech_score": self.technical.score,
            "news_score": self.news.score,
            "chain_score": self.onchain.score,
            "bb_touch": self.technical.bb_touch,
            "rsi": self.technical.rsi_value,
            "volume_spike": self.technical.volume_spike,
            "btc_ok": self.btc_filter_ok,
        }


# =============================================================================
# PRICE DATA FETCHER
# =============================================================================

class PriceDataFetcher:
    """Fetch OHLCV data from exchanges."""

    ENDPOINTS = [
        "https://api.bybit.com/v5/market/kline",
        "https://api.binance.com/api/v3/klines",
    ]

    def __init__(self):
        self.cache = {}

    def fetch_ohlcv(self, symbol: str, days: int = 365) -> pd.DataFrame:
        """Fetch daily OHLCV data."""
        cache_key = f"{symbol}_{days}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Try Bybit first
        df = self._fetch_bybit(symbol, days)
        if df.empty:
            df = self._fetch_binance(symbol, days)

        if not df.empty:
            self.cache[cache_key] = df

        return df

    def _fetch_bybit(self, symbol: str, days: int) -> pd.DataFrame:
        """Fetch from Bybit."""
        try:
            end_time = int(datetime.now().timestamp() * 1000)
            start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

            params = {
                "category": "spot",
                "symbol": symbol,
                "interval": "D",
                "start": start_time,
                "end": end_time,
                "limit": 1000,
            }

            response = requests.get(self.ENDPOINTS[0], params=params, timeout=15)
            data = response.json()

            if data.get("retCode") == 0 and data.get("result", {}).get("list"):
                rows = []
                for candle in data["result"]["list"]:
                    rows.append({
                        "timestamp": pd.to_datetime(int(candle[0]), unit="ms"),
                        "open": float(candle[1]),
                        "high": float(candle[2]),
                        "low": float(candle[3]),
                        "close": float(candle[4]),
                        "volume": float(candle[5]),
                    })

                df = pd.DataFrame(rows)
                df = df.sort_values("timestamp").reset_index(drop=True)
                return df
        except Exception as e:
            print(f"Bybit fetch error: {e}")

        return pd.DataFrame()

    def _fetch_binance(self, symbol: str, days: int) -> pd.DataFrame:
        """Fetch from Binance."""
        try:
            params = {
                "symbol": symbol,
                "interval": "1d",
                "startTime": int((datetime.now() - timedelta(days=days)).timestamp() * 1000),
                "endTime": int(datetime.now().timestamp() * 1000),
                "limit": 1000,
            }

            response = requests.get(self.ENDPOINTS[1], params=params, timeout=15)
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
        except Exception as e:
            print(f"Binance fetch error: {e}")

        return pd.DataFrame()


# =============================================================================
# TECHNICAL ANALYZER
# =============================================================================

class TechnicalAnalyzer:
    """Calculate technical indicators."""

    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        return series.rolling(window=period, min_periods=1).mean()

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False, min_periods=1).mean()

    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.ewm(span=period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(span=period, adjust=False, min_periods=period).mean()

        rs = avg_gain / avg_loss.replace(0, np.inf)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def bollinger_bands(series: pd.Series, period: int = 20, std: float = 1.5):
        middle = series.rolling(window=period, min_periods=1).mean()
        std_dev = series.rolling(window=period, min_periods=1).std()
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        return upper, middle, lower

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        ], axis=1).max(axis=1)
        return tr.ewm(span=period, adjust=False, min_periods=period).mean()

    def analyze(self, df: pd.DataFrame, bb_period: int = 20, bb_std: float = 1.5) -> TechnicalSignal:
        """Generate technical signal from OHLCV data."""
        if df.empty or len(df) < 200:
            return TechnicalSignal()

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # Current values
        idx = len(df) - 1
        price = close.iloc[idx]

        # Bollinger Bands
        upper, middle, lower = self.bollinger_bands(close, bb_period, bb_std)
        bb_lower = lower.iloc[idx]
        bb_middle = middle.iloc[idx]

        bb_touch = price <= bb_lower
        bb_penetration = ((bb_lower - price) / bb_lower * 100) if bb_lower > 0 else 0

        # RSI
        rsi_values = self.rsi(close)
        rsi_val = rsi_values.iloc[idx]
        rsi_oversold = rsi_val < 35

        # Volume
        vol_ma = volume.rolling(20).mean()
        vol_ratio = volume.iloc[idx] / vol_ma.iloc[idx] if vol_ma.iloc[idx] > 0 else 1
        volume_spike = vol_ratio >= 2.0

        # 200 MA filter
        ma_200 = self.sma(close, 200)
        above_200ma = price > ma_200.iloc[idx]

        # ATR
        atr_values = self.atr(high, low, close)
        atr_val = atr_values.iloc[idx]
        atr_pct = (atr_val / price * 100) if price > 0 else 0

        return TechnicalSignal(
            bb_touch=bb_touch,
            bb_penetration_pct=max(0, bb_penetration),
            rsi_oversold=rsi_oversold,
            rsi_value=rsi_val,
            volume_spike=volume_spike,
            volume_ratio=vol_ratio,
            above_200ma=above_200ma,
            price=price,
            atr=atr_val,
            atr_pct=atr_pct,
        )


# =============================================================================
# NEWS ANALYZER
# =============================================================================

class NewsAnalyzer:
    """Fetch and analyze crypto news."""

    CRYPTOPANIC_URL = "https://cryptopanic.com/api/v1/posts/"

    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.cache = {}

    def fetch_news(self, coin: str, days: int = 7) -> List[Dict]:
        """Fetch news from CryptoPanic (free tier)."""
        # For demo, use free public feed
        try:
            params = {
                "auth_token": self.api_key or "demo",
                "currencies": coin.lower(),
                "kind": "news",
                "public": "true",
            }

            response = requests.get(
                self.CRYPTOPANIC_URL,
                params=params,
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("results", [])
        except Exception as e:
            print(f"News fetch error: {e}")

        return []

    def analyze(self, coin: str, config: Dict) -> NewsSignal:
        """Analyze news for a coin."""
        news = self.fetch_news(coin)

        positive = []
        negative = []

        keywords = config.get("keywords", [])
        neg_keywords = config.get("negative_keywords", [])

        for item in news[:20]:  # Last 20 news items
            title = item.get("title", "").lower()

            # Check for positive signals
            for kw in keywords:
                if kw.lower() in title:
                    # Check it's not negative
                    is_negative = any(nk.lower() in title for nk in neg_keywords)
                    if not is_negative:
                        positive.append(item.get("title", ""))
                        break

            # Check for negative signals
            for nk in neg_keywords:
                if nk.lower() in title:
                    negative.append(item.get("title", ""))
                    break

        latest = news[0].get("title", "") if news else ""
        event_date = None
        if news:
            try:
                event_date = datetime.fromisoformat(news[0].get("published_at", "").replace("Z", "+00:00"))
            except:
                pass

        return NewsSignal(
            positive_news=positive[:5],
            negative_news=negative[:5],
            news_score=len(positive) - len(negative) * 2,
            latest_event=latest,
            event_date=event_date,
        )


# =============================================================================
# ON-CHAIN ANALYZER (Simulated - would need real API keys)
# =============================================================================

class OnChainAnalyzer:
    """Analyze on-chain metrics."""

    WHALE_ALERT_URL = "https://api.whale-alert.io/v1/transactions"

    def __init__(self, api_key: str = None):
        self.api_key = api_key

    def analyze(self, coin: str, config: Dict) -> OnChainSignal:
        """Analyze on-chain signals for a coin."""
        # Note: In production, use Whale Alert API, Glassnode, etc.
        # For demo, we simulate based on technical conditions

        # Simulate whale detection (in real implementation, use API)
        signal = OnChainSignal()

        # If we had Whale Alert API:
        # transfers = self._fetch_whale_transfers(coin, days=7)
        # signal.whale_count = len([t for t in transfers if t["amount_usd"] > config["whale_threshold_usd"]])
        # signal.whale_accumulation = signal.whale_count >= 3

        # For now, return empty signal (user can add API key later)
        return signal

    def _fetch_whale_transfers(self, coin: str, days: int = 7) -> List[Dict]:
        """Fetch whale transfers (requires API key)."""
        if not self.api_key:
            return []

        try:
            params = {
                "api_key": self.api_key,
                "currency": coin.lower(),
                "min_value": 500000,
                "start": int((datetime.now() - timedelta(days=days)).timestamp()),
            }

            response = requests.get(self.WHALE_ALERT_URL, params=params, timeout=10)
            if response.status_code == 200:
                return response.json().get("transactions", [])
        except:
            pass

        return []


# =============================================================================
# MARKET ANALYZER (Main)
# =============================================================================

class MarketAnalyzer:
    """Main market analyzer combining all signals."""

    def __init__(self, cryptopanic_key: str = None, whale_alert_key: str = None):
        self.price_fetcher = PriceDataFetcher()
        self.tech_analyzer = TechnicalAnalyzer()
        self.news_analyzer = NewsAnalyzer(cryptopanic_key)
        self.chain_analyzer = OnChainAnalyzer(whale_alert_key)
        self.btc_above_200ma = True  # Global filter

    def _check_btc_filter(self) -> bool:
        """Check if BTC is above 200 MA."""
        try:
            df = self.price_fetcher.fetch_ohlcv("BTCUSDT", days=250)
            if not df.empty:
                ma_200 = df["close"].rolling(200).mean().iloc[-1]
                price = df["close"].iloc[-1]
                return price > ma_200
        except:
            pass
        return True  # Default to OK if can't fetch

    def analyze_coin(self, coin: str, days: int = 365) -> MarketSignal:
        """Analyze a single coin."""
        config = COINS.get(coin, {})
        symbol = config.get("symbol", f"{coin}USDT")

        # Fetch price data
        df = self.price_fetcher.fetch_ohlcv(symbol, days)

        # Technical analysis
        tech_signal = self.tech_analyzer.analyze(df)

        # News analysis
        news_signal = self.news_analyzer.analyze(coin, config)

        # On-chain analysis
        chain_signal = self.chain_analyzer.analyze(coin, config)

        # BTC filter
        btc_ok = self._check_btc_filter()

        return MarketSignal(
            coin=coin,
            timestamp=datetime.now(),
            technical=tech_signal,
            news=news_signal,
            onchain=chain_signal,
            btc_filter_ok=btc_ok,
        )

    def analyze_all(self) -> List[MarketSignal]:
        """Analyze all configured coins."""
        signals = []
        for coin in COINS:
            print(f"  Analyzing {coin}...")
            signal = self.analyze_coin(coin)
            signals.append(signal)
            time.sleep(0.5)  # Rate limiting
        return signals

    def generate_report(self, signals: List[MarketSignal]) -> str:
        """Generate text report."""
        lines = [
            "=" * 70,
            "MARKET ANALYSIS REPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 70,
            "",
        ]

        # Sort by score
        signals.sort(key=lambda x: x.total_score, reverse=True)

        for sig in signals:
            emoji = "🟢" if sig.is_buy_signal else "🟡" if sig.total_score >= 2 else "⚪"

            lines.append(f"{emoji} {sig.coin}")
            lines.append(f"   Price: ${sig.technical.price:.4f}")
            lines.append(f"   Total Score: {sig.total_score} {'(BUY SIGNAL!)' if sig.is_buy_signal else ''}")
            lines.append(f"   Technical: {sig.technical.score}/4")
            lines.append(f"      - BB Touch: {'✓' if sig.technical.bb_touch else '✗'}")
            lines.append(f"      - RSI ({sig.technical.rsi_value:.1f}): {'✓ Oversold' if sig.technical.rsi_oversold else '✗'}")
            lines.append(f"      - Volume Spike ({sig.technical.volume_ratio:.1f}x): {'✓' if sig.technical.volume_spike else '✗'}")
            lines.append(f"      - Above 200 MA: {'✓' if sig.technical.above_200ma else '✗'}")
            lines.append(f"   News Score: {sig.news.score}")
            if sig.news.positive_news:
                lines.append(f"      + {sig.news.positive_news[0][:50]}...")
            if sig.news.negative_news:
                lines.append(f"      - {sig.news.negative_news[0][:50]}...")
            lines.append(f"   On-Chain: {sig.onchain.score}")
            lines.append(f"   BTC Filter: {'✓' if sig.btc_filter_ok else '✗'}")
            lines.append("")

        # Trading signals
        buy_signals = [s for s in signals if s.is_buy_signal]
        if buy_signals:
            lines.append("=" * 70)
            lines.append("🚨 BUY SIGNALS DETECTED:")
            for sig in buy_signals:
                lines.append(f"   {sig.coin} @ ${sig.technical.price:.4f}")
                lines.append(f"      Stop Loss: ${sig.technical.price - sig.technical.atr * 2:.4f} (2x ATR)")
                lines.append(f"      Take Profit: ${sig.technical.price * 1.10:.4f} (+10%)")
            lines.append("=" * 70)

        return "\n".join(lines)


# =============================================================================
# BACKTESTER
# =============================================================================

class Backtester:
    """Backtest the scoring strategy."""

    def __init__(self):
        self.price_fetcher = PriceDataFetcher()
        self.tech_analyzer = TechnicalAnalyzer()

    def backtest_coin(self, coin: str, days: int = 365) -> Dict:
        """Backtest scoring strategy on historical data."""
        config = COINS.get(coin, {})
        symbol = config.get("symbol", f"{coin}USDT")

        df = self.price_fetcher.fetch_ohlcv(symbol, days)
        if df.empty or len(df) < 220:
            return {"error": "Insufficient data"}

        trades = []
        position = None

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # Pre-calculate indicators
        upper, middle, lower = self.tech_analyzer.bollinger_bands(close, 20, 1.5)
        rsi = self.tech_analyzer.rsi(close, 14)
        vol_ma = volume.rolling(20).mean()
        ma_200 = close.rolling(200).mean()
        atr = self.tech_analyzer.atr(high, low, close, 14)

        for i in range(220, len(df)):
            price = close.iloc[i]

            # Calculate score
            score = 0

            # BB touch
            if price <= lower.iloc[i]:
                score += 1
                # BB penetration
                penetration = (lower.iloc[i] - price) / lower.iloc[i] * 100
                if penetration >= 1:
                    score += 1

            # RSI oversold
            if rsi.iloc[i] < 35:
                score += 1

            # Volume spike
            if volume.iloc[i] >= vol_ma.iloc[i] * 2:
                score += 1

            # 200 MA filter
            above_ma = price > ma_200.iloc[i]

            # Entry
            if position is None and score >= 3 and above_ma:
                position = {
                    "entry_price": price,
                    "entry_date": df["timestamp"].iloc[i],
                    "entry_idx": i,
                    "score": score,
                    "stop_loss": price - atr.iloc[i] * 2,
                }

            # Exit
            elif position is not None:
                days_held = i - position["entry_idx"]

                # Take profit at middle band
                if price >= middle.iloc[i]:
                    pnl = (price - position["entry_price"]) / position["entry_price"] * 100
                    trades.append({
                        "entry": position["entry_price"],
                        "exit": price,
                        "pnl": pnl,
                        "days": days_held,
                        "reason": "TARGET",
                    })
                    position = None

                # Stop loss
                elif price <= position["stop_loss"]:
                    pnl = (price - position["entry_price"]) / position["entry_price"] * 100
                    trades.append({
                        "entry": position["entry_price"],
                        "exit": price,
                        "pnl": pnl,
                        "days": days_held,
                        "reason": "STOP",
                    })
                    position = None

                # Time exit (22 days)
                elif days_held >= 22:
                    pnl = (price - position["entry_price"]) / position["entry_price"] * 100
                    trades.append({
                        "entry": position["entry_price"],
                        "exit": price,
                        "pnl": pnl,
                        "days": days_held,
                        "reason": "TIME",
                    })
                    position = None

        # Calculate metrics
        if not trades:
            return {"trades": 0, "error": "No trades generated"}

        pnls = [t["pnl"] for t in trades]
        wins = len([p for p in pnls if p > 0])

        return {
            "coin": coin,
            "trades": len(trades),
            "wins": wins,
            "win_rate": wins / len(trades) * 100,
            "total_pnl": sum(pnls),
            "avg_pnl": np.mean(pnls),
            "max_pnl": max(pnls),
            "min_pnl": min(pnls),
            "avg_days": np.mean([t["days"] for t in trades]),
            "by_reason": {
                "TARGET": len([t for t in trades if t["reason"] == "TARGET"]),
                "STOP": len([t for t in trades if t["reason"] == "STOP"]),
                "TIME": len([t for t in trades if t["reason"] == "TIME"]),
            }
        }

    def backtest_all(self) -> Dict:
        """Backtest all coins."""
        results = {}
        for coin in COINS:
            print(f"  Backtesting {coin}...")
            results[coin] = self.backtest_coin(coin)
        return results

    def print_results(self, results: Dict):
        """Print backtest results."""
        print("\n" + "=" * 70)
        print("BACKTEST RESULTS (Scoring Strategy)")
        print("=" * 70)

        total_pnl = 0
        total_trades = 0

        for coin, r in results.items():
            if "error" in r:
                print(f"\n{coin}: {r['error']}")
                continue

            total_pnl += r["total_pnl"]
            total_trades += r["trades"]

            print(f"\n{coin}:")
            print(f"  Trades: {r['trades']}")
            print(f"  Win Rate: {r['win_rate']:.1f}%")
            print(f"  Total P&L: {r['total_pnl']:+.2f}%")
            print(f"  Avg P&L: {r['avg_pnl']:+.2f}%")
            print(f"  Best/Worst: {r['max_pnl']:+.1f}% / {r['min_pnl']:+.1f}%")
            print(f"  Avg Days: {r['avg_days']:.1f}")
            print(f"  Exits: Target={r['by_reason']['TARGET']}, Stop={r['by_reason']['STOP']}, Time={r['by_reason']['TIME']}")

        print("\n" + "-" * 70)
        print(f"TOTAL: {total_trades} trades, {total_pnl:+.2f}% combined P&L")
        print("=" * 70)


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Market Analyzer")
    parser.add_argument("--coin", type=str, help="Single coin to analyze")
    parser.add_argument("--days", type=int, default=365, help="Days of history")
    parser.add_argument("--backtest", action="store_true", help="Run backtest")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    if args.backtest:
        print("Running backtest...")
        backtester = Backtester()
        results = backtester.backtest_all()
        backtester.print_results(results)
        return

    print("=" * 70)
    print("MARKET ANALYZER")
    print("Analyzing XRP, SOL, ATOM...")
    print("=" * 70)

    analyzer = MarketAnalyzer()

    if args.coin:
        signals = [analyzer.analyze_coin(args.coin.upper(), args.days)]
    else:
        signals = analyzer.analyze_all()

    if args.json:
        output = [s.to_dict() for s in signals]
        print(json.dumps(output, indent=2))
    else:
        report = analyzer.generate_report(signals)
        print(report)


if __name__ == "__main__":
    main()
