"""
Advanced Strategy Validator

Components:
1. Rolling Walk-Forward Validator - Test strategy stability across time periods
2. Market Regime Classifier - Bull/Bear/Sideways/Volatile detection
3. Signal Audit & Quality Tracker - Measure signal precision/recall
4. Event Reliability Engine - Score news credibility
5. Regime-aware Scoring Engine - Adaptive signal weights

Usage:
    python examples/advanced_validator.py --walkforward
    python examples/advanced_validator.py --regime
    python examples/advanced_validator.py --audit
    python examples/advanced_validator.py --full
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import json
import sqlite3
import os


# =============================================================================
# CONFIGURATION
# =============================================================================

COINS = ["XRPUSDT", "SOLUSDT", "ATOMUSDT"]
DATA_DIR = "data"
DB_PATH = f"{DATA_DIR}/signal_audit.db"


# =============================================================================
# 1. MARKET REGIME CLASSIFIER
# =============================================================================

class MarketRegime(Enum):
    BULL = "BULL"           # Trending up, low volatility
    BEAR = "BEAR"           # Trending down
    SIDEWAYS = "SIDEWAYS"   # Range-bound
    VOLATILE = "VOLATILE"   # High volatility, no clear trend


@dataclass
class RegimeSignal:
    """Market regime classification."""
    regime: MarketRegime
    confidence: float
    btc_trend: str
    volatility_percentile: float
    adx_value: float
    ma_cross: str  # "golden" / "death" / "none"

    def to_dict(self) -> Dict:
        return {
            "regime": self.regime.value,
            "confidence": self.confidence,
            "btc_trend": self.btc_trend,
            "volatility_pct": self.volatility_percentile,
            "adx": self.adx_value,
            "ma_cross": self.ma_cross,
        }


class RegimeClassifier:
    """Classify market regime based on BTC and volatility."""

    def __init__(self):
        self.cache = {}

    def _fetch_btc_data(self, days: int = 250) -> pd.DataFrame:
        """Fetch BTC data for regime analysis."""
        try:
            params = {
                "symbol": "BTCUSDT",
                "interval": "1d",
                "limit": days,
            }
            response = requests.get(
                "https://api.binance.us/api/v3/klines",
                params=params,
                timeout=15
            )
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
                return df
        except Exception as e:
            print(f"BTC fetch error: {e}")
        return pd.DataFrame()

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ADX (Average Directional Index)."""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # True Range
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)

        # Directional Movement
        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        # Smoothed averages
        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)

        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()

        return adx

    def classify(self, df: pd.DataFrame = None) -> RegimeSignal:
        """Classify current market regime."""
        if df is None or df.empty:
            df = self._fetch_btc_data()

        if df.empty or len(df) < 200:
            return RegimeSignal(
                regime=MarketRegime.SIDEWAYS,
                confidence=0.0,
                btc_trend="unknown",
                volatility_percentile=50.0,
                adx_value=0.0,
                ma_cross="none",
            )

        close = df["close"]

        # Moving averages
        sma_50 = close.rolling(50).mean()
        sma_200 = close.rolling(200).mean()

        current_price = close.iloc[-1]
        ma_50 = sma_50.iloc[-1]
        ma_200 = sma_200.iloc[-1]

        # MA cross
        if ma_50 > ma_200:
            ma_cross = "golden"
        elif ma_50 < ma_200:
            ma_cross = "death"
        else:
            ma_cross = "none"

        # Trend direction
        if current_price > ma_50 > ma_200:
            btc_trend = "bullish"
        elif current_price < ma_50 < ma_200:
            btc_trend = "bearish"
        else:
            btc_trend = "mixed"

        # Volatility (30-day realized volatility percentile)
        returns = close.pct_change()
        vol_30d = returns.rolling(30).std() * np.sqrt(365) * 100
        vol_current = vol_30d.iloc[-1]
        vol_percentile = (vol_30d < vol_current).mean() * 100

        # ADX for trend strength
        adx = self._calculate_adx(df)
        adx_current = adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 20

        # Classify regime
        if vol_percentile > 80:
            regime = MarketRegime.VOLATILE
            confidence = vol_percentile / 100
        elif adx_current > 25:
            if btc_trend == "bullish":
                regime = MarketRegime.BULL
                confidence = min(adx_current / 40, 1.0)
            else:
                regime = MarketRegime.BEAR
                confidence = min(adx_current / 40, 1.0)
        else:
            regime = MarketRegime.SIDEWAYS
            confidence = 1 - (adx_current / 25)

        return RegimeSignal(
            regime=regime,
            confidence=round(confidence, 2),
            btc_trend=btc_trend,
            volatility_percentile=round(vol_percentile, 1),
            adx_value=round(adx_current, 1),
            ma_cross=ma_cross,
        )

    def get_regime_history(self, days: int = 365) -> pd.DataFrame:
        """Get regime classification for each day."""
        df = self._fetch_btc_data(days + 200)
        if df.empty:
            return pd.DataFrame()

        regimes = []
        for i in range(200, len(df)):
            window = df.iloc[:i+1]
            regime = self.classify(window)
            regimes.append({
                "date": df["timestamp"].iloc[i],
                "price": df["close"].iloc[i],
                "regime": regime.regime.value,
                "adx": regime.adx_value,
                "volatility_pct": regime.volatility_percentile,
            })

        return pd.DataFrame(regimes)


# =============================================================================
# 2. ROLLING WALK-FORWARD VALIDATOR
# =============================================================================

@dataclass
class WindowResult:
    """Result for a single walk-forward window."""
    window_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_trades: int
    test_trades: int
    train_return: float
    test_return: float
    test_win_rate: float
    test_max_drawdown: float
    is_profitable: bool


@dataclass
class WalkForwardResult:
    """Overall walk-forward validation result."""
    coin: str
    total_windows: int
    profitable_windows: int
    consistency_rate: float
    avg_test_return: float
    total_test_return: float
    avg_win_rate: float
    windows: List[WindowResult] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "coin": self.coin,
            "total_windows": self.total_windows,
            "profitable_windows": self.profitable_windows,
            "consistency_rate": self.consistency_rate,
            "avg_test_return": self.avg_test_return,
            "total_test_return": self.total_test_return,
            "avg_win_rate": self.avg_win_rate,
        }


class WalkForwardValidator:
    """Rolling walk-forward validation for strategy testing."""

    def __init__(self, train_months: int = 6, test_months: int = 3, step_months: int = 1):
        self.train_months = train_months
        self.test_months = test_months
        self.step_months = step_months

    def _fetch_data(self, symbol: str, days: int = 1100) -> pd.DataFrame:
        """Fetch historical data."""
        try:
            params = {
                "symbol": symbol,
                "interval": "1d",
                "limit": days,
            }
            response = requests.get(
                "https://api.binance.us/api/v3/klines",
                params=params,
                timeout=15
            )
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
                return df
        except Exception as e:
            print(f"Data fetch error: {e}")
        return pd.DataFrame()

    def _run_strategy(self, df: pd.DataFrame, bb_period: int = 20, bb_std: float = 1.5) -> List[Dict]:
        """Run Bollinger strategy on data and return trades."""
        if len(df) < 220:
            return []

        close = df["close"]
        high = df["high"]
        low = df["low"]

        # Bollinger Bands
        middle = close.rolling(bb_period).mean()
        std = close.rolling(bb_period).std()
        lower = middle - (std * bb_std)

        # 200 MA
        ma_200 = close.rolling(200).mean()

        # ATR for stop loss
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()

        trades = []
        position = None

        for i in range(220, len(df)):
            price = close.iloc[i]

            # Entry: price below lower band + above 200 MA
            if position is None:
                if price <= lower.iloc[i] and price > ma_200.iloc[i]:
                    position = {
                        "entry_price": price,
                        "entry_idx": i,
                        "stop_loss": price - atr.iloc[i] * 2,
                    }

            # Exit conditions
            elif position is not None:
                days_held = i - position["entry_idx"]

                # Take profit at middle band
                if price >= middle.iloc[i]:
                    pnl = (price - position["entry_price"]) / position["entry_price"] * 100
                    trades.append({"pnl": pnl, "days": days_held, "reason": "TARGET"})
                    position = None

                # Stop loss
                elif price <= position["stop_loss"]:
                    pnl = (price - position["entry_price"]) / position["entry_price"] * 100
                    trades.append({"pnl": pnl, "days": days_held, "reason": "STOP"})
                    position = None

                # Time exit
                elif days_held >= 22:
                    pnl = (price - position["entry_price"]) / position["entry_price"] * 100
                    trades.append({"pnl": pnl, "days": days_held, "reason": "TIME"})
                    position = None

        return trades

    def validate(self, symbol: str) -> WalkForwardResult:
        """Run walk-forward validation on a symbol."""
        df = self._fetch_data(symbol)
        if df.empty or len(df) < 400:
            return WalkForwardResult(
                coin=symbol.replace("USDT", ""),
                total_windows=0,
                profitable_windows=0,
                consistency_rate=0.0,
                avg_test_return=0.0,
                total_test_return=0.0,
                avg_win_rate=0.0,
            )

        # Calculate window sizes in days
        train_days = self.train_months * 30
        test_days = self.test_months * 30
        step_days = self.step_months * 30
        window_size = train_days + test_days

        windows = []
        window_id = 0

        # Rolling windows
        start_idx = 0
        while start_idx + window_size <= len(df):
            train_end_idx = start_idx + train_days
            test_end_idx = start_idx + window_size

            train_df = df.iloc[start_idx:train_end_idx].copy()
            test_df = df.iloc[train_end_idx:test_end_idx].copy()

            # Need enough data for indicators
            if len(train_df) < 220:
                start_idx += step_days
                continue

            # Run strategy on both periods
            train_trades = self._run_strategy(df.iloc[start_idx:train_end_idx + 220])
            test_trades = self._run_strategy(df.iloc[train_end_idx - 220:test_end_idx])

            train_return = sum(t["pnl"] for t in train_trades) if train_trades else 0
            test_return = sum(t["pnl"] for t in test_trades) if test_trades else 0
            test_wins = len([t for t in test_trades if t["pnl"] > 0])
            test_win_rate = (test_wins / len(test_trades) * 100) if test_trades else 0

            # Max drawdown (simplified)
            max_dd = min([t["pnl"] for t in test_trades]) if test_trades else 0

            window_result = WindowResult(
                window_id=window_id,
                train_start=df["timestamp"].iloc[start_idx],
                train_end=df["timestamp"].iloc[train_end_idx - 1],
                test_start=df["timestamp"].iloc[train_end_idx],
                test_end=df["timestamp"].iloc[min(test_end_idx - 1, len(df) - 1)],
                train_trades=len(train_trades),
                test_trades=len(test_trades),
                train_return=round(train_return, 2),
                test_return=round(test_return, 2),
                test_win_rate=round(test_win_rate, 1),
                test_max_drawdown=round(max_dd, 2),
                is_profitable=test_return > 0,
            )

            windows.append(window_result)
            window_id += 1
            start_idx += step_days

        # Calculate overall metrics
        profitable = len([w for w in windows if w.is_profitable])
        total = len(windows)

        return WalkForwardResult(
            coin=symbol.replace("USDT", ""),
            total_windows=total,
            profitable_windows=profitable,
            consistency_rate=round(profitable / total * 100, 1) if total > 0 else 0,
            avg_test_return=round(np.mean([w.test_return for w in windows]), 2) if windows else 0,
            total_test_return=round(sum(w.test_return for w in windows), 2),
            avg_win_rate=round(np.mean([w.test_win_rate for w in windows]), 1) if windows else 0,
            windows=windows,
        )


# =============================================================================
# 3. SIGNAL AUDIT & QUALITY TRACKER
# =============================================================================

class SignalAuditor:
    """Track and audit signal quality."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                coin TEXT,
                signal_type TEXT,
                score INTEGER,
                price REAL,
                bb_touch INTEGER,
                rsi_oversold INTEGER,
                volume_spike INTEGER,
                above_200ma INTEGER,
                regime TEXT,
                price_7d REAL,
                price_14d REAL,
                price_30d REAL,
                return_7d REAL,
                return_14d REAL,
                return_30d REAL,
                is_win_7d INTEGER,
                is_win_14d INTEGER,
                is_win_30d INTEGER
            )
        """)

        conn.commit()
        conn.close()

    def log_signal(self, signal_data: Dict):
        """Log a signal to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO signals (
                timestamp, coin, signal_type, score, price,
                bb_touch, rsi_oversold, volume_spike, above_200ma, regime
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            signal_data.get("timestamp", datetime.now().isoformat()),
            signal_data.get("coin"),
            signal_data.get("signal_type"),
            signal_data.get("score", 0),
            signal_data.get("price", 0),
            signal_data.get("bb_touch", 0),
            signal_data.get("rsi_oversold", 0),
            signal_data.get("volume_spike", 0),
            signal_data.get("above_200ma", 0),
            signal_data.get("regime", ""),
        ))

        conn.commit()
        conn.close()

    def update_outcomes(self, signal_id: int, price_7d: float, price_14d: float, price_30d: float, entry_price: float):
        """Update signal with actual outcomes."""
        return_7d = ((price_7d - entry_price) / entry_price) * 100
        return_14d = ((price_14d - entry_price) / entry_price) * 100
        return_30d = ((price_30d - entry_price) / entry_price) * 100

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE signals SET
                price_7d = ?, price_14d = ?, price_30d = ?,
                return_7d = ?, return_14d = ?, return_30d = ?,
                is_win_7d = ?, is_win_14d = ?, is_win_30d = ?
            WHERE id = ?
        """, (
            price_7d, price_14d, price_30d,
            return_7d, return_14d, return_30d,
            1 if return_7d > 0 else 0,
            1 if return_14d > 0 else 0,
            1 if return_30d > 0 else 0,
            signal_id,
        ))

        conn.commit()
        conn.close()

    def get_precision_report(self) -> Dict:
        """Get precision metrics for signals."""
        conn = sqlite3.connect(self.db_path)

        # Overall precision
        df = pd.read_sql_query("""
            SELECT * FROM signals WHERE return_14d IS NOT NULL
        """, conn)

        if df.empty:
            conn.close()
            return {"error": "No signals with outcomes"}

        report = {
            "total_signals": len(df),
            "precision_7d": round((df["is_win_7d"].sum() / len(df)) * 100, 1),
            "precision_14d": round((df["is_win_14d"].sum() / len(df)) * 100, 1),
            "precision_30d": round((df["is_win_30d"].sum() / len(df)) * 100, 1),
            "avg_return_14d": round(df["return_14d"].mean(), 2),
        }

        # By signal component
        for component in ["bb_touch", "rsi_oversold", "volume_spike", "above_200ma"]:
            subset = df[df[component] == 1]
            if len(subset) > 0:
                report[f"{component}_precision_14d"] = round((subset["is_win_14d"].sum() / len(subset)) * 100, 1)
                report[f"{component}_avg_return"] = round(subset["return_14d"].mean(), 2)

        # By regime
        for regime in ["BULL", "BEAR", "SIDEWAYS", "VOLATILE"]:
            subset = df[df["regime"] == regime]
            if len(subset) > 0:
                report[f"regime_{regime.lower()}_precision"] = round((subset["is_win_14d"].sum() / len(subset)) * 100, 1)

        conn.close()
        return report


# =============================================================================
# 4. EVENT RELIABILITY ENGINE
# =============================================================================

@dataclass
class ScoredEvent:
    """Event with credibility score."""
    headline: str
    source: str
    source_trust: float  # 0-1
    event_type: str
    coin: str
    independent_sources: int
    credibility_score: float
    timestamp: datetime


class EventReliabilityEngine:
    """Score event credibility and source trust."""

    # Source trust scores (0-1)
    SOURCE_TRUST = {
        "reuters": 0.95,
        "bloomberg": 0.95,
        "coindesk": 0.85,
        "cointelegraph": 0.80,
        "decrypt": 0.80,
        "theblock": 0.85,
        "cryptopanic": 0.70,
        "twitter": 0.50,
        "reddit": 0.40,
        "unknown": 0.30,
    }

    def __init__(self):
        self.seen_events = {}  # For deduplication

    def _get_source_trust(self, source: str) -> float:
        """Get trust score for a source."""
        source_lower = source.lower()
        for key, trust in self.SOURCE_TRUST.items():
            if key in source_lower:
                return trust
        return self.SOURCE_TRUST["unknown"]

    def _is_duplicate(self, headline: str) -> bool:
        """Check if headline is duplicate (simplified)."""
        # Simple dedup based on key words
        words = set(headline.lower().split())
        for seen_words in self.seen_events.values():
            overlap = len(words & seen_words) / max(len(words), 1)
            if overlap > 0.7:
                return True
        return False

    def score_event(self, headline: str, source: str, event_type: str, coin: str) -> ScoredEvent:
        """Score an event for credibility."""
        # Check duplicate
        if self._is_duplicate(headline):
            # Increment independent source count
            pass

        source_trust = self._get_source_trust(source)

        # Event type weight
        event_weights = {
            "legal_positive": 1.5,
            "legal_negative": 1.5,
            "partnership": 1.2,
            "upgrade": 1.1,
            "airdrop": 1.3,
            "outage": 1.4,
            "listing": 1.2,
        }
        event_weight = event_weights.get(event_type, 1.0)

        # Calculate credibility
        credibility = source_trust * event_weight

        # Store for dedup
        self.seen_events[headline] = set(headline.lower().split())

        return ScoredEvent(
            headline=headline,
            source=source,
            source_trust=source_trust,
            event_type=event_type,
            coin=coin,
            independent_sources=1,  # Would track across multiple sources
            credibility_score=round(credibility, 2),
            timestamp=datetime.now(),
        )

    def filter_reliable_events(self, events: List[Dict], min_credibility: float = 0.6) -> List[ScoredEvent]:
        """Filter events by credibility threshold."""
        scored = []
        for event in events:
            scored_event = self.score_event(
                headline=event.get("headline", ""),
                source=event.get("source", "unknown"),
                event_type=event.get("event_type", "unknown"),
                coin=event.get("coin", ""),
            )
            if scored_event.credibility_score >= min_credibility:
                scored.append(scored_event)
        return scored


# =============================================================================
# 5. REGIME-AWARE SCORING ENGINE
# =============================================================================

class RegimeAwareScoringEngine:
    """Adaptive signal scoring based on market regime."""

    # Weight profiles per regime
    REGIME_WEIGHTS = {
        MarketRegime.BULL: {
            "fundamental": 2,
            "onchain": 2,
            "technical": 3,  # Momentum matters more
            "threshold": 5,
        },
        MarketRegime.BEAR: {
            "fundamental": 3,
            "onchain": 3,
            "technical": 1,  # Be more selective
            "threshold": 7,
        },
        MarketRegime.SIDEWAYS: {
            "fundamental": 2,
            "onchain": 3,
            "technical": 2,  # Mean reversion works
            "threshold": 4,
        },
        MarketRegime.VOLATILE: {
            "fundamental": 4,
            "onchain": 2,
            "technical": 1,  # Wait for clarity
            "threshold": 8,
        },
    }

    def __init__(self):
        self.regime_classifier = RegimeClassifier()

    def calculate_score(
        self,
        technical_score: int,
        fundamental_score: int,
        onchain_score: int,
        regime: MarketRegime = None,
    ) -> Dict:
        """Calculate regime-aware score."""
        if regime is None:
            regime_signal = self.regime_classifier.classify()
            regime = regime_signal.regime

        weights = self.REGIME_WEIGHTS.get(regime, self.REGIME_WEIGHTS[MarketRegime.SIDEWAYS])

        weighted_tech = technical_score * weights["technical"]
        weighted_fund = fundamental_score * weights["fundamental"]
        weighted_chain = onchain_score * weights["onchain"]

        total_score = weighted_tech + weighted_fund + weighted_chain
        threshold = weights["threshold"]

        return {
            "regime": regime.value,
            "total_score": total_score,
            "threshold": threshold,
            "is_buy_signal": total_score >= threshold,
            "breakdown": {
                "technical": f"{technical_score} x {weights['technical']} = {weighted_tech}",
                "fundamental": f"{fundamental_score} x {weights['fundamental']} = {weighted_fund}",
                "onchain": f"{onchain_score} x {weights['onchain']} = {weighted_chain}",
            },
            "weights_used": weights,
        }


# =============================================================================
# MAIN REPORT GENERATOR
# =============================================================================

def generate_full_report():
    """Generate comprehensive validation report."""
    print("=" * 80)
    print("ADVANCED STRATEGY VALIDATION REPORT")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # 1. Market Regime
    print("\n" + "-" * 80)
    print("1. CURRENT MARKET REGIME")
    print("-" * 80)

    classifier = RegimeClassifier()
    regime = classifier.classify()

    print(f"\n   Regime: {regime.regime.value}")
    print(f"   Confidence: {regime.confidence * 100:.0f}%")
    print(f"   BTC Trend: {regime.btc_trend}")
    print(f"   Volatility Percentile: {regime.volatility_percentile:.1f}%")
    print(f"   ADX: {regime.adx_value:.1f}")
    print(f"   MA Cross: {regime.ma_cross}")

    # Strategy recommendation based on regime
    recommendations = {
        MarketRegime.BULL: "Favor momentum strategies, increase position sizes",
        MarketRegime.BEAR: "Be defensive, tighten stop losses, reduce exposure",
        MarketRegime.SIDEWAYS: "Mean reversion works well, Bollinger strategy optimal",
        MarketRegime.VOLATILE: "Reduce trading, wait for clarity, widen stops",
    }
    print(f"\n   Recommendation: {recommendations.get(regime.regime, 'No specific recommendation')}")

    # 2. Walk-Forward Validation
    print("\n" + "-" * 80)
    print("2. WALK-FORWARD VALIDATION (6mo train / 3mo test)")
    print("-" * 80)

    validator = WalkForwardValidator(train_months=6, test_months=3, step_months=1)

    for symbol in COINS:
        print(f"\n   Validating {symbol}...")
        result = validator.validate(symbol)

        print(f"\n   {result.coin}:")
        print(f"      Windows: {result.total_windows}")
        print(f"      Profitable: {result.profitable_windows}/{result.total_windows}")
        print(f"      Consistency Rate: {result.consistency_rate}%")
        print(f"      Avg Test Return: {result.avg_test_return}%")
        print(f"      Total Test Return: {result.total_test_return}%")
        print(f"      Avg Win Rate: {result.avg_win_rate}%")

        # Stability assessment
        if result.consistency_rate >= 70:
            status = "✅ STABLE"
        elif result.consistency_rate >= 50:
            status = "⚠️ MODERATE"
        else:
            status = "❌ UNSTABLE"
        print(f"      Status: {status}")

    # 3. Regime-Aware Scoring Example
    print("\n" + "-" * 80)
    print("3. REGIME-AWARE SCORING")
    print("-" * 80)

    scorer = RegimeAwareScoringEngine()

    # Example signal
    example_score = scorer.calculate_score(
        technical_score=2,
        fundamental_score=1,
        onchain_score=1,
        regime=regime.regime,
    )

    print(f"\n   Current Regime: {example_score['regime']}")
    print(f"   Example Signal (tech=2, fund=1, chain=1):")
    print(f"      Total Score: {example_score['total_score']}")
    print(f"      Threshold: {example_score['threshold']}")
    print(f"      Is Buy: {'YES' if example_score['is_buy_signal'] else 'NO'}")
    print(f"\n   Score Breakdown:")
    for layer, calc in example_score["breakdown"].items():
        print(f"      {layer}: {calc}")

    # 4. Event Reliability
    print("\n" + "-" * 80)
    print("4. EVENT RELIABILITY SCORING")
    print("-" * 80)

    engine = EventReliabilityEngine()

    example_events = [
        {"headline": "SEC rules in favor of Ripple", "source": "Reuters", "event_type": "legal_positive", "coin": "XRP"},
        {"headline": "Solana announces major upgrade", "source": "CoinDesk", "event_type": "upgrade", "coin": "SOL"},
        {"headline": "XRP moon soon!", "source": "Twitter", "event_type": "unknown", "coin": "XRP"},
    ]

    print("\n   Event Credibility Scores:")
    for event in example_events:
        scored = engine.score_event(**event)
        status = "✅" if scored.credibility_score >= 0.6 else "❌"
        print(f"   {status} [{scored.source}] {scored.headline[:40]}...")
        print(f"      Trust: {scored.source_trust:.2f} | Credibility: {scored.credibility_score:.2f}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"""
   Market Regime: {regime.regime.value}
   Strategy Status: {'ACTIVE' if regime.regime in [MarketRegime.SIDEWAYS, MarketRegime.BULL] else 'CAUTIOUS'}

   Key Insights:
   - Bollinger mean-reversion works best in SIDEWAYS markets
   - Current regime suggests: {recommendations.get(regime.regime, 'standard approach')}
   - Walk-forward validation shows strategy consistency across time
   - Use regime-adaptive thresholds for better risk management
""")

    print("=" * 80)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Advanced Strategy Validator")
    parser.add_argument("--walkforward", action="store_true", help="Run walk-forward validation only")
    parser.add_argument("--regime", action="store_true", help="Show regime classification only")
    parser.add_argument("--audit", action="store_true", help="Show signal audit report")
    parser.add_argument("--full", action="store_true", help="Generate full report")

    args = parser.parse_args()

    if args.walkforward:
        print("Running walk-forward validation...")
        validator = WalkForwardValidator()
        for symbol in COINS:
            result = validator.validate(symbol)
            print(f"{result.coin}: {result.consistency_rate}% consistency, {result.total_test_return}% total return")

    elif args.regime:
        classifier = RegimeClassifier()
        regime = classifier.classify()
        print(f"Regime: {regime.regime.value}")
        print(f"BTC Trend: {regime.btc_trend}")
        print(f"Volatility: {regime.volatility_percentile}%")
        print(f"ADX: {regime.adx_value}")

    elif args.audit:
        auditor = SignalAuditor()
        report = auditor.get_precision_report()
        print(json.dumps(report, indent=2))

    else:
        generate_full_report()


if __name__ == "__main__":
    main()
