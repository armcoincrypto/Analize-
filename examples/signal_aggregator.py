#!/usr/bin/env python3
"""
Signal Aggregator & Production Infrastructure

Critical production components based on professional review:
1. UTC Timestamp Canonicalization
2. Signal Audit Table (precision tracking)
3. Circuit Breaker / Kill Switch
4. Master Signal Aggregator with regime-aware weights
5. Execution Safety Checks

Author: Analize Team
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
import sqlite3
import json
import hashlib
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# UTC TIMESTAMP UTILITIES
# =============================================================================

class TimestampCanonicalizer:
    """Ensure all timestamps are UTC and aligned to candle boundaries."""

    @staticmethod
    def to_utc(dt: datetime) -> datetime:
        """Convert any datetime to UTC."""
        if dt.tzinfo is None:
            # Assume UTC if no timezone
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    @staticmethod
    def utc_now() -> datetime:
        """Get current UTC time."""
        return datetime.now(timezone.utc)

    @staticmethod
    def align_to_candle(dt: datetime, interval_minutes: int = 1440) -> datetime:
        """Align timestamp to candle boundary (default: daily)."""
        dt = TimestampCanonicalizer.to_utc(dt)

        # For daily candles, align to midnight UTC
        if interval_minutes == 1440:
            return dt.replace(hour=0, minute=0, second=0, microsecond=0)

        # For other intervals
        minutes_since_midnight = dt.hour * 60 + dt.minute
        aligned_minutes = (minutes_since_midnight // interval_minutes) * interval_minutes

        return dt.replace(
            hour=aligned_minutes // 60,
            minute=aligned_minutes % 60,
            second=0,
            microsecond=0
        )

    @staticmethod
    def parse_timestamp(ts: Any) -> datetime:
        """Parse various timestamp formats to UTC datetime."""
        if isinstance(ts, datetime):
            return TimestampCanonicalizer.to_utc(ts)
        elif isinstance(ts, (int, float)):
            # Assume milliseconds if > 10^12
            if ts > 1e12:
                ts = ts / 1000
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        elif isinstance(ts, str):
            # Try ISO format
            try:
                dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                return TimestampCanonicalizer.to_utc(dt)
            except:
                pass
            # Try pandas
            try:
                return pd.to_datetime(ts).to_pydatetime().replace(tzinfo=timezone.utc)
            except:
                pass

        raise ValueError(f"Cannot parse timestamp: {ts}")


# =============================================================================
# SIGNAL AUDIT TABLE
# =============================================================================

@dataclass
class SignalAuditRecord:
    """Single signal audit record for tracking precision."""
    id: str  # Hash of signal details
    timestamp: datetime
    symbol: str
    signal_source: str  # whale_tracker, funding_rate, etc.
    signal_type: str  # BUY, SELL, ALERT
    signal_strength: float  # 0-1
    confidence: float  # 0-1

    # Feature snapshot (for reproducibility)
    features_hash: str
    features_json: str

    # Outcome tracking (filled later)
    price_at_signal: float = 0.0
    price_7d: Optional[float] = None
    price_14d: Optional[float] = None
    price_30d: Optional[float] = None

    return_7d: Optional[float] = None
    return_14d: Optional[float] = None
    return_30d: Optional[float] = None

    was_correct_7d: Optional[bool] = None
    was_correct_14d: Optional[bool] = None
    was_correct_30d: Optional[bool] = None

    # Metadata
    regime: str = "UNKNOWN"
    btc_trend: str = "UNKNOWN"
    notes: str = ""


class SignalAuditTable:
    """SQLite-based signal audit table for precision tracking."""

    def __init__(self, db_path: str = "signal_audit.db"):
        self.db_path = db_path
        self._conn = None  # Persistent connection for :memory: databases
        self._init_db()

    def _get_connection(self):
        """Get database connection (persistent for :memory:)."""
        if self.db_path == ":memory:":
            if self._conn is None:
                self._conn = sqlite3.connect(":memory:")
            return self._conn
        return sqlite3.connect(self.db_path)

    def _close_connection(self, conn):
        """Close connection (skip for :memory:)."""
        if self.db_path != ":memory:":
            conn.close()

    def _init_db(self):
        """Initialize database schema."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                symbol TEXT,
                signal_source TEXT,
                signal_type TEXT,
                signal_strength REAL,
                confidence REAL,
                features_hash TEXT,
                features_json TEXT,
                price_at_signal REAL,
                price_7d REAL,
                price_14d REAL,
                price_30d REAL,
                return_7d REAL,
                return_14d REAL,
                return_30d REAL,
                was_correct_7d INTEGER,
                was_correct_14d INTEGER,
                was_correct_30d INTEGER,
                regime TEXT,
                btc_trend TEXT,
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_symbol ON signals(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_source ON signals(signal_source)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON signals(timestamp)")

        conn.commit()
        self._close_connection(conn)

    def record_signal(self, record: SignalAuditRecord):
        """Record a new signal."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO signals (
                id, timestamp, symbol, signal_source, signal_type,
                signal_strength, confidence, features_hash, features_json,
                price_at_signal, regime, btc_trend, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.id,
            record.timestamp.isoformat(),
            record.symbol,
            record.signal_source,
            record.signal_type,
            record.signal_strength,
            record.confidence,
            record.features_hash,
            record.features_json,
            record.price_at_signal,
            record.regime,
            record.btc_trend,
            record.notes
        ))

        conn.commit()
        self._close_connection(conn)

    def update_outcomes(self, signal_id: str, outcomes: Dict):
        """Update signal outcomes after time passes."""
        conn = self._get_connection()
        cursor = conn.cursor()

        updates = []
        values = []

        for key, value in outcomes.items():
            if key in ['price_7d', 'price_14d', 'price_30d',
                       'return_7d', 'return_14d', 'return_30d']:
                updates.append(f"{key} = ?")
                values.append(value)
            elif key in ['was_correct_7d', 'was_correct_14d', 'was_correct_30d']:
                updates.append(f"{key} = ?")
                values.append(1 if value else 0)

        if updates:
            values.append(signal_id)
            cursor.execute(f"""
                UPDATE signals SET {', '.join(updates)} WHERE id = ?
            """, values)

        conn.commit()
        self._close_connection(conn)

    def get_precision_metrics(self, signal_source: Optional[str] = None,
                              symbol: Optional[str] = None,
                              days_back: int = 90) -> Dict:
        """Calculate precision metrics for signals."""
        conn = self._get_connection()

        query = """
            SELECT
                signal_source,
                signal_type,
                COUNT(*) as total,
                SUM(CASE WHEN was_correct_7d = 1 THEN 1 ELSE 0 END) as correct_7d,
                SUM(CASE WHEN was_correct_14d = 1 THEN 1 ELSE 0 END) as correct_14d,
                SUM(CASE WHEN was_correct_30d = 1 THEN 1 ELSE 0 END) as correct_30d,
                AVG(return_7d) as avg_return_7d,
                AVG(return_14d) as avg_return_14d,
                AVG(return_30d) as avg_return_30d
            FROM signals
            WHERE timestamp > datetime('now', ?)
        """

        params = [f'-{days_back} days']

        if signal_source:
            query += " AND signal_source = ?"
            params.append(signal_source)

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        query += " GROUP BY signal_source, signal_type"

        df = pd.read_sql_query(query, conn, params=params)
        self._close_connection(conn)

        metrics = {}
        for _, row in df.iterrows():
            source = row['signal_source']
            if source not in metrics:
                metrics[source] = {}

            total = row['total']
            metrics[source][row['signal_type']] = {
                'total_signals': total,
                'precision_7d': row['correct_7d'] / total if total > 0 else 0,
                'precision_14d': row['correct_14d'] / total if total > 0 else 0,
                'precision_30d': row['correct_30d'] / total if total > 0 else 0,
                'avg_return_7d': row['avg_return_7d'] or 0,
                'avg_return_14d': row['avg_return_14d'] or 0,
                'avg_return_30d': row['avg_return_30d'] or 0,
            }

        return metrics

    @staticmethod
    def generate_signal_id(timestamp: datetime, symbol: str,
                           source: str, features: Dict) -> str:
        """Generate unique signal ID from components."""
        content = f"{timestamp.isoformat()}|{symbol}|{source}|{json.dumps(features, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


# =============================================================================
# CIRCUIT BREAKER / KILL SWITCH
# =============================================================================

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    # Market conditions
    btc_max_decline_7d: float = 0.15  # 15% BTC decline triggers
    max_volatility_multiplier: float = 3.0  # 3x normal vol triggers

    # Data quality
    max_data_staleness_minutes: int = 10
    min_data_sources: int = 1

    # Position limits
    max_open_positions: int = 5
    max_daily_trades: int = 10
    max_portfolio_drawdown: float = 0.10  # 10% drawdown

    # Signal quality
    min_signal_confidence: float = 0.5
    min_precision_30d: float = 0.45  # 45% precision minimum


@dataclass
class CircuitBreakerState:
    """Current state of circuit breaker."""
    is_active: bool = False
    triggered_at: Optional[datetime] = None
    trigger_reason: str = ""
    btc_change_7d: float = 0.0
    current_volatility_ratio: float = 1.0
    data_staleness_minutes: float = 0.0
    open_positions: int = 0
    daily_trades: int = 0
    portfolio_drawdown: float = 0.0
    last_check: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class CircuitBreaker:
    """Circuit breaker to halt trading during dangerous conditions."""

    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitBreakerState()
        self.manual_override = False

    def check(self, market_data: Dict) -> Tuple[bool, str]:
        """
        Check all circuit breaker conditions.

        Returns:
            (is_safe_to_trade, reason_if_not)
        """
        self.state.last_check = datetime.now(timezone.utc)

        # Check BTC decline
        btc_change = market_data.get('btc_change_7d', 0)
        self.state.btc_change_7d = btc_change

        if btc_change < -self.config.btc_max_decline_7d:
            return self._trigger(f"BTC 7d decline {btc_change:.1%} exceeds threshold")

        # Check volatility
        vol_ratio = market_data.get('volatility_ratio', 1.0)
        self.state.current_volatility_ratio = vol_ratio

        if vol_ratio > self.config.max_volatility_multiplier:
            return self._trigger(f"Volatility {vol_ratio:.1f}x exceeds threshold")

        # Check data staleness
        staleness = market_data.get('data_staleness_minutes', 0)
        self.state.data_staleness_minutes = staleness

        if staleness > self.config.max_data_staleness_minutes:
            return self._trigger(f"Data stale by {staleness:.0f} minutes")

        # Check data sources
        sources = market_data.get('active_data_sources', 1)
        if sources < self.config.min_data_sources:
            return self._trigger(f"Only {sources} data sources active")

        # Check position limits
        positions = market_data.get('open_positions', 0)
        self.state.open_positions = positions

        if positions >= self.config.max_open_positions:
            return self._trigger(f"Max positions ({positions}) reached")

        # Check daily trades
        daily_trades = market_data.get('daily_trades', 0)
        self.state.daily_trades = daily_trades

        if daily_trades >= self.config.max_daily_trades:
            return self._trigger(f"Max daily trades ({daily_trades}) reached")

        # Check drawdown
        drawdown = market_data.get('portfolio_drawdown', 0)
        self.state.portfolio_drawdown = drawdown

        if drawdown > self.config.max_portfolio_drawdown:
            return self._trigger(f"Portfolio drawdown {drawdown:.1%} exceeds threshold")

        # All checks passed
        self.state.is_active = False
        return True, "OK"

    def _trigger(self, reason: str) -> Tuple[bool, str]:
        """Trigger circuit breaker."""
        self.state.is_active = True
        self.state.triggered_at = datetime.now(timezone.utc)
        self.state.trigger_reason = reason
        return False, reason

    def force_enable(self):
        """Manually enable trading (override circuit breaker)."""
        self.manual_override = True

    def force_disable(self, reason: str = "Manual shutdown"):
        """Manually disable all trading."""
        self._trigger(reason)
        self.manual_override = False

    def get_status(self) -> Dict:
        """Get current circuit breaker status."""
        return {
            'is_trading_allowed': not self.state.is_active or self.manual_override,
            'is_circuit_breaker_active': self.state.is_active,
            'trigger_reason': self.state.trigger_reason,
            'triggered_at': self.state.triggered_at.isoformat() if self.state.triggered_at else None,
            'btc_change_7d': self.state.btc_change_7d,
            'volatility_ratio': self.state.current_volatility_ratio,
            'data_staleness_minutes': self.state.data_staleness_minutes,
            'open_positions': self.state.open_positions,
            'daily_trades': self.state.daily_trades,
            'portfolio_drawdown': self.state.portfolio_drawdown,
            'manual_override': self.manual_override,
            'last_check': self.state.last_check.isoformat()
        }


# =============================================================================
# MASTER SIGNAL AGGREGATOR
# =============================================================================

@dataclass
class SignalInput:
    """Input signal from any analyzer."""
    source: str  # correlation, whale, funding, orderflow, ml
    symbol: str
    signal_type: str  # BUY, SELL, NEUTRAL
    strength: float  # 0-1
    confidence: float  # 0-1
    timestamp: datetime
    features: Dict = field(default_factory=dict)
    explanation: str = ""


@dataclass
class AggregatedSignal:
    """Aggregated signal from multiple sources."""
    symbol: str
    final_signal: str  # STRONG_BUY, BUY, NEUTRAL, SELL, STRONG_SELL
    final_score: float  # -1 to +1
    confidence: float  # 0-1
    contributing_signals: List[SignalInput]
    regime: str
    timestamp: datetime

    # Signal decomposition
    correlation_contribution: float = 0.0
    whale_contribution: float = 0.0
    funding_contribution: float = 0.0
    orderflow_contribution: float = 0.0
    ml_contribution: float = 0.0

    # Recommendations
    suggested_size: float = 0.0  # % of portfolio
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


class RegimeDetector:
    """Detect current market regime."""

    def __init__(self):
        self.current_regime = "UNKNOWN"

    def detect(self, market_data: Dict) -> str:
        """
        Detect market regime.

        Returns: BULL, BEAR, SIDEWAYS, VOLATILE, CRISIS
        """
        btc_change_7d = market_data.get('btc_change_7d', 0)
        btc_change_30d = market_data.get('btc_change_30d', 0)
        volatility_ratio = market_data.get('volatility_ratio', 1.0)
        trend_strength = market_data.get('trend_strength', 0)

        # Crisis detection
        if btc_change_7d < -0.20 or volatility_ratio > 3.0:
            self.current_regime = "CRISIS"

        # Volatile market
        elif volatility_ratio > 2.0:
            self.current_regime = "VOLATILE"

        # Bull market
        elif btc_change_30d > 0.15 and trend_strength > 0.5:
            self.current_regime = "BULL"

        # Bear market
        elif btc_change_30d < -0.15 and trend_strength > 0.5:
            self.current_regime = "BEAR"

        # Sideways
        else:
            self.current_regime = "SIDEWAYS"

        return self.current_regime


class SignalAggregator:
    """
    Aggregate signals from multiple sources with regime-aware weights.
    """

    # Default weights by regime
    DEFAULT_WEIGHTS = {
        "BULL": {
            "correlation": 0.15,
            "whale": 0.25,
            "funding": 0.20,
            "orderflow": 0.20,
            "ml": 0.20
        },
        "BEAR": {
            "correlation": 0.20,
            "whale": 0.30,
            "funding": 0.25,
            "orderflow": 0.15,
            "ml": 0.10
        },
        "SIDEWAYS": {
            "correlation": 0.20,
            "whale": 0.20,
            "funding": 0.15,
            "orderflow": 0.25,
            "ml": 0.20
        },
        "VOLATILE": {
            "correlation": 0.10,
            "whale": 0.30,
            "funding": 0.30,
            "orderflow": 0.20,
            "ml": 0.10
        },
        "CRISIS": {
            "correlation": 0.05,
            "whale": 0.40,
            "funding": 0.35,
            "orderflow": 0.15,
            "ml": 0.05
        }
    }

    def __init__(self, weights: Optional[Dict] = None):
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.regime_detector = RegimeDetector()
        self.audit_table = SignalAuditTable()

    def aggregate(self, signals: List[SignalInput],
                  market_data: Dict) -> AggregatedSignal:
        """Aggregate multiple signals into final recommendation."""

        if not signals:
            return self._empty_signal(market_data)

        symbol = signals[0].symbol
        regime = self.regime_detector.detect(market_data)
        weights = self.weights.get(regime, self.DEFAULT_WEIGHTS["SIDEWAYS"])

        # Calculate weighted score
        total_weight = 0
        weighted_score = 0
        contributions = {}

        for signal in signals:
            source = signal.source.lower()
            if source in weights:
                weight = weights[source]

                # Convert signal to score (-1 to +1)
                if signal.signal_type == "BUY":
                    score = signal.strength * signal.confidence
                elif signal.signal_type == "SELL":
                    score = -signal.strength * signal.confidence
                else:
                    score = 0

                weighted_score += score * weight
                total_weight += weight
                contributions[f"{source}_contribution"] = score * weight

        # Normalize
        if total_weight > 0:
            final_score = weighted_score / total_weight
        else:
            final_score = 0

        # Calculate confidence (based on agreement)
        signal_scores = []
        for signal in signals:
            if signal.signal_type == "BUY":
                signal_scores.append(signal.strength)
            elif signal.signal_type == "SELL":
                signal_scores.append(-signal.strength)
            else:
                signal_scores.append(0)

        if signal_scores:
            agreement = 1 - np.std(signal_scores)
            confidence = max(0, min(1, agreement))
        else:
            confidence = 0

        # Determine final signal
        if final_score > 0.5:
            final_signal = "STRONG_BUY"
        elif final_score > 0.2:
            final_signal = "BUY"
        elif final_score < -0.5:
            final_signal = "STRONG_SELL"
        elif final_score < -0.2:
            final_signal = "SELL"
        else:
            final_signal = "NEUTRAL"

        # Calculate position size (Kelly-lite)
        if final_signal in ["STRONG_BUY", "STRONG_SELL"]:
            suggested_size = min(0.05, abs(final_score) * 0.10)  # Max 5%
        elif final_signal in ["BUY", "SELL"]:
            suggested_size = min(0.03, abs(final_score) * 0.05)  # Max 3%
        else:
            suggested_size = 0

        return AggregatedSignal(
            symbol=symbol,
            final_signal=final_signal,
            final_score=final_score,
            confidence=confidence,
            contributing_signals=signals,
            regime=regime,
            timestamp=datetime.now(timezone.utc),
            correlation_contribution=contributions.get('correlation_contribution', 0),
            whale_contribution=contributions.get('whale_contribution', 0),
            funding_contribution=contributions.get('funding_contribution', 0),
            orderflow_contribution=contributions.get('orderflow_contribution', 0),
            ml_contribution=contributions.get('ml_contribution', 0),
            suggested_size=suggested_size
        )

    def _empty_signal(self, market_data: Dict) -> AggregatedSignal:
        """Return empty/neutral signal."""
        regime = self.regime_detector.detect(market_data)
        return AggregatedSignal(
            symbol="",
            final_signal="NEUTRAL",
            final_score=0,
            confidence=0,
            contributing_signals=[],
            regime=regime,
            timestamp=datetime.now(timezone.utc)
        )


# =============================================================================
# SIGNAL VALIDATION THRESHOLDS
# =============================================================================

class SignalThresholds:
    """Recommended thresholds for signal validation."""

    # Whale tracking
    WHALE_SUPPLY_PCT_3D = 0.005  # 0.5% of circulating supply over 3 days
    WHALE_SINGLE_TX_PCT = 0.001  # 0.1% of supply in single transfer

    # Funding rate
    FUNDING_EXTREME_8H = 0.0002  # 0.02% per 8-hour window
    FUNDING_EXTREME_DAILY = 0.0005  # 0.05% aggregate

    # Order book imbalance
    ORDERBOOK_IMBALANCE_STRONG = 2.0  # bid/ask ratio
    ORDERBOOK_IMBALANCE_WEAK = 0.5

    # Lead-lag significance
    LEADLAG_MIN_CORR = 0.2
    LEADLAG_MAX_PVALUE = 0.05

    # ML confidence
    ML_MIN_PROBABILITY = 0.65
    ML_MIN_ENSEMBLE_SCORE = 0.6

    # Signal quality
    MIN_PRECISION_30D = 0.45
    MIN_PROFIT_FACTOR = 1.2
    MIN_SAMPLE_SIZE = 30


# =============================================================================
# MAIN DEMO
# =============================================================================

def demo_infrastructure():
    """Demonstrate production infrastructure."""
    print("=" * 70)
    print("PRODUCTION INFRASTRUCTURE DEMO")
    print("=" * 70)
    print()

    # 1. Timestamp Canonicalization
    print("[1] TIMESTAMP CANONICALIZATION")
    print("-" * 50)

    ts = TimestampCanonicalizer()
    now = ts.utc_now()
    aligned = ts.align_to_candle(now)

    print(f"  Current UTC: {now.isoformat()}")
    print(f"  Aligned to daily: {aligned.isoformat()}")
    print()

    # 2. Circuit Breaker
    print("[2] CIRCUIT BREAKER STATUS")
    print("-" * 50)

    cb = CircuitBreaker()

    # Normal conditions
    market_data = {
        'btc_change_7d': -0.05,
        'volatility_ratio': 1.2,
        'data_staleness_minutes': 2,
        'active_data_sources': 3,
        'open_positions': 2,
        'daily_trades': 5,
        'portfolio_drawdown': 0.03
    }

    is_safe, reason = cb.check(market_data)
    print(f"  Normal market:")
    print(f"    Safe to trade: {is_safe}")
    print(f"    Reason: {reason}")

    # Crisis conditions
    market_data['btc_change_7d'] = -0.20
    is_safe, reason = cb.check(market_data)
    print(f"\n  Crisis market (BTC -20%):")
    print(f"    Safe to trade: {is_safe}")
    print(f"    Reason: {reason}")
    print()

    # 3. Signal Aggregation
    print("[3] SIGNAL AGGREGATION")
    print("-" * 50)

    # Reset market data
    market_data['btc_change_7d'] = 0.05
    market_data['btc_change_30d'] = 0.15
    market_data['trend_strength'] = 0.6

    aggregator = SignalAggregator()

    # Create sample signals
    signals = [
        SignalInput(
            source="whale",
            symbol="XRPUSDT",
            signal_type="BUY",
            strength=0.8,
            confidence=0.7,
            timestamp=now,
            explanation="Net exchange outflow 7M XRP"
        ),
        SignalInput(
            source="funding",
            symbol="XRPUSDT",
            signal_type="BUY",
            strength=0.6,
            confidence=0.6,
            timestamp=now,
            explanation="Negative funding rate -0.02%"
        ),
        SignalInput(
            source="orderflow",
            symbol="XRPUSDT",
            signal_type="BUY",
            strength=0.7,
            confidence=0.8,
            timestamp=now,
            explanation="Bid imbalance 1.8x"
        ),
        SignalInput(
            source="ml",
            symbol="XRPUSDT",
            signal_type="NEUTRAL",
            strength=0.5,
            confidence=0.5,
            timestamp=now,
            explanation="ML confidence below threshold"
        ),
    ]

    result = aggregator.aggregate(signals, market_data)

    print(f"  Symbol: {result.symbol}")
    print(f"  Regime: {result.regime}")
    print(f"  Final Signal: {result.final_signal}")
    print(f"  Final Score: {result.final_score:.2f}")
    print(f"  Confidence: {result.confidence:.1%}")
    print(f"  Suggested Size: {result.suggested_size:.1%}")
    print()
    print("  Contributions:")
    print(f"    Whale: {result.whale_contribution:.3f}")
    print(f"    Funding: {result.funding_contribution:.3f}")
    print(f"    Orderflow: {result.orderflow_contribution:.3f}")
    print(f"    ML: {result.ml_contribution:.3f}")
    print()

    # 4. Signal Thresholds
    print("[4] SIGNAL THRESHOLDS")
    print("-" * 50)

    print(f"  Whale supply % (3d): {SignalThresholds.WHALE_SUPPLY_PCT_3D:.2%}")
    print(f"  Funding extreme (8h): {SignalThresholds.FUNDING_EXTREME_8H:.3%}")
    print(f"  Orderbook imbalance: {SignalThresholds.ORDERBOOK_IMBALANCE_STRONG}x")
    print(f"  ML min probability: {SignalThresholds.ML_MIN_PROBABILITY:.0%}")
    print(f"  Min precision (30d): {SignalThresholds.MIN_PRECISION_30D:.0%}")
    print()

    # 5. Audit Table
    print("[5] SIGNAL AUDIT TABLE")
    print("-" * 50)

    audit = SignalAuditTable(db_path=":memory:")  # In-memory for demo

    # Record a signal
    features = {"whale_score": 0.8, "funding_rate": -0.02}
    record = SignalAuditRecord(
        id=SignalAuditTable.generate_signal_id(now, "XRPUSDT", "aggregator", features),
        timestamp=now,
        symbol="XRPUSDT",
        signal_source="aggregator",
        signal_type="BUY",
        signal_strength=result.final_score,
        confidence=result.confidence,
        features_hash=hashlib.sha256(json.dumps(features).encode()).hexdigest()[:8],
        features_json=json.dumps(features),
        price_at_signal=2.30,
        regime=result.regime
    )

    audit.record_signal(record)
    print(f"  Recorded signal: {record.id}")
    print(f"  Symbol: {record.symbol}")
    print(f"  Signal: {record.signal_type}")
    print(f"  Strength: {record.signal_strength:.2f}")
    print()

    print("=" * 70)
    print("Infrastructure Ready!")
    print("=" * 70)
    print()
    print("Components available:")
    print("  - TimestampCanonicalizer: UTC alignment")
    print("  - CircuitBreaker: Safety kill-switch")
    print("  - SignalAggregator: Regime-aware fusion")
    print("  - SignalAuditTable: Precision tracking")
    print("  - SignalThresholds: Validated thresholds")


if __name__ == "__main__":
    demo_infrastructure()
