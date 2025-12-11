#!/usr/bin/env python3
"""
Quant Database - Self-Learning Trading Brain

Comprehensive database layer for storing all signals, prices, predictions,
and trade outcomes. Enables cross-correlation analysis, regime detection,
signal scoring, and failure pattern identification.

6 Core Tables:
1. Price & Market Data - OHLCV, volume, market cap
2. Technical Signals - RSI, MACD, Bollinger, etc.
3. News & Event Signals - Sentiment, events
4. Machine Learning Predictions - Model outputs with confidence
5. Whale Activity - Large wallet movements, exchange flows
6. Trades & Outcomes - Entry/exit with P&L tracking

Author: Cloud AI Analyzer
"""

import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import statistics


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class SignalType(Enum):
    """Types of trading signals."""
    TECHNICAL = "technical"
    NEWS = "news"
    ML = "ml"
    WHALE = "whale"
    FUNDING = "funding"
    ORDERFLOW = "orderflow"
    CORRELATION = "correlation"


class SignalDirection(Enum):
    """Signal direction."""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    NEUTRAL = "NEUTRAL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


class TradeOutcome(Enum):
    """Trade outcome classification."""
    WIN = "WIN"
    LOSS = "LOSS"
    BREAKEVEN = "BREAKEVEN"
    OPEN = "OPEN"


class MarketRegime(Enum):
    """Market regime classification."""
    BULL = "BULL"
    BEAR = "BEAR"
    SIDEWAYS = "SIDEWAYS"
    VOLATILE = "VOLATILE"
    CRISIS = "CRISIS"


@dataclass
class PriceData:
    """Price and market data record."""
    symbol: str
    timestamp: str  # UTC ISO format
    open: float
    high: float
    low: float
    close: float
    volume: float
    market_cap: Optional[float] = None
    volume_24h: Optional[float] = None
    price_change_24h: Optional[float] = None
    source: str = "unknown"


@dataclass
class TechnicalSignal:
    """Technical analysis signal record."""
    symbol: str
    timestamp: str
    signal_name: str  # RSI, MACD, BB, etc.
    signal_value: float
    signal_direction: str  # STRONG_BUY, BUY, NEUTRAL, SELL, STRONG_SELL
    confidence: float  # 0.0 to 1.0
    timeframe: str  # 1h, 4h, 1d
    parameters: Optional[str] = None  # JSON string of indicator params


@dataclass
class NewsSignal:
    """News and event signal record."""
    symbol: str
    timestamp: str
    headline: str
    sentiment_score: float  # -1.0 to 1.0
    signal_direction: str
    confidence: float
    source: str
    category: str  # regulation, partnership, hack, etc.
    url: Optional[str] = None


@dataclass
class MLPrediction:
    """Machine learning prediction record."""
    symbol: str
    timestamp: str
    model_name: str
    predicted_direction: str
    predicted_return: float  # Expected % return
    confidence: float
    feature_importance: Optional[str] = None  # JSON string
    model_version: str = "1.0"
    validation_score: Optional[float] = None


@dataclass
class WhaleActivity:
    """Whale activity record."""
    symbol: str
    timestamp: str
    activity_type: str  # exchange_inflow, exchange_outflow, large_transfer
    amount_usd: float
    signal_direction: str
    confidence: float
    from_address: Optional[str] = None
    to_address: Optional[str] = None
    exchange: Optional[str] = None
    tx_hash: Optional[str] = None


@dataclass
class Trade:
    """Trade record with outcomes."""
    symbol: str
    entry_timestamp: str
    entry_price: float
    position_size: float
    direction: str  # LONG or SHORT
    exit_timestamp: Optional[str] = None
    exit_price: Optional[float] = None
    pnl_percent: Optional[float] = None
    pnl_usd: Optional[float] = None
    outcome: str = "OPEN"
    signals_used: Optional[str] = None  # JSON array of signal IDs
    market_regime: Optional[str] = None
    notes: Optional[str] = None


# =============================================================================
# QUANT DATABASE CLASS
# =============================================================================

class QuantDatabase:
    """
    Comprehensive quant database for signal storage, analysis, and learning.

    Features:
    - Store all signals, prices, predictions, and trades
    - Cross-correlation analysis between signals
    - Signal performance scoring (which signals are profitable)
    - Regime detection and analysis
    - Failure pattern identification
    - Time-series analytics
    """

    def __init__(self, db_path: str = "quant_signals.db"):
        """
        Initialize the quant database.

        Args:
            db_path: Path to SQLite database file (use ":memory:" for in-memory)
        """
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._init_database()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection (persistent for :memory:)."""
        if self.db_path == ":memory:":
            if self._conn is None:
                self._conn = sqlite3.connect(":memory:")
                self._conn.row_factory = sqlite3.Row
            return self._conn
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _close_connection(self, conn: sqlite3.Connection):
        """Close connection if not in-memory."""
        if self.db_path != ":memory:":
            conn.close()

    def _init_database(self):
        """Initialize all database tables."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # =================================================================
        # TABLE 1: Price & Market Data
        # =================================================================
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS price_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                market_cap REAL,
                volume_24h REAL,
                price_change_24h REAL,
                source TEXT DEFAULT 'unknown',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timestamp, source)
            )
        """)

        # Indexes for price data
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_price_symbol_time
            ON price_data(symbol, timestamp)
        """)

        # =================================================================
        # TABLE 2: Technical Signals
        # =================================================================
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS technical_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                signal_name TEXT NOT NULL,
                signal_value REAL NOT NULL,
                signal_direction TEXT NOT NULL,
                confidence REAL NOT NULL,
                timeframe TEXT NOT NULL,
                parameters TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_tech_symbol_time
            ON technical_signals(symbol, timestamp)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_tech_signal_name
            ON technical_signals(signal_name, signal_direction)
        """)

        # =================================================================
        # TABLE 3: News & Event Signals
        # =================================================================
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS news_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                headline TEXT NOT NULL,
                sentiment_score REAL NOT NULL,
                signal_direction TEXT NOT NULL,
                confidence REAL NOT NULL,
                source TEXT NOT NULL,
                category TEXT NOT NULL,
                url TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_news_symbol_time
            ON news_signals(symbol, timestamp)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_news_category
            ON news_signals(category, sentiment_score)
        """)

        # =================================================================
        # TABLE 4: Machine Learning Predictions
        # =================================================================
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ml_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                model_name TEXT NOT NULL,
                predicted_direction TEXT NOT NULL,
                predicted_return REAL NOT NULL,
                confidence REAL NOT NULL,
                feature_importance TEXT,
                model_version TEXT DEFAULT '1.0',
                validation_score REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ml_symbol_time
            ON ml_predictions(symbol, timestamp)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ml_model
            ON ml_predictions(model_name, model_version)
        """)

        # =================================================================
        # TABLE 5: Whale Activity
        # =================================================================
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS whale_activity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                activity_type TEXT NOT NULL,
                amount_usd REAL NOT NULL,
                signal_direction TEXT NOT NULL,
                confidence REAL NOT NULL,
                from_address TEXT,
                to_address TEXT,
                exchange TEXT,
                tx_hash TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(tx_hash)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_whale_symbol_time
            ON whale_activity(symbol, timestamp)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_whale_type
            ON whale_activity(activity_type, amount_usd)
        """)

        # =================================================================
        # TABLE 6: Trades & Outcomes
        # =================================================================
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                entry_timestamp TEXT NOT NULL,
                entry_price REAL NOT NULL,
                position_size REAL NOT NULL,
                direction TEXT NOT NULL,
                exit_timestamp TEXT,
                exit_price REAL,
                pnl_percent REAL,
                pnl_usd REAL,
                outcome TEXT DEFAULT 'OPEN',
                signals_used TEXT,
                market_regime TEXT,
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_trades_symbol
            ON trades(symbol, outcome)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_trades_regime
            ON trades(market_regime, outcome)
        """)

        # =================================================================
        # ANALYTICS TABLE: Signal Performance Tracking
        # =================================================================
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signal_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_type TEXT NOT NULL,
                signal_name TEXT NOT NULL,
                symbol TEXT NOT NULL,
                market_regime TEXT,
                total_signals INTEGER DEFAULT 0,
                correct_signals INTEGER DEFAULT 0,
                accuracy REAL DEFAULT 0.0,
                avg_return REAL DEFAULT 0.0,
                sharpe_ratio REAL,
                max_drawdown REAL,
                last_updated TEXT,
                UNIQUE(signal_type, signal_name, symbol, market_regime)
            )
        """)

        # =================================================================
        # ANALYTICS TABLE: Market Regime History
        # =================================================================
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_regimes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                regime TEXT NOT NULL,
                btc_price REAL,
                volatility REAL,
                correlation_avg REAL,
                confidence REAL DEFAULT 0.0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_regime_time
            ON market_regimes(timestamp)
        """)

        conn.commit()
        self._close_connection(conn)
        print(f"[QuantDB] Database initialized: {self.db_path}")

    # =========================================================================
    # INSERT METHODS
    # =========================================================================

    def insert_price(self, data: PriceData) -> int:
        """Insert price data record."""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT OR REPLACE INTO price_data
                (symbol, timestamp, open, high, low, close, volume,
                 market_cap, volume_24h, price_change_24h, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data.symbol, data.timestamp, data.open, data.high,
                data.low, data.close, data.volume, data.market_cap,
                data.volume_24h, data.price_change_24h, data.source
            ))
            conn.commit()
            return cursor.lastrowid
        finally:
            self._close_connection(conn)

    def insert_technical_signal(self, signal: TechnicalSignal) -> int:
        """Insert technical analysis signal."""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT INTO technical_signals
                (symbol, timestamp, signal_name, signal_value,
                 signal_direction, confidence, timeframe, parameters)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                signal.symbol, signal.timestamp, signal.signal_name,
                signal.signal_value, signal.signal_direction,
                signal.confidence, signal.timeframe, signal.parameters
            ))
            conn.commit()
            return cursor.lastrowid
        finally:
            self._close_connection(conn)

    def insert_news_signal(self, signal: NewsSignal) -> int:
        """Insert news/event signal."""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT INTO news_signals
                (symbol, timestamp, headline, sentiment_score,
                 signal_direction, confidence, source, category, url)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                signal.symbol, signal.timestamp, signal.headline,
                signal.sentiment_score, signal.signal_direction,
                signal.confidence, signal.source, signal.category, signal.url
            ))
            conn.commit()
            return cursor.lastrowid
        finally:
            self._close_connection(conn)

    def insert_ml_prediction(self, pred: MLPrediction) -> int:
        """Insert ML prediction."""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT INTO ml_predictions
                (symbol, timestamp, model_name, predicted_direction,
                 predicted_return, confidence, feature_importance,
                 model_version, validation_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pred.symbol, pred.timestamp, pred.model_name,
                pred.predicted_direction, pred.predicted_return,
                pred.confidence, pred.feature_importance,
                pred.model_version, pred.validation_score
            ))
            conn.commit()
            return cursor.lastrowid
        finally:
            self._close_connection(conn)

    def insert_whale_activity(self, activity: WhaleActivity) -> int:
        """Insert whale activity record."""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT OR IGNORE INTO whale_activity
                (symbol, timestamp, activity_type, amount_usd,
                 signal_direction, confidence, from_address,
                 to_address, exchange, tx_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                activity.symbol, activity.timestamp, activity.activity_type,
                activity.amount_usd, activity.signal_direction,
                activity.confidence, activity.from_address,
                activity.to_address, activity.exchange, activity.tx_hash
            ))
            conn.commit()
            return cursor.lastrowid
        finally:
            self._close_connection(conn)

    def insert_trade(self, trade: Trade) -> int:
        """Insert trade record."""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT INTO trades
                (symbol, entry_timestamp, entry_price, position_size,
                 direction, exit_timestamp, exit_price, pnl_percent,
                 pnl_usd, outcome, signals_used, market_regime, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade.symbol, trade.entry_timestamp, trade.entry_price,
                trade.position_size, trade.direction, trade.exit_timestamp,
                trade.exit_price, trade.pnl_percent, trade.pnl_usd,
                trade.outcome, trade.signals_used, trade.market_regime,
                trade.notes
            ))
            conn.commit()
            return cursor.lastrowid
        finally:
            self._close_connection(conn)

    def close_trade(self, trade_id: int, exit_price: float,
                    exit_timestamp: Optional[str] = None) -> bool:
        """Close an open trade and calculate P&L."""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # Get the trade
            cursor.execute("SELECT * FROM trades WHERE id = ?", (trade_id,))
            row = cursor.fetchone()
            if not row:
                return False

            entry_price = row['entry_price']
            direction = row['direction']
            position_size = row['position_size']

            # Calculate P&L
            if direction == "LONG":
                pnl_percent = ((exit_price - entry_price) / entry_price) * 100
            else:  # SHORT
                pnl_percent = ((entry_price - exit_price) / entry_price) * 100

            pnl_usd = position_size * (pnl_percent / 100)

            # Determine outcome
            if pnl_percent > 0.5:
                outcome = "WIN"
            elif pnl_percent < -0.5:
                outcome = "LOSS"
            else:
                outcome = "BREAKEVEN"

            exit_ts = exit_timestamp or datetime.utcnow().isoformat() + "Z"

            cursor.execute("""
                UPDATE trades
                SET exit_timestamp = ?, exit_price = ?, pnl_percent = ?,
                    pnl_usd = ?, outcome = ?
                WHERE id = ?
            """, (exit_ts, exit_price, pnl_percent, pnl_usd, outcome, trade_id))

            conn.commit()
            return True
        finally:
            self._close_connection(conn)

    def record_market_regime(self, regime: str, btc_price: float,
                            volatility: float, correlation_avg: float,
                            confidence: float = 0.8) -> int:
        """Record current market regime."""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            timestamp = datetime.utcnow().isoformat() + "Z"
            cursor.execute("""
                INSERT INTO market_regimes
                (timestamp, regime, btc_price, volatility,
                 correlation_avg, confidence)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (timestamp, regime, btc_price, volatility,
                  correlation_avg, confidence))
            conn.commit()
            return cursor.lastrowid
        finally:
            self._close_connection(conn)

    # =========================================================================
    # QUERY METHODS
    # =========================================================================

    def get_prices(self, symbol: str, start_time: Optional[str] = None,
                   end_time: Optional[str] = None, limit: int = 1000) -> List[Dict]:
        """Get price data for a symbol."""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            query = "SELECT * FROM price_data WHERE symbol = ?"
            params = [symbol]

            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
        finally:
            self._close_connection(conn)

    def get_signals_for_symbol(self, symbol: str,
                               signal_types: Optional[List[str]] = None,
                               hours_back: int = 24) -> Dict[str, List[Dict]]:
        """Get all signals for a symbol within time window."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cutoff = (datetime.utcnow() - timedelta(hours=hours_back)).isoformat() + "Z"
        result = {}

        try:
            # Technical signals
            if not signal_types or "technical" in signal_types:
                cursor.execute("""
                    SELECT * FROM technical_signals
                    WHERE symbol = ? AND timestamp >= ?
                    ORDER BY timestamp DESC
                """, (symbol, cutoff))
                result["technical"] = [dict(row) for row in cursor.fetchall()]

            # News signals
            if not signal_types or "news" in signal_types:
                cursor.execute("""
                    SELECT * FROM news_signals
                    WHERE symbol = ? AND timestamp >= ?
                    ORDER BY timestamp DESC
                """, (symbol, cutoff))
                result["news"] = [dict(row) for row in cursor.fetchall()]

            # ML predictions
            if not signal_types or "ml" in signal_types:
                cursor.execute("""
                    SELECT * FROM ml_predictions
                    WHERE symbol = ? AND timestamp >= ?
                    ORDER BY timestamp DESC
                """, (symbol, cutoff))
                result["ml"] = [dict(row) for row in cursor.fetchall()]

            # Whale activity
            if not signal_types or "whale" in signal_types:
                cursor.execute("""
                    SELECT * FROM whale_activity
                    WHERE symbol = ? AND timestamp >= ?
                    ORDER BY timestamp DESC
                """, (symbol, cutoff))
                result["whale"] = [dict(row) for row in cursor.fetchall()]

            return result
        finally:
            self._close_connection(conn)

    def get_open_trades(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get all open trades."""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            if symbol:
                cursor.execute("""
                    SELECT * FROM trades
                    WHERE outcome = 'OPEN' AND symbol = ?
                    ORDER BY entry_timestamp DESC
                """, (symbol,))
            else:
                cursor.execute("""
                    SELECT * FROM trades WHERE outcome = 'OPEN'
                    ORDER BY entry_timestamp DESC
                """)
            return [dict(row) for row in cursor.fetchall()]
        finally:
            self._close_connection(conn)

    def get_current_regime(self) -> Optional[Dict]:
        """Get the most recent market regime."""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT * FROM market_regimes
                ORDER BY timestamp DESC LIMIT 1
            """)
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            self._close_connection(conn)

    # =========================================================================
    # ANALYTICS METHODS
    # =========================================================================

    def calculate_signal_accuracy(self, signal_type: str, signal_name: str,
                                  symbol: Optional[str] = None,
                                  days_back: int = 30) -> Dict[str, Any]:
        """
        Calculate accuracy of a specific signal type.

        Compares signal predictions against actual price movements.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cutoff = (datetime.utcnow() - timedelta(days=days_back)).isoformat() + "Z"

        try:
            # Get signals based on type
            if signal_type == "technical":
                query = """
                    SELECT ts.*, pd.close as signal_price
                    FROM technical_signals ts
                    LEFT JOIN price_data pd ON ts.symbol = pd.symbol
                        AND DATE(ts.timestamp) = DATE(pd.timestamp)
                    WHERE ts.signal_name = ? AND ts.timestamp >= ?
                """
                params = [signal_name, cutoff]
            elif signal_type == "ml":
                query = """
                    SELECT mp.*, pd.close as signal_price
                    FROM ml_predictions mp
                    LEFT JOIN price_data pd ON mp.symbol = pd.symbol
                        AND DATE(mp.timestamp) = DATE(pd.timestamp)
                    WHERE mp.model_name = ? AND mp.timestamp >= ?
                """
                params = [signal_name, cutoff]
            else:
                return {"error": f"Unknown signal type: {signal_type}"}

            if symbol:
                query = query.replace("WHERE", f"WHERE {'ts' if signal_type == 'technical' else 'mp'}.symbol = '{symbol}' AND")

            cursor.execute(query, params)
            signals = cursor.fetchall()

            if not signals:
                return {
                    "signal_type": signal_type,
                    "signal_name": signal_name,
                    "total_signals": 0,
                    "accuracy": None,
                    "message": "No signals found"
                }

            # Calculate accuracy by checking if direction matched price movement
            correct = 0
            total = 0
            returns = []

            for signal in signals:
                signal_dict = dict(signal)
                signal_symbol = signal_dict['symbol']
                signal_time = signal_dict['timestamp']

                if signal_type == "technical":
                    direction = signal_dict['signal_direction']
                else:
                    direction = signal_dict['predicted_direction']

                # Get price 24h later
                cursor.execute("""
                    SELECT close FROM price_data
                    WHERE symbol = ? AND timestamp > ?
                    ORDER BY timestamp ASC LIMIT 1
                """, (signal_symbol, signal_time))

                future_row = cursor.fetchone()
                if not future_row or not signal_dict.get('signal_price'):
                    continue

                future_price = future_row['close']
                signal_price = signal_dict['signal_price']
                price_return = ((future_price - signal_price) / signal_price) * 100
                returns.append(price_return)

                # Check if signal was correct
                total += 1
                if direction in ["STRONG_BUY", "BUY"] and price_return > 0:
                    correct += 1
                elif direction in ["STRONG_SELL", "SELL"] and price_return < 0:
                    correct += 1
                elif direction == "NEUTRAL" and abs(price_return) < 1:
                    correct += 1

            accuracy = (correct / total * 100) if total > 0 else None
            avg_return = statistics.mean(returns) if returns else None

            return {
                "signal_type": signal_type,
                "signal_name": signal_name,
                "symbol": symbol or "ALL",
                "total_signals": total,
                "correct_signals": correct,
                "accuracy": round(accuracy, 2) if accuracy else None,
                "avg_return": round(avg_return, 4) if avg_return else None,
                "days_analyzed": days_back
            }
        finally:
            self._close_connection(conn)

    def get_signal_performance_by_regime(self,
                                         signal_type: str,
                                         days_back: int = 90) -> Dict[str, Dict]:
        """
        Analyze signal performance broken down by market regime.

        Returns accuracy and returns for each regime.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cutoff = (datetime.utcnow() - timedelta(days=days_back)).isoformat() + "Z"

        try:
            # Get trades with their market regimes
            cursor.execute("""
                SELECT market_regime, outcome, pnl_percent, signals_used
                FROM trades
                WHERE entry_timestamp >= ? AND outcome != 'OPEN'
                AND signals_used LIKE ?
            """, (cutoff, f'%{signal_type}%'))

            trades = cursor.fetchall()

            regime_stats = {}
            for trade in trades:
                regime = trade['market_regime'] or 'UNKNOWN'
                if regime not in regime_stats:
                    regime_stats[regime] = {
                        "total_trades": 0,
                        "wins": 0,
                        "losses": 0,
                        "total_pnl": 0.0,
                        "returns": []
                    }

                regime_stats[regime]["total_trades"] += 1
                if trade['outcome'] == 'WIN':
                    regime_stats[regime]["wins"] += 1
                elif trade['outcome'] == 'LOSS':
                    regime_stats[regime]["losses"] += 1

                if trade['pnl_percent']:
                    regime_stats[regime]["total_pnl"] += trade['pnl_percent']
                    regime_stats[regime]["returns"].append(trade['pnl_percent'])

            # Calculate win rates and averages
            for regime, stats in regime_stats.items():
                total = stats["total_trades"]
                if total > 0:
                    stats["win_rate"] = round(stats["wins"] / total * 100, 2)
                    stats["avg_pnl"] = round(stats["total_pnl"] / total, 4)
                    if len(stats["returns"]) > 1:
                        stats["sharpe"] = round(
                            statistics.mean(stats["returns"]) /
                            statistics.stdev(stats["returns"]) if statistics.stdev(stats["returns"]) > 0 else 0,
                            2
                        )
                    else:
                        stats["sharpe"] = None
                del stats["returns"]  # Remove raw data from output

            return regime_stats
        finally:
            self._close_connection(conn)

    def find_signal_correlations(self, days_back: int = 30) -> List[Dict]:
        """
        Find correlations between different signal types.

        Identifies which signals tend to appear together and their
        combined effectiveness.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cutoff = (datetime.utcnow() - timedelta(days=days_back)).isoformat() + "Z"

        try:
            # Get all signals grouped by symbol and date
            cursor.execute("""
                SELECT
                    symbol,
                    DATE(timestamp) as signal_date,
                    'technical' as signal_type,
                    signal_name,
                    signal_direction
                FROM technical_signals
                WHERE timestamp >= ?
                UNION ALL
                SELECT
                    symbol,
                    DATE(timestamp) as signal_date,
                    'ml' as signal_type,
                    model_name as signal_name,
                    predicted_direction as signal_direction
                FROM ml_predictions
                WHERE timestamp >= ?
                UNION ALL
                SELECT
                    symbol,
                    DATE(timestamp) as signal_date,
                    'whale' as signal_type,
                    activity_type as signal_name,
                    signal_direction
                FROM whale_activity
                WHERE timestamp >= ?
                ORDER BY symbol, signal_date
            """, (cutoff, cutoff, cutoff))

            signals = cursor.fetchall()

            # Group signals by symbol+date
            signal_groups = {}
            for sig in signals:
                key = f"{sig['symbol']}_{sig['signal_date']}"
                if key not in signal_groups:
                    signal_groups[key] = []
                signal_groups[key].append({
                    "type": sig['signal_type'],
                    "name": sig['signal_name'],
                    "direction": sig['signal_direction']
                })

            # Find co-occurrences
            co_occurrences = {}
            for key, group in signal_groups.items():
                if len(group) >= 2:
                    # Create pairs
                    for i in range(len(group)):
                        for j in range(i + 1, len(group)):
                            sig1 = f"{group[i]['type']}:{group[i]['name']}"
                            sig2 = f"{group[j]['type']}:{group[j]['name']}"
                            pair = tuple(sorted([sig1, sig2]))

                            if pair not in co_occurrences:
                                co_occurrences[pair] = {
                                    "signal_1": pair[0],
                                    "signal_2": pair[1],
                                    "count": 0,
                                    "same_direction": 0
                                }

                            co_occurrences[pair]["count"] += 1
                            if group[i]['direction'] == group[j]['direction']:
                                co_occurrences[pair]["same_direction"] += 1

            # Format results
            results = []
            for pair, data in co_occurrences.items():
                if data["count"] >= 3:  # Minimum occurrences
                    data["agreement_rate"] = round(
                        data["same_direction"] / data["count"] * 100, 2
                    )
                    results.append(data)

            # Sort by count
            results.sort(key=lambda x: x["count"], reverse=True)
            return results[:20]  # Top 20 correlations
        finally:
            self._close_connection(conn)

    def detect_failure_patterns(self, days_back: int = 60) -> Dict[str, Any]:
        """
        Identify patterns in losing trades.

        Analyzes what conditions lead to losses:
        - Which signals preceded losses
        - Which regimes had most losses
        - Time-of-day patterns
        - Signal combination patterns
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cutoff = (datetime.utcnow() - timedelta(days=days_back)).isoformat() + "Z"

        try:
            # Get losing trades
            cursor.execute("""
                SELECT * FROM trades
                WHERE entry_timestamp >= ? AND outcome = 'LOSS'
                ORDER BY pnl_percent ASC
            """, (cutoff,))

            losing_trades = [dict(row) for row in cursor.fetchall()]

            if not losing_trades:
                return {"message": "No losing trades found", "patterns": []}

            # Analyze patterns
            patterns = {
                "by_regime": {},
                "by_symbol": {},
                "by_signal_type": {},
                "worst_losses": [],
                "common_signals": {}
            }

            for trade in losing_trades:
                # By regime
                regime = trade['market_regime'] or 'UNKNOWN'
                if regime not in patterns["by_regime"]:
                    patterns["by_regime"][regime] = {"count": 0, "total_loss": 0}
                patterns["by_regime"][regime]["count"] += 1
                patterns["by_regime"][regime]["total_loss"] += trade['pnl_percent'] or 0

                # By symbol
                symbol = trade['symbol']
                if symbol not in patterns["by_symbol"]:
                    patterns["by_symbol"][symbol] = {"count": 0, "total_loss": 0}
                patterns["by_symbol"][symbol]["count"] += 1
                patterns["by_symbol"][symbol]["total_loss"] += trade['pnl_percent'] or 0

                # By signal type
                if trade['signals_used']:
                    try:
                        signals = json.loads(trade['signals_used'])
                        for sig in signals:
                            sig_type = sig.get('type', 'unknown')
                            if sig_type not in patterns["by_signal_type"]:
                                patterns["by_signal_type"][sig_type] = {"count": 0, "total_loss": 0}
                            patterns["by_signal_type"][sig_type]["count"] += 1
                            patterns["by_signal_type"][sig_type]["total_loss"] += trade['pnl_percent'] or 0
                    except json.JSONDecodeError:
                        pass

            # Worst 5 losses
            patterns["worst_losses"] = [
                {
                    "symbol": t['symbol'],
                    "pnl_percent": t['pnl_percent'],
                    "regime": t['market_regime'],
                    "entry_time": t['entry_timestamp']
                }
                for t in losing_trades[:5]
            ]

            # Calculate averages
            for regime, data in patterns["by_regime"].items():
                data["avg_loss"] = round(data["total_loss"] / data["count"], 4)

            for symbol, data in patterns["by_symbol"].items():
                data["avg_loss"] = round(data["total_loss"] / data["count"], 4)

            for sig_type, data in patterns["by_signal_type"].items():
                data["avg_loss"] = round(data["total_loss"] / data["count"], 4)

            return {
                "total_losing_trades": len(losing_trades),
                "total_loss_percent": round(sum(t['pnl_percent'] or 0 for t in losing_trades), 4),
                "patterns": patterns,
                "recommendations": self._generate_failure_recommendations(patterns)
            }
        finally:
            self._close_connection(conn)

    def _generate_failure_recommendations(self, patterns: Dict) -> List[str]:
        """Generate actionable recommendations from failure patterns."""
        recommendations = []

        # Find worst regime
        if patterns["by_regime"]:
            worst_regime = min(
                patterns["by_regime"].items(),
                key=lambda x: x[1]["avg_loss"]
            )
            if worst_regime[1]["count"] >= 3:
                recommendations.append(
                    f"REDUCE position size in {worst_regime[0]} regime "
                    f"(avg loss: {worst_regime[1]['avg_loss']:.2f}%)"
                )

        # Find worst symbol
        if patterns["by_symbol"]:
            worst_symbol = min(
                patterns["by_symbol"].items(),
                key=lambda x: x[1]["avg_loss"]
            )
            if worst_symbol[1]["count"] >= 3:
                recommendations.append(
                    f"REVIEW {worst_symbol[0]} trades - high loss rate "
                    f"({worst_symbol[1]['count']} losses, "
                    f"avg: {worst_symbol[1]['avg_loss']:.2f}%)"
                )

        # Find worst signal type
        if patterns["by_signal_type"]:
            worst_signal = min(
                patterns["by_signal_type"].items(),
                key=lambda x: x[1]["avg_loss"]
            )
            if worst_signal[1]["count"] >= 3:
                recommendations.append(
                    f"LOWER weight for {worst_signal[0]} signals "
                    f"(associated with {worst_signal[1]['count']} losses)"
                )

        if not recommendations:
            recommendations.append("Insufficient data to generate recommendations")

        return recommendations

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio and performance summary."""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # Trade statistics
            cursor.execute("""
                SELECT
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN outcome = 'LOSS' THEN 1 ELSE 0 END) as losses,
                    SUM(CASE WHEN outcome = 'OPEN' THEN 1 ELSE 0 END) as open_trades,
                    SUM(pnl_usd) as total_pnl_usd,
                    AVG(pnl_percent) as avg_pnl_percent
                FROM trades
            """)
            trade_stats = dict(cursor.fetchone())

            # Current regime
            cursor.execute("""
                SELECT regime, confidence FROM market_regimes
                ORDER BY timestamp DESC LIMIT 1
            """)
            regime_row = cursor.fetchone()
            current_regime = dict(regime_row) if regime_row else {"regime": "UNKNOWN"}

            # Signals summary (last 24h)
            cutoff = (datetime.utcnow() - timedelta(hours=24)).isoformat() + "Z"

            cursor.execute("""
                SELECT COUNT(*) as count FROM technical_signals
                WHERE timestamp >= ?
            """, (cutoff,))
            tech_count = cursor.fetchone()['count']

            cursor.execute("""
                SELECT COUNT(*) as count FROM ml_predictions
                WHERE timestamp >= ?
            """, (cutoff,))
            ml_count = cursor.fetchone()['count']

            cursor.execute("""
                SELECT COUNT(*) as count FROM whale_activity
                WHERE timestamp >= ?
            """, (cutoff,))
            whale_count = cursor.fetchone()['count']

            # Win rate by symbol
            cursor.execute("""
                SELECT
                    symbol,
                    COUNT(*) as trades,
                    SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
                    SUM(pnl_percent) as total_pnl
                FROM trades
                WHERE outcome != 'OPEN'
                GROUP BY symbol
                ORDER BY total_pnl DESC
            """)
            symbol_performance = [dict(row) for row in cursor.fetchall()]

            for sp in symbol_performance:
                sp["win_rate"] = round(sp["wins"] / sp["trades"] * 100, 2) if sp["trades"] > 0 else 0

            return {
                "trade_statistics": {
                    "total_trades": trade_stats["total_trades"],
                    "wins": trade_stats["wins"],
                    "losses": trade_stats["losses"],
                    "open_positions": trade_stats["open_trades"],
                    "win_rate": round(
                        trade_stats["wins"] /
                        (trade_stats["wins"] + trade_stats["losses"]) * 100, 2
                    ) if (trade_stats["wins"] + trade_stats["losses"]) > 0 else None,
                    "total_pnl_usd": round(trade_stats["total_pnl_usd"] or 0, 2),
                    "avg_pnl_percent": round(trade_stats["avg_pnl_percent"] or 0, 4)
                },
                "current_regime": current_regime,
                "signals_24h": {
                    "technical": tech_count,
                    "ml_predictions": ml_count,
                    "whale_activity": whale_count,
                    "total": tech_count + ml_count + whale_count
                },
                "symbol_performance": symbol_performance[:10]
            }
        finally:
            self._close_connection(conn)

    # =========================================================================
    # INTEGRATION METHODS (Connect with existing tools)
    # =========================================================================

    def store_correlation_analysis(self, symbol: str,
                                   correlation_data: Dict) -> int:
        """Store correlation analyzer results as technical signal."""
        signal = TechnicalSignal(
            symbol=symbol,
            timestamp=datetime.utcnow().isoformat() + "Z",
            signal_name="correlation",
            signal_value=correlation_data.get("correlation_coefficient", 0),
            signal_direction=correlation_data.get("signal", "NEUTRAL"),
            confidence=abs(correlation_data.get("correlation_coefficient", 0)),
            timeframe="1d",
            parameters=json.dumps(correlation_data)
        )
        return self.insert_technical_signal(signal)

    def store_funding_rate(self, symbol: str,
                          funding_data: Dict) -> int:
        """Store funding rate analyzer results as technical signal."""
        signal = TechnicalSignal(
            symbol=symbol,
            timestamp=datetime.utcnow().isoformat() + "Z",
            signal_name="funding_rate",
            signal_value=funding_data.get("rate", 0),
            signal_direction=funding_data.get("signal", "NEUTRAL"),
            confidence=funding_data.get("confidence", 0.5),
            timeframe="8h",
            parameters=json.dumps(funding_data)
        )
        return self.insert_technical_signal(signal)

    def store_orderflow_signal(self, symbol: str,
                              orderflow_data: Dict) -> int:
        """Store order flow analyzer results as technical signal."""
        signal = TechnicalSignal(
            symbol=symbol,
            timestamp=datetime.utcnow().isoformat() + "Z",
            signal_name="orderflow",
            signal_value=orderflow_data.get("imbalance_ratio", 0),
            signal_direction=orderflow_data.get("signal", "NEUTRAL"),
            confidence=orderflow_data.get("confidence", 0.5),
            timeframe="1h",
            parameters=json.dumps(orderflow_data)
        )
        return self.insert_technical_signal(signal)

    def store_ml_signal(self, symbol: str, model_name: str,
                       ml_result: Dict) -> int:
        """Store ML signal generator results."""
        pred = MLPrediction(
            symbol=symbol,
            timestamp=datetime.utcnow().isoformat() + "Z",
            model_name=model_name,
            predicted_direction=ml_result.get("direction", "NEUTRAL"),
            predicted_return=ml_result.get("predicted_return", 0),
            confidence=ml_result.get("confidence", 0.5),
            feature_importance=json.dumps(ml_result.get("feature_importance")),
            model_version=ml_result.get("version", "1.0"),
            validation_score=ml_result.get("validation_score")
        )
        return self.insert_ml_prediction(pred)

    def store_whale_signal(self, symbol: str,
                          whale_data: Dict) -> int:
        """Store whale tracker results."""
        activity = WhaleActivity(
            symbol=symbol,
            timestamp=datetime.utcnow().isoformat() + "Z",
            activity_type=whale_data.get("type", "large_transfer"),
            amount_usd=whale_data.get("amount_usd", 0),
            signal_direction=whale_data.get("signal", "NEUTRAL"),
            confidence=whale_data.get("confidence", 0.5),
            from_address=whale_data.get("from_address"),
            to_address=whale_data.get("to_address"),
            exchange=whale_data.get("exchange"),
            tx_hash=whale_data.get("tx_hash")
        )
        return self.insert_whale_activity(activity)

    def get_aggregated_signals(self, symbol: str,
                               hours_back: int = 24) -> Dict[str, Any]:
        """
        Get aggregated view of all signals for a symbol.

        Returns a weighted consensus signal based on all available
        signal types within the time window.
        """
        signals = self.get_signals_for_symbol(symbol, hours_back=hours_back)

        # Weight configuration by regime
        current_regime = self.get_current_regime()
        regime = current_regime.get("regime", "SIDEWAYS") if current_regime else "SIDEWAYS"

        weights = {
            "BULL": {"technical": 0.25, "ml": 0.25, "whale": 0.25, "news": 0.25},
            "BEAR": {"technical": 0.20, "ml": 0.20, "whale": 0.35, "news": 0.25},
            "SIDEWAYS": {"technical": 0.30, "ml": 0.30, "whale": 0.20, "news": 0.20},
            "VOLATILE": {"technical": 0.15, "ml": 0.25, "whale": 0.35, "news": 0.25},
            "CRISIS": {"technical": 0.10, "ml": 0.15, "whale": 0.40, "news": 0.35}
        }.get(regime, {"technical": 0.25, "ml": 0.25, "whale": 0.25, "news": 0.25})

        # Calculate weighted score for each signal type
        direction_scores = {
            "STRONG_BUY": 1.0,
            "BUY": 0.5,
            "NEUTRAL": 0.0,
            "SELL": -0.5,
            "STRONG_SELL": -1.0
        }

        signal_scores = {}

        # Technical signals
        if signals.get("technical"):
            tech_scores = []
            for sig in signals["technical"]:
                score = direction_scores.get(sig["signal_direction"], 0)
                tech_scores.append(score * sig["confidence"])
            if tech_scores:
                signal_scores["technical"] = statistics.mean(tech_scores)

        # ML predictions
        if signals.get("ml"):
            ml_scores = []
            for pred in signals["ml"]:
                score = direction_scores.get(pred["predicted_direction"], 0)
                ml_scores.append(score * pred["confidence"])
            if ml_scores:
                signal_scores["ml"] = statistics.mean(ml_scores)

        # Whale activity
        if signals.get("whale"):
            whale_scores = []
            for act in signals["whale"]:
                score = direction_scores.get(act["signal_direction"], 0)
                whale_scores.append(score * act["confidence"])
            if whale_scores:
                signal_scores["whale"] = statistics.mean(whale_scores)

        # News signals
        if signals.get("news"):
            news_scores = []
            for news in signals["news"]:
                score = direction_scores.get(news["signal_direction"], 0)
                news_scores.append(score * news["confidence"])
            if news_scores:
                signal_scores["news"] = statistics.mean(news_scores)

        # Calculate weighted consensus
        if not signal_scores:
            return {
                "symbol": symbol,
                "consensus_score": 0,
                "consensus_direction": "NEUTRAL",
                "confidence": 0,
                "regime": regime,
                "signal_count": 0
            }

        weighted_sum = 0
        total_weight = 0

        for sig_type, score in signal_scores.items():
            weight = weights.get(sig_type, 0.25)
            weighted_sum += score * weight
            total_weight += weight

        consensus_score = weighted_sum / total_weight if total_weight > 0 else 0

        # Determine direction
        if consensus_score >= 0.5:
            direction = "STRONG_BUY"
        elif consensus_score >= 0.2:
            direction = "BUY"
        elif consensus_score <= -0.5:
            direction = "STRONG_SELL"
        elif consensus_score <= -0.2:
            direction = "SELL"
        else:
            direction = "NEUTRAL"

        total_signals = sum(len(s) for s in signals.values())

        return {
            "symbol": symbol,
            "consensus_score": round(consensus_score, 4),
            "consensus_direction": direction,
            "confidence": min(abs(consensus_score) * 1.5, 1.0),
            "regime": regime,
            "signal_breakdown": signal_scores,
            "signal_count": total_signals,
            "weights_used": weights
        }

    # =========================================================================
    # DATABASE MAINTENANCE
    # =========================================================================

    def vacuum_old_data(self, days_to_keep: int = 90):
        """Remove old data to keep database size manageable."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cutoff = (datetime.utcnow() - timedelta(days=days_to_keep)).isoformat() + "Z"

        try:
            tables = [
                "price_data", "technical_signals", "news_signals",
                "ml_predictions", "whale_activity", "market_regimes"
            ]

            deleted = {}
            for table in tables:
                cursor.execute(f"""
                    DELETE FROM {table} WHERE timestamp < ?
                """, (cutoff,))
                deleted[table] = cursor.rowcount

            # Keep all trades for historical analysis

            conn.commit()
            cursor.execute("VACUUM")

            print(f"[QuantDB] Cleaned up data older than {days_to_keep} days")
            return deleted
        finally:
            self._close_connection(conn)

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            stats = {}

            tables = [
                ("price_data", "Price Records"),
                ("technical_signals", "Technical Signals"),
                ("news_signals", "News Signals"),
                ("ml_predictions", "ML Predictions"),
                ("whale_activity", "Whale Activities"),
                ("trades", "Trades"),
                ("market_regimes", "Regime Records")
            ]

            for table, name in tables:
                cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
                stats[name] = cursor.fetchone()['count']

            # Get date range
            cursor.execute("""
                SELECT MIN(timestamp) as earliest, MAX(timestamp) as latest
                FROM price_data
            """)
            date_range = cursor.fetchone()

            return {
                "record_counts": stats,
                "date_range": {
                    "earliest": date_range['earliest'],
                    "latest": date_range['latest']
                },
                "database_path": self.db_path
            }
        finally:
            self._close_connection(conn)


# =============================================================================
# DEMO AND TESTING
# =============================================================================

def demo():
    """Demonstrate database capabilities."""
    print("=" * 70)
    print("QUANT DATABASE - Self-Learning Trading Brain")
    print("=" * 70)

    # Initialize database
    db = QuantDatabase(":memory:")  # Use in-memory for demo

    # Insert sample price data
    print("\n[1] Inserting sample price data...")
    for i in range(5):
        price = PriceData(
            symbol="BTCUSDT",
            timestamp=(datetime.utcnow() - timedelta(days=i)).isoformat() + "Z",
            open=50000 + i * 100,
            high=51000 + i * 100,
            low=49000 + i * 100,
            close=50500 + i * 100,
            volume=1000000000,
            source="binance"
        )
        db.insert_price(price)
    print("  ✓ Inserted 5 price records")

    # Insert technical signals
    print("\n[2] Inserting technical signals...")
    signals_data = [
        ("BTCUSDT", "RSI", 35, "BUY", 0.7),
        ("BTCUSDT", "MACD", 0.5, "BUY", 0.65),
        ("ETHUSDT", "RSI", 72, "SELL", 0.75),
        ("XRPUSDT", "BB", -0.8, "STRONG_BUY", 0.8),
    ]

    for symbol, name, value, direction, conf in signals_data:
        signal = TechnicalSignal(
            symbol=symbol,
            timestamp=datetime.utcnow().isoformat() + "Z",
            signal_name=name,
            signal_value=value,
            signal_direction=direction,
            confidence=conf,
            timeframe="4h"
        )
        db.insert_technical_signal(signal)
    print(f"  ✓ Inserted {len(signals_data)} technical signals")

    # Insert ML predictions
    print("\n[3] Inserting ML predictions...")
    ml_data = [
        ("BTCUSDT", "RandomForest", "BUY", 2.5, 0.72),
        ("ETHUSDT", "GradientBoosting", "SELL", -1.8, 0.68),
        ("XRPUSDT", "Ensemble", "STRONG_BUY", 5.2, 0.81),
    ]

    for symbol, model, direction, ret, conf in ml_data:
        pred = MLPrediction(
            symbol=symbol,
            timestamp=datetime.utcnow().isoformat() + "Z",
            model_name=model,
            predicted_direction=direction,
            predicted_return=ret,
            confidence=conf
        )
        db.insert_ml_prediction(pred)
    print(f"  ✓ Inserted {len(ml_data)} ML predictions")

    # Insert whale activity
    print("\n[4] Inserting whale activity...")
    whale_data = [
        ("BTCUSDT", "exchange_outflow", 50000000, "BUY", 0.75),
        ("ETHUSDT", "exchange_inflow", 25000000, "SELL", 0.7),
        ("XRPUSDT", "large_transfer", 100000000, "NEUTRAL", 0.5),
    ]

    for i, (symbol, atype, amount, direction, conf) in enumerate(whale_data):
        activity = WhaleActivity(
            symbol=symbol,
            timestamp=datetime.utcnow().isoformat() + "Z",
            activity_type=atype,
            amount_usd=amount,
            signal_direction=direction,
            confidence=conf,
            tx_hash=f"0xdemo{i}"
        )
        db.insert_whale_activity(activity)
    print(f"  ✓ Inserted {len(whale_data)} whale activities")

    # Insert trades
    print("\n[5] Inserting sample trades...")
    trade = Trade(
        symbol="BTCUSDT",
        entry_timestamp=(datetime.utcnow() - timedelta(hours=24)).isoformat() + "Z",
        entry_price=50000,
        position_size=1000,
        direction="LONG",
        signals_used=json.dumps([
            {"type": "technical", "name": "RSI", "direction": "BUY"},
            {"type": "ml", "name": "RandomForest", "direction": "BUY"}
        ]),
        market_regime="BULL"
    )
    trade_id = db.insert_trade(trade)
    print(f"  ✓ Opened trade #{trade_id}")

    # Close the trade
    db.close_trade(trade_id, exit_price=51500)
    print("  ✓ Closed trade with profit")

    # Record market regime
    print("\n[6] Recording market regime...")
    db.record_market_regime(
        regime="BULL",
        btc_price=51500,
        volatility=0.02,
        correlation_avg=0.75,
        confidence=0.85
    )
    print("  ✓ Recorded BULL regime")

    # Query demonstrations
    print("\n" + "=" * 70)
    print("ANALYTICS DEMONSTRATIONS")
    print("=" * 70)

    # Get signals for symbol
    print("\n[A] Signals for BTCUSDT (last 24h):")
    signals = db.get_signals_for_symbol("BTCUSDT", hours_back=24)
    for sig_type, sig_list in signals.items():
        if sig_list:
            print(f"  {sig_type}: {len(sig_list)} signals")

    # Get aggregated signals
    print("\n[B] Aggregated Signal Consensus:")
    for symbol in ["BTCUSDT", "ETHUSDT", "XRPUSDT"]:
        agg = db.get_aggregated_signals(symbol)
        print(f"  {symbol}: {agg['consensus_direction']} "
              f"(score: {agg['consensus_score']:.2f}, "
              f"confidence: {agg['confidence']:.2f})")

    # Portfolio summary
    print("\n[C] Portfolio Summary:")
    summary = db.get_portfolio_summary()
    trade_stats = summary["trade_statistics"]
    print(f"  Total Trades: {trade_stats['total_trades']}")
    print(f"  Win Rate: {trade_stats['win_rate']}%")
    print(f"  Total P&L: ${trade_stats['total_pnl_usd']}")
    print(f"  Current Regime: {summary['current_regime']['regime']}")

    # Signal correlations
    print("\n[D] Signal Correlations Found:")
    correlations = db.find_signal_correlations(days_back=30)
    if correlations:
        for corr in correlations[:3]:
            print(f"  {corr['signal_1']} <-> {corr['signal_2']}: "
                  f"{corr['count']} co-occurrences, "
                  f"{corr['agreement_rate']}% agreement")
    else:
        print("  (Insufficient data for correlation analysis)")

    # Database stats
    print("\n[E] Database Statistics:")
    stats = db.get_database_stats()
    for name, count in stats["record_counts"].items():
        print(f"  {name}: {count} records")

    print("\n" + "=" * 70)
    print("DATABASE READY FOR PRODUCTION USE")
    print("=" * 70)
    print("\nUsage:")
    print("  db = QuantDatabase('quant_signals.db')  # Persistent")
    print("  db = QuantDatabase(':memory:')          # In-memory testing")
    print("\nKey Methods:")
    print("  - insert_*()           : Store signals, trades, prices")
    print("  - get_*()              : Query data and signals")
    print("  - calculate_signal_accuracy() : Score signal performance")
    print("  - detect_failure_patterns()   : Find what causes losses")
    print("  - get_aggregated_signals()    : Weighted consensus signal")
    print("  - get_portfolio_summary()     : Full portfolio analysis")


if __name__ == "__main__":
    demo()
