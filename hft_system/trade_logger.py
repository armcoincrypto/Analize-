"""
Trade Logger for HFT System
===========================
Stores all trades, signals, and market data in SQLite for analysis.

Tables:
- trades: All completed trades with entry/exit details
- signals: All generated signals (even rejected ones)
- market_snapshots: Periodic market state snapshots
- daily_stats: Daily performance summaries
"""

import sqlite3
import logging
import time
import json
from typing import Dict, Optional, List
from datetime import datetime, date
from pathlib import Path

from .config import SYSTEM_CONFIG
from .signal_engine import Signal, ConditionResult
from .execution_engine import TradeResult, ExitResult
from .risk_controller import RiskDecision

logger = logging.getLogger(__name__)


class TradeLogger:
    """
    SQLite-based trade and signal logger.

    All data persisted for post-analysis.
    """

    def __init__(self, db_path: str = None):
        self.db_path = db_path or SYSTEM_CONFIG.db_path
        self.conn = None
        self._init_database()

    def _init_database(self):
        """Initialize database with required tables."""
        # Allow multi-thread access and set timeout to avoid "database is locked"
        self.conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            timeout=30.0  # Wait up to 30s for lock
        )
        self.conn.row_factory = sqlite3.Row
        # Enable WAL mode for better concurrency
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=30000")

        cursor = self.conn.cursor()

        # Trades table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT UNIQUE,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                quantity REAL NOT NULL,
                entry_time INTEGER NOT NULL,
                exit_time INTEGER,
                pnl REAL,
                pnl_pct REAL,
                hold_time_sec REAL,
                exit_reason TEXT,
                mfe REAL,
                mae REAL,
                conditions_met INTEGER,
                signal_confidence REAL,
                status TEXT DEFAULT 'open',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Signals table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                conditions_met INTEGER NOT NULL,
                confidence REAL NOT NULL,
                entry_price REAL NOT NULL,
                condition_details TEXT,
                risk_approved INTEGER,
                risk_reason TEXT,
                trade_executed INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Market snapshots table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                price REAL NOT NULL,
                price_change_60s REAL,
                volume_spike REAL,
                orderbook_imbalance REAL,
                funding_rate REAL,
                rsi REAL,
                btc_volatility REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Daily stats table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT UNIQUE NOT NULL,
                total_trades INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                gross_pnl REAL DEFAULT 0,
                net_pnl REAL DEFAULT 0,
                win_rate REAL DEFAULT 0,
                avg_win REAL DEFAULT 0,
                avg_loss REAL DEFAULT 0,
                profit_factor REAL DEFAULT 0,
                max_drawdown REAL DEFAULT 0,
                signals_generated INTEGER DEFAULT 0,
                signals_rejected INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Tick data table - stores every price update for analysis
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tick_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                price REAL NOT NULL,
                volume REAL,
                is_buyer_maker INTEGER
            )
        """)

        # Orderbook snapshots - for microstructure analysis
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS orderbook_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                best_bid REAL,
                best_ask REAL,
                bid_volume REAL,
                ask_volume REAL,
                spread_pct REAL,
                imbalance REAL
            )
        """)

        # Indicator values - for strategy optimization
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS indicator_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                price REAL,
                rsi REAL,
                volume_spike REAL,
                price_change_60s REAL,
                funding_rate REAL,
                orderbook_imbalance REAL,
                conditions_met INTEGER
            )
        """)

        # Strategy signals - log ALL potential signals for multi-strategy analysis
        # Now with extended outcome tracking up to 10 minutes
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategy_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                price REAL NOT NULL,
                conditions_met INTEGER NOT NULL,
                price_cond INTEGER DEFAULT 0,
                volume_cond INTEGER DEFAULT 0,
                orderbook_cond INTEGER DEFAULT 0,
                funding_cond INTEGER DEFAULT 0,
                rsi_cond INTEGER DEFAULT 0,
                price_after_60s REAL,
                price_after_90s REAL,
                price_after_2m REAL,
                price_after_3m REAL,
                price_after_5m REAL,
                price_after_10m REAL,
                pnl_60s REAL,
                pnl_90s REAL,
                pnl_2m REAL,
                pnl_3m REAL,
                pnl_5m REAL,
                pnl_10m REAL,
                max_up_10m REAL,
                max_down_10m REAL
            )
        """)

        # Trade flow - track every trade for buy/sell pressure analysis
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trade_flow (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                price REAL NOT NULL,
                quantity REAL NOT NULL,
                is_buyer_maker INTEGER,
                trade_value REAL
            )
        """)

        # CVD (Cumulative Volume Delta) snapshots
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cvd_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                cvd_1m REAL,
                cvd_5m REAL,
                cvd_15m REAL,
                buy_volume_1m REAL,
                sell_volume_1m REAL
            )
        """)

        # Trade entry flow - MICROSTRUCTURE: capture trade flow at entry/exit for analysis
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trade_entry_flow (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT NOT NULL,
                snapshot_type TEXT DEFAULT 'entry',
                timestamp INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                aggressive_buy_vol REAL,
                aggressive_sell_vol REAL,
                trades_per_second REAL,
                net_delta REAL,
                delta_pct REAL,
                delta_1s REAL,
                delta_3s REAL,
                trade_count INTEGER,
                orderbook_imbalance REAL,
                spread_pct REAL,
                FOREIGN KEY (trade_id) REFERENCES trades(trade_id)
            )
        """)

        # TASK 6: Trade quality metrics - consolidated view for analysis
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trade_quality_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT UNIQUE NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,

                -- Entry conditions
                imbalance_at_entry REAL,
                spread_at_entry REAL,
                delta_at_entry REAL,
                buy_sell_ratio_entry REAL,

                -- Exit conditions
                imbalance_at_exit REAL,
                spread_at_exit REAL,
                delta_at_exit REAL,
                buy_sell_ratio_exit REAL,

                -- Trade metrics
                time_in_trade_sec REAL,
                exit_reason TEXT,
                pnl_pct REAL,

                -- Slippage tracking
                expected_entry_price REAL,
                actual_entry_price REAL,
                entry_slippage_pct REAL,
                expected_exit_price REAL,
                actual_exit_price REAL,
                exit_slippage_pct REAL,
                total_cost_pct REAL,

                -- Quality flags
                exited_within_5s INTEGER DEFAULT 0,
                exited_by_invalidation INTEGER DEFAULT 0,

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (trade_id) REFERENCES trades(trade_id)
            )
        """)

        # TASK 8: Trade causality - WHY the move happened
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trade_causality (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT UNIQUE NOT NULL,
                symbol TEXT NOT NULL,
                timestamp INTEGER NOT NULL,

                -- Orderbook state (top 5 levels)
                ob_bid1_price REAL,
                ob_bid1_size REAL,
                ob_ask1_price REAL,
                ob_ask1_size REAL,
                ob_top5_bid_volume REAL,
                ob_top5_ask_volume REAL,
                ob_imbalance_ratio REAL,
                ob_spread_pct REAL,

                -- CVD (Cumulative Volume Delta) momentum
                cvd_delta_1s REAL,
                cvd_delta_3s REAL,
                cvd_delta_5s REAL,

                -- Price velocity (micro moves)
                price_velocity_1s REAL,
                price_velocity_3s REAL,
                price_velocity_5s REAL,

                -- Correlated symbols
                btc_price_change_1s REAL,
                btc_price_change_3s REAL,
                btc_delta_1s REAL,

                -- Primary causal factor
                primary_cause TEXT,
                cause_strength REAL,

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (trade_id) REFERENCES trades(trade_id)
            )
        """)

        # TASK 9: Trade edge validation - WHEN edge is real vs fake
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trade_edge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT UNIQUE NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,

                -- Edge timing metrics
                seconds_to_max_favorable REAL,
                max_favorable_pct REAL,
                seconds_to_ob_decay REAL,
                ob_decay_amount REAL,
                delta_persistence_sec REAL,

                -- Entry state
                entry_imbalance REAL,
                entry_delta REAL,
                entry_spread REAL,

                -- Edge quality
                edge_duration_sec REAL,
                historical_median_edge_sec REAL,
                real_edge_flag INTEGER DEFAULT 0,

                -- Final outcome
                final_pnl_pct REAL,
                exit_reason TEXT,

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (trade_id) REFERENCES trades(trade_id)
            )
        """)

        # TASK 10: Market regime classification - WHICH market context
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_regime (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                symbol TEXT NOT NULL,

                -- Volatility metrics
                volatility_1m REAL,
                volatility_5m REAL,
                volatility_ratio REAL,

                -- Range metrics
                range_expansion REAL,
                range_compression REAL,

                -- Trend metrics
                trend_strength REAL,
                trend_direction TEXT,

                -- Orderbook patterns
                ob_imbalance_stability REAL,
                ob_depth_ratio REAL,

                -- Trade flow patterns
                flow_consistency REAL,
                large_trade_ratio REAL,

                -- Regime classification
                regime TEXT NOT NULL,
                regime_confidence REAL,

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Add regime columns to trades table if not exists
        try:
            cursor.execute("ALTER TABLE trades ADD COLUMN regime_at_entry TEXT")
        except sqlite3.OperationalError:
            pass  # Column already exists
        try:
            cursor.execute("ALTER TABLE trades ADD COLUMN regime_at_exit TEXT")
        except sqlite3.OperationalError:
            pass  # Column already exists

        # Create indexes for faster queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp ON market_snapshots(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tick_timestamp ON tick_data(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tick_symbol ON tick_data(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_indicator_timestamp ON indicator_snapshots(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_strategy_signals_timestamp ON strategy_signals(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_strategy_signals_conditions ON strategy_signals(conditions_met)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trade_flow_timestamp ON trade_flow(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trade_flow_symbol ON trade_flow(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cvd_timestamp ON cvd_snapshots(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trade_entry_flow_trade_id ON trade_entry_flow(trade_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trade_quality_trade_id ON trade_quality_metrics(trade_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trade_quality_exit_reason ON trade_quality_metrics(exit_reason)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trade_causality_trade_id ON trade_causality(trade_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trade_causality_cause ON trade_causality(primary_cause)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trade_edge_trade_id ON trade_edge(trade_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trade_edge_symbol ON trade_edge(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trade_edge_real_flag ON trade_edge(real_edge_flag)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_regime_timestamp ON market_regime(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_regime_symbol ON market_regime(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_regime_regime ON market_regime(regime)")

        self.conn.commit()
        logger.info(f"Database initialized: {self.db_path}")

    def log_signal(
        self,
        signal: Signal,
        risk_decision: RiskDecision,
        executed: bool = False
    ):
        """Log a generated signal."""
        try:
            condition_details = json.dumps([
                {
                    "name": c.name,
                    "triggered": c.triggered,
                    "value": c.value,
                    "threshold": c.threshold,
                    "message": c.message
                }
                for c in signal.conditions
            ])

            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO signals (
                    timestamp, symbol, signal_type, conditions_met,
                    confidence, entry_price, condition_details,
                    risk_approved, risk_reason, trade_executed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                signal.timestamp,
                signal.symbol,
                signal.signal_type.value,
                signal.conditions_met,
                signal.confidence,
                signal.entry_price,
                condition_details,
                1 if risk_decision.approved else 0,
                risk_decision.reason.value,
                1 if executed else 0
            ))

            self.conn.commit()

        except Exception as e:
            logger.error(f"Error logging signal: {e}")

    def log_trade_entry(self, result: TradeResult, signal: Signal):
        """Log trade entry."""
        try:
            trade_id = result.order_id or f"{result.symbol}_{result.timestamp}"

            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO trades (
                    trade_id, symbol, side, entry_price, quantity,
                    entry_time, conditions_met, signal_confidence, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'open')
            """, (
                trade_id,
                result.symbol,
                result.side,
                result.entry_price,
                result.quantity,
                result.timestamp,
                signal.conditions_met,
                signal.confidence
            ))

            self.conn.commit()
            logger.debug(f"Trade entry logged: {trade_id}")

        except Exception as e:
            logger.error(f"Error logging trade entry: {e}")

    def log_trade_exit(self, result: ExitResult, mfe: float = 0, mae: float = 0):
        """Log trade exit and update trade record."""
        try:
            cursor = self.conn.cursor()

            # Find the open trade
            cursor.execute("""
                UPDATE trades SET
                    exit_price = ?,
                    exit_time = ?,
                    pnl = ?,
                    pnl_pct = ?,
                    hold_time_sec = ?,
                    exit_reason = ?,
                    mfe = ?,
                    mae = ?,
                    status = 'closed'
                WHERE symbol = ? AND status = 'open'
            """, (
                result.exit_price,
                result.timestamp,
                result.pnl,
                result.pnl_pct,
                result.hold_time_sec,
                result.reason.value,
                mfe,
                mae,
                result.symbol
            ))

            self.conn.commit()
            logger.debug(f"Trade exit logged: {result.symbol}")

            # Update daily stats
            self._update_daily_stats()

        except Exception as e:
            logger.error(f"Error logging trade exit: {e}")

    def log_market_snapshot(self, symbol: str, data: Dict):
        """Log periodic market snapshot."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO market_snapshots (
                    timestamp, symbol, price, price_change_60s,
                    volume_spike, orderbook_imbalance, funding_rate,
                    rsi, btc_volatility
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                int(time.time() * 1000),
                symbol,
                data.get("price", 0),
                data.get("price_change_60s", 0),
                data.get("volume_spike", 0),
                data.get("orderbook_imbalance", 0),
                data.get("funding_rate", 0),
                data.get("rsi", 0),
                data.get("btc_volatility", 0)
            ))

            self.conn.commit()

        except Exception as e:
            logger.error(f"Error logging market snapshot: {e}")

    def log_tick(self, symbol: str, price: float, volume: float, is_buyer_maker: bool):
        """Log individual tick/trade data."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO tick_data (timestamp, symbol, price, volume, is_buyer_maker)
                VALUES (?, ?, ?, ?, ?)
            """, (
                int(time.time() * 1000),
                symbol,
                price,
                volume,
                1 if is_buyer_maker else 0
            ))
            # Don't commit every tick - batch commit
        except Exception as e:
            pass  # Silent fail for performance

    def log_orderbook_snapshot(self, symbol: str, data: Dict):
        """Log orderbook snapshot for microstructure analysis."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO orderbook_snapshots (
                    timestamp, symbol, best_bid, best_ask,
                    bid_volume, ask_volume, spread_pct, imbalance
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                int(time.time() * 1000),
                symbol,
                data.get("best_bid", 0),
                data.get("best_ask", 0),
                data.get("bid_volume", 0),
                data.get("ask_volume", 0),
                data.get("spread_pct", 0),
                data.get("imbalance", 0)
            ))
        except Exception as e:
            pass  # Silent fail for performance

    def log_indicator_snapshot(self, symbol: str, data: Dict):
        """Log all indicator values for strategy analysis."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO indicator_snapshots (
                    timestamp, symbol, price, rsi, volume_spike,
                    price_change_60s, funding_rate, orderbook_imbalance, conditions_met
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                int(time.time() * 1000),
                symbol,
                data.get("price", 0),
                data.get("rsi", 0),
                data.get("volume_spike", 0),
                data.get("price_change_60s", 0),
                data.get("funding_rate", 0),
                data.get("orderbook_imbalance", 0),
                data.get("conditions_met", 0)
            ))
        except Exception as e:
            pass  # Silent fail for performance

    def log_strategy_signal(self, symbol: str, price: float, conditions: Dict):
        """
        Log potential strategy signal for multi-strategy analysis.

        Logs when ANY condition is met (1/5 or higher), so later
        we can analyze which threshold (1/5, 2/5, 3/5, etc.) is optimal.
        """
        try:
            conditions_met = sum(1 for v in conditions.values() if v)

            if conditions_met >= 1:  # Log when at least 1 condition met
                cursor = self.conn.cursor()
                cursor.execute("""
                    INSERT INTO strategy_signals (
                        timestamp, symbol, price, conditions_met,
                        price_cond, volume_cond, orderbook_cond,
                        funding_cond, rsi_cond
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    int(time.time() * 1000),
                    symbol,
                    price,
                    conditions_met,
                    1 if conditions.get("price_movement", False) else 0,
                    1 if conditions.get("volume_spike", False) else 0,
                    1 if conditions.get("orderbook_imbalance", False) else 0,
                    1 if conditions.get("derivatives_signal", False) else 0,
                    1 if conditions.get("micro_indicator", False) else 0
                ))
        except Exception as e:
            pass  # Silent fail for performance

    def batch_commit(self):
        """Commit batched writes (call periodically)."""
        try:
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error in batch commit: {e}")

    def update_signal_outcomes(self, symbol: str, current_price: float):
        """
        Update old strategy signals with what price became.

        Tracks outcomes at: 60s, 90s, 2m, 3m, 5m, 10m
        This lets us analyze which timeframe is most profitable.
        """
        try:
            now = int(time.time() * 1000)
            cursor = self.conn.cursor()

            # Define time windows (target_ms, window_ms, column_price, column_pnl)
            time_windows = [
                (60000, 5000, "price_after_60s", "pnl_60s"),
                (90000, 5000, "price_after_90s", "pnl_90s"),
                (120000, 5000, "price_after_2m", "pnl_2m"),
                (180000, 5000, "price_after_3m", "pnl_3m"),
                (300000, 10000, "price_after_5m", "pnl_5m"),
                (600000, 15000, "price_after_10m", "pnl_10m"),
            ]

            for target_ms, window_ms, price_col, pnl_col in time_windows:
                cursor.execute(f"""
                    UPDATE strategy_signals
                    SET {price_col} = ?,
                        {pnl_col} = ((? - price) / price) * 100
                    WHERE symbol = ?
                      AND {price_col} IS NULL
                      AND timestamp BETWEEN ? AND ?
                """, (
                    current_price,
                    current_price,
                    symbol,
                    now - target_ms - window_ms,
                    now - target_ms + window_ms
                ))

        except Exception as e:
            pass  # Silent fail for performance

    def log_trade_flow(self, symbol: str, price: float, quantity: float, is_buyer_maker: bool):
        """
        Log individual trade for buy/sell pressure analysis.

        is_buyer_maker=True means the buyer was the maker (seller was aggressor = selling pressure)
        is_buyer_maker=False means the seller was the maker (buyer was aggressor = buying pressure)
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO trade_flow (timestamp, symbol, price, quantity, is_buyer_maker, trade_value)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                int(time.time() * 1000),
                symbol,
                price,
                quantity,
                1 if is_buyer_maker else 0,
                price * quantity
            ))
        except Exception as e:
            pass  # Silent fail

    def log_cvd_snapshot(self, symbol: str, cvd_data: dict):
        """
        Log CVD (Cumulative Volume Delta) snapshot.

        CVD = Sum of (buy volume - sell volume)
        Positive CVD = more buying pressure
        Negative CVD = more selling pressure
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO cvd_snapshots (
                    timestamp, symbol, cvd_1m, cvd_5m, cvd_15m,
                    buy_volume_1m, sell_volume_1m
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                int(time.time() * 1000),
                symbol,
                cvd_data.get("cvd_1m", 0),
                cvd_data.get("cvd_5m", 0),
                cvd_data.get("cvd_15m", 0),
                cvd_data.get("buy_volume_1m", 0),
                cvd_data.get("sell_volume_1m", 0)
            ))
        except Exception as e:
            pass  # Silent fail

    def log_trade_entry_flow(
        self,
        trade_id: str,
        symbol: str,
        trade_flow: dict,
        orderbook_imbalance: float = None,
        spread_pct: float = None,
        snapshot_type: str = "entry"
    ):
        """
        Log MICROSTRUCTURE trade flow snapshot at entry or exit.

        Links trade flow data to specific trade for analysis:
        - aggressive_buy_vol: Market buys hitting asks
        - aggressive_sell_vol: Market sells hitting bids
        - trades_per_second: Trading activity
        - net_delta: buy - sell volume
        - delta_1s, delta_3s: Short-term momentum
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO trade_entry_flow (
                    trade_id, snapshot_type, timestamp, symbol,
                    aggressive_buy_vol, aggressive_sell_vol,
                    trades_per_second, net_delta, delta_pct,
                    delta_1s, delta_3s, trade_count,
                    orderbook_imbalance, spread_pct
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_id,
                snapshot_type,
                int(time.time() * 1000),
                symbol,
                trade_flow.get("aggressive_buy_vol", 0),
                trade_flow.get("aggressive_sell_vol", 0),
                trade_flow.get("trades_per_second", 0),
                trade_flow.get("net_delta", 0),
                trade_flow.get("delta_pct", 0),
                trade_flow.get("delta_1s", 0),
                trade_flow.get("delta_3s", 0),
                trade_flow.get("trade_count", 0),
                orderbook_imbalance,
                spread_pct
            ))
            self.conn.commit()
            logger.debug(f"Trade flow logged for {trade_id}: delta={trade_flow.get('net_delta', 0):.2f}")
        except Exception as e:
            logger.error(f"Error logging trade entry flow: {e}")

    def log_trade_quality_metrics(
        self,
        trade_id: str,
        symbol: str,
        side: str,
        entry_flow: dict,
        exit_flow: dict,
        entry_orderbook: dict,
        exit_orderbook: dict,
        time_in_trade_sec: float,
        exit_reason: str,
        pnl_pct: float,
        expected_entry_price: float,
        actual_entry_price: float,
        expected_exit_price: float,
        actual_exit_price: float
    ):
        """
        TASK 6: Log consolidated trade quality metrics for analysis.

        This is the PRIMARY table for evaluating trade quality.
        Called when trade is closed to capture all metrics.
        """
        try:
            # Calculate buy/sell ratios
            entry_buy = entry_flow.get("aggressive_buy_vol", 0)
            entry_sell = entry_flow.get("aggressive_sell_vol", 0)
            entry_total = entry_buy + entry_sell
            buy_sell_ratio_entry = entry_buy / entry_total if entry_total > 0 else 0.5

            exit_buy = exit_flow.get("aggressive_buy_vol", 0)
            exit_sell = exit_flow.get("aggressive_sell_vol", 0)
            exit_total = exit_buy + exit_sell
            buy_sell_ratio_exit = exit_buy / exit_total if exit_total > 0 else 0.5

            # Calculate slippage
            entry_slippage = 0
            if expected_entry_price > 0:
                entry_slippage = (actual_entry_price - expected_entry_price) / expected_entry_price * 100

            exit_slippage = 0
            if expected_exit_price > 0:
                exit_slippage = (actual_exit_price - expected_exit_price) / expected_exit_price * 100

            # Total cost = entry slippage + exit slippage + spread
            entry_spread = entry_orderbook.get("spread_pct", 0) or 0
            exit_spread = exit_orderbook.get("spread_pct", 0) or 0
            total_cost = abs(entry_slippage) + abs(exit_slippage) + (entry_spread + exit_spread) / 2

            # Quality flags
            exited_within_5s = 1 if time_in_trade_sec <= 5 else 0

            # Invalidation exits: OB_FLIP, DELTA_NEGATIVE, SPREAD_WIDEN, NO_MOVEMENT, FLOW_NEUTRAL
            invalidation_reasons = ["ob_flip", "delta_negative", "spread_widen", "no_movement", "flow_neutral"]
            exited_by_invalidation = 1 if exit_reason.lower() in invalidation_reasons else 0

            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO trade_quality_metrics (
                    trade_id, symbol, side,
                    imbalance_at_entry, spread_at_entry, delta_at_entry, buy_sell_ratio_entry,
                    imbalance_at_exit, spread_at_exit, delta_at_exit, buy_sell_ratio_exit,
                    time_in_trade_sec, exit_reason, pnl_pct,
                    expected_entry_price, actual_entry_price, entry_slippage_pct,
                    expected_exit_price, actual_exit_price, exit_slippage_pct,
                    total_cost_pct, exited_within_5s, exited_by_invalidation
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_id,
                symbol,
                side,
                entry_orderbook.get("imbalance", 0),
                entry_orderbook.get("spread_pct", 0),
                entry_flow.get("net_delta", 0),
                buy_sell_ratio_entry,
                exit_orderbook.get("imbalance", 0),
                exit_orderbook.get("spread_pct", 0),
                exit_flow.get("net_delta", 0),
                buy_sell_ratio_exit,
                time_in_trade_sec,
                exit_reason,
                pnl_pct,
                expected_entry_price,
                actual_entry_price,
                entry_slippage,
                expected_exit_price,
                actual_exit_price,
                exit_slippage,
                total_cost,
                exited_within_5s,
                exited_by_invalidation
            ))
            self.conn.commit()
            logger.debug(f"Trade quality metrics logged: {trade_id}")
        except Exception as e:
            logger.error(f"Error logging trade quality metrics: {e}")

    def log_trade_causality(
        self,
        trade_id: str,
        symbol: str,
        orderbook_data: dict,
        cvd_data: dict,
        price_velocity: dict,
        btc_data: dict
    ) -> str:
        """
        TASK 8: Log WHY the trade signal happened.

        Captures causal context:
        - Orderbook state (top 5 levels)
        - CVD delta (buy/sell pressure momentum)
        - Price velocity (micro moves)
        - BTC correlation

        Returns: primary_cause (the top factor)
        """
        try:
            # Determine primary causal factor
            causes = []

            # Check orderbook imbalance
            ob_imbalance = orderbook_data.get("imbalance_ratio", 0.5)
            if ob_imbalance > 0.65:
                causes.append(("ob_bullish_imbalance", ob_imbalance - 0.5))
            elif ob_imbalance < 0.35:
                causes.append(("ob_bearish_imbalance", 0.5 - ob_imbalance))

            # Check CVD momentum
            cvd_1s = cvd_data.get("delta_1s", 0)
            cvd_3s = cvd_data.get("delta_3s", 0)
            if abs(cvd_1s) > 1000:  # Significant volume delta
                if cvd_1s > 0:
                    causes.append(("cvd_buy_pressure", cvd_1s / 10000))
                else:
                    causes.append(("cvd_sell_pressure", abs(cvd_1s) / 10000))

            # Check price velocity
            velocity_1s = price_velocity.get("velocity_1s", 0)
            velocity_3s = price_velocity.get("velocity_3s", 0)
            if abs(velocity_1s) > 0.01:  # >0.01% in 1s is significant
                if velocity_1s > 0:
                    causes.append(("price_momentum_up", velocity_1s))
                else:
                    causes.append(("price_momentum_down", abs(velocity_1s)))

            # Check BTC correlation
            btc_change = btc_data.get("price_change_1s", 0)
            if abs(btc_change) > 0.02:  # BTC moved >0.02%
                if btc_change > 0:
                    causes.append(("btc_bullish", btc_change))
                else:
                    causes.append(("btc_bearish", abs(btc_change)))

            # Sort by strength and get primary cause
            causes.sort(key=lambda x: x[1], reverse=True)
            primary_cause = causes[0][0] if causes else "unknown"
            cause_strength = causes[0][1] if causes else 0

            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO trade_causality (
                    trade_id, symbol, timestamp,
                    ob_bid1_price, ob_bid1_size, ob_ask1_price, ob_ask1_size,
                    ob_top5_bid_volume, ob_top5_ask_volume, ob_imbalance_ratio, ob_spread_pct,
                    cvd_delta_1s, cvd_delta_3s, cvd_delta_5s,
                    price_velocity_1s, price_velocity_3s, price_velocity_5s,
                    btc_price_change_1s, btc_price_change_3s, btc_delta_1s,
                    primary_cause, cause_strength
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_id,
                symbol,
                int(time.time() * 1000),
                orderbook_data.get("bid1_price", 0),
                orderbook_data.get("bid1_size", 0),
                orderbook_data.get("ask1_price", 0),
                orderbook_data.get("ask1_size", 0),
                orderbook_data.get("top5_bid_volume", 0),
                orderbook_data.get("top5_ask_volume", 0),
                orderbook_data.get("imbalance_ratio", 0),
                orderbook_data.get("spread_pct", 0),
                cvd_data.get("delta_1s", 0),
                cvd_data.get("delta_3s", 0),
                cvd_data.get("delta_5s", 0),
                price_velocity.get("velocity_1s", 0),
                price_velocity.get("velocity_3s", 0),
                price_velocity.get("velocity_5s", 0),
                btc_data.get("price_change_1s", 0),
                btc_data.get("price_change_3s", 0),
                btc_data.get("delta_1s", 0),
                primary_cause,
                cause_strength
            ))
            self.conn.commit()
            logger.info(f"Trade causality: {trade_id} -> {primary_cause} (strength: {cause_strength:.4f})")
            return primary_cause
        except Exception as e:
            logger.error(f"Error logging trade causality: {e}")
            return "error"

    def get_causality_stats(self) -> Dict:
        """Get aggregate stats on trade causality factors."""
        try:
            cursor = self.conn.cursor()

            # Get breakdown by primary cause
            cursor.execute("""
                SELECT
                    primary_cause,
                    COUNT(*) as count,
                    AVG(cause_strength) as avg_strength
                FROM trade_causality
                GROUP BY primary_cause
                ORDER BY count DESC
            """)

            cause_breakdown = {
                r[0]: {
                    "count": r[1],
                    "avg_strength": round(r[2], 4) if r[2] else 0
                }
                for r in cursor.fetchall()
            }

            # Get total trades with causality data
            cursor.execute("SELECT COUNT(*) FROM trade_causality")
            total = cursor.fetchone()[0]

            return {
                "total_trades_with_causality": total,
                "cause_breakdown": cause_breakdown
            }
        except Exception as e:
            logger.error(f"Error getting causality stats: {e}")
            return {"error": str(e)}

    def get_historical_median_edge(self, symbol: str) -> float:
        """Get historical median edge duration for symbol."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT edge_duration_sec
                FROM trade_edge
                WHERE symbol = ? AND edge_duration_sec > 0
                ORDER BY edge_duration_sec
            """, (symbol,))
            rows = cursor.fetchall()

            if not rows:
                return 3.0  # Default 3 seconds if no history

            # Calculate median
            n = len(rows)
            mid = n // 2
            if n % 2 == 0:
                return (rows[mid - 1][0] + rows[mid][0]) / 2
            else:
                return rows[mid][0]
        except Exception as e:
            logger.error(f"Error getting historical median edge: {e}")
            return 3.0

    def log_trade_edge(
        self,
        trade_id: str,
        symbol: str,
        side: str,
        seconds_to_max_favorable: float,
        max_favorable_pct: float,
        seconds_to_ob_decay: float,
        ob_decay_amount: float,
        delta_persistence_sec: float,
        entry_imbalance: float,
        entry_delta: float,
        entry_spread: float,
        edge_duration_sec: float,
        final_pnl_pct: float,
        exit_reason: str
    ):
        """
        TASK 9: Log edge validation metrics.

        Determines if this trade had a real edge or fake edge:
        - real_edge_flag = 1 if edge_duration > historical median
        """
        try:
            # Get historical median for this symbol
            historical_median = self.get_historical_median_edge(symbol)

            # Determine if edge was real
            real_edge_flag = 1 if edge_duration_sec > historical_median else 0

            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO trade_edge (
                    trade_id, symbol, side,
                    seconds_to_max_favorable, max_favorable_pct,
                    seconds_to_ob_decay, ob_decay_amount, delta_persistence_sec,
                    entry_imbalance, entry_delta, entry_spread,
                    edge_duration_sec, historical_median_edge_sec, real_edge_flag,
                    final_pnl_pct, exit_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_id,
                symbol,
                side,
                seconds_to_max_favorable,
                max_favorable_pct,
                seconds_to_ob_decay,
                ob_decay_amount,
                delta_persistence_sec,
                entry_imbalance,
                entry_delta,
                entry_spread,
                edge_duration_sec,
                historical_median,
                real_edge_flag,
                final_pnl_pct,
                exit_reason
            ))
            self.conn.commit()

            edge_status = "REAL" if real_edge_flag else "FAKE"
            logger.info(
                f"Trade edge: {trade_id} -> {edge_status} edge | "
                f"Duration: {edge_duration_sec:.1f}s (median: {historical_median:.1f}s)"
            )
            return real_edge_flag
        except Exception as e:
            logger.error(f"Error logging trade edge: {e}")
            return 0

    def get_edge_stats(self) -> Dict:
        """
        TASK 9: Get aggregate edge validation statistics.

        Returns:
        - % of signals with real edge
        - Average edge duration
        - Edge decay patterns
        """
        try:
            cursor = self.conn.cursor()

            # Get overall stats
            cursor.execute("""
                SELECT
                    COUNT(*) as total_trades,
                    SUM(real_edge_flag) as real_edge_count,
                    AVG(edge_duration_sec) as avg_edge_duration,
                    AVG(seconds_to_max_favorable) as avg_time_to_max,
                    AVG(max_favorable_pct) as avg_max_favorable,
                    AVG(seconds_to_ob_decay) as avg_ob_decay_time,
                    AVG(delta_persistence_sec) as avg_delta_persistence
                FROM trade_edge
            """)
            row = cursor.fetchone()

            if not row or row[0] == 0:
                return {"message": "No edge data yet"}

            total = row[0]
            real_count = row[1] or 0
            pct_real_edge = (real_count / total * 100) if total > 0 else 0

            # Get per-symbol breakdown
            cursor.execute("""
                SELECT
                    symbol,
                    COUNT(*) as trades,
                    SUM(real_edge_flag) * 100.0 / COUNT(*) as pct_real,
                    AVG(edge_duration_sec) as avg_duration
                FROM trade_edge
                GROUP BY symbol
                ORDER BY trades DESC
            """)
            symbol_breakdown = {
                r[0]: {
                    "trades": r[1],
                    "pct_real_edge": round(r[2], 1) if r[2] else 0,
                    "avg_edge_duration": round(r[3], 2) if r[3] else 0
                }
                for r in cursor.fetchall()
            }

            # Get real vs fake edge PnL comparison
            cursor.execute("""
                SELECT
                    real_edge_flag,
                    AVG(final_pnl_pct) as avg_pnl,
                    COUNT(*) as count
                FROM trade_edge
                GROUP BY real_edge_flag
            """)
            edge_pnl = {}
            for r in cursor.fetchall():
                edge_type = "real" if r[0] == 1 else "fake"
                edge_pnl[edge_type] = {
                    "avg_pnl_pct": round(r[1], 4) if r[1] else 0,
                    "count": r[2]
                }

            return {
                "total_trades": total,
                "pct_real_edge": round(pct_real_edge, 1),
                "pct_fake_edge": round(100 - pct_real_edge, 1),
                "avg_edge_duration_sec": round(row[2], 2) if row[2] else 0,
                "avg_time_to_max_favorable_sec": round(row[3], 2) if row[3] else 0,
                "avg_max_favorable_pct": round(row[4], 4) if row[4] else 0,
                "avg_ob_decay_time_sec": round(row[5], 2) if row[5] else 0,
                "avg_delta_persistence_sec": round(row[6], 2) if row[6] else 0,
                "symbol_breakdown": symbol_breakdown,
                "edge_pnl_comparison": edge_pnl
            }
        except Exception as e:
            logger.error(f"Error getting edge stats: {e}")
            return {"error": str(e)}

    def get_microstructure_stats(self) -> Dict:
        """
        TASK 7: Get microstructure performance stats.

        Returns metrics that matter for microstructure trading:
        - Average profit per trade (expect small)
        - % of trades exited within 5 seconds
        - % of trades exited by invalidation
        - Slippage + spread impact

        Success = small consistent edge, not big wins.
        """
        try:
            cursor = self.conn.cursor()

            # Get aggregate stats from trade_quality_metrics
            cursor.execute("""
                SELECT
                    COUNT(*) as total_trades,
                    AVG(pnl_pct) as avg_pnl_pct,
                    AVG(time_in_trade_sec) as avg_hold_time,
                    SUM(exited_within_5s) * 100.0 / COUNT(*) as pct_exited_within_5s,
                    SUM(exited_by_invalidation) * 100.0 / COUNT(*) as pct_exited_by_invalidation,
                    AVG(total_cost_pct) as avg_total_cost,
                    AVG(entry_slippage_pct) as avg_entry_slippage,
                    AVG(exit_slippage_pct) as avg_exit_slippage,
                    AVG(spread_at_entry) as avg_entry_spread,
                    AVG(spread_at_exit) as avg_exit_spread
                FROM trade_quality_metrics
            """)
            row = cursor.fetchone()

            if not row or row[0] == 0:
                return {"message": "No trade quality data yet"}

            # Get exit reason breakdown
            cursor.execute("""
                SELECT exit_reason, COUNT(*) as count,
                       AVG(pnl_pct) as avg_pnl,
                       AVG(time_in_trade_sec) as avg_time
                FROM trade_quality_metrics
                GROUP BY exit_reason
                ORDER BY count DESC
            """)
            exit_breakdown = {
                r[0]: {
                    "count": r[1],
                    "avg_pnl_pct": round(r[2], 4) if r[2] else 0,
                    "avg_time_sec": round(r[3], 2) if r[3] else 0
                }
                for r in cursor.fetchall()
            }

            # Calculate net edge after costs
            avg_pnl = row[1] or 0
            avg_cost = row[5] or 0
            net_edge = avg_pnl - avg_cost

            return {
                "total_trades": row[0],
                "avg_pnl_pct": round(row[1], 4) if row[1] else 0,
                "avg_hold_time_sec": round(row[2], 2) if row[2] else 0,
                "pct_exited_within_5s": round(row[3], 1) if row[3] else 0,
                "pct_exited_by_invalidation": round(row[4], 1) if row[4] else 0,
                "avg_total_cost_pct": round(row[5], 4) if row[5] else 0,
                "avg_entry_slippage_pct": round(row[6], 4) if row[6] else 0,
                "avg_exit_slippage_pct": round(row[7], 4) if row[7] else 0,
                "avg_spread_pct": round((row[8] or 0 + row[9] or 0) / 2, 4),
                "net_edge_after_costs_pct": round(net_edge, 4),
                "exit_breakdown": exit_breakdown,
                "evaluation": self._evaluate_microstructure_performance(
                    row[0], avg_pnl, row[2], row[3], net_edge
                )
            }
        except Exception as e:
            logger.error(f"Error getting microstructure stats: {e}")
            return {"error": str(e)}

    def _evaluate_microstructure_performance(
        self,
        total_trades: int,
        avg_pnl: float,
        avg_hold_time: float,
        pct_quick_exit: float,
        net_edge: float
    ) -> str:
        """Evaluate performance based on microstructure criteria."""
        issues = []
        goods = []

        # Check trade duration (expect 2-20 seconds)
        if avg_hold_time and avg_hold_time > 20:
            issues.append(f"Hold time too long ({avg_hold_time:.1f}s, expect 2-20s)")
        elif avg_hold_time and avg_hold_time <= 20:
            goods.append(f"Good hold time ({avg_hold_time:.1f}s)")

        # Check quick exit rate (expect high)
        if pct_quick_exit and pct_quick_exit < 30:
            issues.append(f"Low quick exit rate ({pct_quick_exit:.1f}%, expect >30%)")
        elif pct_quick_exit and pct_quick_exit >= 50:
            goods.append(f"Good quick exit rate ({pct_quick_exit:.1f}%)")

        # Check net edge (must be positive after costs)
        if net_edge < 0:
            issues.append(f"Negative edge after costs ({net_edge:.4f}%)")
        elif net_edge > 0.005:
            goods.append(f"Positive edge ({net_edge:.4f}%)")

        # Check if enough trades
        if total_trades < 50:
            issues.append(f"Need more trades for reliable stats ({total_trades})")

        if issues:
            return "NEEDS WORK: " + "; ".join(issues)
        elif goods:
            return "GOOD: " + "; ".join(goods)
        else:
            return "INSUFFICIENT DATA"

    def _update_daily_stats(self):
        """Update daily statistics."""
        try:
            today = str(date.today())
            cursor = self.conn.cursor()

            # Get today's trades
            cursor.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) as losses,
                    SUM(pnl) as gross_pnl,
                    AVG(CASE WHEN pnl > 0 THEN pnl ELSE NULL END) as avg_win,
                    AVG(CASE WHEN pnl <= 0 THEN pnl ELSE NULL END) as avg_loss
                FROM trades
                WHERE DATE(created_at) = DATE('now') AND status = 'closed'
            """)

            row = cursor.fetchone()

            if row and row["total"] > 0:
                total = row["total"]
                wins = row["wins"] or 0
                losses = row["losses"] or 0
                gross_pnl = row["gross_pnl"] or 0
                avg_win = row["avg_win"] or 0
                avg_loss = abs(row["avg_loss"] or 0)

                win_rate = (wins / total * 100) if total > 0 else 0
                profit_factor = (avg_win * wins) / (avg_loss * losses) if losses > 0 and avg_loss > 0 else 0

                # Get signal stats
                cursor.execute("""
                    SELECT
                        COUNT(*) as generated,
                        SUM(CASE WHEN risk_approved = 0 THEN 1 ELSE 0 END) as rejected
                    FROM signals
                    WHERE DATE(created_at) = DATE('now')
                """)
                sig_row = cursor.fetchone()
                signals_generated = sig_row["generated"] or 0
                signals_rejected = sig_row["rejected"] or 0

                # Upsert daily stats
                cursor.execute("""
                    INSERT INTO daily_stats (
                        date, total_trades, wins, losses, gross_pnl,
                        win_rate, avg_win, avg_loss, profit_factor,
                        signals_generated, signals_rejected
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(date) DO UPDATE SET
                        total_trades = excluded.total_trades,
                        wins = excluded.wins,
                        losses = excluded.losses,
                        gross_pnl = excluded.gross_pnl,
                        win_rate = excluded.win_rate,
                        avg_win = excluded.avg_win,
                        avg_loss = excluded.avg_loss,
                        profit_factor = excluded.profit_factor,
                        signals_generated = excluded.signals_generated,
                        signals_rejected = excluded.signals_rejected
                """, (
                    today, total, wins, losses, gross_pnl,
                    win_rate, avg_win, avg_loss, profit_factor,
                    signals_generated, signals_rejected
                ))

                self.conn.commit()

        except Exception as e:
            logger.error(f"Error updating daily stats: {e}")

    def get_recent_trades(self, limit: int = 50) -> List[Dict]:
        """Get recent trades."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM trades
            ORDER BY entry_time DESC
            LIMIT ?
        """, (limit,))

        return [dict(row) for row in cursor.fetchall()]

    def get_daily_stats(self, days: int = 30) -> List[Dict]:
        """Get daily stats for last N days."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM daily_stats
            ORDER BY date DESC
            LIMIT ?
        """, (days,))

        return [dict(row) for row in cursor.fetchall()]

    def get_performance_summary(self) -> Dict:
        """Get overall performance summary."""
        cursor = self.conn.cursor()

        # Total stats
        cursor.execute("""
            SELECT
                COUNT(*) as total_trades,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) as losses,
                SUM(pnl) as total_pnl,
                AVG(pnl) as avg_pnl,
                AVG(CASE WHEN pnl > 0 THEN pnl ELSE NULL END) as avg_win,
                AVG(CASE WHEN pnl <= 0 THEN pnl ELSE NULL END) as avg_loss,
                AVG(hold_time_sec) as avg_hold_time,
                AVG(mfe) as avg_mfe,
                AVG(mae) as avg_mae
            FROM trades
            WHERE status = 'closed'
        """)

        row = cursor.fetchone()

        if not row or row["total_trades"] == 0:
            return {"message": "No trades yet"}

        total = row["total_trades"]
        wins = row["wins"] or 0
        losses = row["losses"] or 0
        avg_win = row["avg_win"] or 0
        avg_loss = abs(row["avg_loss"] or 0)

        win_rate = (wins / total * 100) if total > 0 else 0
        profit_factor = (avg_win * wins) / (avg_loss * losses) if losses > 0 and avg_loss > 0 else 0

        # Exit reason breakdown
        cursor.execute("""
            SELECT exit_reason, COUNT(*) as count
            FROM trades
            WHERE status = 'closed'
            GROUP BY exit_reason
        """)
        exit_reasons = {row["exit_reason"]: row["count"] for row in cursor.fetchall()}

        # Per-symbol stats
        cursor.execute("""
            SELECT
                symbol,
                COUNT(*) as trades,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                SUM(pnl) as total_pnl
            FROM trades
            WHERE status = 'closed'
            GROUP BY symbol
        """)
        per_symbol = {
            row["symbol"]: {
                "trades": row["trades"],
                "wins": row["wins"],
                "pnl": row["total_pnl"]
            }
            for row in cursor.fetchall()
        }

        return {
            "total_trades": total,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "total_pnl": row["total_pnl"],
            "avg_pnl": row["avg_pnl"],
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "avg_hold_time_sec": row["avg_hold_time"],
            "avg_mfe": row["avg_mfe"],
            "avg_mae": row["avg_mae"],
            "exit_reasons": exit_reasons,
            "per_symbol": per_symbol
        }

    def log_market_regime(
        self,
        symbol: str,
        regime: str,
        metrics: dict,
        confidence: float = 0.5
    ):
        """
        TASK 10: Log market regime classification.

        Regime types:
        - low_vol_chop: Low volatility, no trend, range-bound
        - high_vol_trend: High volatility with directional move
        - mean_reversion: Overextended price returning to mean
        - liquidity_vacuum: Low depth, wide spreads, choppy
        - news_spike: Sudden volume/volatility spike
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO market_regime (
                    timestamp, symbol,
                    volatility_1m, volatility_5m, volatility_ratio,
                    range_expansion, range_compression,
                    trend_strength, trend_direction,
                    ob_imbalance_stability, ob_depth_ratio,
                    flow_consistency, large_trade_ratio,
                    regime, regime_confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                int(time.time() * 1000),
                symbol,
                metrics.get("volatility_1m", 0),
                metrics.get("volatility_5m", 0),
                metrics.get("volatility_ratio", 0),
                metrics.get("range_expansion", 0),
                metrics.get("range_compression", 0),
                metrics.get("trend_strength", 0),
                metrics.get("trend_direction", "neutral"),
                metrics.get("ob_imbalance_stability", 0),
                metrics.get("ob_depth_ratio", 0),
                metrics.get("flow_consistency", 0),
                metrics.get("large_trade_ratio", 0),
                regime,
                confidence
            ))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error logging market regime: {e}")

    def get_current_regime(self, symbol: str) -> str:
        """Get the most recent regime classification for a symbol."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT regime FROM market_regime
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT 1
            """, (symbol,))
            row = cursor.fetchone()
            return row[0] if row else "unknown"
        except Exception as e:
            logger.error(f"Error getting current regime: {e}")
            return "unknown"

    def update_trade_regime(self, symbol: str, regime: str, is_entry: bool = True):
        """Update regime at entry or exit for the open trade."""
        try:
            cursor = self.conn.cursor()
            column = "regime_at_entry" if is_entry else "regime_at_exit"
            cursor.execute(f"""
                UPDATE trades
                SET {column} = ?
                WHERE symbol = ? AND status = 'open'
            """, (regime, symbol))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error updating trade regime: {e}")

    def get_regime_stats(self) -> Dict:
        """
        TASK 10: Get aggregate regime statistics.

        Returns:
        - Distribution of trades by regime
        - PnL by regime
        - Which regimes are most profitable
        """
        try:
            cursor = self.conn.cursor()

            # Get regime distribution and PnL
            cursor.execute("""
                SELECT
                    regime_at_entry,
                    COUNT(*) as trade_count,
                    AVG(pnl_pct) as avg_pnl,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as win_rate
                FROM trades
                WHERE status = 'closed' AND regime_at_entry IS NOT NULL
                GROUP BY regime_at_entry
                ORDER BY trade_count DESC
            """)
            regime_performance = {}
            for row in cursor.fetchall():
                regime_performance[row[0]] = {
                    "trades": row[1],
                    "avg_pnl_pct": round(row[2], 4) if row[2] else 0,
                    "win_rate": round(row[3], 1) if row[3] else 0
                }

            # Get recent regime distribution (last 1 hour)
            one_hour_ago = int((time.time() - 3600) * 1000)
            cursor.execute("""
                SELECT regime, COUNT(*) as count
                FROM market_regime
                WHERE timestamp > ?
                GROUP BY regime
                ORDER BY count DESC
            """, (one_hour_ago,))
            recent_regimes = {row[0]: row[1] for row in cursor.fetchall()}

            # Total trades with regime data
            cursor.execute("""
                SELECT COUNT(*) FROM trades
                WHERE status = 'closed' AND regime_at_entry IS NOT NULL
            """)
            total = cursor.fetchone()[0]

            return {
                "total_trades_with_regime": total,
                "regime_performance": regime_performance,
                "recent_regime_distribution": recent_regimes
            }
        except Exception as e:
            logger.error(f"Error getting regime stats: {e}")
            return {"error": str(e)}

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
