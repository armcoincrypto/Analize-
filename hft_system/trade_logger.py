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

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
