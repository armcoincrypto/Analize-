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
import threading
from typing import Dict, Optional, List
from datetime import datetime, date
from pathlib import Path

from .config import SYSTEM_CONFIG, COST_MODEL
from .signal_engine import Signal, ConditionResult
from .execution_engine import TradeResult, ExitResult
from .risk_controller import RiskDecision
from .symbol_utils import normalize_symbol

logger = logging.getLogger(__name__)


class TradeLogger:
    """
    SQLite-based trade and signal logger.

    All data persisted for post-analysis.
    """

    def __init__(self, db_path: str = None):
        self.db_path = db_path or SYSTEM_CONFIG.db_path
        self.conn = None
        self._db_lock = threading.Lock()  # Thread safety for database writes
        self._maker_telemetry_writer = None  # Separate writer for maker telemetry
        self._init_database()

    def _safe_execute(self, operation_name: str, func):
        """Execute a database operation with lock and silent error handling for locks."""
        with self._db_lock:
            try:
                func()
                self.conn.commit()
            except sqlite3.OperationalError as e:
                if "database is locked" not in str(e):
                    logger.error(f"Error {operation_name}: {e}")
            except Exception as e:
                logger.error(f"Error {operation_name}: {e}")

    def _execute_with_retry(self, sql: str, params: tuple = (), max_retries: int = 5) -> bool:
        """Execute SQL with retry logic for lock contention."""
        delay_ms = 50
        for attempt in range(max_retries):
            try:
                with self._db_lock:
                    cursor = self.conn.cursor()
                    cursor.execute(sql, params)
                    self.conn.commit()
                    return True
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower() or "busy" in str(e).lower():
                    if attempt < max_retries - 1:
                        time.sleep(delay_ms / 1000.0)
                        delay_ms = min(delay_ms * 2, 2000)
                        continue
                    logger.warning(f"SQL retry exhausted after {max_retries} attempts: {e}")
                    return False
                raise
        return False

    def _init_database(self):
        """Initialize database with required tables."""
        # Import and use shared db helper
        from .db import open_sqlite
        self.conn = open_sqlite(self.db_path)
        self.conn.row_factory = sqlite3.Row

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
                -- Cost model fields (realistic trading costs)
                entry_fee_pct REAL DEFAULT 0,
                exit_fee_pct REAL DEFAULT 0,
                fees_paid_pct REAL DEFAULT 0,
                spread_cost_pct REAL DEFAULT 0,
                slippage_pct REAL DEFAULT 0,
                total_costs_pct REAL DEFAULT 0,
                pnl_after_costs_pct REAL DEFAULT 0,
                -- Pocket tracking (which gate pocket allowed this trade)
                pocket_id TEXT,
                -- Execution mode tracking (taker vs maker)
                execution_mode TEXT DEFAULT 'taker',
                -- Cause policy tracking (FULL vs PROBE)
                is_probe_cause INTEGER DEFAULT 0,
                cause_class TEXT DEFAULT 'FULL',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Add cost model columns to existing tables (migration)
        try:
            cursor.execute("ALTER TABLE trades ADD COLUMN entry_fee_pct REAL DEFAULT 0")
            cursor.execute("ALTER TABLE trades ADD COLUMN exit_fee_pct REAL DEFAULT 0")
            cursor.execute("ALTER TABLE trades ADD COLUMN fees_paid_pct REAL DEFAULT 0")
            cursor.execute("ALTER TABLE trades ADD COLUMN spread_cost_pct REAL DEFAULT 0")
            cursor.execute("ALTER TABLE trades ADD COLUMN slippage_pct REAL DEFAULT 0")
            cursor.execute("ALTER TABLE trades ADD COLUMN total_costs_pct REAL DEFAULT 0")
            cursor.execute("ALTER TABLE trades ADD COLUMN pnl_after_costs_pct REAL DEFAULT 0")
            cursor.execute("ALTER TABLE trades ADD COLUMN pocket_id TEXT")
            cursor.execute("ALTER TABLE trades ADD COLUMN execution_mode TEXT DEFAULT 'taker'")
            cursor.execute("ALTER TABLE trades ADD COLUMN is_probe_cause INTEGER DEFAULT 0")
            cursor.execute("ALTER TABLE trades ADD COLUMN cause_class TEXT DEFAULT 'FULL'")
        except:
            pass  # Columns already exist

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

        # TASK 11: Blocked signals - No-Trade Zone detection
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS blocked_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                entry_price REAL,

                -- Block reason details
                block_reason TEXT NOT NULL,
                block_details TEXT,

                -- Market conditions at block time
                spread_pct REAL,
                spread_change_1s REAL,
                delta_variance REAL,
                ob_volume_instability REAL,
                liquidity_depth REAL,

                -- What would have happened
                price_after_5s REAL,
                price_after_10s REAL,
                would_have_pnl REAL,

                -- Regime at block time
                regime TEXT,

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # TASK 12: Position sizing decisions - adaptive sizing log
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS position_sizing (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                trade_id TEXT NOT NULL,
                symbol TEXT NOT NULL,

                -- Base sizing
                base_position_pct REAL,
                base_position_value REAL,
                base_quantity REAL,

                -- Confidence scoring
                confidence_score REAL,
                confidence_tier TEXT,
                orderbook_score REAL,
                regime_score REAL,
                causality_score REAL,
                edge_persistence_score REAL,

                -- Adjusted sizing
                size_multiplier REAL,
                final_position_pct REAL,
                final_position_value REAL,
                final_quantity REAL,

                -- Reasoning
                adjustment_reason TEXT,
                confidence_reasoning TEXT,

                -- Context at decision time
                regime TEXT,
                orderbook_imbalance REAL,
                spread_pct REAL,
                primary_cause TEXT,

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # RESEARCH_GATE: Winner gate blocks tracking for threshold tuning
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS winner_gate_blocks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER NOT NULL,
                symbol TEXT NOT NULL,

                -- Block reason
                block_reason TEXT NOT NULL,

                -- Market conditions at block time
                regime TEXT,
                confidence_tier TEXT,
                imbalance REAL,
                spread_pct REAL,

                -- For analyzing what we missed
                entry_price REAL,
                signal_type TEXT,

                -- Research mode flag
                research_mode INTEGER DEFAULT 0,

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # PART 7: Maker order telemetry - track maker order fill behavior
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS maker_order_telemetry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id TEXT UNIQUE NOT NULL,
                trade_id TEXT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,

                -- Order details
                limit_price REAL NOT NULL,
                quantity REAL NOT NULL,
                order_type TEXT DEFAULT 'limit',

                -- Timing telemetry
                posted_ts INTEGER NOT NULL,
                filled_ts INTEGER,
                cancelled_ts INTEGER,
                time_to_fill_ms INTEGER,

                -- Fill telemetry
                fill_ratio REAL DEFAULT 0,
                filled_quantity REAL DEFAULT 0,
                avg_fill_price REAL,
                slippage_from_limit_pct REAL,

                -- Market context at post time
                best_bid_at_post REAL,
                best_ask_at_post REAL,
                spread_at_post_pct REAL,
                mid_price_at_post REAL,
                queue_position_estimate INTEGER,

                -- Market context at fill/cancel time
                best_bid_at_fill REAL,
                best_ask_at_fill REAL,
                spread_at_fill_pct REAL,
                price_moved_pct REAL,

                -- Cancel reason (if cancelled)
                cancel_reason TEXT,

                -- Fill model validation
                price_crossed_limit INTEGER DEFAULT 0,
                cross_depth_bps REAL,
                would_fill_conservative INTEGER DEFAULT 0,

                -- Outcome tracking
                status TEXT DEFAULT 'pending',

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
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
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_blocked_signals_timestamp ON blocked_signals(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_blocked_signals_symbol ON blocked_signals(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_blocked_signals_reason ON blocked_signals(block_reason)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_position_sizing_trade_id ON position_sizing(trade_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_position_sizing_symbol ON position_sizing(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_position_sizing_tier ON position_sizing(confidence_tier)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_winner_gate_blocks_timestamp ON winner_gate_blocks(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_winner_gate_blocks_symbol ON winner_gate_blocks(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_winner_gate_blocks_reason ON winner_gate_blocks(block_reason)")

        # Maker order telemetry indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_maker_order_order_id ON maker_order_telemetry(order_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_maker_order_trade_id ON maker_order_telemetry(trade_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_maker_order_symbol ON maker_order_telemetry(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_maker_order_status ON maker_order_telemetry(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_maker_order_posted_ts ON maker_order_telemetry(posted_ts)")

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
            # Normalize symbol to full format (e.g., XRP -> XRPUSDT)
            normalized_symbol = normalize_symbol(signal.symbol)

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
                normalized_symbol,
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
            pass  # Silently ignore database lock errors

    def log_trade_entry(self, result: TradeResult, signal: Signal, is_probe: bool = False, cause_class: str = "FULL"):
        """Log trade entry with cause policy tracking."""
        try:
            # Normalize symbol to full format (e.g., XRP -> XRPUSDT) for consistent storage
            normalized_symbol = normalize_symbol(result.symbol)
            trade_id = result.order_id or f"{normalized_symbol}_{result.timestamp}"

            # Get execution_mode from TradeResult (defaults to "taker" if not set)
            execution_mode = getattr(result, 'execution_mode', 'taker')

            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO trades (
                    trade_id, symbol, side, entry_price, quantity,
                    entry_time, conditions_met, signal_confidence, status,
                    is_probe_cause, cause_class, execution_mode
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'open', ?, ?, ?)
            """, (
                trade_id,
                normalized_symbol,
                result.side,
                result.entry_price,
                result.quantity,
                result.timestamp,
                signal.conditions_met,
                signal.confidence,
                1 if is_probe else 0,
                cause_class,
                execution_mode
            ))

            self.conn.commit()
            logger.debug(f"Trade entry logged: {trade_id} | probe={is_probe} | class={cause_class} | mode={execution_mode}")

        except Exception as e:
            pass  # Silently ignore database lock errors

    def log_trade_exit(self, result: ExitResult, mfe: float = 0, mae: float = 0,
                        spread_at_exit: float = 0, imbalance_at_exit: float = 0.5,
                        pocket_id: str = None):
        """Log trade exit and update trade record with realistic cost model."""
        try:
            cursor = self.conn.cursor()

            # Calculate realistic trading costs
            if COST_MODEL.enabled:
                entry_fee = COST_MODEL.entry_fee_pct
                exit_fee = COST_MODEL.exit_fee_pct
                fees_paid = entry_fee + exit_fee  # Combined fees
                spread_cost = COST_MODEL.spread_cost_pct

                # Slippage: higher when imbalance is extreme (OB looks good but disappears)
                imbalance_extremity = abs(imbalance_at_exit - 0.5) * 2  # 0 to 1
                slippage = COST_MODEL.base_slippage_pct + (
                    imbalance_extremity * COST_MODEL.imbalance_slippage_factor
                )

                # Total round-trip costs
                total_costs = fees_paid + (2 * spread_cost) + slippage
                pnl_after_costs = result.pnl_pct - total_costs
            else:
                entry_fee = 0
                exit_fee = 0
                fees_paid = 0
                spread_cost = 0
                slippage = 0
                total_costs = 0
                pnl_after_costs = result.pnl_pct

            # Get execution mode for logging (handle "auto" mode)
            if COST_MODEL.enabled:
                execution_mode = COST_MODEL.execution_mode
                # If auto mode, we'd need the spread at exit to determine actual mode
                # For now, log the configured mode - actual mode is determined at entry
                if execution_mode == "auto":
                    # Default to taker for auto mode logging (actual decision made at entry)
                    execution_mode = "taker"
            else:
                execution_mode = "taker"

            # Update trade with exit info and costs
            # NOTE: Do NOT overwrite execution_mode - it was set correctly at entry
            # Only set execution_mode if it's NULL (backward compatibility)
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
                    entry_fee_pct = ?,
                    exit_fee_pct = ?,
                    fees_paid_pct = ?,
                    spread_cost_pct = ?,
                    slippage_pct = ?,
                    total_costs_pct = ?,
                    pnl_after_costs_pct = ?,
                    pocket_id = ?,
                    execution_mode = COALESCE(execution_mode, ?),
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
                entry_fee,
                exit_fee,
                fees_paid,
                spread_cost,
                slippage,
                total_costs,
                pnl_after_costs,
                pocket_id,
                execution_mode,
                result.symbol
            ))

            self.conn.commit()

            # Log with cost info
            if COST_MODEL.enabled:
                logger.debug(f"Trade exit logged: {result.symbol} | PnL: {result.pnl_pct:.4f}% | "
                           f"Costs: {total_costs:.4f}% | Net: {pnl_after_costs:.4f}%")
            else:
                logger.debug(f"Trade exit logged: {result.symbol}")

            # Update daily stats
            self._update_daily_stats()

        except Exception as e:
            pass  # Silently ignore database lock errors

    def log_market_snapshot(self, symbol: str, data: Dict):
        """Log periodic market snapshot."""
        try:
            normalized_symbol = normalize_symbol(symbol)
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO market_snapshots (
                    timestamp, symbol, price, price_change_60s,
                    volume_spike, orderbook_imbalance, funding_rate,
                    rsi, btc_volatility
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                int(time.time() * 1000),
                normalized_symbol,
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
            pass  # Silently ignore database lock errors

    def log_tick(self, symbol: str, price: float, volume: float, is_buyer_maker: bool):
        """Log individual tick/trade data."""
        try:
            normalized_symbol = normalize_symbol(symbol)
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO tick_data (timestamp, symbol, price, volume, is_buyer_maker)
                VALUES (?, ?, ?, ?, ?)
            """, (
                int(time.time() * 1000),
                normalized_symbol,
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
            normalized_symbol = normalize_symbol(symbol)
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO orderbook_snapshots (
                    timestamp, symbol, best_bid, best_ask,
                    bid_volume, ask_volume, spread_pct, imbalance
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                int(time.time() * 1000),
                normalized_symbol,
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
            normalized_symbol = normalize_symbol(symbol)
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO indicator_snapshots (
                    timestamp, symbol, price, rsi, volume_spike,
                    price_change_60s, funding_rate, orderbook_imbalance, conditions_met
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                int(time.time() * 1000),
                normalized_symbol,
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
            normalized_symbol = normalize_symbol(symbol)
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
                    normalized_symbol,
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
            with self._db_lock:
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
            normalized_symbol = normalize_symbol(symbol)
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO trade_flow (timestamp, symbol, price, quantity, is_buyer_maker, trade_value)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                int(time.time() * 1000),
                normalized_symbol,
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
            normalized_symbol = normalize_symbol(symbol)
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO cvd_snapshots (
                    timestamp, symbol, cvd_1m, cvd_5m, cvd_15m,
                    buy_volume_1m, sell_volume_1m
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                int(time.time() * 1000),
                normalized_symbol,
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
            normalized_symbol = normalize_symbol(symbol)
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
                normalized_symbol,
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
            pass  # Silently ignore database lock errors

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

            normalized_symbol = normalize_symbol(symbol)
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
                normalized_symbol,
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
            pass  # Silently ignore database lock errors

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

            normalized_symbol = normalize_symbol(symbol)
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
                normalized_symbol,
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
            pass  # Silently ignore database lock errors
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
            normalized_symbol = normalize_symbol(symbol)

            # Get historical median for this symbol
            historical_median = self.get_historical_median_edge(normalized_symbol)

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
                normalized_symbol,
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
            pass  # Silently ignore database lock errors
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
            pass  # Silently ignore database lock errors

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
            normalized_symbol = normalize_symbol(symbol)
            with self._db_lock:  # Thread-safe database access
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
                    normalized_symbol,
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
            pass  # Silently ignore database lock errors

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
            pass  # Silently ignore database lock errors

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

    def log_blocked_signal(
        self,
        symbol: str,
        signal_type: str,
        entry_price: float,
        block_reason: str,
        block_details: str,
        market_conditions: dict,
        regime: str = "unknown"
    ):
        """
        TASK 11: Log a signal that was blocked by no-trade zone detection.

        Block reasons:
        - spread_unstable: Spread changed >X% in 1s
        - delta_noise: Delta variance too high
        - ob_unstable: Top5 OB volume unstable
        - low_liquidity: Depth below minimum threshold
        - regime_avoided: Blocked due to bad regime
        """
        try:
            normalized_symbol = normalize_symbol(symbol)
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO blocked_signals (
                    timestamp, symbol, signal_type, entry_price,
                    block_reason, block_details,
                    spread_pct, spread_change_1s, delta_variance,
                    ob_volume_instability, liquidity_depth, regime
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                int(time.time() * 1000),
                normalized_symbol,
                signal_type,
                entry_price,
                block_reason,
                block_details,
                market_conditions.get("spread_pct", 0),
                market_conditions.get("spread_change_1s", 0),
                market_conditions.get("delta_variance", 0),
                market_conditions.get("ob_volume_instability", 0),
                market_conditions.get("liquidity_depth", 0),
                regime
            ))
            self.conn.commit()
            logger.debug(f"Blocked signal logged: {symbol} - {block_reason}")
        except Exception as e:
            pass  # Silently ignore database lock errors

    def log_winner_gate_block(
        self,
        symbol: str,
        signal_type: str,
        entry_price: float,
        block_reason: str,
        regime: str,
        confidence_tier: str,
        imbalance: float,
        spread_pct: float,
        research_mode: bool = False
    ):
        """
        RESEARCH_GATE: Log a signal blocked by winner gate for threshold analysis.
        """
        try:
            normalized_symbol = normalize_symbol(symbol)
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO winner_gate_blocks (
                    timestamp, symbol, block_reason, regime,
                    confidence_tier, imbalance, spread_pct,
                    entry_price, signal_type, research_mode
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                int(time.time() * 1000),
                normalized_symbol,
                block_reason,
                regime,
                confidence_tier,
                imbalance,
                spread_pct,
                entry_price,
                signal_type,
                1 if research_mode else 0
            ))
            self.conn.commit()
        except Exception as e:
            pass  # Silently ignore database lock errors

    def update_blocked_signal_outcome(self, symbol: str, timestamp_ms: int,
                                       price_after_5s: float, price_after_10s: float,
                                       would_have_pnl: float):
        """Update blocked signal with what would have happened."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                UPDATE blocked_signals
                SET price_after_5s = ?, price_after_10s = ?, would_have_pnl = ?
                WHERE symbol = ? AND timestamp = ?
            """, (price_after_5s, price_after_10s, would_have_pnl, symbol, timestamp_ms))
            self.conn.commit()
        except Exception as e:
            pass  # Silently ignore database lock errors

    def get_blocked_signal_stats(self) -> Dict:
        """
        TASK 11: Get statistics on blocked signals.

        Returns:
        - Total blocked signals
        - Block reason breakdown
        - Estimated savings (if would_have_pnl is tracked)
        - % of signals blocked
        """
        try:
            cursor = self.conn.cursor()

            # Get total blocked and breakdown by reason
            cursor.execute("""
                SELECT
                    block_reason,
                    COUNT(*) as count,
                    AVG(would_have_pnl) as avg_would_have_pnl
                FROM blocked_signals
                GROUP BY block_reason
                ORDER BY count DESC
            """)
            reason_breakdown = {}
            total_blocked = 0
            total_saved = 0
            for row in cursor.fetchall():
                reason_breakdown[row[0]] = {
                    "count": row[1],
                    "avg_would_have_pnl": round(row[2], 4) if row[2] else None
                }
                total_blocked += row[1]
                if row[2] and row[2] < 0:
                    total_saved += abs(row[2]) * row[1]

            # Get total signals (executed + blocked)
            cursor.execute("SELECT COUNT(*) FROM signals")
            total_signals = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM blocked_signals")
            blocked_count = cursor.fetchone()[0]

            # Calculate block rate
            block_rate = (blocked_count / (total_signals + blocked_count) * 100) if (total_signals + blocked_count) > 0 else 0

            # Get per-symbol breakdown
            cursor.execute("""
                SELECT symbol, COUNT(*) as count
                FROM blocked_signals
                GROUP BY symbol
                ORDER BY count DESC
            """)
            symbol_breakdown = {row[0]: row[1] for row in cursor.fetchall()}

            # Calculate estimated savings from blocking bad trades
            cursor.execute("""
                SELECT
                    COUNT(*) as bad_blocks,
                    AVG(would_have_pnl) as avg_loss_avoided
                FROM blocked_signals
                WHERE would_have_pnl < 0
            """)
            row = cursor.fetchone()
            bad_blocks = row[0] if row[0] else 0
            avg_loss_avoided = row[1] if row[1] else 0

            return {
                "total_blocked": total_blocked,
                "block_rate_pct": round(block_rate, 1),
                "reason_breakdown": reason_breakdown,
                "symbol_breakdown": symbol_breakdown,
                "bad_trades_avoided": bad_blocks,
                "avg_loss_avoided_pct": round(avg_loss_avoided, 4) if avg_loss_avoided else 0,
                "estimated_savings_pct": round(total_saved, 4)
            }
        except Exception as e:
            logger.error(f"Error getting blocked signal stats: {e}")
            return {"error": str(e)}

    def log_position_sizing(
        self,
        trade_id: str,
        symbol: str,
        base_quantity: float,
        base_value: float,
        final_quantity: float,
        final_value: float,
        confidence_data: dict,
        context_data: dict = None,
        # New: Multiplier tracking for single-probe audit
        base_size: float = None,
        applied_multipliers: list = None,
        final_size: float = None,
        is_probe_trade: bool = False
    ):
        """
        TASK 12: Log position sizing decision with confidence scoring.

        Records both base sizing and confidence-adjusted final sizing
        for analysis of adaptive sizing performance.

        New multiplier tracking (Part 2 - Single Probe):
        - base_size: Original size before any multipliers
        - applied_multipliers: List of multipliers applied (e.g., ["cause_probe=0.10", "confidence=1.5"])
        - final_size: Final size after all multipliers
        - is_probe_trade: True if this is a PROBE cause trade (0.10x size)
        """
        try:
            normalized_symbol = normalize_symbol(symbol)
            context_data = context_data or {}
            base_pct = (base_value / 10000) * 100  # Assume $10k capital
            final_pct = (final_value / 10000) * 100

            # Serialize multipliers list to JSON
            multipliers_json = json.dumps(applied_multipliers or [])

            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO position_sizing (
                    timestamp, trade_id, symbol,
                    base_position_pct, base_position_value, base_quantity,
                    confidence_score, confidence_tier,
                    orderbook_score, regime_score, causality_score, edge_persistence_score,
                    size_multiplier, final_position_pct, final_position_value, final_quantity,
                    adjustment_reason, confidence_reasoning,
                    regime, orderbook_imbalance, spread_pct, primary_cause,
                    base_size, applied_multipliers, final_size, is_probe_trade
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                int(time.time() * 1000),
                trade_id,
                normalized_symbol,
                round(base_pct, 4),
                round(base_value, 2),
                round(base_quantity, 6),
                round(confidence_data.get("total_score", 0), 2),
                confidence_data.get("tier", "unknown"),
                round(confidence_data.get("orderbook_score", 0), 4),
                round(confidence_data.get("regime_score", 0), 4),
                round(confidence_data.get("causality_score", 0), 4),
                round(confidence_data.get("edge_persistence_score", 0), 4),
                round(confidence_data.get("size_multiplier", 1.0), 2),
                round(final_pct, 4),
                round(final_value, 2),
                round(final_quantity, 6),
                confidence_data.get("adjustment_reason", ""),
                confidence_data.get("reasoning", ""),
                context_data.get("regime", "unknown"),
                round(context_data.get("orderbook_imbalance", 0), 4),
                round(context_data.get("spread_pct", 0), 4),
                context_data.get("primary_cause", "unknown"),
                round(base_size, 6) if base_size else None,
                multipliers_json,
                round(final_size, 6) if final_size else None,
                1 if is_probe_trade else 0
            ))
            self.conn.commit()
            probe_label = " [PROBE]" if is_probe_trade else ""
            logger.debug(f"Position sizing logged: {trade_id} - tier={confidence_data.get('tier')}{probe_label}")
        except Exception as e:
            pass  # Silently ignore database lock errors

    def get_position_sizing_stats(self) -> Dict:
        """
        TASK 12: Get aggregate position sizing statistics.

        Returns:
        - Tier distribution (HIGH/MEDIUM/LOW)
        - Performance by tier (did HIGH confidence trades perform better?)
        - Average sizing adjustments
        """
        try:
            cursor = self.conn.cursor()

            # Get tier distribution
            cursor.execute("""
                SELECT
                    confidence_tier,
                    COUNT(*) as count,
                    AVG(confidence_score) as avg_score,
                    AVG(size_multiplier) as avg_multiplier
                FROM position_sizing
                GROUP BY confidence_tier
                ORDER BY avg_score DESC
            """)
            tier_distribution = {}
            for row in cursor.fetchall():
                tier_distribution[row[0]] = {
                    "count": row[1],
                    "avg_score": round(row[2], 1) if row[2] else 0,
                    "avg_multiplier": round(row[3], 2) if row[3] else 1.0
                }

            # Get performance by tier (join with trades table)
            cursor.execute("""
                SELECT
                    ps.confidence_tier,
                    COUNT(*) as trades,
                    AVG(t.pnl_pct) as avg_pnl_pct,
                    SUM(CASE WHEN t.pnl_pct > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as win_rate
                FROM position_sizing ps
                LEFT JOIN trades t ON ps.trade_id = t.trade_id
                WHERE t.status = 'closed'
                GROUP BY ps.confidence_tier
                ORDER BY avg_pnl_pct DESC
            """)
            tier_performance = {}
            for row in cursor.fetchall():
                tier_performance[row[0]] = {
                    "trades": row[1],
                    "avg_pnl_pct": round(row[2], 4) if row[2] else 0,
                    "win_rate": round(row[3], 1) if row[3] else 0
                }

            # Get component score analysis
            cursor.execute("""
                SELECT
                    AVG(orderbook_score) as avg_ob,
                    AVG(regime_score) as avg_regime,
                    AVG(causality_score) as avg_causality,
                    AVG(edge_persistence_score) as avg_edge
                FROM position_sizing
            """)
            row = cursor.fetchone()
            component_averages = {
                "orderbook_score": round(row[0], 3) if row[0] else 0,
                "regime_score": round(row[1], 3) if row[1] else 0,
                "causality_score": round(row[2], 3) if row[2] else 0,
                "edge_persistence_score": round(row[3], 3) if row[3] else 0
            }

            # Get sizing impact analysis
            cursor.execute("""
                SELECT
                    AVG(base_position_value) as avg_base_value,
                    AVG(final_position_value) as avg_final_value,
                    AVG(size_multiplier) as avg_multiplier
                FROM position_sizing
            """)
            row = cursor.fetchone()
            sizing_summary = {
                "avg_base_value": round(row[0], 2) if row[0] else 0,
                "avg_final_value": round(row[1], 2) if row[1] else 0,
                "avg_multiplier": round(row[2], 2) if row[2] else 1.0,
                "avg_adjustment_pct": round((row[2] - 1) * 100, 1) if row[2] else 0
            }

            return {
                "tier_distribution": tier_distribution,
                "tier_performance": tier_performance,
                "component_averages": component_averages,
                "sizing_summary": sizing_summary
            }
        except Exception as e:
            logger.error(f"Error getting position sizing stats: {e}")
            return {"error": str(e)}

    def log_maker_order_posted(
        self,
        order_id: str,
        symbol: str,
        side: str,
        limit_price: float,
        quantity: float,
        market_data: dict,
        trade_id: str = None
    ):
        """
        PART 7: Log maker order when posted.

        Records initial order state for fill analysis:
        - Limit price relative to market
        - Spread at post time
        - Queue position estimate
        """
        normalized_symbol = normalize_symbol(symbol)
        best_bid = market_data.get("best_bid", 0)
        best_ask = market_data.get("best_ask", 0)
        mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else 0
        spread_pct = ((best_ask - best_bid) / mid_price * 100) if mid_price > 0 else 0
        queue_position = market_data.get("queue_position_estimate", 0)
        posted_ts = int(time.time() * 1000)

        sql = """
            INSERT INTO maker_order_telemetry (
                order_id, trade_id, symbol, side,
                limit_price, quantity, order_type,
                posted_ts,
                best_bid_at_post, best_ask_at_post, spread_at_post_pct,
                mid_price_at_post, queue_position_estimate,
                status
            ) VALUES (?, ?, ?, ?, ?, ?, 'limit', ?, ?, ?, ?, ?, ?, 'pending')
        """
        params = (
            order_id,
            trade_id,
            normalized_symbol,
            side,
            limit_price,
            quantity,
            posted_ts,
            best_bid,
            best_ask,
            round(spread_pct, 4),
            mid_price,
            queue_position
        )

        success = self._execute_with_retry(sql, params)
        if success:
            logger.debug(f"Maker order posted: {order_id} @ {limit_price}")
        else:
            logger.warning(f"Failed to persist maker order posted: {order_id}")

    def log_maker_order_filled(
        self,
        order_id: str,
        filled_quantity: float,
        avg_fill_price: float,
        market_data: dict,
        price_crossed_limit: bool = False,
        cross_depth_bps: float = 0,
        symbol: str = None,
        side: str = None,
        limit_price: float = None
    ):
        """
        PART 7: Log maker order fill.

        Records:
        - Time to fill
        - Fill ratio
        - Price movement during order life
        - Whether price crossed limit (for conservative fill model validation)

        Added fallback: If order not found in DB (due to race/lock), insert it now.
        """
        filled_ts = int(time.time() * 1000)
        best_bid = market_data.get("best_bid", 0)
        best_ask = market_data.get("best_ask", 0)
        spread_pct = 0
        mid_price = 0
        if best_bid and best_ask:
            mid_price = (best_bid + best_ask) / 2
            spread_pct = (best_ask - best_bid) / mid_price * 100 if mid_price > 0 else 0

        # Try to get original order details with retry
        row = None
        for attempt in range(3):
            try:
                with self._db_lock:
                    cursor = self.conn.cursor()
                    cursor.execute("""
                        SELECT posted_ts, limit_price, quantity, mid_price_at_post, symbol, side
                        FROM maker_order_telemetry
                        WHERE order_id = ?
                    """, (order_id,))
                    row = cursor.fetchone()
                break
            except sqlite3.OperationalError as e:
                if attempt < 2:
                    time.sleep(0.05 * (attempt + 1))
                    continue
                logger.warning(f"Failed to query maker order {order_id}: {e}")

        if not row:
            # FALLBACK: Insert the order now with filled status
            # This handles race conditions where posted wasn't persisted
            logger.info(f"Maker order {order_id} not found, inserting with filled status")
            use_symbol = symbol or "UNKNOWN"
            use_side = side or "UNKNOWN"
            use_limit_price = limit_price or avg_fill_price

            insert_sql = """
                INSERT INTO maker_order_telemetry (
                    order_id, symbol, side, limit_price, quantity, order_type,
                    posted_ts, filled_ts, filled_quantity, avg_fill_price,
                    best_bid_at_post, best_ask_at_post, spread_at_post_pct,
                    mid_price_at_post, best_bid_at_fill, best_ask_at_fill,
                    spread_at_fill_pct, fill_ratio, status
                ) VALUES (?, ?, ?, ?, ?, 'limit', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'filled')
            """
            insert_params = (
                order_id, use_symbol, use_side, use_limit_price, filled_quantity,
                filled_ts - 500,  # Assume posted 500ms before fill
                filled_ts, filled_quantity, avg_fill_price,
                best_bid, best_ask, round(spread_pct, 4), mid_price,
                best_bid, best_ask, round(spread_pct, 4), 1.0
            )
            success = self._execute_with_retry(insert_sql, insert_params)
            if success:
                logger.debug(f"Maker order filled (fallback insert): {order_id}")
            return

        # Normal path: update existing order
        posted_ts, db_limit_price, quantity, mid_at_post, db_symbol, db_side = row
        time_to_fill_ms = filled_ts - posted_ts
        fill_ratio = filled_quantity / quantity if quantity > 0 else 1.0
        use_limit = db_limit_price or limit_price or avg_fill_price
        slippage = ((avg_fill_price - use_limit) / use_limit * 100) if use_limit > 0 else 0
        current_mid = mid_price
        price_moved_pct = ((current_mid - mid_at_post) / mid_at_post * 100) if mid_at_post > 0 else 0

        update_sql = """
            UPDATE maker_order_telemetry SET
                filled_ts = ?,
                time_to_fill_ms = ?,
                fill_ratio = ?,
                filled_quantity = ?,
                avg_fill_price = ?,
                slippage_from_limit_pct = ?,
                best_bid_at_fill = ?,
                best_ask_at_fill = ?,
                spread_at_fill_pct = ?,
                price_moved_pct = ?,
                price_crossed_limit = ?,
                cross_depth_bps = ?,
                status = ?
            WHERE order_id = ?
        """
        update_params = (
            filled_ts,
            time_to_fill_ms,
            round(fill_ratio, 4),
            filled_quantity,
            avg_fill_price,
            round(slippage, 6),
            best_bid,
            best_ask,
            round(spread_pct, 4),
            round(price_moved_pct, 4),
            1 if price_crossed_limit else 0,
            round(cross_depth_bps, 2),
            'filled' if fill_ratio >= 0.99 else 'partial',
            order_id
        )
        success = self._execute_with_retry(update_sql, update_params)
        if success:
            logger.debug(f"Maker order filled: {order_id} | {fill_ratio*100:.0f}% in {time_to_fill_ms}ms")
        else:
            logger.warning(f"Failed to update maker order filled: {order_id}")

    def log_maker_order_cancelled(
        self,
        order_id: str,
        cancel_reason: str,
        market_data: dict,
        symbol: str = None,
        side: str = None,
        limit_price: float = None
    ):
        """
        PART 7: Log maker order cancellation.

        Cancel reasons:
        - timeout: Order didn't fill within time limit
        - price_moved: Market moved away from limit
        - signal_invalid: Original signal invalidated
        - manual: Manual cancellation

        Added fallback: If order not found, insert with cancelled status.
        """
        cancelled_ts = int(time.time() * 1000)
        best_bid = market_data.get("best_bid", 0)
        best_ask = market_data.get("best_ask", 0)
        spread_pct = 0
        mid_price = 0
        if best_bid and best_ask:
            mid_price = (best_bid + best_ask) / 2
            spread_pct = (best_ask - best_bid) / mid_price * 100 if mid_price > 0 else 0

        # Try to get original order details with retry
        row = None
        for attempt in range(3):
            try:
                with self._db_lock:
                    cursor = self.conn.cursor()
                    cursor.execute("""
                        SELECT posted_ts, mid_price_at_post, symbol, side, limit_price
                        FROM maker_order_telemetry
                        WHERE order_id = ?
                    """, (order_id,))
                    row = cursor.fetchone()
                break
            except sqlite3.OperationalError as e:
                if attempt < 2:
                    time.sleep(0.05 * (attempt + 1))
                    continue

        if not row:
            # FALLBACK: Insert the order now with cancelled status
            logger.info(f"Maker order {order_id} not found, inserting with cancelled status")
            use_symbol = symbol or "UNKNOWN"
            use_side = side or "UNKNOWN"
            use_limit_price = limit_price or mid_price

            insert_sql = """
                INSERT INTO maker_order_telemetry (
                    order_id, symbol, side, limit_price, quantity, order_type,
                    posted_ts, cancelled_ts,
                    best_bid_at_post, best_ask_at_post, spread_at_post_pct,
                    mid_price_at_post, cancel_reason, status
                ) VALUES (?, ?, ?, ?, 0, 'limit', ?, ?, ?, ?, ?, ?, ?, 'cancelled')
            """
            insert_params = (
                order_id, use_symbol, use_side, use_limit_price,
                cancelled_ts - 500,  # Assume posted 500ms before cancel
                cancelled_ts,
                best_bid, best_ask, round(spread_pct, 4), mid_price,
                cancel_reason
            )
            success = self._execute_with_retry(insert_sql, insert_params)
            if success:
                logger.debug(f"Maker order cancelled (fallback insert): {order_id}")
            return

        # Normal path: update existing order
        posted_ts, mid_at_post, db_symbol, db_side, db_limit_price = row
        time_alive_ms = cancelled_ts - posted_ts
        current_mid = mid_price
        price_moved_pct = ((current_mid - mid_at_post) / mid_at_post * 100) if mid_at_post > 0 else 0

        update_sql = """
            UPDATE maker_order_telemetry SET
                cancelled_ts = ?,
                time_to_fill_ms = ?,
                best_bid_at_fill = ?,
                best_ask_at_fill = ?,
                spread_at_fill_pct = ?,
                price_moved_pct = ?,
                cancel_reason = ?,
                status = 'cancelled'
            WHERE order_id = ?
        """
        update_params = (
            cancelled_ts,
            time_alive_ms,
            best_bid,
            best_ask,
            round(spread_pct, 4),
            round(price_moved_pct, 4),
            cancel_reason,
            order_id
        )
        success = self._execute_with_retry(update_sql, update_params)
        if success:
            logger.debug(f"Maker order cancelled: {order_id} | reason={cancel_reason}")
        else:
            logger.warning(f"Failed to update maker order cancelled: {order_id}")

    def update_maker_fill_model_validation(
        self,
        order_id: str,
        would_fill_conservative: bool
    ):
        """
        PART 7: Update whether this order would have filled under conservative model.

        Used for backtesting fill model accuracy.
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                UPDATE maker_order_telemetry SET
                    would_fill_conservative = ?
                WHERE order_id = ?
            """, (1 if would_fill_conservative else 0, order_id))
            self.conn.commit()
        except Exception as e:
            pass

    def get_maker_fill_stats(self, days: int = 7) -> dict:
        """
        PART 7: Get maker order fill statistics.

        Returns:
        - Fill rate (% of orders fully filled)
        - Average time to fill
        - Cancel rate and reasons
        - Fill model accuracy (conservative vs actual)
        """
        try:
            cursor = self.conn.cursor()
            cutoff_ms = int((time.time() - days * 86400) * 1000)

            # Overall stats
            cursor.execute("""
                SELECT
                    COUNT(*) as total_orders,
                    SUM(CASE WHEN status = 'filled' THEN 1 ELSE 0 END) as filled,
                    SUM(CASE WHEN status = 'partial' THEN 1 ELSE 0 END) as partial,
                    SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) as cancelled,
                    AVG(CASE WHEN status IN ('filled', 'partial') THEN time_to_fill_ms ELSE NULL END) as avg_time_to_fill,
                    AVG(CASE WHEN status IN ('filled', 'partial') THEN fill_ratio ELSE NULL END) as avg_fill_ratio,
                    AVG(CASE WHEN status IN ('filled', 'partial') THEN slippage_from_limit_pct ELSE NULL END) as avg_slippage
                FROM maker_order_telemetry
                WHERE posted_ts > ?
            """, (cutoff_ms,))
            row = cursor.fetchone()

            if not row or row[0] == 0:
                return {"message": "No maker order data"}

            total = row[0]
            filled = row[1] or 0
            partial = row[2] or 0
            cancelled = row[3] or 0
            fill_rate = ((filled + partial) / total * 100) if total > 0 else 0

            # Cancel reason breakdown
            cursor.execute("""
                SELECT cancel_reason, COUNT(*) as count
                FROM maker_order_telemetry
                WHERE posted_ts > ? AND status = 'cancelled'
                GROUP BY cancel_reason
                ORDER BY count DESC
            """, (cutoff_ms,))
            cancel_reasons = {r[0]: r[1] for r in cursor.fetchall()}

            # Conservative fill model validation
            cursor.execute("""
                SELECT
                    SUM(CASE WHEN status = 'filled' AND would_fill_conservative = 1 THEN 1 ELSE 0 END) as conservative_correct_fill,
                    SUM(CASE WHEN status = 'filled' AND would_fill_conservative = 0 THEN 1 ELSE 0 END) as conservative_missed,
                    SUM(CASE WHEN status IN ('cancelled', 'partial') AND would_fill_conservative = 1 THEN 1 ELSE 0 END) as conservative_false_positive,
                    SUM(CASE WHEN status IN ('cancelled', 'partial') AND would_fill_conservative = 0 THEN 1 ELSE 0 END) as conservative_correct_reject
                FROM maker_order_telemetry
                WHERE posted_ts > ?
            """, (cutoff_ms,))
            fill_model_row = cursor.fetchone()

            # Price cross stats
            cursor.execute("""
                SELECT
                    AVG(CASE WHEN price_crossed_limit = 1 THEN cross_depth_bps ELSE NULL END) as avg_cross_depth,
                    SUM(CASE WHEN price_crossed_limit = 1 AND status = 'filled' THEN 1 ELSE 0 END) as filled_with_cross,
                    SUM(CASE WHEN price_crossed_limit = 0 AND status = 'filled' THEN 1 ELSE 0 END) as filled_no_cross
                FROM maker_order_telemetry
                WHERE posted_ts > ?
            """, (cutoff_ms,))
            cross_row = cursor.fetchone()

            # Per-symbol breakdown
            cursor.execute("""
                SELECT
                    symbol,
                    COUNT(*) as orders,
                    AVG(fill_ratio) as avg_fill_ratio,
                    AVG(time_to_fill_ms) as avg_time_ms
                FROM maker_order_telemetry
                WHERE posted_ts > ? AND status != 'pending'
                GROUP BY symbol
                ORDER BY orders DESC
            """, (cutoff_ms,))
            symbol_breakdown = {
                r[0]: {
                    "orders": r[1],
                    "avg_fill_ratio": round(r[2], 3) if r[2] else 0,
                    "avg_time_ms": round(r[3], 0) if r[3] else 0
                }
                for r in cursor.fetchall()
            }

            return {
                "period_days": days,
                "total_orders": total,
                "filled": filled,
                "partial": partial,
                "cancelled": cancelled,
                "fill_rate_pct": round(fill_rate, 1),
                "avg_time_to_fill_ms": round(row[4], 0) if row[4] else 0,
                "avg_fill_ratio": round(row[5], 3) if row[5] else 0,
                "avg_slippage_pct": round(row[6], 4) if row[6] else 0,
                "cancel_reasons": cancel_reasons,
                "fill_model_validation": {
                    "conservative_correct_fill": fill_model_row[0] or 0,
                    "conservative_missed": fill_model_row[1] or 0,
                    "conservative_false_positive": fill_model_row[2] or 0,
                    "conservative_correct_reject": fill_model_row[3] or 0
                },
                "price_cross_stats": {
                    "avg_cross_depth_bps": round(cross_row[0], 2) if cross_row[0] else 0,
                    "filled_with_cross": cross_row[1] or 0,
                    "filled_no_cross": cross_row[2] or 0
                },
                "symbol_breakdown": symbol_breakdown
            }
        except Exception as e:
            logger.error(f"Error getting maker fill stats: {e}")
            return {"error": str(e)}

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
