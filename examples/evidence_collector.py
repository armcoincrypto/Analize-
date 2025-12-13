#!/usr/bin/env python3
"""
Evidence Collector - The ONLY System That Matters for Profitability

This module implements the 7 critical systems for collecting behavioral
and outcome data that actually determines if a strategy is profitable.

Systems:
1. Trade Outcome Database - Full trade context (entry, during, exit)
2. Signal Effectiveness Scorecard - Which signal combinations work
3. Regime-Specific Statistics - Performance by market regime
4. False Signal Log - Learn when NOT to trade
5. Time-of-Day Analysis - Best/worst trading times
6. Capital Erosion Monitor - Kill switch for drawdowns
7. Strategy Confidence Meter - Permission to trade

Author: Cloud AI Analyzer
"""

import sqlite3
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from enum import Enum
import pandas as pd
import numpy as np

# Database path
DB_PATH = Path(__file__).parent / "evidence.db"


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class ExitReason(Enum):
    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    TIME_STOP = "time_stop"
    REGIME_CHANGE = "regime_change"
    MANUAL = "manual"
    SESSION_END = "session_end"
    SIGNAL_REVERSAL = "signal_reversal"


class MarketRegime(Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    CRISIS = "crisis"


@dataclass
class SignalContext:
    """Signals active at trade entry."""
    bollinger_touch: bool = False
    bollinger_penetration_pct: float = 0.0
    rsi_value: float = 50.0
    rsi_signal: str = "neutral"  # oversold, overbought, neutral
    stochastic_value: float = 50.0
    stochastic_signal: str = "neutral"
    macd_signal: str = "neutral"  # bullish, bearish, neutral
    funding_rate: float = 0.0
    funding_signal: str = "neutral"  # buy, sell, neutral
    whale_flow: float = 0.0  # positive = inflow, negative = outflow
    whale_signal: str = "neutral"
    order_flow_imbalance: float = 1.0  # bid/ask ratio
    order_flow_signal: str = "neutral"
    news_events: int = 0
    news_sentiment: str = "neutral"  # positive, negative, neutral
    ml_prediction: str = "hold"
    ml_confidence: float = 0.0
    correlation_regime: str = "moderate"

    def active_signals(self) -> List[str]:
        """Return list of active buy signals."""
        signals = []
        if self.bollinger_touch:
            signals.append("BB")
        if self.rsi_signal == "oversold":
            signals.append("RSI")
        if self.stochastic_signal == "oversold":
            signals.append("STOCH")
        if self.macd_signal == "bullish":
            signals.append("MACD")
        if self.funding_signal in ["buy", "strong_buy"]:
            signals.append("FUNDING")
        if self.whale_signal == "buy":
            signals.append("WHALE")
        if self.order_flow_signal in ["buy", "strong_buy"]:
            signals.append("ORDERFLOW")
        if self.news_sentiment == "positive":
            signals.append("NEWS")
        if self.ml_prediction == "buy":
            signals.append("ML")
        return signals

    def signal_combination_key(self) -> str:
        """Return sorted combination key for grouping."""
        return "+".join(sorted(self.active_signals())) or "NONE"


@dataclass
class TradeOutcome:
    """Complete trade record with full context."""
    # Identity
    trade_id: str
    symbol: str
    side: str  # long, short

    # Entry context
    entry_time: datetime
    entry_price: float
    entry_regime: str
    entry_signals: SignalContext
    entry_confidence: float

    # During trade
    max_favorable_excursion: float = 0.0  # Best unrealized profit %
    max_adverse_excursion: float = 0.0    # Worst unrealized loss %
    time_in_trade_seconds: int = 0
    regime_changes: int = 0

    # Exit
    exit_time: Optional[datetime] = None
    exit_price: float = 0.0
    exit_reason: str = ""
    slippage_pct: float = 0.0
    fees_pct: float = 0.0

    # Result
    gross_pnl_pct: float = 0.0
    net_pnl_pct: float = 0.0
    is_win: bool = False


@dataclass
class FalseSignal:
    """Record of signal that did NOT result in expected move."""
    timestamp: datetime
    symbol: str
    signal_type: str  # e.g., "BB_touch_no_bounce"
    expected_direction: str  # up, down
    actual_move_pct: float

    # Context at signal
    regime: str
    funding_rate: float
    order_flow_imbalance: float
    whale_flow: float
    news_events: int
    rsi: float
    stochastic: float

    # What happened after
    price_after_1h_pct: float = 0.0
    price_after_4h_pct: float = 0.0
    price_after_24h_pct: float = 0.0


# =============================================================================
# 1. TRADE OUTCOME DATABASE
# =============================================================================

class TradeOutcomeDatabase:
    """
    Stores complete trade context for every trade (paper or real).

    This is THE most important data for profitability analysis.
    """

    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(DB_PATH)
        self._init_db()

    def _init_db(self):
        """Initialize database tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Main trades table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,

                -- Entry
                entry_time TEXT NOT NULL,
                entry_price REAL NOT NULL,
                entry_regime TEXT,
                entry_signals_json TEXT,
                entry_confidence REAL,

                -- During
                mfe REAL DEFAULT 0,
                mae REAL DEFAULT 0,
                time_in_trade_seconds INTEGER DEFAULT 0,
                regime_changes INTEGER DEFAULT 0,

                -- Exit
                exit_time TEXT,
                exit_price REAL,
                exit_reason TEXT,
                slippage_pct REAL DEFAULT 0,
                fees_pct REAL DEFAULT 0,

                -- Result
                gross_pnl_pct REAL,
                net_pnl_pct REAL,
                is_win INTEGER,

                -- Meta
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Signal effectiveness tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signal_combinations (
                combination_key TEXT PRIMARY KEY,
                total_trades INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                total_pnl_pct REAL DEFAULT 0,
                avg_mfe REAL DEFAULT 0,
                avg_mae REAL DEFAULT 0,
                max_drawdown REAL DEFAULT 0,
                last_updated TEXT
            )
        """)

        # Regime statistics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS regime_stats (
                regime TEXT PRIMARY KEY,
                total_trades INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                total_pnl_pct REAL DEFAULT 0,
                avg_return REAL DEFAULT 0,
                win_rate REAL DEFAULT 0,
                last_updated TEXT
            )
        """)

        # False signals log
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS false_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                expected_direction TEXT,
                actual_move_pct REAL,

                regime TEXT,
                funding_rate REAL,
                order_flow_imbalance REAL,
                whale_flow REAL,
                news_events INTEGER,
                rsi REAL,
                stochastic REAL,

                price_after_1h_pct REAL,
                price_after_4h_pct REAL,
                price_after_24h_pct REAL
            )
        """)

        # Time analysis
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS time_stats (
                hour INTEGER,
                day_of_week INTEGER,
                total_trades INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                total_pnl_pct REAL DEFAULT 0,
                avg_return REAL DEFAULT 0,
                PRIMARY KEY (hour, day_of_week)
            )
        """)

        # Capital erosion tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS capital_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                capital REAL NOT NULL,
                drawdown_pct REAL DEFAULT 0,
                consecutive_losses INTEGER DEFAULT 0,
                daily_pnl REAL DEFAULT 0,
                cumulative_fees REAL DEFAULT 0
            )
        """)

        conn.commit()
        conn.close()

        print(f"[EVIDENCE] Trade Outcome Database initialized: {self.db_path}")

    def record_trade(self, trade: TradeOutcome) -> None:
        """Record a completed trade with full context."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        signals_json = json.dumps(asdict(trade.entry_signals))

        cursor.execute("""
            INSERT OR REPLACE INTO trades (
                trade_id, symbol, side,
                entry_time, entry_price, entry_regime, entry_signals_json, entry_confidence,
                mfe, mae, time_in_trade_seconds, regime_changes,
                exit_time, exit_price, exit_reason, slippage_pct, fees_pct,
                gross_pnl_pct, net_pnl_pct, is_win
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade.trade_id, trade.symbol, trade.side,
            trade.entry_time.isoformat(), trade.entry_price, trade.entry_regime,
            signals_json, trade.entry_confidence,
            trade.max_favorable_excursion, trade.max_adverse_excursion,
            trade.time_in_trade_seconds, trade.regime_changes,
            trade.exit_time.isoformat() if trade.exit_time else None,
            trade.exit_price, trade.exit_reason, trade.slippage_pct, trade.fees_pct,
            trade.gross_pnl_pct, trade.net_pnl_pct, 1 if trade.is_win else 0
        ))

        # Update signal combination stats
        combo_key = trade.entry_signals.signal_combination_key()
        self._update_signal_combination(cursor, combo_key, trade)

        # Update regime stats
        self._update_regime_stats(cursor, trade.entry_regime, trade)

        # Update time stats
        self._update_time_stats(cursor, trade)

        conn.commit()
        conn.close()

        print(f"[EVIDENCE] Trade recorded: {trade.trade_id} | {trade.symbol} | "
              f"{'WIN' if trade.is_win else 'LOSS'} | {trade.net_pnl_pct:+.2f}%")

    def _update_signal_combination(self, cursor, combo_key: str, trade: TradeOutcome):
        """Update signal combination statistics."""
        cursor.execute("SELECT * FROM signal_combinations WHERE combination_key = ?", (combo_key,))
        row = cursor.fetchone()

        if row:
            cursor.execute("""
                UPDATE signal_combinations SET
                    total_trades = total_trades + 1,
                    wins = wins + ?,
                    losses = losses + ?,
                    total_pnl_pct = total_pnl_pct + ?,
                    avg_mfe = (avg_mfe * total_trades + ?) / (total_trades + 1),
                    avg_mae = (avg_mae * total_trades + ?) / (total_trades + 1),
                    last_updated = ?
                WHERE combination_key = ?
            """, (
                1 if trade.is_win else 0,
                0 if trade.is_win else 1,
                trade.net_pnl_pct,
                trade.max_favorable_excursion,
                trade.max_adverse_excursion,
                datetime.now().isoformat(),
                combo_key
            ))
        else:
            cursor.execute("""
                INSERT INTO signal_combinations
                (combination_key, total_trades, wins, losses, total_pnl_pct, avg_mfe, avg_mae, last_updated)
                VALUES (?, 1, ?, ?, ?, ?, ?, ?)
            """, (
                combo_key,
                1 if trade.is_win else 0,
                0 if trade.is_win else 1,
                trade.net_pnl_pct,
                trade.max_favorable_excursion,
                trade.max_adverse_excursion,
                datetime.now().isoformat()
            ))

    def _update_regime_stats(self, cursor, regime: str, trade: TradeOutcome):
        """Update regime-specific statistics."""
        cursor.execute("SELECT * FROM regime_stats WHERE regime = ?", (regime,))
        row = cursor.fetchone()

        if row:
            total = row[1] + 1
            wins = row[2] + (1 if trade.is_win else 0)
            cursor.execute("""
                UPDATE regime_stats SET
                    total_trades = ?,
                    wins = ?,
                    losses = losses + ?,
                    total_pnl_pct = total_pnl_pct + ?,
                    avg_return = (total_pnl_pct + ?) / ?,
                    win_rate = ? * 100.0 / ?,
                    last_updated = ?
                WHERE regime = ?
            """, (
                total, wins,
                0 if trade.is_win else 1,
                trade.net_pnl_pct,
                trade.net_pnl_pct, total,
                wins, total,
                datetime.now().isoformat(),
                regime
            ))
        else:
            cursor.execute("""
                INSERT INTO regime_stats
                (regime, total_trades, wins, losses, total_pnl_pct, avg_return, win_rate, last_updated)
                VALUES (?, 1, ?, ?, ?, ?, ?, ?)
            """, (
                regime,
                1 if trade.is_win else 0,
                0 if trade.is_win else 1,
                trade.net_pnl_pct,
                trade.net_pnl_pct,
                100.0 if trade.is_win else 0.0,
                datetime.now().isoformat()
            ))

    def _update_time_stats(self, cursor, trade: TradeOutcome):
        """Update time-of-day statistics."""
        hour = trade.entry_time.hour
        dow = trade.entry_time.weekday()

        cursor.execute("""
            INSERT INTO time_stats (hour, day_of_week, total_trades, wins, total_pnl_pct, avg_return)
            VALUES (?, ?, 1, ?, ?, ?)
            ON CONFLICT(hour, day_of_week) DO UPDATE SET
                total_trades = total_trades + 1,
                wins = wins + ?,
                total_pnl_pct = total_pnl_pct + ?,
                avg_return = (total_pnl_pct + ?) / (total_trades + 1)
        """, (
            hour, dow,
            1 if trade.is_win else 0, trade.net_pnl_pct, trade.net_pnl_pct,
            1 if trade.is_win else 0, trade.net_pnl_pct, trade.net_pnl_pct
        ))

    def get_trade_count(self) -> int:
        """Get total number of recorded trades."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM trades")
        count = cursor.fetchone()[0]
        conn.close()
        return count


# =============================================================================
# 2. SIGNAL EFFECTIVENESS SCORECARD
# =============================================================================

class SignalEffectivenessScorecard:
    """
    Answers: "Which signal combinations ACTUALLY work?"

    Not "is RSI good?" but "BB + funding + whale → win rate?"
    """

    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(DB_PATH)

    def get_scorecard(self) -> pd.DataFrame:
        """Get signal combination effectiveness scorecard."""
        conn = sqlite3.connect(self.db_path)

        df = pd.read_sql_query("""
            SELECT
                combination_key as "Signal Combination",
                total_trades as "Trades",
                ROUND(wins * 100.0 / NULLIF(total_trades, 0), 1) as "Win %",
                ROUND(total_pnl_pct / NULLIF(total_trades, 0), 2) as "Expectancy",
                ROUND(avg_mfe, 2) as "Avg MFE %",
                ROUND(avg_mae, 2) as "Avg MAE %",
                ROUND(avg_mfe / NULLIF(ABS(avg_mae), 0.01), 2) as "MFE/MAE Ratio"
            FROM signal_combinations
            WHERE total_trades >= 3
            ORDER BY "Win %" DESC, "Expectancy" DESC
        """, conn)

        conn.close()
        return df

    def get_best_combinations(self, min_trades: int = 5, min_win_rate: float = 55.0) -> List[str]:
        """Get signal combinations that meet profitability threshold."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT combination_key
            FROM signal_combinations
            WHERE total_trades >= ?
              AND (wins * 100.0 / total_trades) >= ?
              AND (total_pnl_pct / total_trades) > 0
            ORDER BY (wins * 100.0 / total_trades) DESC
        """, (min_trades, min_win_rate))

        results = [row[0] for row in cursor.fetchall()]
        conn.close()
        return results

    def get_worst_combinations(self, min_trades: int = 5) -> List[str]:
        """Get signal combinations to AVOID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT combination_key
            FROM signal_combinations
            WHERE total_trades >= ?
              AND (wins * 100.0 / total_trades) < 45
            ORDER BY (wins * 100.0 / total_trades) ASC
        """, (min_trades,))

        results = [row[0] for row in cursor.fetchall()]
        conn.close()
        return results

    def print_scorecard(self):
        """Print formatted scorecard."""
        df = self.get_scorecard()

        print("\n" + "=" * 80)
        print("SIGNAL EFFECTIVENESS SCORECARD")
        print("Which signal combinations actually work?")
        print("=" * 80)

        if df.empty:
            print("\n  [NO DATA] Run paper_trader.py to collect trade data")
            print("  Minimum 150-300 trades recommended for statistical significance")
        else:
            print(df.to_string(index=False))

            best = self.get_best_combinations()
            worst = self.get_worst_combinations()

            if best:
                print(f"\n  ✅ PROFITABLE COMBINATIONS: {', '.join(best[:5])}")
            if worst:
                print(f"  ❌ AVOID THESE: {', '.join(worst[:5])}")

        print("=" * 80)


# =============================================================================
# 3. REGIME-SPECIFIC STATISTICS
# =============================================================================

class RegimeStatistics:
    """
    Answers: "Does this strategy work in VOLATILE? TRENDING? SIDEWAYS?"

    This alone can double profitability by avoiding bad regimes.
    """

    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(DB_PATH)

    def get_regime_stats(self) -> pd.DataFrame:
        """Get performance statistics by market regime."""
        conn = sqlite3.connect(self.db_path)

        df = pd.read_sql_query("""
            SELECT
                regime as "Regime",
                total_trades as "Trades",
                ROUND(win_rate, 1) as "Win %",
                ROUND(avg_return, 2) as "Avg Return %",
                ROUND(total_pnl_pct, 2) as "Total P&L %"
            FROM regime_stats
            ORDER BY win_rate DESC
        """, conn)

        conn.close()
        return df

    def should_trade_in_regime(self, regime: str, min_win_rate: float = 50.0) -> Tuple[bool, str]:
        """Determine if trading is advisable in current regime."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT total_trades, win_rate, avg_return
            FROM regime_stats
            WHERE regime = ?
        """, (regime.lower(),))

        row = cursor.fetchone()
        conn.close()

        if not row or row[0] < 10:
            return True, f"Insufficient data for {regime} regime (need 10+ trades)"

        trades, win_rate, avg_return = row

        if win_rate >= min_win_rate and avg_return > 0:
            return True, f"✅ {regime}: {win_rate:.1f}% win rate, {avg_return:+.2f}% avg return"
        else:
            return False, f"❌ {regime}: {win_rate:.1f}% win rate, {avg_return:+.2f}% avg return - AVOID"

    def print_regime_stats(self):
        """Print formatted regime statistics."""
        df = self.get_regime_stats()

        print("\n" + "=" * 60)
        print("REGIME-SPECIFIC STATISTICS")
        print("Does the strategy work in each market condition?")
        print("=" * 60)

        if df.empty:
            print("\n  [NO DATA] Run paper_trader.py to collect trade data")
        else:
            print(df.to_string(index=False))

            # Recommendations
            print("\n  RECOMMENDATIONS:")
            for _, row in df.iterrows():
                regime = row['Regime']
                win_pct = row['Win %'] if pd.notna(row['Win %']) else 0
                avg_ret = row['Avg Return %'] if pd.notna(row['Avg Return %']) else 0

                if win_pct >= 55 and avg_ret > 0:
                    print(f"    ✅ {regime}: TRADE NORMALLY")
                elif win_pct >= 50:
                    print(f"    ⚠️  {regime}: REDUCE SIZE")
                else:
                    print(f"    ❌ {regime}: AVOID TRADING")

        print("=" * 60)


# =============================================================================
# 4. FALSE SIGNAL LOG
# =============================================================================

class FalseSignalLog:
    """
    Logs signals that did NOT result in expected move.

    Answers: "When does price touch lower Bollinger but NOT bounce?"

    Avoiding bad trades is MORE important than finding good ones.
    """

    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(DB_PATH)

    def log_false_signal(self, signal: FalseSignal) -> None:
        """Record a false signal for analysis."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO false_signals (
                timestamp, symbol, signal_type, expected_direction, actual_move_pct,
                regime, funding_rate, order_flow_imbalance, whale_flow, news_events,
                rsi, stochastic, price_after_1h_pct, price_after_4h_pct, price_after_24h_pct
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            signal.timestamp.isoformat(), signal.symbol, signal.signal_type,
            signal.expected_direction, signal.actual_move_pct,
            signal.regime, signal.funding_rate, signal.order_flow_imbalance,
            signal.whale_flow, signal.news_events, signal.rsi, signal.stochastic,
            signal.price_after_1h_pct, signal.price_after_4h_pct, signal.price_after_24h_pct
        ))

        conn.commit()
        conn.close()

        print(f"[FALSE SIGNAL] {signal.signal_type} on {signal.symbol} - "
              f"Expected {signal.expected_direction}, got {signal.actual_move_pct:+.2f}%")

    def get_false_signal_patterns(self) -> pd.DataFrame:
        """Analyze patterns in false signals to learn when NOT to trade."""
        conn = sqlite3.connect(self.db_path)

        df = pd.read_sql_query("""
            SELECT
                signal_type as "Signal Type",
                regime as "Regime",
                COUNT(*) as "Count",
                ROUND(AVG(actual_move_pct), 2) as "Avg Move %",
                ROUND(AVG(funding_rate) * 100, 4) as "Avg Funding %",
                ROUND(AVG(order_flow_imbalance), 2) as "Avg Order Flow",
                ROUND(AVG(rsi), 1) as "Avg RSI"
            FROM false_signals
            GROUP BY signal_type, regime
            HAVING COUNT(*) >= 3
            ORDER BY COUNT(*) DESC
        """, conn)

        conn.close()
        return df

    def get_filter_rules(self) -> List[str]:
        """Generate filter rules based on false signal patterns."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        rules = []

        # Check if BB touch fails with positive funding
        cursor.execute("""
            SELECT COUNT(*), AVG(funding_rate)
            FROM false_signals
            WHERE signal_type LIKE '%BB%' AND funding_rate > 0.0001
        """)
        row = cursor.fetchone()
        if row[0] and row[0] >= 5:
            rules.append(f"AVOID BB buy when funding > 0.01% (failed {row[0]} times)")

        # Check if signals fail in volatile regime
        cursor.execute("""
            SELECT COUNT(*)
            FROM false_signals
            WHERE regime = 'volatile'
        """)
        row = cursor.fetchone()
        if row[0] and row[0] >= 10:
            rules.append(f"REDUCE trading in VOLATILE regime (failed {row[0]} times)")

        # Check order flow patterns
        cursor.execute("""
            SELECT COUNT(*)
            FROM false_signals
            WHERE order_flow_imbalance < 0.8
        """)
        row = cursor.fetchone()
        if row[0] and row[0] >= 5:
            rules.append(f"AVOID buy when order flow < 0.8 (failed {row[0]} times)")

        conn.close()
        return rules

    def print_analysis(self):
        """Print false signal analysis."""
        df = self.get_false_signal_patterns()
        rules = self.get_filter_rules()

        print("\n" + "=" * 70)
        print("FALSE SIGNAL ANALYSIS")
        print("Learn when NOT to trade")
        print("=" * 70)

        if df.empty:
            print("\n  [NO DATA] False signals will be logged as trades are monitored")
        else:
            print("\nFALSE SIGNAL PATTERNS:")
            print(df.to_string(index=False))

        if rules:
            print("\n  FILTER RULES (based on data):")
            for rule in rules:
                print(f"    • {rule}")

        print("=" * 70)


# =============================================================================
# 5. TIME-OF-DAY & WEEK ANALYSIS
# =============================================================================

class TimeAnalysis:
    """
    Analyzes best/worst trading times.

    Many strategies lose money simply because of bad timing, not bad logic.
    """

    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(DB_PATH)

    def get_hourly_stats(self) -> pd.DataFrame:
        """Get performance by hour of day."""
        conn = sqlite3.connect(self.db_path)

        df = pd.read_sql_query("""
            SELECT
                hour as "Hour (UTC)",
                SUM(total_trades) as "Trades",
                ROUND(SUM(wins) * 100.0 / NULLIF(SUM(total_trades), 0), 1) as "Win %",
                ROUND(SUM(total_pnl_pct) / NULLIF(SUM(total_trades), 0), 2) as "Avg Return %"
            FROM time_stats
            GROUP BY hour
            ORDER BY hour
        """, conn)

        conn.close()
        return df

    def get_daily_stats(self) -> pd.DataFrame:
        """Get performance by day of week."""
        conn = sqlite3.connect(self.db_path)

        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

        df = pd.read_sql_query("""
            SELECT
                day_of_week,
                SUM(total_trades) as "Trades",
                ROUND(SUM(wins) * 100.0 / NULLIF(SUM(total_trades), 0), 1) as "Win %",
                ROUND(SUM(total_pnl_pct) / NULLIF(SUM(total_trades), 0), 2) as "Avg Return %"
            FROM time_stats
            GROUP BY day_of_week
            ORDER BY day_of_week
        """, conn)

        conn.close()

        if not df.empty:
            df['Day'] = df['day_of_week'].apply(lambda x: days[x] if x < 7 else 'Unknown')
            df = df[['Day', 'Trades', 'Win %', 'Avg Return %']]

        return df

    def get_best_times(self) -> Tuple[List[int], List[int]]:
        """Get best and worst hours to trade."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Best hours
        cursor.execute("""
            SELECT hour
            FROM time_stats
            GROUP BY hour
            HAVING SUM(total_trades) >= 5
               AND (SUM(wins) * 100.0 / SUM(total_trades)) >= 55
            ORDER BY (SUM(wins) * 100.0 / SUM(total_trades)) DESC
            LIMIT 5
        """)
        best = [row[0] for row in cursor.fetchall()]

        # Worst hours
        cursor.execute("""
            SELECT hour
            FROM time_stats
            GROUP BY hour
            HAVING SUM(total_trades) >= 5
               AND (SUM(wins) * 100.0 / SUM(total_trades)) < 45
            ORDER BY (SUM(wins) * 100.0 / SUM(total_trades)) ASC
            LIMIT 5
        """)
        worst = [row[0] for row in cursor.fetchall()]

        conn.close()
        return best, worst

    def print_analysis(self):
        """Print time analysis."""
        hourly = self.get_hourly_stats()
        daily = self.get_daily_stats()
        best, worst = self.get_best_times()

        print("\n" + "=" * 60)
        print("TIME-OF-DAY & WEEK ANALYSIS")
        print("Best and worst trading times")
        print("=" * 60)

        if hourly.empty:
            print("\n  [NO DATA] Run paper_trader.py to collect trade data")
        else:
            print("\nBY HOUR (UTC):")
            print(hourly.to_string(index=False))

            print("\nBY DAY OF WEEK:")
            print(daily.to_string(index=False))

            if best:
                print(f"\n  ✅ BEST HOURS (UTC): {best}")
            if worst:
                print(f"  ❌ WORST HOURS (UTC): {worst}")

        print("=" * 60)


# =============================================================================
# 6. CAPITAL EROSION MONITOR
# =============================================================================

class CapitalErosionMonitor:
    """
    Tracks capital health and triggers kill switches.

    Protects from death by a thousand cuts.
    """

    def __init__(self, db_path: str = None, initial_capital: float = 10000.0):
        self.db_path = db_path or str(DB_PATH)
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        self.consecutive_losses = 0
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.cumulative_fees = 0.0

        # Kill switch thresholds
        self.max_drawdown_pct = 15.0  # Stop at 15% drawdown
        self.max_consecutive_losses = 5
        self.max_daily_loss_pct = 5.0
        self.max_daily_trades = 10

    def update(self, trade_pnl: float, fees: float, is_win: bool) -> Tuple[bool, str]:
        """
        Update capital state and check kill switches.

        Returns: (should_continue_trading, reason)
        """
        # Update state
        self.current_capital += trade_pnl
        self.cumulative_fees += fees
        self.daily_pnl += trade_pnl
        self.daily_trades += 1

        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital

        if is_win:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1

        # Calculate metrics
        drawdown_pct = (self.peak_capital - self.current_capital) / self.peak_capital * 100
        daily_loss_pct = -self.daily_pnl / self.initial_capital * 100 if self.daily_pnl < 0 else 0

        # Log to database
        self._log_state(drawdown_pct)

        # Check kill switches
        if drawdown_pct >= self.max_drawdown_pct:
            return False, f"🛑 KILL SWITCH: Drawdown {drawdown_pct:.1f}% >= {self.max_drawdown_pct}%"

        if self.consecutive_losses >= self.max_consecutive_losses:
            return False, f"🛑 KILL SWITCH: {self.consecutive_losses} consecutive losses"

        if daily_loss_pct >= self.max_daily_loss_pct:
            return False, f"🛑 KILL SWITCH: Daily loss {daily_loss_pct:.1f}% >= {self.max_daily_loss_pct}%"

        if self.daily_trades >= self.max_daily_trades:
            return False, f"⚠️ PAUSE: Daily trade limit ({self.max_daily_trades}) reached"

        return True, "OK"

    def _log_state(self, drawdown_pct: float):
        """Log capital state to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO capital_history
            (timestamp, capital, drawdown_pct, consecutive_losses, daily_pnl, cumulative_fees)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            self.current_capital,
            drawdown_pct,
            self.consecutive_losses,
            self.daily_pnl,
            self.cumulative_fees
        ))

        conn.commit()
        conn.close()

    def reset_daily(self):
        """Reset daily counters (call at start of new trading day)."""
        self.daily_trades = 0
        self.daily_pnl = 0.0

    def get_status(self) -> Dict[str, Any]:
        """Get current capital status."""
        drawdown_pct = (self.peak_capital - self.current_capital) / self.peak_capital * 100

        return {
            "current_capital": self.current_capital,
            "initial_capital": self.initial_capital,
            "peak_capital": self.peak_capital,
            "total_return_pct": (self.current_capital - self.initial_capital) / self.initial_capital * 100,
            "drawdown_pct": drawdown_pct,
            "consecutive_losses": self.consecutive_losses,
            "daily_trades": self.daily_trades,
            "daily_pnl": self.daily_pnl,
            "cumulative_fees": self.cumulative_fees,
            "fee_drag_pct": self.cumulative_fees / self.initial_capital * 100
        }

    def print_status(self):
        """Print capital status."""
        status = self.get_status()

        print("\n" + "=" * 60)
        print("CAPITAL EROSION MONITOR")
        print("Kill switches and drawdown tracking")
        print("=" * 60)

        print(f"""
  Capital Status:
    Initial:     ${status['initial_capital']:,.2f}
    Current:     ${status['current_capital']:,.2f}
    Peak:        ${status['peak_capital']:,.2f}
    Total Return: {status['total_return_pct']:+.2f}%

  Risk Metrics:
    Drawdown:    {status['drawdown_pct']:.2f}% (max: {self.max_drawdown_pct}%)
    Consec Losses: {status['consecutive_losses']} (max: {self.max_consecutive_losses})
    Daily Trades: {status['daily_trades']} (max: {self.max_daily_trades})
    Daily P&L:   ${status['daily_pnl']:+.2f}

  Fee Impact:
    Total Fees:  ${status['cumulative_fees']:.2f}
    Fee Drag:    {status['fee_drag_pct']:.2f}%
""")

        # Status indicator
        if status['drawdown_pct'] < 5:
            print("  Status: 🟢 HEALTHY")
        elif status['drawdown_pct'] < 10:
            print("  Status: 🟡 CAUTION")
        else:
            print("  Status: 🔴 CRITICAL")

        print("=" * 60)


# =============================================================================
# 7. STRATEGY CONFIDENCE METER
# =============================================================================

class StrategyConfidenceMeter:
    """
    Calculates permission to trade based on multiple factors.

    This is not prediction — it is permission to trade.

    Confidence levels:
    - < 60%: NO TRADE
    - 60-75%: SMALL SIZE (25% of normal)
    - > 75%: FULL SIZE
    """

    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(DB_PATH)
        self.scorecard = SignalEffectivenessScorecard(db_path)
        self.regime_stats = RegimeStatistics(db_path)

    def calculate_confidence(
        self,
        signal_combination: str,
        current_regime: str,
        recent_performance: Optional[Dict] = None
    ) -> Tuple[float, str, Dict[str, float]]:
        """
        Calculate trading confidence based on multiple factors.

        Returns: (confidence_pct, size_recommendation, breakdown)
        """
        breakdown = {}

        # 1. Signal alignment score (0-40 points)
        signal_score = self._get_signal_score(signal_combination)
        breakdown['signal_alignment'] = signal_score

        # 2. Regime compatibility (0-30 points)
        regime_score = self._get_regime_score(current_regime)
        breakdown['regime_compatibility'] = regime_score

        # 3. Recent performance (0-30 points)
        if recent_performance:
            perf_score = self._get_performance_score(recent_performance)
        else:
            perf_score = 15.0  # Neutral if no data
        breakdown['recent_performance'] = perf_score

        # Total confidence
        total = signal_score + regime_score + perf_score
        confidence = min(100.0, max(0.0, total))

        # Size recommendation
        if confidence < 60:
            size = "NO TRADE"
        elif confidence < 75:
            size = "SMALL SIZE (25%)"
        else:
            size = "FULL SIZE (100%)"

        return confidence, size, breakdown

    def _get_signal_score(self, combination: str) -> float:
        """Score based on historical signal effectiveness."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT total_trades, wins, total_pnl_pct
            FROM signal_combinations
            WHERE combination_key = ?
        """, (combination,))

        row = cursor.fetchone()
        conn.close()

        if not row or row[0] < 5:
            return 20.0  # Neutral for unknown combinations

        trades, wins, total_pnl = row
        win_rate = wins / trades * 100
        expectancy = total_pnl / trades

        score = 0.0
        if win_rate >= 60:
            score += 20.0
        elif win_rate >= 55:
            score += 15.0
        elif win_rate >= 50:
            score += 10.0
        else:
            score += 5.0

        if expectancy > 0.5:
            score += 20.0
        elif expectancy > 0:
            score += 15.0
        elif expectancy > -0.5:
            score += 10.0
        else:
            score += 0.0

        return score

    def _get_regime_score(self, regime: str) -> float:
        """Score based on regime compatibility."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT total_trades, win_rate, avg_return
            FROM regime_stats
            WHERE regime = ?
        """, (regime.lower(),))

        row = cursor.fetchone()
        conn.close()

        if not row or row[0] < 10:
            return 15.0  # Neutral for unknown regimes

        trades, win_rate, avg_return = row

        score = 0.0
        if win_rate >= 60:
            score += 15.0
        elif win_rate >= 55:
            score += 10.0
        elif win_rate >= 50:
            score += 5.0
        else:
            score += 0.0

        if avg_return > 0.5:
            score += 15.0
        elif avg_return > 0:
            score += 10.0
        elif avg_return > -0.5:
            score += 5.0
        else:
            score += 0.0

        return score

    def _get_performance_score(self, recent: Dict) -> float:
        """Score based on recent performance."""
        score = 15.0  # Base score

        # Adjust based on recent wins
        recent_win_rate = recent.get('win_rate', 50)
        if recent_win_rate >= 60:
            score += 10.0
        elif recent_win_rate >= 50:
            score += 5.0
        elif recent_win_rate < 40:
            score -= 10.0

        # Adjust based on drawdown
        drawdown = recent.get('drawdown_pct', 0)
        if drawdown < 5:
            score += 5.0
        elif drawdown > 10:
            score -= 10.0

        return max(0.0, min(30.0, score))

    def get_trade_permission(
        self,
        signal_combination: str,
        current_regime: str,
        recent_performance: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Get complete trade permission with reasoning."""
        confidence, size_rec, breakdown = self.calculate_confidence(
            signal_combination, current_regime, recent_performance
        )

        return {
            "confidence": confidence,
            "size_recommendation": size_rec,
            "can_trade": confidence >= 60,
            "breakdown": breakdown,
            "signals": signal_combination,
            "regime": current_regime
        }

    def print_permission(self, signal_combination: str, current_regime: str):
        """Print trade permission status."""
        permission = self.get_trade_permission(signal_combination, current_regime)

        print("\n" + "=" * 60)
        print("STRATEGY CONFIDENCE METER")
        print("Permission to trade")
        print("=" * 60)

        print(f"""
  Signals: {permission['signals']}
  Regime:  {permission['regime']}

  Confidence Breakdown:
    Signal Alignment:    {permission['breakdown']['signal_alignment']:.1f}/40
    Regime Compatibility: {permission['breakdown']['regime_compatibility']:.1f}/30
    Recent Performance:  {permission['breakdown']['recent_performance']:.1f}/30

  TOTAL CONFIDENCE: {permission['confidence']:.1f}%

  RECOMMENDATION: {permission['size_recommendation']}
""")

        if permission['can_trade']:
            print("  ✅ PERMISSION GRANTED")
        else:
            print("  ❌ PERMISSION DENIED - Wait for better conditions")

        print("=" * 60)


# =============================================================================
# MASTER EVIDENCE COLLECTOR
# =============================================================================

class EvidenceCollector:
    """
    Master class that integrates all 7 evidence collection systems.
    """

    def __init__(self, db_path: str = None, initial_capital: float = 10000.0):
        self.db_path = db_path or str(DB_PATH)

        # Initialize all systems
        self.trade_db = TradeOutcomeDatabase(self.db_path)
        self.scorecard = SignalEffectivenessScorecard(self.db_path)
        self.regime_stats = RegimeStatistics(self.db_path)
        self.false_signals = FalseSignalLog(self.db_path)
        self.time_analysis = TimeAnalysis(self.db_path)
        self.capital_monitor = CapitalErosionMonitor(self.db_path, initial_capital)
        self.confidence_meter = StrategyConfidenceMeter(self.db_path)

        print("\n" + "=" * 70)
        print("EVIDENCE COLLECTOR INITIALIZED")
        print("7 systems ready to collect behavioral & outcome data")
        print("=" * 70)
        print(f"  Database: {self.db_path}")
        print(f"  Initial Capital: ${initial_capital:,.2f}")
        print("=" * 70)

    def record_trade(self, trade: TradeOutcome) -> Tuple[bool, str]:
        """Record a trade and update all systems."""
        # Record trade
        self.trade_db.record_trade(trade)

        # Update capital monitor
        trade_pnl = trade.net_pnl_pct / 100 * self.capital_monitor.current_capital
        fees = trade.fees_pct / 100 * self.capital_monitor.current_capital
        can_continue, reason = self.capital_monitor.update(trade_pnl, fees, trade.is_win)

        return can_continue, reason

    def log_false_signal(self, signal: FalseSignal):
        """Log a false signal."""
        self.false_signals.log_false_signal(signal)

    def get_trade_permission(self, signals: SignalContext, regime: str) -> Dict:
        """Get permission to trade based on all evidence."""
        combo_key = signals.signal_combination_key()
        return self.confidence_meter.get_trade_permission(combo_key, regime)

    def print_full_report(self):
        """Print complete evidence report."""
        trade_count = self.trade_db.get_trade_count()

        print("\n")
        print("█" * 70)
        print("█" + " " * 68 + "█")
        print("█" + "    EVIDENCE COLLECTION REPORT".center(68) + "█")
        print("█" + f"    Total Trades Recorded: {trade_count}".center(68) + "█")
        print("█" + " " * 68 + "█")
        print("█" * 70)

        if trade_count < 30:
            print(f"""
  ⚠️  INSUFFICIENT DATA

  You have {trade_count} trades recorded.
  Minimum recommended: 150-300 trades

  Keep running paper_trader.py to collect more data!
""")

        # Print all reports
        self.scorecard.print_scorecard()
        self.regime_stats.print_regime_stats()
        self.false_signals.print_analysis()
        self.time_analysis.print_analysis()
        self.capital_monitor.print_status()

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for quick overview."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*), SUM(is_win), AVG(net_pnl_pct) FROM trades")
        row = cursor.fetchone()

        conn.close()

        if not row or not row[0]:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "avg_return": 0,
                "ready_for_live": False,
                "recommendation": "Collect more data"
            }

        total, wins, avg_return = row
        win_rate = (wins or 0) / total * 100

        return {
            "total_trades": total,
            "win_rate": win_rate,
            "avg_return": avg_return or 0,
            "ready_for_live": total >= 150 and win_rate >= 55 and (avg_return or 0) > 0,
            "recommendation": self._get_recommendation(total, win_rate, avg_return or 0)
        }

    def _get_recommendation(self, trades: int, win_rate: float, avg_return: float) -> str:
        """Get recommendation based on current data."""
        if trades < 50:
            return f"Keep collecting data ({trades}/150 minimum)"
        elif trades < 150:
            return f"Good progress! {trades}/150 trades, keep going"
        elif win_rate < 50:
            return "Strategy needs improvement (win rate < 50%)"
        elif avg_return <= 0:
            return "Strategy not profitable (avg return <= 0)"
        elif win_rate < 55:
            return "Marginal edge - consider optimization"
        else:
            return "✅ Strategy shows edge - ready for small live test"


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Main entry point for evidence collector."""
    import argparse

    parser = argparse.ArgumentParser(description="Evidence Collector - Behavioral & Outcome Data")
    parser.add_argument("--report", action="store_true", help="Print full evidence report")
    parser.add_argument("--summary", action="store_true", help="Print quick summary")
    parser.add_argument("--scorecard", action="store_true", help="Print signal effectiveness scorecard")
    parser.add_argument("--regimes", action="store_true", help="Print regime statistics")
    parser.add_argument("--times", action="store_true", help="Print time analysis")
    parser.add_argument("--capital", action="store_true", help="Print capital status")
    parser.add_argument("--confidence", action="store_true", help="Calculate trade confidence")
    args = parser.parse_args()

    collector = EvidenceCollector()

    if args.report:
        collector.print_full_report()
    elif args.summary:
        stats = collector.get_summary_stats()
        print("\n" + "=" * 50)
        print("EVIDENCE SUMMARY")
        print("=" * 50)
        print(f"  Total Trades: {stats['total_trades']}")
        print(f"  Win Rate: {stats['win_rate']:.1f}%")
        print(f"  Avg Return: {stats['avg_return']:.2f}%")
        print(f"  Ready for Live: {'Yes' if stats['ready_for_live'] else 'No'}")
        print(f"\n  Recommendation: {stats['recommendation']}")
        print("=" * 50)
    elif args.scorecard:
        collector.scorecard.print_scorecard()
    elif args.regimes:
        collector.regime_stats.print_regime_stats()
    elif args.times:
        collector.time_analysis.print_analysis()
    elif args.capital:
        collector.capital_monitor.print_status()
    elif args.confidence:
        # Example confidence calculation
        collector.confidence_meter.print_permission("BB+FUNDING", "sideways")
    else:
        # Default: print summary
        collector.print_full_report()


if __name__ == "__main__":
    main()
