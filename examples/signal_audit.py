#!/usr/bin/env python3
"""
Signal Audit Dashboard - Track and Analyze Signal Performance

Provides comprehensive signal auditing capabilities:
1. Signal precision/recall by type and coin
2. Hit-rate tracking (7d, 14d, 30d)
3. Signal provenance and input tracking
4. Outcome analysis
5. Weekly/monthly performance reports

Author: Cloud AI Analyzer
"""

import sqlite3
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json


# =============================================================================
# CONFIGURATION
# =============================================================================

DB_PATH = Path(__file__).parent / "quant_signals.db"


@dataclass
class SignalPerformance:
    """Performance metrics for a signal type."""
    signal_type: str
    total_signals: int
    correct_predictions: int
    incorrect_predictions: int
    pending: int
    precision_7d: float
    precision_14d: float
    precision_30d: float
    avg_confidence: float
    avg_return_when_correct: float
    avg_return_when_incorrect: float


@dataclass
class SignalAuditRecord:
    """Complete audit record for a signal."""
    signal_id: int
    timestamp: datetime
    symbol: str
    signal_type: str
    signal_name: str
    direction: str
    confidence: float
    inputs: Dict
    outcome_7d: Optional[str]
    outcome_14d: Optional[str]
    return_7d: Optional[float]
    return_14d: Optional[float]
    reason_tags: List[str]


# =============================================================================
# SIGNAL AUDIT CLASS
# =============================================================================

class SignalAudit:
    """
    Signal auditing and performance tracking system.

    Features:
    - Track all signals with full provenance
    - Calculate precision/recall by signal type
    - Monitor hit rates over multiple timeframes
    - Generate performance reports
    - Identify underperforming signals
    """

    def __init__(self, db_path: str = None):
        """Initialize signal audit system."""
        self.db_path = db_path or str(DB_PATH)
        self._init_audit_tables()
        print("  [AUDIT] Signal audit system initialized")

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_audit_tables(self):
        """Initialize audit-specific tables."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Signal audit table with full provenance
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signal_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id INTEGER,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                signal_name TEXT NOT NULL,
                direction TEXT NOT NULL,
                confidence REAL NOT NULL,
                inputs_json TEXT,
                price_at_signal REAL,
                price_7d REAL,
                price_14d REAL,
                price_30d REAL,
                outcome_7d TEXT,
                outcome_14d TEXT,
                outcome_30d TEXT,
                return_7d REAL,
                return_14d REAL,
                return_30d REAL,
                reason_tags TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Performance summary table (updated periodically)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signal_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                symbol TEXT,
                total_signals INTEGER DEFAULT 0,
                correct_7d INTEGER DEFAULT 0,
                incorrect_7d INTEGER DEFAULT 0,
                correct_14d INTEGER DEFAULT 0,
                incorrect_14d INTEGER DEFAULT 0,
                correct_30d INTEGER DEFAULT 0,
                incorrect_30d INTEGER DEFAULT 0,
                avg_confidence REAL DEFAULT 0,
                avg_return_correct REAL DEFAULT 0,
                avg_return_incorrect REAL DEFAULT 0,
                UNIQUE(date, signal_type, symbol)
            )
        """)

        # Create indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_symbol_type
            ON signal_audit(symbol, signal_type)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_timestamp
            ON signal_audit(timestamp)
        """)

        conn.commit()
        conn.close()

    # =========================================================================
    # SIGNAL RECORDING
    # =========================================================================

    def record_signal(
        self,
        symbol: str,
        signal_type: str,
        signal_name: str,
        direction: str,
        confidence: float,
        price: float,
        inputs: Dict = None,
        reason_tags: List[str] = None,
        signal_id: int = None
    ) -> int:
        """
        Record a new signal for audit tracking.

        Args:
            symbol: Trading symbol
            signal_type: Type (technical, news, ml, whale, funding, etc.)
            signal_name: Specific indicator/model name
            direction: BUY, SELL, or NEUTRAL
            confidence: Confidence score 0-1
            price: Price at signal generation
            inputs: Dictionary of inputs used to generate signal
            reason_tags: List of reason tags
            signal_id: Optional reference to original signal ID

        Returns:
            Audit record ID
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT INTO signal_audit
                (signal_id, timestamp, symbol, signal_type, signal_name,
                 direction, confidence, inputs_json, price_at_signal, reason_tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                signal_id,
                datetime.utcnow().isoformat() + "Z",
                symbol,
                signal_type,
                signal_name,
                direction,
                confidence,
                json.dumps(inputs or {}),
                price,
                json.dumps(reason_tags or [])
            ))
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def update_outcomes(self, symbol: str = None):
        """
        Update signal outcomes based on price changes.

        Should be run periodically (daily) to update 7d/14d/30d outcomes.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        now = datetime.utcnow()

        # Get signals that need outcome updates
        query = """
            SELECT id, timestamp, symbol, direction, price_at_signal
            FROM signal_audit
            WHERE (outcome_7d IS NULL OR outcome_14d IS NULL OR outcome_30d IS NULL)
        """
        if symbol:
            query += f" AND symbol = '{symbol}'"

        cursor.execute(query)
        signals = cursor.fetchall()

        updated = 0
        for signal in signals:
            signal_time = datetime.fromisoformat(signal['timestamp'].replace('Z', ''))
            age_days = (now - signal_time).days

            # Get current price
            current_price = self._fetch_price(signal['symbol'])
            if not current_price:
                continue

            entry_price = signal['price_at_signal']
            direction = signal['direction']

            # Calculate return
            if direction == "BUY":
                return_pct = ((current_price - entry_price) / entry_price) * 100
            elif direction == "SELL":
                return_pct = ((entry_price - current_price) / entry_price) * 100
            else:
                return_pct = 0

            # Determine outcome (>0.5% profit = correct)
            outcome = "CORRECT" if return_pct > 0.5 else ("INCORRECT" if return_pct < -0.5 else "NEUTRAL")

            updates = []
            params = []

            # Update 7d outcome
            if age_days >= 7 and signal['outcome_7d'] is None:
                updates.append("outcome_7d = ?, return_7d = ?, price_7d = ?")
                params.extend([outcome, return_pct, current_price])

            # Update 14d outcome
            if age_days >= 14 and signal['outcome_14d'] is None:
                updates.append("outcome_14d = ?, return_14d = ?, price_14d = ?")
                params.extend([outcome, return_pct, current_price])

            # Update 30d outcome
            if age_days >= 30 and signal['outcome_30d'] is None:
                updates.append("outcome_30d = ?, return_30d = ?, price_30d = ?")
                params.extend([outcome, return_pct, current_price])

            if updates:
                params.append(signal['id'])
                cursor.execute(f"""
                    UPDATE signal_audit
                    SET {', '.join(updates)}, updated_at = ?
                    WHERE id = ?
                """, params[:-1] + [datetime.utcnow().isoformat(), signal['id']])
                updated += 1

        conn.commit()
        conn.close()

        if updated > 0:
            print(f"  [AUDIT] Updated {updated} signal outcomes")

    def _fetch_price(self, symbol: str) -> Optional[float]:
        """Fetch current price for a symbol."""
        import requests
        try:
            url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return float(response.json()['price'])
        except:
            pass
        return None

    # =========================================================================
    # PERFORMANCE ANALYSIS
    # =========================================================================

    def get_performance_by_type(self, days: int = 30) -> Dict[str, SignalPerformance]:
        """
        Get performance metrics grouped by signal type.

        Args:
            days: Look-back period

        Returns:
            Dictionary of signal_type -> SignalPerformance
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        since = (datetime.utcnow() - timedelta(days=days)).isoformat()

        cursor.execute("""
            SELECT
                signal_type,
                COUNT(*) as total,
                SUM(CASE WHEN outcome_7d = 'CORRECT' THEN 1 ELSE 0 END) as correct_7d,
                SUM(CASE WHEN outcome_7d = 'INCORRECT' THEN 1 ELSE 0 END) as incorrect_7d,
                SUM(CASE WHEN outcome_14d = 'CORRECT' THEN 1 ELSE 0 END) as correct_14d,
                SUM(CASE WHEN outcome_14d = 'INCORRECT' THEN 1 ELSE 0 END) as incorrect_14d,
                SUM(CASE WHEN outcome_7d IS NULL THEN 1 ELSE 0 END) as pending,
                AVG(confidence) as avg_conf,
                AVG(CASE WHEN outcome_14d = 'CORRECT' THEN return_14d ELSE NULL END) as avg_ret_correct,
                AVG(CASE WHEN outcome_14d = 'INCORRECT' THEN return_14d ELSE NULL END) as avg_ret_incorrect
            FROM signal_audit
            WHERE timestamp >= ?
            GROUP BY signal_type
        """, (since,))

        results = {}
        for row in cursor.fetchall():
            total_resolved_7d = row['correct_7d'] + row['incorrect_7d']
            total_resolved_14d = row['correct_14d'] + row['incorrect_14d']

            results[row['signal_type']] = SignalPerformance(
                signal_type=row['signal_type'],
                total_signals=row['total'],
                correct_predictions=row['correct_14d'],
                incorrect_predictions=row['incorrect_14d'],
                pending=row['pending'],
                precision_7d=row['correct_7d'] / total_resolved_7d * 100 if total_resolved_7d > 0 else 0,
                precision_14d=row['correct_14d'] / total_resolved_14d * 100 if total_resolved_14d > 0 else 0,
                precision_30d=0,  # Calculate separately if needed
                avg_confidence=row['avg_conf'] or 0,
                avg_return_when_correct=row['avg_ret_correct'] or 0,
                avg_return_when_incorrect=row['avg_ret_incorrect'] or 0,
            )

        conn.close()
        return results

    def get_performance_by_symbol(self, days: int = 30) -> Dict[str, Dict]:
        """Get performance metrics grouped by symbol."""
        conn = self._get_connection()
        cursor = conn.cursor()

        since = (datetime.utcnow() - timedelta(days=days)).isoformat()

        cursor.execute("""
            SELECT
                symbol,
                COUNT(*) as total,
                SUM(CASE WHEN outcome_14d = 'CORRECT' THEN 1 ELSE 0 END) as correct,
                SUM(CASE WHEN outcome_14d = 'INCORRECT' THEN 1 ELSE 0 END) as incorrect,
                AVG(confidence) as avg_conf,
                AVG(return_14d) as avg_return
            FROM signal_audit
            WHERE timestamp >= ?
            GROUP BY symbol
        """, (since,))

        results = {}
        for row in cursor.fetchall():
            total_resolved = row['correct'] + row['incorrect']
            results[row['symbol']] = {
                'total_signals': row['total'],
                'correct': row['correct'],
                'incorrect': row['incorrect'],
                'precision': row['correct'] / total_resolved * 100 if total_resolved > 0 else 0,
                'avg_confidence': row['avg_conf'] or 0,
                'avg_return': row['avg_return'] or 0,
            }

        conn.close()
        return results

    def get_recent_signals(self, limit: int = 20, symbol: str = None) -> List[Dict]:
        """Get recent signals with outcomes."""
        conn = self._get_connection()
        cursor = conn.cursor()

        query = """
            SELECT
                id, timestamp, symbol, signal_type, signal_name,
                direction, confidence, price_at_signal,
                outcome_7d, outcome_14d, return_7d, return_14d,
                reason_tags
            FROM signal_audit
        """
        if symbol:
            query += f" WHERE symbol = '{symbol}'"
        query += " ORDER BY timestamp DESC LIMIT ?"

        cursor.execute(query, (limit,))

        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row['id'],
                'timestamp': row['timestamp'],
                'symbol': row['symbol'],
                'signal_type': row['signal_type'],
                'signal_name': row['signal_name'],
                'direction': row['direction'],
                'confidence': row['confidence'],
                'price': row['price_at_signal'],
                'outcome_7d': row['outcome_7d'],
                'outcome_14d': row['outcome_14d'],
                'return_7d': row['return_7d'],
                'return_14d': row['return_14d'],
                'reason_tags': json.loads(row['reason_tags']) if row['reason_tags'] else [],
            })

        conn.close()
        return results

    def get_underperforming_signals(self, min_signals: int = 10, max_precision: float = 55) -> List[Dict]:
        """
        Identify signal types that are underperforming.

        Args:
            min_signals: Minimum signals to consider
            max_precision: Precision threshold below which signals are flagged

        Returns:
            List of underperforming signal types with metrics
        """
        performance = self.get_performance_by_type(days=30)

        underperforming = []
        for sig_type, perf in performance.items():
            if perf.total_signals >= min_signals and perf.precision_14d < max_precision:
                underperforming.append({
                    'signal_type': sig_type,
                    'precision_14d': perf.precision_14d,
                    'total_signals': perf.total_signals,
                    'avg_confidence': perf.avg_confidence,
                    'recommendation': 'DISABLE' if perf.precision_14d < 45 else 'REDUCE_WEIGHT'
                })

        return sorted(underperforming, key=lambda x: x['precision_14d'])

    # =========================================================================
    # REPORTING
    # =========================================================================

    def print_performance_report(self, days: int = 30):
        """Print comprehensive performance report."""
        print("\n" + "=" * 70)
        print(f"SIGNAL AUDIT REPORT - Last {days} Days")
        print("=" * 70)

        # Performance by signal type
        print("\n[1] PERFORMANCE BY SIGNAL TYPE")
        print("─" * 70)
        print(f"{'Type':<15} {'Total':>8} {'Prec_7d':>10} {'Prec_14d':>10} {'Conf':>8} {'Status':<12}")
        print("─" * 70)

        perf_by_type = self.get_performance_by_type(days)

        for sig_type, perf in sorted(perf_by_type.items(), key=lambda x: -x[1].precision_14d):
            status = "✓ GOOD" if perf.precision_14d >= 60 else (
                "⚠ WEAK" if perf.precision_14d >= 50 else "✗ DISABLE"
            )
            print(f"{sig_type:<15} {perf.total_signals:>8} {perf.precision_7d:>9.1f}% "
                  f"{perf.precision_14d:>9.1f}% {perf.avg_confidence:>7.2f} {status:<12}")

        # Performance by symbol
        print("\n[2] PERFORMANCE BY SYMBOL")
        print("─" * 70)
        print(f"{'Symbol':<12} {'Total':>8} {'Correct':>10} {'Precision':>12} {'Avg Return':>12}")
        print("─" * 70)

        perf_by_symbol = self.get_performance_by_symbol(days)

        for symbol, data in sorted(perf_by_symbol.items(), key=lambda x: -x[1]['precision']):
            print(f"{symbol:<12} {data['total_signals']:>8} {data['correct']:>10} "
                  f"{data['precision']:>11.1f}% {data['avg_return']:>11.2f}%")

        # Underperforming signals
        print("\n[3] UNDERPERFORMING SIGNALS (Action Required)")
        print("─" * 70)

        underperforming = self.get_underperforming_signals()
        if underperforming:
            for item in underperforming:
                print(f"  ⚠ {item['signal_type']}: {item['precision_14d']:.1f}% precision "
                      f"({item['total_signals']} signals) -> {item['recommendation']}")
        else:
            print("  ✓ No underperforming signals detected")

        # Recent signals
        print("\n[4] RECENT SIGNALS (Last 10)")
        print("─" * 70)

        recent = self.get_recent_signals(limit=10)
        for sig in recent:
            outcome = sig['outcome_14d'] or 'PENDING'
            outcome_icon = "✓" if outcome == 'CORRECT' else ("✗" if outcome == 'INCORRECT' else "?")
            print(f"  {outcome_icon} {sig['timestamp'][:16]} {sig['symbol']:<10} "
                  f"{sig['signal_type']:<12} {sig['direction']:<5} "
                  f"conf={sig['confidence']:.2f} ret={sig['return_14d'] or 0:+.2f}%")

        print("\n" + "=" * 70)

    def export_to_json(self, filepath: str = None) -> str:
        """Export audit data to JSON."""
        filepath = filepath or str(Path(__file__).parent / "signal_audit_export.json")

        data = {
            'export_time': datetime.utcnow().isoformat(),
            'performance_by_type': {},
            'performance_by_symbol': self.get_performance_by_symbol(),
            'underperforming': self.get_underperforming_signals(),
            'recent_signals': self.get_recent_signals(limit=100),
        }

        # Convert SignalPerformance objects to dicts
        for sig_type, perf in self.get_performance_by_type().items():
            data['performance_by_type'][sig_type] = {
                'total_signals': perf.total_signals,
                'precision_7d': perf.precision_7d,
                'precision_14d': perf.precision_14d,
                'avg_confidence': perf.avg_confidence,
            }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        print(f"  [AUDIT] Exported to {filepath}")
        return filepath


# =============================================================================
# DEMO / TEST
# =============================================================================

def demo_signal_audit():
    """Demonstrate signal audit functionality."""
    print("\n" + "=" * 70)
    print("SIGNAL AUDIT DEMO")
    print("=" * 70)

    audit = SignalAudit()

    # Record some sample signals
    print("\n[1] Recording sample signals...")

    signals = [
        ("XRPUSDT", "technical", "RSI", "BUY", 0.75, 2.03),
        ("XRPUSDT", "whale", "exchange_outflow", "BUY", 0.70, 2.03),
        ("BTCUSDT", "technical", "MACD", "SELL", 0.65, 97500),
        ("ETHUSDT", "funding", "negative_funding", "BUY", 0.60, 3650),
        ("XRPUSDT", "ml", "xgboost", "BUY", 0.72, 2.03),
    ]

    for symbol, sig_type, name, direction, conf, price in signals:
        audit.record_signal(
            symbol=symbol,
            signal_type=sig_type,
            signal_name=name,
            direction=direction,
            confidence=conf,
            price=price,
            inputs={'demo': True},
            reason_tags=['demo_signal']
        )
        print(f"    Recorded: {symbol} {sig_type} {direction}")

    # Print report
    audit.print_performance_report(days=30)


if __name__ == "__main__":
    demo_signal_audit()
