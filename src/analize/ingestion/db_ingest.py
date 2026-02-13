"""
ScalperBot database ingestion.

Reads trades, positions, and config from ScalperBot's SQLite/Postgres database.
"""

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator
from uuid import uuid4

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from analize.config import get_settings
from analize.utils.time import utcnow
from analize.models.signals import (
    CandleData,
    ExecutionData,
    FilterResult,
    PolicyMode,
    Signal,
    SignalCreate,
    TradingMode,
)


class ScalperBotDBIngestor:
    """Ingests data from ScalperBot database."""

    def __init__(self, db_url: str | None = None):
        settings = get_settings()
        self.db_url = db_url or settings.database.scalperbot_db_url
        self._engine: Engine | None = None

    @property
    def engine(self) -> Engine:
        """Get or create database engine."""
        if self._engine is None:
            self._engine = create_engine(self.db_url)
        return self._engine

    def close(self) -> None:
        """Close database connection."""
        if self._engine:
            self._engine.dispose()
            self._engine = None

    def test_connection(self) -> bool:
        """Test database connection."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    def get_tables(self) -> list[str]:
        """Get list of tables in the database."""
        from sqlalchemy import inspect

        inspector = inspect(self.engine)
        return inspector.get_table_names()

    def get_trades(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        symbol: str | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """
        Get trades from ScalperBot database.

        Args:
            start_date: Filter trades after this date
            end_date: Filter trades before this date
            symbol: Filter by symbol
            limit: Maximum number of trades to return

        Returns:
            DataFrame of trades
        """
        query = "SELECT * FROM trades WHERE 1=1"
        params: dict[str, Any] = {}

        if start_date:
            query += " AND created_at >= :start_date"
            params["start_date"] = start_date

        if end_date:
            query += " AND created_at <= :end_date"
            params["end_date"] = end_date

        if symbol:
            query += " AND symbol = :symbol"
            params["symbol"] = symbol

        query += " ORDER BY created_at DESC"

        if limit:
            query += f" LIMIT {limit}"

        with self.engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params=params)

        return df

    def get_positions(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        symbol: str | None = None,
        status: str | None = None,
    ) -> pd.DataFrame:
        """
        Get positions from ScalperBot database.

        Args:
            start_date: Filter positions after this date
            end_date: Filter positions before this date
            symbol: Filter by symbol
            status: Filter by status (open, closed)

        Returns:
            DataFrame of positions
        """
        query = "SELECT * FROM positions WHERE 1=1"
        params: dict[str, Any] = {}

        if start_date:
            query += " AND opened_at >= :start_date"
            params["start_date"] = start_date

        if end_date:
            query += " AND opened_at <= :end_date"
            params["end_date"] = end_date

        if symbol:
            query += " AND symbol = :symbol"
            params["symbol"] = symbol

        if status:
            query += " AND status = :status"
            params["status"] = status

        query += " ORDER BY opened_at DESC"

        with self.engine.connect() as conn:
            try:
                df = pd.read_sql(text(query), conn, params=params)
            except Exception:
                # Table might not exist or have different schema
                df = pd.DataFrame()

        return df

    def get_config_snapshots(self, limit: int = 10) -> pd.DataFrame:
        """Get recent config snapshots."""
        query = """
            SELECT * FROM config_snapshots
            ORDER BY created_at DESC
            LIMIT :limit
        """

        with self.engine.connect() as conn:
            try:
                df = pd.read_sql(text(query), conn, params={"limit": limit})
            except Exception:
                df = pd.DataFrame()

        return df

    def get_signals_raw(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        symbol: str | None = None,
    ) -> pd.DataFrame:
        """
        Get raw signals from ScalperBot database.

        Note: This assumes ScalperBot has a signals table. Adapt as needed.
        """
        query = "SELECT * FROM signals WHERE 1=1"
        params: dict[str, Any] = {}

        if start_date:
            query += " AND timestamp >= :start_date"
            params["start_date"] = start_date

        if end_date:
            query += " AND timestamp <= :end_date"
            params["end_date"] = end_date

        if symbol:
            query += " AND symbol = :symbol"
            params["symbol"] = symbol

        query += " ORDER BY timestamp DESC"

        with self.engine.connect() as conn:
            try:
                df = pd.read_sql(text(query), conn, params=params)
            except Exception:
                df = pd.DataFrame()

        return df

    def get_pnl_history(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pd.DataFrame:
        """Get PnL history."""
        query = """
            SELECT
                DATE(closed_at) as date,
                symbol,
                COUNT(*) as trade_count,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) as losses,
                SUM(pnl) as total_pnl,
                AVG(pnl) as avg_pnl
            FROM positions
            WHERE status = 'closed'
        """
        params: dict[str, Any] = {}

        if start_date:
            query += " AND closed_at >= :start_date"
            params["start_date"] = start_date

        if end_date:
            query += " AND closed_at <= :end_date"
            params["end_date"] = end_date

        query += " GROUP BY DATE(closed_at), symbol ORDER BY date DESC"

        with self.engine.connect() as conn:
            try:
                df = pd.read_sql(text(query), conn, params=params)
            except Exception:
                df = pd.DataFrame()

        return df

    def convert_trade_to_signal(self, trade_row: pd.Series) -> Signal | None:
        """
        Convert a trade row from ScalperBot DB to a Signal object.

        This method should be customized based on ScalperBot's actual schema.
        """
        try:
            # Create candle data (placeholder - actual data should come from market data)
            candle = CandleData(
                timestamp=trade_row.get("created_at", utcnow()),
                open=float(trade_row.get("entry_price", 0)),
                high=float(trade_row.get("entry_price", 0)),
                low=float(trade_row.get("entry_price", 0)),
                close=float(trade_row.get("entry_price", 0)),
                volume=float(trade_row.get("quantity", 0)),
                timeframe="1m",
            )

            # Create execution data
            execution = ExecutionData(
                order_id=str(trade_row.get("order_id", "")),
                exec_price=float(trade_row.get("entry_price", 0)),
                exec_qty=float(trade_row.get("quantity", 0)),
                filled=True,
                slippage_pct=float(trade_row.get("slippage", 0)) if "slippage" in trade_row else None,
            )

            # Parse filters if available
            filters_passed = []
            if "filters" in trade_row and trade_row["filters"]:
                import json
                try:
                    filters_data = json.loads(trade_row["filters"])
                    for f in filters_data:
                        filters_passed.append(FilterResult(
                            filter_name=f.get("name", "unknown"),
                            passed=f.get("passed", True),
                            value=f.get("value"),
                            threshold=f.get("threshold"),
                        ))
                except Exception:
                    pass

            # Determine mode
            mode = TradingMode.DRY_RUN
            if trade_row.get("mode", "").upper() == "LIVE":
                mode = TradingMode.LIVE

            # Create signal
            signal = Signal(
                signal_id=uuid4(),
                timestamp_utc=trade_row.get("created_at", utcnow()),
                symbol=str(trade_row.get("symbol", "UNKNOWN")),
                pair_id=str(trade_row.get("pair_id", "")),
                timeframe="1m",
                mode=mode,
                branch=trade_row.get("branch"),
                strategy_version=trade_row.get("strategy_version"),
                candle=candle,
                filters_passed=filters_passed,
                policy_mode=PolicyMode.MODERATE,
                expected_tp_pct=float(trade_row.get("tp_pct", 0)) if "tp_pct" in trade_row else None,
                expected_sl_pct=float(trade_row.get("sl_pct", 0)) if "sl_pct" in trade_row else None,
                position_size_usd=float(trade_row.get("position_size", 0)) if "position_size" in trade_row else None,
                execution=execution,
            )

            return signal

        except Exception as e:
            # Log error and return None
            print(f"Error converting trade to signal: {e}")
            return None

    def ingest_trades_as_signals(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        symbol: str | None = None,
    ) -> list[Signal]:
        """
        Ingest trades from ScalperBot DB and convert to Signal objects.

        Returns:
            List of Signal objects
        """
        trades_df = self.get_trades(start_date, end_date, symbol)
        signals = []

        for _, row in trades_df.iterrows():
            signal = self.convert_trade_to_signal(row)
            if signal:
                signals.append(signal)

        return signals

    def compute_data_hash(self, df: pd.DataFrame) -> str:
        """Compute a hash of the data for reproducibility."""
        data_str = df.to_json(orient="records", date_format="iso")
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]

    def get_stats(self) -> dict[str, Any]:
        """Get database statistics."""
        stats = {
            "connected": self.test_connection(),
            "tables": [],
            "trade_count": 0,
            "position_count": 0,
            "date_range": {"min": None, "max": None},
            "symbols": [],
        }

        if not stats["connected"]:
            return stats

        stats["tables"] = self.get_tables()

        with self.engine.connect() as conn:
            # Trade count
            try:
                result = conn.execute(text("SELECT COUNT(*) FROM trades"))
                stats["trade_count"] = result.scalar() or 0
            except Exception:
                pass

            # Position count
            try:
                result = conn.execute(text("SELECT COUNT(*) FROM positions"))
                stats["position_count"] = result.scalar() or 0
            except Exception:
                pass

            # Date range
            try:
                result = conn.execute(text("SELECT MIN(created_at), MAX(created_at) FROM trades"))
                row = result.fetchone()
                if row:
                    stats["date_range"]["min"] = row[0].isoformat() if row[0] else None
                    stats["date_range"]["max"] = row[1].isoformat() if row[1] else None
            except Exception:
                pass

            # Symbols
            try:
                result = conn.execute(text("SELECT DISTINCT symbol FROM trades ORDER BY symbol"))
                stats["symbols"] = [row[0] for row in result.fetchall()]
            except Exception:
                pass

        return stats
