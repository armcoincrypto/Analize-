"""
Parquet file storage for ML-ready signal records.

Handles:
- Writing signal records to daily-partitioned Parquet files
- Reading and querying Parquet files
- Schema management and validation
"""

from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from analize.config import get_settings
from analize.models.signals import SignalRecord


# Define PyArrow schema for signal records
SIGNAL_SCHEMA = pa.schema([
    ("signal_id", pa.string()),
    ("timestamp_utc", pa.timestamp("us", tz="UTC")),
    ("symbol", pa.string()),
    ("pair_id", pa.string()),
    ("timeframe", pa.string()),
    ("mode", pa.string()),
    ("branch", pa.string()),
    ("strategy_version", pa.string()),
    ("price_open", pa.float64()),
    ("price_high", pa.float64()),
    ("price_low", pa.float64()),
    ("price_close", pa.float64()),
    ("volume", pa.float64()),
    ("spread", pa.float64()),
    ("spread_bps", pa.float64()),
    ("bid", pa.float64()),
    ("ask", pa.float64()),
    ("orderbook_imbalance_top5", pa.float64()),
    ("candle_body_pct", pa.float64()),
    ("candle_color", pa.string()),
    ("ema_1h_20", pa.float64()),
    ("ema_1h_slope", pa.float64()),
    ("rsi_1h", pa.float64()),
    ("atr_1h", pa.float64()),
    ("atr_1h_pct", pa.float64()),
    ("rsi_4h", pa.float64()),
    ("atr_4h", pa.float64()),
    ("rsi_1m", pa.float64()),
    ("atr_1m", pa.float64()),
    ("bb_width", pa.float64()),
    ("bb_expand_rate", pa.float64()),
    ("vol_zscore_20", pa.float64()),
    ("filters_passed", pa.string()),  # Comma-separated list
    ("filters_passed_count", pa.int32()),
    ("filters_total_count", pa.int32()),
    ("policy_mode", pa.string()),
    ("percentile_threshold", pa.float64()),
    ("expansion_threshold", pa.float64()),
    ("breakout_buffer_bps", pa.float64()),
    ("expected_tp_pct", pa.float64()),
    ("expected_sl_pct", pa.float64()),
    ("position_size_usd", pa.float64()),
    ("risk_pct_account", pa.float64()),
    ("exec_price", pa.float64()),
    ("exec_qty", pa.float64()),
    ("filled", pa.bool_()),
    ("slippage_pct", pa.float64()),
    ("order_id", pa.string()),
    ("mfe_pct", pa.float64()),
    ("mae_pct", pa.float64()),
    ("pnl_usd", pa.float64()),
    ("pnl_pct", pa.float64()),
    ("exit_price", pa.float64()),
    ("exit_time", pa.timestamp("us", tz="UTC")),
    ("exit_reason", pa.string()),
    ("hit_tp_1m", pa.bool_()),
    ("hit_tp_5m", pa.bool_()),
    ("hit_tp_15m", pa.bool_()),
    ("hit_tp_60m", pa.bool_()),
    ("hit_tp_240m", pa.bool_()),
    ("data_hash", pa.string()),
    ("notes", pa.string()),
    ("created_at", pa.timestamp("us", tz="UTC")),
])


class ParquetStorage:
    """Manages Parquet file storage for signal records."""

    def __init__(self, base_path: Path | None = None):
        settings = get_settings()
        self.base_path = base_path or settings.storage.processed_data_path
        self.base_path = Path(self.base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_partition_path(self, dt: date, symbol: str | None = None) -> Path:
        """Get the path for a daily partition."""
        partition = self.base_path / f"year={dt.year}" / f"month={dt.month:02d}" / f"day={dt.day:02d}"
        if symbol:
            partition = partition / f"symbol={symbol}"
        return partition

    def _get_file_path(self, dt: date, symbol: str) -> Path:
        """Get the file path for a specific date and symbol."""
        partition = self._get_partition_path(dt, symbol)
        partition.mkdir(parents=True, exist_ok=True)
        return partition / f"signals_{dt.isoformat()}_{symbol}.parquet"

    def write_signals(
        self,
        signals: list[SignalRecord],
        partition_by_symbol: bool = True,
    ) -> list[Path]:
        """
        Write signal records to Parquet files.

        Args:
            signals: List of signal records to write
            partition_by_symbol: Whether to partition by symbol

        Returns:
            List of file paths written
        """
        if not signals:
            return []

        # Convert to DataFrame
        records = [s.to_flat_dict() for s in signals]
        df = pd.DataFrame(records)

        # Convert UUIDs to strings
        df["signal_id"] = df["signal_id"].astype(str)

        # Ensure timestamp columns are datetime
        for col in ["timestamp_utc", "exit_time", "created_at"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=True)

        # Convert enums to strings
        for col in ["mode", "policy_mode", "exit_reason", "candle_color"]:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: x.value if hasattr(x, "value") else x)

        written_files = []

        if partition_by_symbol:
            # Group by date and symbol
            df["_date"] = df["timestamp_utc"].dt.date
            for (dt, symbol), group in df.groupby(["_date", "symbol"]):
                file_path = self._get_file_path(dt, symbol)
                group = group.drop(columns=["_date"])

                # Append if file exists, otherwise create
                if file_path.exists():
                    existing = pq.read_table(file_path)
                    new_table = pa.Table.from_pandas(group, schema=SIGNAL_SCHEMA, preserve_index=False)
                    combined = pa.concat_tables([existing, new_table])
                    pq.write_table(combined, file_path)
                else:
                    table = pa.Table.from_pandas(group, schema=SIGNAL_SCHEMA, preserve_index=False)
                    pq.write_table(table, file_path)

                written_files.append(file_path)
        else:
            # Write all to single file per date
            df["_date"] = df["timestamp_utc"].dt.date
            for dt, group in df.groupby("_date"):
                file_path = self._get_partition_path(dt) / f"signals_{dt.isoformat()}.parquet"
                file_path.parent.mkdir(parents=True, exist_ok=True)
                group = group.drop(columns=["_date"])

                table = pa.Table.from_pandas(group, schema=SIGNAL_SCHEMA, preserve_index=False)
                pq.write_table(table, file_path)
                written_files.append(file_path)

        return written_files

    def read_signals(
        self,
        start_date: date,
        end_date: date,
        symbols: list[str] | None = None,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Read signal records from Parquet files.

        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            symbols: Optional list of symbols to filter
            columns: Optional list of columns to read

        Returns:
            DataFrame of signal records
        """
        tables = []
        current = start_date

        while current <= end_date:
            if symbols:
                for symbol in symbols:
                    file_path = self._get_file_path(current, symbol)
                    if file_path.exists():
                        table = pq.read_table(file_path, columns=columns)
                        tables.append(table)
            else:
                # Read all symbols for the date
                partition = self._get_partition_path(current)
                if partition.exists():
                    for symbol_dir in partition.iterdir():
                        if symbol_dir.is_dir() and symbol_dir.name.startswith("symbol="):
                            for file_path in symbol_dir.glob("*.parquet"):
                                table = pq.read_table(file_path, columns=columns)
                                tables.append(table)

            current = date(
                current.year + (current.month // 12),
                (current.month % 12) + 1,
                1,
            ) if current.day == 1 else date(current.year, current.month, current.day + 1)

        if not tables:
            return pd.DataFrame()

        combined = pa.concat_tables(tables)
        return combined.to_pandas()

    def get_available_dates(self, symbol: str | None = None) -> list[date]:
        """Get list of dates with available data."""
        dates = []

        for year_dir in self.base_path.glob("year=*"):
            year = int(year_dir.name.split("=")[1])
            for month_dir in year_dir.glob("month=*"):
                month = int(month_dir.name.split("=")[1])
                for day_dir in month_dir.glob("day=*"):
                    day = int(day_dir.name.split("=")[1])
                    dt = date(year, month, day)

                    if symbol:
                        file_path = self._get_file_path(dt, symbol)
                        if file_path.exists():
                            dates.append(dt)
                    else:
                        if any(day_dir.glob("**/*.parquet")):
                            dates.append(dt)

        return sorted(dates)

    def get_available_symbols(self, dt: date | None = None) -> list[str]:
        """Get list of symbols with available data."""
        symbols = set()

        if dt:
            partition = self._get_partition_path(dt)
            if partition.exists():
                for symbol_dir in partition.glob("symbol=*"):
                    symbols.add(symbol_dir.name.split("=")[1])
        else:
            for parquet_file in self.base_path.glob("**/symbol=*/*.parquet"):
                symbol_part = parquet_file.parent.name
                if symbol_part.startswith("symbol="):
                    symbols.add(symbol_part.split("=")[1])

        return sorted(symbols)

    def delete_partition(self, dt: date, symbol: str | None = None) -> bool:
        """Delete a partition."""
        import shutil

        if symbol:
            file_path = self._get_file_path(dt, symbol)
            if file_path.exists():
                file_path.unlink()
                return True
        else:
            partition = self._get_partition_path(dt)
            if partition.exists():
                shutil.rmtree(partition)
                return True
        return False

    def get_stats(self) -> dict[str, Any]:
        """Get storage statistics."""
        total_files = 0
        total_size = 0
        total_records = 0
        symbols = set()
        dates = set()

        for parquet_file in self.base_path.glob("**/*.parquet"):
            total_files += 1
            total_size += parquet_file.stat().st_size

            # Extract symbol from path
            if "symbol=" in str(parquet_file):
                symbol = parquet_file.parent.name.split("=")[1]
                symbols.add(symbol)

            # Extract date from path
            parts = str(parquet_file).split("/")
            for part in parts:
                if part.startswith("day="):
                    day = int(part.split("=")[1])
                    month = int([p for p in parts if p.startswith("month=")][0].split("=")[1])
                    year = int([p for p in parts if p.startswith("year=")][0].split("=")[1])
                    dates.add(date(year, month, day))

            # Count records
            try:
                metadata = pq.read_metadata(parquet_file)
                total_records += metadata.num_rows
            except Exception:
                pass

        return {
            "total_files": total_files,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "total_records": total_records,
            "unique_symbols": len(symbols),
            "unique_dates": len(dates),
            "symbols": sorted(symbols),
            "date_range": {
                "min": min(dates).isoformat() if dates else None,
                "max": max(dates).isoformat() if dates else None,
            },
        }
