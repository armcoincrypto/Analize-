"""
File-based data ingestion.

Handles:
- CSV files
- Parquet files
- JSON/JSONL files
- Automatic schema detection
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Generator

import pandas as pd
import pyarrow.parquet as pq

from analize.models.signals import Signal, SignalRecord


class FileIngestor:
    """Ingests data from various file formats."""

    SUPPORTED_FORMATS = {
        ".csv": "csv",
        ".parquet": "parquet",
        ".pq": "parquet",
        ".json": "json",
        ".jsonl": "jsonl",
        ".ndjson": "jsonl",
    }

    def __init__(self, base_path: Path | str | None = None):
        self.base_path = Path(base_path) if base_path else Path.cwd()

    def detect_format(self, file_path: Path | str) -> str | None:
        """Detect file format from extension."""
        path = Path(file_path)
        suffix = path.suffix.lower()

        # Handle compressed files
        if suffix == ".gz":
            inner_suffix = Path(path.stem).suffix.lower()
            return self.SUPPORTED_FORMATS.get(inner_suffix)

        return self.SUPPORTED_FORMATS.get(suffix)

    def read_file(
        self,
        file_path: Path | str,
        format: str | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Read a file into a DataFrame.

        Args:
            file_path: Path to file
            format: File format (auto-detected if None)
            **kwargs: Additional arguments passed to reader

        Returns:
            DataFrame with file contents
        """
        path = Path(file_path)
        format = format or self.detect_format(path)

        if format is None:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        if format == "csv":
            return pd.read_csv(path, **kwargs)
        elif format == "parquet":
            return pd.read_parquet(path, **kwargs)
        elif format == "json":
            return pd.read_json(path, **kwargs)
        elif format == "jsonl":
            return pd.read_json(path, lines=True, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def read_json_lines(
        self,
        file_path: Path | str,
    ) -> Generator[dict[str, Any], None, None]:
        """
        Read JSON lines file as generator.

        Args:
            file_path: Path to JSONL file

        Yields:
            Parsed JSON objects
        """
        path = Path(file_path)

        opener = open
        if path.suffix == ".gz":
            import gzip
            opener = gzip.open

        with opener(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue

    def write_file(
        self,
        df: pd.DataFrame,
        file_path: Path | str,
        format: str | None = None,
        **kwargs: Any,
    ) -> Path:
        """
        Write DataFrame to file.

        Args:
            df: DataFrame to write
            file_path: Output path
            format: File format (auto-detected if None)
            **kwargs: Additional arguments passed to writer

        Returns:
            Path to written file
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        format = format or self.detect_format(path)
        if format is None:
            format = "csv"  # Default

        if format == "csv":
            df.to_csv(path, index=False, **kwargs)
        elif format == "parquet":
            df.to_parquet(path, index=False, **kwargs)
        elif format == "json":
            df.to_json(path, orient="records", **kwargs)
        elif format == "jsonl":
            df.to_json(path, orient="records", lines=True, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")

        return path

    def ingest_signals_csv(
        self,
        file_path: Path | str,
        column_mapping: dict[str, str] | None = None,
    ) -> list[SignalRecord]:
        """
        Ingest signals from a CSV file.

        Args:
            file_path: Path to CSV file
            column_mapping: Optional mapping of CSV columns to SignalRecord fields

        Returns:
            List of SignalRecord objects
        """
        df = self.read_file(file_path)

        # Apply column mapping
        if column_mapping:
            df = df.rename(columns=column_mapping)

        # Standardize column names
        df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

        signals = []

        for _, row in df.iterrows():
            try:
                # Convert row to SignalRecord (with defaults for missing fields)
                record_data = {
                    "signal_id": row.get("signal_id", str(hash(str(row)))),
                    "timestamp_utc": pd.to_datetime(
                        row.get("timestamp_utc", row.get("timestamp", datetime.utcnow()))
                    ),
                    "symbol": str(row.get("symbol", "UNKNOWN")),
                    "timeframe": str(row.get("timeframe", "1m")),
                    "price_open": float(row.get("price_open", row.get("open", 0))),
                    "price_high": float(row.get("price_high", row.get("high", 0))),
                    "price_low": float(row.get("price_low", row.get("low", 0))),
                    "price_close": float(row.get("price_close", row.get("close", 0))),
                    "volume": float(row.get("volume", 0)),
                }

                # Add optional fields if present
                optional_fields = [
                    "spread", "spread_bps", "bid", "ask", "orderbook_imbalance_top5",
                    "ema_1h_20", "rsi_1h", "atr_1h", "rsi_1m", "bb_width",
                    "vol_zscore_20", "expected_tp_pct", "expected_sl_pct",
                    "exec_price", "slippage_pct", "mfe_pct", "mae_pct",
                    "pnl_usd", "pnl_pct",
                ]

                for field in optional_fields:
                    if field in row and pd.notna(row[field]):
                        record_data[field] = float(row[field])

                signal = SignalRecord(**record_data)
                signals.append(signal)

            except Exception as e:
                # Skip invalid rows
                print(f"Warning: Could not parse row: {e}")
                continue

        return signals

    def ingest_signals_parquet(
        self,
        file_path: Path | str,
    ) -> list[SignalRecord]:
        """
        Ingest signals from a Parquet file.

        Args:
            file_path: Path to Parquet file

        Returns:
            List of SignalRecord objects
        """
        df = self.read_file(file_path, format="parquet")
        return self._dataframe_to_signals(df)

    def _dataframe_to_signals(self, df: pd.DataFrame) -> list[SignalRecord]:
        """Convert DataFrame to list of SignalRecord objects."""
        signals = []

        for _, row in df.iterrows():
            try:
                record_dict = row.to_dict()

                # Handle NaN values
                cleaned = {}
                for k, v in record_dict.items():
                    if pd.isna(v):
                        cleaned[k] = None
                    else:
                        cleaned[k] = v

                signal = SignalRecord(**cleaned)
                signals.append(signal)
            except Exception:
                continue

        return signals

    def merge_files(
        self,
        file_paths: list[Path | str],
        output_path: Path | str,
        dedupe_column: str | None = "signal_id",
    ) -> Path:
        """
        Merge multiple files into one.

        Args:
            file_paths: List of file paths to merge
            output_path: Output file path
            dedupe_column: Column to use for deduplication (None to skip)

        Returns:
            Path to merged file
        """
        dfs = []

        for path in file_paths:
            try:
                df = self.read_file(path)
                dfs.append(df)
            except Exception as e:
                print(f"Warning: Could not read {path}: {e}")

        if not dfs:
            raise ValueError("No files could be read")

        merged = pd.concat(dfs, ignore_index=True)

        # Deduplicate
        if dedupe_column and dedupe_column in merged.columns:
            merged = merged.drop_duplicates(subset=[dedupe_column], keep="last")

        return self.write_file(merged, output_path)

    def validate_schema(
        self,
        df: pd.DataFrame,
        required_columns: list[str],
    ) -> tuple[bool, list[str]]:
        """
        Validate DataFrame has required columns.

        Args:
            df: DataFrame to validate
            required_columns: List of required column names

        Returns:
            Tuple of (is_valid, missing_columns)
        """
        df_columns = set(df.columns.str.lower())
        required = set(c.lower() for c in required_columns)

        missing = required - df_columns

        return len(missing) == 0, list(missing)

    def compute_file_hash(self, file_path: Path | str) -> str:
        """Compute SHA256 hash of file."""
        hasher = hashlib.sha256()

        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)

        return hasher.hexdigest()[:16]

    def get_file_stats(self, file_path: Path | str) -> dict[str, Any]:
        """Get statistics about a file."""
        path = Path(file_path)

        stats = {
            "path": str(path),
            "exists": path.exists(),
            "size_bytes": 0,
            "format": self.detect_format(path),
            "row_count": 0,
            "column_count": 0,
            "columns": [],
            "hash": None,
        }

        if not path.exists():
            return stats

        stats["size_bytes"] = path.stat().st_size
        stats["hash"] = self.compute_file_hash(path)

        try:
            df = self.read_file(path)
            stats["row_count"] = len(df)
            stats["column_count"] = len(df.columns)
            stats["columns"] = list(df.columns)
        except Exception:
            pass

        return stats

    def list_files(
        self,
        directory: Path | str | None = None,
        pattern: str = "*",
        recursive: bool = False,
    ) -> list[Path]:
        """
        List files in directory matching pattern.

        Args:
            directory: Directory to search (uses base_path if None)
            pattern: Glob pattern
            recursive: Whether to search recursively

        Returns:
            List of file paths
        """
        dir_path = Path(directory) if directory else self.base_path

        if recursive:
            files = list(dir_path.rglob(pattern))
        else:
            files = list(dir_path.glob(pattern))

        # Filter to supported formats
        supported = []
        for f in files:
            if f.is_file() and self.detect_format(f) is not None:
                supported.append(f)

        return sorted(supported)
