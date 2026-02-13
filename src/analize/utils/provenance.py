"""
Data provenance and lineage tracking for reproducibility.

Ensures every analysis run is fully auditable with:
- Data hash and source tracking
- Time range and symbol coverage
- Missing data detection and backfill
- Lineage chain for derived datasets
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pandas as pd

from analize.config import get_settings
from analize.utils.time import utcnow


@dataclass
class DataSource:
    """Represents a data source with metadata."""

    source_id: str
    source_type: str  # "db", "file", "api", "derived"
    source_path: str | None = None
    source_query: str | None = None

    # Parent sources for derived data
    parent_sources: list[str] = field(default_factory=list)


@dataclass
class DataLineage:
    """
    Complete lineage record for a dataset.

    This is stored with every analysis run for full reproducibility.
    """

    lineage_id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=utcnow)

    # Data identification
    data_hash: str = ""
    row_count: int = 0
    column_count: int = 0

    # Source tracking
    sources: list[DataSource] = field(default_factory=list)

    # Time coverage
    time_range_start: datetime | None = None
    time_range_end: datetime | None = None
    symbols: list[str] = field(default_factory=list)

    # Quality metrics
    missing_periods: list[dict[str, Any]] = field(default_factory=list)
    duplicate_count: int = 0
    null_count: dict[str, int] = field(default_factory=dict)

    # Code version
    code_version: str | None = None
    git_commit: str | None = None

    # Processing info
    processing_steps: list[str] = field(default_factory=list)
    parameters_used: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "lineage_id": str(self.lineage_id),
            "created_at": self.created_at.isoformat(),
            "data_hash": self.data_hash,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "sources": [
                {
                    "source_id": s.source_id,
                    "source_type": s.source_type,
                    "source_path": s.source_path,
                    "parent_sources": s.parent_sources,
                }
                for s in self.sources
            ],
            "time_range_start": self.time_range_start.isoformat() if self.time_range_start else None,
            "time_range_end": self.time_range_end.isoformat() if self.time_range_end else None,
            "symbols": self.symbols,
            "missing_periods": self.missing_periods,
            "duplicate_count": self.duplicate_count,
            "null_count": self.null_count,
            "code_version": self.code_version,
            "git_commit": self.git_commit,
            "processing_steps": self.processing_steps,
            "parameters_used": self.parameters_used,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DataLineage":
        """Create from dictionary."""
        lineage = cls(
            lineage_id=UUID(data["lineage_id"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            data_hash=data["data_hash"],
            row_count=data["row_count"],
            column_count=data["column_count"],
            symbols=data.get("symbols", []),
            missing_periods=data.get("missing_periods", []),
            duplicate_count=data.get("duplicate_count", 0),
            null_count=data.get("null_count", {}),
            code_version=data.get("code_version"),
            git_commit=data.get("git_commit"),
            processing_steps=data.get("processing_steps", []),
            parameters_used=data.get("parameters_used", {}),
        )

        if data.get("time_range_start"):
            lineage.time_range_start = datetime.fromisoformat(data["time_range_start"])
        if data.get("time_range_end"):
            lineage.time_range_end = datetime.fromisoformat(data["time_range_end"])

        for s in data.get("sources", []):
            lineage.sources.append(DataSource(
                source_id=s["source_id"],
                source_type=s["source_type"],
                source_path=s.get("source_path"),
                parent_sources=s.get("parent_sources", []),
            ))

        return lineage


class ProvenanceTracker:
    """Tracks data provenance and detects quality issues."""

    def __init__(self):
        settings = get_settings()
        self.lineage_path = settings.storage.processed_data_path / "lineage"
        self.lineage_path.mkdir(parents=True, exist_ok=True)

        # Get git info
        self.git_commit = settings.git_commit_hash
        self.code_version = settings.strategy_version

    def compute_dataframe_hash(self, df: pd.DataFrame) -> str:
        """Compute deterministic hash of DataFrame."""
        # Sort for determinism
        df_sorted = df.sort_values(by=list(df.columns)).reset_index(drop=True)

        # Convert to JSON string
        data_str = df_sorted.to_json(orient="records", date_format="iso")

        return hashlib.sha256(data_str.encode()).hexdigest()

    def detect_missing_periods(
        self,
        df: pd.DataFrame,
        timestamp_col: str = "timestamp",
        expected_interval_minutes: int = 1,
        symbol_col: str | None = "symbol",
    ) -> list[dict[str, Any]]:
        """
        Detect gaps in time series data.

        Returns list of missing period descriptions.
        """
        missing_periods = []

        if timestamp_col not in df.columns:
            return missing_periods

        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

        # Group by symbol if present
        groups = df.groupby(symbol_col) if symbol_col and symbol_col in df.columns else [(None, df)]

        expected_delta = timedelta(minutes=expected_interval_minutes)
        tolerance = timedelta(seconds=30)  # Allow 30s variance

        for symbol, group in groups:
            group = group.sort_values(timestamp_col)
            timestamps = group[timestamp_col].tolist()

            for i in range(1, len(timestamps)):
                actual_delta = timestamps[i] - timestamps[i - 1]

                if actual_delta > expected_delta + tolerance:
                    missing_count = int(actual_delta / expected_delta) - 1
                    missing_periods.append({
                        "symbol": symbol,
                        "gap_start": timestamps[i - 1].isoformat(),
                        "gap_end": timestamps[i].isoformat(),
                        "expected_interval_minutes": expected_interval_minutes,
                        "actual_interval_minutes": actual_delta.total_seconds() / 60,
                        "missing_candles_estimate": missing_count,
                    })

        return missing_periods

    def detect_duplicates(
        self,
        df: pd.DataFrame,
        key_cols: list[str] | None = None,
    ) -> tuple[int, pd.DataFrame]:
        """
        Detect duplicate rows.

        Returns (duplicate_count, duplicated_rows).
        """
        if key_cols is None:
            key_cols = ["timestamp", "symbol"] if "symbol" in df.columns else ["timestamp"]

        key_cols = [c for c in key_cols if c in df.columns]
        if not key_cols:
            return 0, pd.DataFrame()

        duplicates = df[df.duplicated(subset=key_cols, keep=False)]
        return len(duplicates) // 2, duplicates  # Divide by 2 since each dup appears twice

    def analyze_nulls(self, df: pd.DataFrame) -> dict[str, int]:
        """Analyze null values per column."""
        return {col: int(df[col].isna().sum()) for col in df.columns if df[col].isna().any()}

    def create_lineage(
        self,
        df: pd.DataFrame,
        source: DataSource,
        timestamp_col: str = "timestamp",
        symbol_col: str | None = "symbol",
        expected_interval_minutes: int = 1,
        parameters: dict[str, Any] | None = None,
    ) -> DataLineage:
        """
        Create a complete lineage record for a dataset.

        Args:
            df: DataFrame to analyze
            source: Data source information
            timestamp_col: Name of timestamp column
            symbol_col: Name of symbol column (or None)
            expected_interval_minutes: Expected data frequency
            parameters: Parameters used to generate this data

        Returns:
            DataLineage object with full provenance info
        """
        # Compute hash
        data_hash = self.compute_dataframe_hash(df)

        # Detect issues
        missing_periods = self.detect_missing_periods(
            df, timestamp_col, expected_interval_minutes, symbol_col
        )
        duplicate_count, _ = self.detect_duplicates(df)
        null_count = self.analyze_nulls(df)

        # Extract time range
        time_range_start = None
        time_range_end = None
        if timestamp_col in df.columns:
            df_ts = pd.to_datetime(df[timestamp_col])
            time_range_start = df_ts.min().to_pydatetime()
            time_range_end = df_ts.max().to_pydatetime()

        # Extract symbols
        symbols = []
        if symbol_col and symbol_col in df.columns:
            symbols = df[symbol_col].unique().tolist()

        lineage = DataLineage(
            data_hash=data_hash,
            row_count=len(df),
            column_count=len(df.columns),
            sources=[source],
            time_range_start=time_range_start,
            time_range_end=time_range_end,
            symbols=symbols,
            missing_periods=missing_periods,
            duplicate_count=duplicate_count,
            null_count=null_count,
            code_version=self.code_version,
            git_commit=self.git_commit,
            parameters_used=parameters or {},
        )

        return lineage

    def save_lineage(self, lineage: DataLineage, run_id: str | None = None) -> Path:
        """Save lineage record to file."""
        run_id = run_id or str(lineage.lineage_id)
        file_path = self.lineage_path / f"{run_id}.json"

        with open(file_path, "w") as f:
            json.dump(lineage.to_dict(), f, indent=2, default=str)

        return file_path

    def load_lineage(self, run_id: str) -> DataLineage | None:
        """Load lineage record from file."""
        file_path = self.lineage_path / f"{run_id}.json"

        if not file_path.exists():
            return None

        with open(file_path) as f:
            data = json.load(f)

        return DataLineage.from_dict(data)

    def verify_reproducibility(
        self,
        df: pd.DataFrame,
        expected_hash: str,
    ) -> tuple[bool, str]:
        """
        Verify a dataset matches an expected hash.

        Returns (is_match, actual_hash).
        """
        actual_hash = self.compute_dataframe_hash(df)
        return actual_hash == expected_hash, actual_hash

    def get_data_quality_report(self, lineage: DataLineage) -> dict[str, Any]:
        """Generate a data quality report from lineage."""
        issues = []
        warnings = []

        # Check for missing periods
        if lineage.missing_periods:
            total_missing = sum(p.get("missing_candles_estimate", 0) for p in lineage.missing_periods)
            issues.append(f"{len(lineage.missing_periods)} gaps detected, ~{total_missing} missing candles")

        # Check for duplicates
        if lineage.duplicate_count > 0:
            issues.append(f"{lineage.duplicate_count} duplicate records found")

        # Check for nulls
        if lineage.null_count:
            high_null_cols = [col for col, count in lineage.null_count.items()
                             if count > lineage.row_count * 0.1]
            if high_null_cols:
                warnings.append(f"High null rate in columns: {high_null_cols}")

        # Overall quality score (0-100)
        quality_score = 100

        # Deduct for missing periods
        if lineage.row_count > 0:
            missing_ratio = sum(p.get("missing_candles_estimate", 0) for p in lineage.missing_periods) / lineage.row_count
            quality_score -= min(30, missing_ratio * 100)

        # Deduct for duplicates
        if lineage.row_count > 0:
            dup_ratio = lineage.duplicate_count / lineage.row_count
            quality_score -= min(20, dup_ratio * 100)

        # Deduct for nulls
        total_nulls = sum(lineage.null_count.values())
        total_cells = lineage.row_count * lineage.column_count
        if total_cells > 0:
            null_ratio = total_nulls / total_cells
            quality_score -= min(20, null_ratio * 100)

        return {
            "quality_score": max(0, round(quality_score, 1)),
            "row_count": lineage.row_count,
            "time_range": {
                "start": lineage.time_range_start.isoformat() if lineage.time_range_start else None,
                "end": lineage.time_range_end.isoformat() if lineage.time_range_end else None,
            },
            "symbols": lineage.symbols,
            "issues": issues,
            "warnings": warnings,
            "missing_periods_count": len(lineage.missing_periods),
            "duplicate_count": lineage.duplicate_count,
            "null_columns": list(lineage.null_count.keys()),
            "data_hash": lineage.data_hash,
            "code_version": lineage.code_version,
            "git_commit": lineage.git_commit,
        }


class BackfillManager:
    """Manages automatic data backfill for missing periods."""

    def __init__(self, market_data_ingestor=None):
        self.market_data = market_data_ingestor
        self.provenance = ProvenanceTracker()

    async def check_and_backfill(
        self,
        df: pd.DataFrame,
        symbol: str,
        expected_interval_minutes: int = 1,
        auto_fetch: bool = True,
    ) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
        """
        Check for missing data and optionally backfill.

        Args:
            df: Existing DataFrame
            symbol: Symbol to backfill
            expected_interval_minutes: Expected data frequency
            auto_fetch: Whether to automatically fetch missing data

        Returns:
            (Updated DataFrame, list of backfill operations performed)
        """
        missing_periods = self.provenance.detect_missing_periods(
            df,
            timestamp_col="timestamp",
            expected_interval_minutes=expected_interval_minutes,
            symbol_col=None,
        )

        backfill_ops = []

        if not missing_periods or not auto_fetch or not self.market_data:
            return df, backfill_ops

        # Fetch missing data
        for gap in missing_periods:
            try:
                gap_start = datetime.fromisoformat(gap["gap_start"])
                gap_end = datetime.fromisoformat(gap["gap_end"])

                # Fetch from market data
                candles = await self.market_data.fetch_all_klines(
                    symbol=symbol,
                    timeframe=f"{expected_interval_minutes}m",
                    start_time=gap_start,
                    end_time=gap_end,
                )

                if candles:
                    # Convert to DataFrame and append
                    new_data = self.market_data.candles_to_dataframe(candles)
                    df = pd.concat([df, new_data], ignore_index=True)

                    backfill_ops.append({
                        "gap_start": gap_start.isoformat(),
                        "gap_end": gap_end.isoformat(),
                        "candles_fetched": len(candles),
                        "status": "success",
                    })

            except Exception as e:
                backfill_ops.append({
                    "gap_start": gap.get("gap_start"),
                    "gap_end": gap.get("gap_end"),
                    "status": "failed",
                    "error": str(e),
                })

        # Sort and deduplicate
        if backfill_ops:
            df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")

        return df, backfill_ops
