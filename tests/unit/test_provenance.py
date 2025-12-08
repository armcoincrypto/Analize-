"""Tests for data provenance and lineage tracking."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from analize.utils.provenance import DataLineage, DataSource, ProvenanceTracker


class TestDataSource:
    """Tests for DataSource dataclass."""

    def test_creation(self) -> None:
        """Test DataSource creation."""
        source = DataSource(
            source_id="src_001",
            source_type="database",
            source_path="/data/signals.db",
        )
        assert source.source_type == "database"
        assert source.source_id == "src_001"


class TestDataLineage:
    """Tests for DataLineage dataclass."""

    def test_creation(self) -> None:
        """Test DataLineage creation."""
        lineage = DataLineage(
            data_hash="abc123",
            row_count=1000,
            column_count=10,
        )
        assert lineage.data_hash == "abc123"
        assert lineage.row_count == 1000
        assert lineage.lineage_id is not None

    def test_to_dict(self) -> None:
        """Test DataLineage serialization."""
        lineage = DataLineage(
            data_hash="abc123",
            row_count=1000,
        )
        d = lineage.to_dict()
        assert d["data_hash"] == "abc123"
        assert d["row_count"] == 1000
        assert "lineage_id" in d


class TestProvenanceTracker:
    """Tests for ProvenanceTracker class."""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Create sample DataFrame for testing."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="h")
        return pd.DataFrame({
            "timestamp": dates,
            "symbol": ["BTC/USDT"] * 100,
            "price": np.random.uniform(40000, 45000, 100),
            "pnl_pct": np.random.uniform(-2, 2, 100),
        })

    @pytest.fixture
    def tracker(self) -> ProvenanceTracker:
        """Create ProvenanceTracker instance."""
        return ProvenanceTracker()

    def test_tracker_initialization(self) -> None:
        """Test tracker can be initialized."""
        tracker = ProvenanceTracker()
        assert tracker is not None

    def test_detect_missing_periods(self, tracker: ProvenanceTracker) -> None:
        """Test missing period detection."""
        # Create data with a gap
        dates = list(pd.date_range(start="2024-01-01", periods=24, freq="h"))
        # Remove 6 hours (hours 10-15)
        dates_with_gap = dates[:10] + dates[16:]

        df = pd.DataFrame({
            "timestamp": dates_with_gap,
            "value": range(len(dates_with_gap)),
        })

        missing = tracker.detect_missing_periods(
            df=df,
            timestamp_col="timestamp",
            expected_interval_minutes=60,  # 1 hour
            symbol_col=None,
        )

        # Should detect the gap
        assert len(missing) > 0
