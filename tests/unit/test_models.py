"""Tests for data models."""

from datetime import datetime
from uuid import uuid4

import pytest

from analize.models.signals import (
    CandleColor,
    CandleData,
    ExitReason,
    FilterResult,
    OrderbookLevel,
    OrderbookSnapshot,
    PolicyMode,
    Signal,
    SignalRecord,
    TradingMode,
)
from analize.models.reports import (
    DailyReport,
    FilterStats,
    OptimizationObjective,
    SymbolSummary,
)
from analize.models.jobs import Job, JobCreate, JobStatus, JobType
from analize.utils.time import utcnow


class TestCandleData:
    """Tests for CandleData model."""

    def test_candle_creation(self) -> None:
        """Test creating a candle."""
        candle = CandleData(
            timestamp=utcnow(),
            open=100.0,
            high=105.0,
            low=98.0,
            close=103.0,
            volume=1000.0,
        )

        assert candle.open == 100.0
        assert candle.high == 105.0
        assert candle.low == 98.0
        assert candle.close == 103.0
        assert candle.volume == 1000.0

    def test_candle_body_pct(self) -> None:
        """Test candle body percentage calculation."""
        candle = CandleData(
            timestamp=utcnow(),
            open=100.0,
            high=110.0,
            low=90.0,
            close=105.0,
            volume=1000.0,
        )

        # Body = |105 - 100| = 5, Range = 110 - 90 = 20
        # Body % = 5/20 * 100 = 25%
        assert candle.body_pct == 25.0

    def test_candle_color(self) -> None:
        """Test candle color determination."""
        # Bullish candle
        green = CandleData(
            timestamp=utcnow(),
            open=100.0, high=110.0, low=95.0, close=108.0, volume=100.0
        )
        assert green.color == CandleColor.GREEN

        # Bearish candle
        red = CandleData(
            timestamp=utcnow(),
            open=100.0, high=105.0, low=90.0, close=92.0, volume=100.0
        )
        assert red.color == CandleColor.RED


class TestOrderbookSnapshot:
    """Tests for OrderbookSnapshot model."""

    def test_spread_calculation(self) -> None:
        """Test spread calculation."""
        ob = OrderbookSnapshot(
            timestamp=utcnow(),
            symbol="BTCUSDT",
            bids=[OrderbookLevel(price=100.0, quantity=10.0)],
            asks=[OrderbookLevel(price=100.1, quantity=10.0)],
        )

        # Use tolerance for floating point comparison
        assert abs(ob.spread - 0.1) < 1e-10
        assert abs(ob.spread_bps - 10.0) < 0.1  # ~10 bps

    def test_imbalance_calculation(self) -> None:
        """Test orderbook imbalance calculation."""
        ob = OrderbookSnapshot(
            timestamp=utcnow(),
            symbol="BTCUSDT",
            bids=[
                OrderbookLevel(price=100.0, quantity=100.0),
                OrderbookLevel(price=99.9, quantity=50.0),
            ],
            asks=[
                OrderbookLevel(price=100.1, quantity=50.0),
                OrderbookLevel(price=100.2, quantity=25.0),
            ],
        )

        # Bid qty = 150, Ask qty = 75, Imbalance = (150-75)/(150+75) = 75/225 = 0.333
        assert abs(ob.imbalance_top5 - 0.333) < 0.01


class TestSignalRecord:
    """Tests for SignalRecord model."""

    def test_signal_record_creation(self) -> None:
        """Test creating a signal record."""
        record = SignalRecord(
            signal_id=uuid4(),
            timestamp_utc=utcnow(),
            symbol="BTCUSDT",
            mode=TradingMode.LIVE,
            price_open=50000.0,
            price_high=50100.0,
            price_low=49900.0,
            price_close=50050.0,
            volume=100.0,
        )

        assert record.symbol == "BTCUSDT"
        assert record.mode == TradingMode.LIVE

    def test_to_flat_dict(self) -> None:
        """Test converting to flat dictionary."""
        record = SignalRecord(
            signal_id=uuid4(),
            timestamp_utc=utcnow(),
            symbol="BTCUSDT",
            mode=TradingMode.LIVE,
            price_open=50000.0,
            price_high=50100.0,
            price_low=49900.0,
            price_close=50050.0,
            volume=100.0,
            filters_passed=["filter1", "filter2"],
        )

        flat = record.to_flat_dict()
        assert "filters_passed" in flat
        assert flat["filters_passed"] == "filter1,filter2"


class TestJob:
    """Tests for Job model."""

    def test_job_creation(self) -> None:
        """Test creating a job."""
        job = Job(
            job_type=JobType.OPTIMIZATION,
            status=JobStatus.PENDING,
            parameters={"symbol": "BTCUSDT"},
        )

        assert job.job_type == JobType.OPTIMIZATION
        assert job.status == JobStatus.PENDING
        assert not job.is_terminal

    def test_job_terminal_state(self) -> None:
        """Test job terminal state detection."""
        job = Job(
            job_type=JobType.ANALYSIS,
            status=JobStatus.COMPLETED,
        )

        assert job.is_terminal

        job.status = JobStatus.FAILED
        assert job.is_terminal

        job.status = JobStatus.RUNNING
        assert not job.is_terminal
