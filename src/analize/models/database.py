"""
SQLAlchemy database models for persistent storage.

These models are used for:
- Metadata storage (jobs, reports, configs)
- Signal records (as backup to Parquet)
- Audit trail
"""

from typing import Any
from uuid import uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base, relationship

from analize.models.jobs import JobPriority, JobStatus, JobType
from analize.utils.time import utcnow_naive
from analize.models.signals import ExitReason, PolicyMode, TradingMode

Base = declarative_base()


class TimestampMixin:
    """Mixin for created_at and updated_at timestamps."""

    created_at = Column(DateTime, default=utcnow_naive, nullable=False)
    updated_at = Column(DateTime, default=utcnow_naive, onupdate=utcnow_naive, nullable=False)


class SignalDB(Base, TimestampMixin):
    """Database model for signal records."""

    __tablename__ = "signals"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    signal_id = Column(UUID(as_uuid=True), unique=True, nullable=False, index=True)
    timestamp_utc = Column(DateTime, nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    pair_id = Column(String(50))
    timeframe = Column(String(10), default="1m")

    # Mode and versioning
    mode = Column(Enum(TradingMode), default=TradingMode.DRY_RUN)
    branch = Column(String(100))
    strategy_version = Column(String(50))

    # Price data
    price_open = Column(Float, nullable=False)
    price_high = Column(Float, nullable=False)
    price_low = Column(Float, nullable=False)
    price_close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)

    # Orderbook
    spread_bps = Column(Float)
    bid = Column(Float)
    ask = Column(Float)
    orderbook_imbalance = Column(Float)

    # Indicators (stored as JSON for flexibility)
    htf_indicators = Column(JSON)
    ltf_indicators = Column(JSON)

    # Filters
    filters_passed = Column(JSON)  # List of filter names
    policy_mode = Column(Enum(PolicyMode))

    # Thresholds
    percentile_threshold = Column(Float)
    expansion_threshold = Column(Float)

    # Position
    expected_tp_pct = Column(Float)
    expected_sl_pct = Column(Float)
    position_size_usd = Column(Float)

    # Execution
    exec_price = Column(Float)
    exec_qty = Column(Float)
    filled = Column(Boolean, default=False)
    slippage_pct = Column(Float)
    order_id = Column(String(100))

    # Outcomes
    mfe_pct = Column(Float)
    mae_pct = Column(Float)
    pnl_usd = Column(Float)
    pnl_pct = Column(Float)
    exit_price = Column(Float)
    exit_time = Column(DateTime)
    exit_reason = Column(Enum(ExitReason))

    # Labels (JSON for flexibility)
    outcome_labels = Column(JSON)

    # Metadata
    data_hash = Column(String(64))
    notes = Column(Text)

    __table_args__ = (
        Index("ix_signals_symbol_timestamp", "symbol", "timestamp_utc"),
        Index("ix_signals_mode_timestamp", "mode", "timestamp_utc"),
    )


class JobDB(Base, TimestampMixin):
    """Database model for background jobs."""

    __tablename__ = "jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    job_id = Column(UUID(as_uuid=True), unique=True, nullable=False, index=True)
    job_type = Column(Enum(JobType), nullable=False, index=True)
    priority = Column(Enum(JobPriority), default=JobPriority.NORMAL)
    status = Column(Enum(JobStatus), default=JobStatus.PENDING, index=True)

    # Parameters
    parameters = Column(JSON)

    # Timing
    scheduled_at = Column(DateTime)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    timeout_seconds = Column(Integer, default=3600)

    # Progress
    progress_current = Column(Integer, default=0)
    progress_total = Column(Integer, default=0)
    progress_description = Column(String(255))

    # Result
    result_success = Column(Boolean)
    result_message = Column(Text)
    result_output = Column(JSON)
    result_files = Column(JSON)  # List of output file paths
    error_type = Column(String(100))
    error_message = Column(Text)
    error_traceback = Column(Text)

    # Worker
    worker_id = Column(String(100))

    # Metadata
    description = Column(Text)
    tags = Column(JSON)
    data_hash = Column(String(64))
    code_version = Column(String(50))

    __table_args__ = (
        Index("ix_jobs_status_created", "status", "created_at"),
        Index("ix_jobs_type_status", "job_type", "status"),
    )


class ReportDB(Base, TimestampMixin):
    """Database model for generated reports."""

    __tablename__ = "reports"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    report_id = Column(UUID(as_uuid=True), unique=True, nullable=False, index=True)
    report_type = Column(String(50), nullable=False, index=True)  # "daily", "weekly"
    report_date = Column(DateTime, nullable=False, index=True)

    # Data versioning
    data_hash = Column(String(64))
    code_tag = Column(String(50))

    # Report content (JSON)
    content = Column(JSON, nullable=False)

    # Summary metrics for quick queries
    total_signals = Column(Integer)
    total_trades = Column(Integer)
    overall_win_rate = Column(Float)
    overall_profit_factor = Column(Float)
    overall_pnl_usd = Column(Float)

    # File path if exported
    file_path = Column(String(500))

    # Status
    is_published = Column(Boolean, default=False)
    published_at = Column(DateTime)

    __table_args__ = (Index("ix_reports_type_date", "report_type", "report_date"),)


class OptimizationRunDB(Base, TimestampMixin):
    """Database model for optimization runs."""

    __tablename__ = "optimization_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    optimization_id = Column(UUID(as_uuid=True), unique=True, nullable=False, index=True)

    # Configuration
    objective = Column(String(50), nullable=False)
    symbols = Column(JSON)
    date_range_start = Column(DateTime, nullable=False)
    date_range_end = Column(DateTime, nullable=False)
    parameters_searched = Column(JSON)

    # Progress
    total_combinations = Column(Integer)
    combinations_evaluated = Column(Integer, default=0)
    status = Column(Enum(JobStatus), default=JobStatus.PENDING)

    # Results
    best_parameters = Column(JSON)
    best_score = Column(Float)
    top_results = Column(JSON)
    walk_forward_results = Column(JSON)

    # Timing
    started_at = Column(DateTime)
    completed_at = Column(DateTime)

    # Reproducibility
    data_hash = Column(String(64))
    code_tag = Column(String(50))


class SuggestionDB(Base, TimestampMixin):
    """Database model for parameter suggestions."""

    __tablename__ = "suggestions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    suggestion_id = Column(UUID(as_uuid=True), unique=True, nullable=False, index=True)

    # What to change
    parameter_name = Column(String(100), nullable=False)
    current_value = Column(String(100))
    suggested_value = Column(String(100))
    change_description = Column(Text)

    # Target
    symbol = Column(String(20), index=True)
    applies_to = Column(JSON)

    # Expected impact
    expected_win_rate_delta = Column(Float)
    expected_pf_delta = Column(Float)
    expected_trade_count_delta = Column(Integer)

    # Confidence
    confidence = Column(String(20))
    p_value = Column(Float)
    sample_size = Column(Integer)

    # Ranking
    rank = Column(Integer)
    objective_score = Column(Float)

    # From optimization run
    optimization_id = Column(UUID(as_uuid=True), ForeignKey("optimization_runs.optimization_id"))

    # Status
    status = Column(String(20), default="pending")  # pending, applied, rejected, tested
    applied_at = Column(DateTime)
    applied_by = Column(String(100))
    test_results = Column(JSON)


class AlertDB(Base, TimestampMixin):
    """Database model for alerts."""

    __tablename__ = "alerts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    alert_id = Column(UUID(as_uuid=True), unique=True, nullable=False, index=True)
    config_id = Column(String(100), index=True)

    # Alert info
    severity = Column(String(20), nullable=False)
    title = Column(String(255), nullable=False)
    message = Column(Text)

    # Context
    metric_name = Column(String(100))
    metric_value = Column(Float)
    threshold = Column(Float)
    symbol = Column(String(20))

    # Status
    acknowledged = Column(Boolean, default=False)
    acknowledged_at = Column(DateTime)
    acknowledged_by = Column(String(100))

    # Notification status
    telegram_sent = Column(Boolean, default=False)
    slack_sent = Column(Boolean, default=False)
    email_sent = Column(Boolean, default=False)


class AuditLogDB(Base):
    """Audit log for tracking all changes and actions."""

    __tablename__ = "audit_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    timestamp = Column(DateTime, default=utcnow_naive, nullable=False, index=True)

    # What happened
    action = Column(String(100), nullable=False, index=True)
    entity_type = Column(String(50))  # "signal", "job", "report", etc.
    entity_id = Column(String(100))

    # Who/what did it
    actor = Column(String(100))  # user, system, worker-id
    actor_type = Column(String(50))  # "user", "system", "scheduler"

    # Details
    details = Column(JSON)
    old_value = Column(JSON)
    new_value = Column(JSON)

    # Context
    ip_address = Column(String(50))
    user_agent = Column(String(500))

    __table_args__ = (Index("ix_audit_action_timestamp", "action", "timestamp"),)


class ConfigSnapshotDB(Base, TimestampMixin):
    """Snapshots of configuration for reproducibility."""

    __tablename__ = "config_snapshots"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    snapshot_id = Column(String(64), unique=True, nullable=False, index=True)

    # What config
    config_type = Column(String(50), nullable=False)  # "scalperbot", "analize", "strategy"
    config_version = Column(String(50))

    # Content
    content = Column(JSON, nullable=False)
    content_hash = Column(String(64), nullable=False)

    # Git info
    git_branch = Column(String(100))
    git_commit = Column(String(64))

    # When it was active
    active_from = Column(DateTime, nullable=False)
    active_to = Column(DateTime)
    is_current = Column(Boolean, default=False)
