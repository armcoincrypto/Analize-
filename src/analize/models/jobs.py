"""
Job and task models for background processing.

These models define the structure for:
- Async job submissions
- Job status tracking
- Job results
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class JobType(str, Enum):
    """Type of background job."""

    INGEST = "INGEST"
    FEATURE_GENERATION = "FEATURE_GENERATION"
    LABELING = "LABELING"
    ANALYSIS = "ANALYSIS"
    OPTIMIZATION = "OPTIMIZATION"
    REPORT_GENERATION = "REPORT_GENERATION"
    BACKTEST = "BACKTEST"
    EXPORT = "EXPORT"


class JobStatus(str, Enum):
    """Job execution status."""

    PENDING = "PENDING"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class JobPriority(str, Enum):
    """Job priority level."""

    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class JobCreate(BaseModel):
    """Model for creating a new job."""

    job_type: JobType
    priority: JobPriority = JobPriority.NORMAL

    # Job parameters
    parameters: dict[str, Any] = Field(default_factory=dict)

    # Scheduling
    scheduled_at: datetime | None = None  # None = run immediately
    timeout_seconds: int = 3600  # 1 hour default

    # Notifications
    notify_on_complete: bool = False
    notify_on_failure: bool = True

    # Metadata
    description: str | None = None
    tags: list[str] = Field(default_factory=list)


class JobProgress(BaseModel):
    """Job progress tracking."""

    current_step: int = 0
    total_steps: int = 0
    step_description: str | None = None
    percent_complete: float = 0.0

    # Timing
    started_at: datetime | None = None
    estimated_completion: datetime | None = None

    # Metrics
    items_processed: int = 0
    items_total: int = 0
    items_failed: int = 0


class JobResult(BaseModel):
    """Job execution result."""

    success: bool
    message: str | None = None

    # Output data
    output: dict[str, Any] = Field(default_factory=dict)

    # File outputs
    output_files: list[str] = Field(default_factory=list)  # Paths to generated files

    # Metrics
    records_processed: int = 0
    records_failed: int = 0
    execution_time_seconds: float = 0.0

    # Errors
    error_type: str | None = None
    error_message: str | None = None
    error_traceback: str | None = None

    # Warnings
    warnings: list[str] = Field(default_factory=list)


class Job(BaseModel):
    """Complete job model with status and results."""

    job_id: UUID = Field(default_factory=uuid4)
    job_type: JobType
    priority: JobPriority = JobPriority.NORMAL
    status: JobStatus = JobStatus.PENDING

    # Parameters
    parameters: dict[str, Any] = Field(default_factory=dict)

    # Timing
    created_at: datetime = Field(default_factory=datetime.utcnow)
    scheduled_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    timeout_seconds: int = 3600

    # Progress
    progress: JobProgress = Field(default_factory=JobProgress)

    # Result (populated on completion)
    result: JobResult | None = None

    # Worker info
    worker_id: str | None = None

    # Notifications
    notify_on_complete: bool = False
    notify_on_failure: bool = True

    # Metadata
    description: str | None = None
    tags: list[str] = Field(default_factory=list)

    # Reproducibility
    data_hash: str | None = None
    code_version: str | None = None

    class Config:
        from_attributes = True

    @property
    def is_terminal(self) -> bool:
        """Check if job is in a terminal state."""
        return self.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)

    @property
    def duration_seconds(self) -> float | None:
        """Get job duration in seconds."""
        if self.started_at is None:
            return None
        end = self.completed_at or datetime.utcnow()
        return (end - self.started_at).total_seconds()


class JobQueue(BaseModel):
    """Job queue status."""

    queue_name: str
    pending_count: int = 0
    running_count: int = 0
    completed_count: int = 0
    failed_count: int = 0

    # Oldest pending job
    oldest_pending_at: datetime | None = None

    # Processing rate
    jobs_per_minute: float = 0.0
    avg_duration_seconds: float = 0.0


class ScheduledJob(BaseModel):
    """Scheduled recurring job configuration."""

    schedule_id: str
    job_type: JobType
    cron_expression: str

    # Job template
    parameters: dict[str, Any] = Field(default_factory=dict)
    priority: JobPriority = JobPriority.NORMAL
    timeout_seconds: int = 3600

    # State
    is_active: bool = True
    last_run_at: datetime | None = None
    next_run_at: datetime | None = None
    last_run_status: JobStatus | None = None

    # Stats
    total_runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
