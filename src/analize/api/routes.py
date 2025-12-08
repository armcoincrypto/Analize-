"""
API routes for Analize.
"""

from datetime import date, datetime
from typing import Any
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel

from analize import __version__
from analize.config import get_settings
from analize.models.jobs import Job, JobCreate, JobStatus, JobType
from analize.models.reports import DailyReport, ParameterSuggestion
from analize.storage.parquet import ParquetStorage

router = APIRouter()


# ============================================================================
# Request/Response Models
# ============================================================================


class StatusResponse(BaseModel):
    """API status response."""

    status: str
    version: str
    environment: str
    ingestion_lag_seconds: int | None = None
    last_analysis_time: datetime | None = None
    storage_stats: dict[str, Any] | None = None


class IngestRequest(BaseModel):
    """Request to trigger data ingestion."""

    source_type: str  # "db", "logs", "csv", "parquet"
    source_path: str | None = None
    symbol: str | None = None
    start_date: date | None = None
    end_date: date | None = None
    force_reprocess: bool = False


class IngestResponse(BaseModel):
    """Response from ingestion request."""

    job_id: UUID
    status: str
    message: str


class SignalsQuery(BaseModel):
    """Query parameters for signals."""

    start_date: date
    end_date: date
    symbols: list[str] | None = None
    limit: int = 1000
    offset: int = 0


class SignalsResponse(BaseModel):
    """Response with signal records."""

    total: int
    limit: int
    offset: int
    signals: list[dict[str, Any]]


class OptimizeRequest(BaseModel):
    """Request to run optimization."""

    symbols: list[str]
    start_date: date
    end_date: date
    objective: str = "profit_factor"  # profit_factor, win_rate, sharpe
    parameters: dict[str, dict[str, Any]] | None = None  # {param_name: {min, max, step}}


class JobResponse(BaseModel):
    """Response with job details."""

    job_id: UUID
    job_type: str
    status: str
    progress_percent: float
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: dict[str, Any] | None = None


class SuggestionsResponse(BaseModel):
    """Response with parameter suggestions."""

    suggestions: list[ParameterSuggestion]
    generated_at: datetime


# ============================================================================
# In-memory job storage (would use database in production)
# ============================================================================

_jobs: dict[UUID, Job] = {}


# ============================================================================
# Status Endpoints
# ============================================================================


@router.get("/", tags=["Status"])
async def root() -> dict[str, str]:
    """API root endpoint."""
    return {"message": "Analize API", "version": __version__}


@router.get("/status", response_model=StatusResponse, tags=["Status"])
async def get_status() -> StatusResponse:
    """Get analyzer health and status."""
    settings = get_settings()

    # Get storage stats
    storage = ParquetStorage()
    try:
        storage_stats = storage.get_stats()
    except Exception:
        storage_stats = None

    return StatusResponse(
        status="healthy",
        version=__version__,
        environment=settings.environment,
        storage_stats=storage_stats,
    )


@router.get("/health", tags=["Status"])
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


# ============================================================================
# Ingestion Endpoints
# ============================================================================


@router.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def trigger_ingest(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
) -> IngestResponse:
    """Trigger data ingestion."""
    from uuid import uuid4

    job_id = uuid4()

    # Create job
    job = Job(
        job_id=job_id,
        job_type=JobType.INGEST,
        status=JobStatus.PENDING,
        parameters={
            "source_type": request.source_type,
            "source_path": request.source_path,
            "symbol": request.symbol,
            "start_date": request.start_date.isoformat() if request.start_date else None,
            "end_date": request.end_date.isoformat() if request.end_date else None,
        },
    )
    _jobs[job_id] = job

    # Queue background task
    background_tasks.add_task(_run_ingestion, job_id, request)

    return IngestResponse(
        job_id=job_id,
        status="queued",
        message=f"Ingestion job queued: {request.source_type}",
    )


async def _run_ingestion(job_id: UUID, request: IngestRequest) -> None:
    """Background task to run ingestion."""
    job = _jobs.get(job_id)
    if not job:
        return

    job.status = JobStatus.RUNNING
    job.started_at = datetime.utcnow()

    try:
        # Perform ingestion based on source type
        if request.source_type == "db":
            from analize.ingestion import ScalperBotDBIngestor
            ingestor = ScalperBotDBIngestor()
            signals = ingestor.ingest_trades_as_signals(
                start_date=datetime.combine(request.start_date, datetime.min.time()) if request.start_date else None,
                end_date=datetime.combine(request.end_date, datetime.max.time()) if request.end_date else None,
                symbol=request.symbol,
            )
            job.result = {
                "success": True,
                "records_processed": len(signals),
            }
        else:
            job.result = {"success": False, "message": f"Unknown source type: {request.source_type}"}

        job.status = JobStatus.COMPLETED
    except Exception as e:
        job.status = JobStatus.FAILED
        job.result = {"success": False, "error": str(e)}
    finally:
        job.completed_at = datetime.utcnow()


# ============================================================================
# Signals Endpoints
# ============================================================================


@router.get("/signals", response_model=SignalsResponse, tags=["Signals"])
async def get_signals(
    start_date: date = Query(..., description="Start date"),
    end_date: date = Query(..., description="End date"),
    symbol: str | None = Query(None, description="Filter by symbol"),
    limit: int = Query(1000, ge=1, le=10000, description="Max records to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
) -> SignalsResponse:
    """Fetch signal records."""
    storage = ParquetStorage()

    symbols = [symbol] if symbol else None
    df = storage.read_signals(start_date, end_date, symbols=symbols)

    total = len(df)
    df = df.iloc[offset : offset + limit]

    return SignalsResponse(
        total=total,
        limit=limit,
        offset=offset,
        signals=df.to_dict("records"),
    )


@router.get("/signals/{signal_id}", tags=["Signals"])
async def get_signal(signal_id: str) -> dict[str, Any]:
    """Get a specific signal by ID."""
    # This would query the database or parquet files
    raise HTTPException(status_code=404, detail="Signal not found")


# ============================================================================
# Reports Endpoints
# ============================================================================


@router.get("/reports/daily", tags=["Reports"])
async def get_daily_report(
    report_date: date = Query(..., description="Report date"),
    symbol: str | None = Query(None, description="Filter by symbol"),
) -> dict[str, Any]:
    """Get daily analysis report."""
    from analize.stats import StatsEngine
    from analize.models.signals import SignalRecord

    storage = ParquetStorage()
    df = storage.read_signals(report_date, report_date, symbols=[symbol] if symbol else None)

    if df.empty:
        return {"error": "No data available for this date"}

    # Convert to SignalRecord objects
    signals = []
    for _, row in df.iterrows():
        try:
            signals.append(SignalRecord(**row.to_dict()))
        except Exception:
            continue

    # Generate report
    engine = StatsEngine()
    report = engine.generate_daily_report(signals, datetime.combine(report_date, datetime.min.time()))

    return report.model_dump()


@router.get("/reports/weekly", tags=["Reports"])
async def get_weekly_report(
    week_start: date = Query(..., description="Week start date"),
    symbol: str | None = Query(None, description="Filter by symbol"),
) -> dict[str, Any]:
    """Get weekly analysis report."""
    from datetime import timedelta

    week_end = week_start + timedelta(days=7)

    # Generate daily reports for the week
    daily_reports = []
    current = week_start
    while current < week_end:
        try:
            report = await get_daily_report(current, symbol)
            if "error" not in report:
                daily_reports.append(report)
        except Exception:
            pass
        current += timedelta(days=1)

    return {
        "week_start": week_start.isoformat(),
        "week_end": week_end.isoformat(),
        "daily_reports": daily_reports,
        "summary": {
            "total_days": len(daily_reports),
        },
    }


# ============================================================================
# Optimization Endpoints
# ============================================================================


@router.post("/optimize", response_model=JobResponse, tags=["Optimization"])
async def start_optimization(
    request: OptimizeRequest,
    background_tasks: BackgroundTasks,
) -> JobResponse:
    """Start parameter optimization job."""
    from uuid import uuid4

    job_id = uuid4()

    job = Job(
        job_id=job_id,
        job_type=JobType.OPTIMIZATION,
        status=JobStatus.PENDING,
        parameters={
            "symbols": request.symbols,
            "start_date": request.start_date.isoformat(),
            "end_date": request.end_date.isoformat(),
            "objective": request.objective,
            "parameter_ranges": request.parameters,
        },
    )
    _jobs[job_id] = job

    background_tasks.add_task(_run_optimization, job_id, request)

    return JobResponse(
        job_id=job_id,
        job_type=job.job_type.value,
        status=job.status.value,
        progress_percent=0.0,
        created_at=job.created_at,
    )


async def _run_optimization(job_id: UUID, request: OptimizeRequest) -> None:
    """Background task to run optimization."""
    from analize.optimizer import GridSearchOptimizer, ParameterSpace
    from analize.models.reports import OptimizationObjective
    from analize.models.signals import SignalRecord

    job = _jobs.get(job_id)
    if not job:
        return

    job.status = JobStatus.RUNNING
    job.started_at = datetime.utcnow()

    try:
        # Load signals
        storage = ParquetStorage()
        df = storage.read_signals(
            request.start_date,
            request.end_date,
            symbols=request.symbols,
        )

        if df.empty:
            job.result = {"success": False, "message": "No signals found"}
            job.status = JobStatus.FAILED
            return

        # Convert to SignalRecord
        signals = []
        for _, row in df.iterrows():
            try:
                signals.append(SignalRecord(**row.to_dict()))
            except Exception:
                continue

        # Define parameter spaces
        param_spaces = []
        if request.parameters:
            for name, config in request.parameters.items():
                param_spaces.append(ParameterSpace.from_range(
                    name=name,
                    start=config.get("min", 0.5),
                    end=config.get("max", 5.0),
                    step=config.get("step", 0.5),
                ))
        else:
            # Default TP/SL search
            param_spaces = [
                ParameterSpace.from_range("tp_pct", 0.5, 5.0, 0.5),
                ParameterSpace.from_range("sl_pct", 0.5, 3.0, 0.5),
            ]

        # Run optimization
        objective = OptimizationObjective(request.objective)
        optimizer = GridSearchOptimizer(objective=objective)
        result = optimizer.optimize_fast(signals, param_spaces)

        job.result = {
            "success": True,
            "best_parameters": result.best_parameters,
            "best_score": result.best_score,
            "top_results": result.top_results,
            "combinations_evaluated": result.combinations_evaluated,
        }
        job.status = JobStatus.COMPLETED

    except Exception as e:
        job.status = JobStatus.FAILED
        job.result = {"success": False, "error": str(e)}
    finally:
        job.completed_at = datetime.utcnow()


# ============================================================================
# Jobs Endpoints
# ============================================================================


@router.get("/jobs/{job_id}", response_model=JobResponse, tags=["Jobs"])
async def get_job(job_id: UUID) -> JobResponse:
    """Get job status and results."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobResponse(
        job_id=job.job_id,
        job_type=job.job_type.value,
        status=job.status.value,
        progress_percent=job.progress.percent_complete,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        result=job.result.model_dump() if job.result else None,
    )


@router.get("/jobs", tags=["Jobs"])
async def list_jobs(
    status: str | None = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=100),
) -> list[JobResponse]:
    """List all jobs."""
    jobs = list(_jobs.values())

    if status:
        jobs = [j for j in jobs if j.status.value == status]

    jobs = sorted(jobs, key=lambda x: x.created_at, reverse=True)[:limit]

    return [
        JobResponse(
            job_id=job.job_id,
            job_type=job.job_type.value,
            status=job.status.value,
            progress_percent=job.progress.percent_complete,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
        )
        for job in jobs
    ]


# ============================================================================
# Suggestions Endpoints
# ============================================================================


@router.get("/suggestions", response_model=SuggestionsResponse, tags=["Suggestions"])
async def get_suggestions(
    top: int = Query(5, ge=1, le=20, description="Number of suggestions"),
    symbol: str | None = Query(None, description="Filter by symbol"),
) -> SuggestionsResponse:
    """Get top parameter suggestions."""
    # In production, this would fetch from database
    # For now, return empty list
    return SuggestionsResponse(
        suggestions=[],
        generated_at=datetime.utcnow(),
    )


# ============================================================================
# Export Endpoints
# ============================================================================


@router.get("/export/signals", tags=["Export"])
async def export_signals(
    start_date: date = Query(...),
    end_date: date = Query(...),
    format: str = Query("csv", description="Export format: csv, parquet, json"),
    symbol: str | None = Query(None),
) -> dict[str, Any]:
    """Export signals to file."""
    from analize.ingestion import FileIngestor

    storage = ParquetStorage()
    df = storage.read_signals(start_date, end_date, symbols=[symbol] if symbol else None)

    if df.empty:
        raise HTTPException(status_code=404, detail="No signals found")

    # Generate export file
    settings = get_settings()
    export_path = settings.storage.reports_path / f"export_{start_date}_{end_date}.{format}"

    ingestor = FileIngestor()
    ingestor.write_file(df, export_path, format=format)

    return {
        "success": True,
        "path": str(export_path),
        "records": len(df),
        "format": format,
    }
