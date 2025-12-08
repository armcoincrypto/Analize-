"""
Prometheus metrics collection for Analize.

Exposes metrics for monitoring:
- Ingestion lag
- Job status
- Analysis performance
- System health
"""

from datetime import datetime
from typing import Any

try:
    from prometheus_client import Counter, Gauge, Histogram, Info, generate_latest, REGISTRY
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


class MetricsCollector:
    """
    Collects and exposes Prometheus metrics.

    Metrics include:
    - analize_ingestion_lag_seconds: Current ingestion lag
    - analize_signals_total: Total signals processed
    - analize_jobs_total: Jobs by type and status
    - analize_optimization_duration_seconds: Optimization job duration
    - analize_drift_alerts_total: Drift alerts by severity
    - analize_api_requests_total: API request counts
    """

    def __init__(self):
        if not PROMETHEUS_AVAILABLE:
            self._enabled = False
            return

        self._enabled = True

        # Info metric
        self.info = Info("analize", "Analize application info")
        self.info.info({
            "version": "0.1.0",
            "environment": "production",
        })

        # Ingestion metrics
        self.ingestion_lag = Gauge(
            "analize_ingestion_lag_seconds",
            "Current data ingestion lag in seconds",
            ["source", "symbol"],
        )

        self.signals_processed = Counter(
            "analize_signals_processed_total",
            "Total signals processed",
            ["symbol", "mode"],
        )

        self.signals_labeled = Counter(
            "analize_signals_labeled_total",
            "Total signals labeled with outcomes",
            ["symbol"],
        )

        # Job metrics
        self.jobs_total = Counter(
            "analize_jobs_total",
            "Total jobs by type and status",
            ["job_type", "status"],
        )

        self.jobs_in_progress = Gauge(
            "analize_jobs_in_progress",
            "Number of jobs currently in progress",
            ["job_type"],
        )

        self.job_duration = Histogram(
            "analize_job_duration_seconds",
            "Job duration in seconds",
            ["job_type"],
            buckets=[1, 5, 10, 30, 60, 120, 300, 600, 1800, 3600],
        )

        # Optimization metrics
        self.optimization_combinations = Gauge(
            "analize_optimization_combinations_total",
            "Total parameter combinations evaluated in optimization",
            ["objective"],
        )

        self.optimization_best_score = Gauge(
            "analize_optimization_best_score",
            "Best score from latest optimization",
            ["objective", "symbol"],
        )

        # Drift detection metrics
        self.drift_alerts = Counter(
            "analize_drift_alerts_total",
            "Total drift alerts by severity",
            ["severity", "metric", "symbol"],
        )

        self.current_win_rate = Gauge(
            "analize_current_win_rate",
            "Current win rate percentage",
            ["symbol"],
        )

        self.current_profit_factor = Gauge(
            "analize_current_profit_factor",
            "Current profit factor",
            ["symbol"],
        )

        # API metrics
        self.api_requests = Counter(
            "analize_api_requests_total",
            "Total API requests",
            ["endpoint", "method", "status_code"],
        )

        self.api_latency = Histogram(
            "analize_api_latency_seconds",
            "API request latency",
            ["endpoint"],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
        )

        # Storage metrics
        self.storage_size_bytes = Gauge(
            "analize_storage_size_bytes",
            "Storage size in bytes",
            ["storage_type"],  # "parquet", "postgres", "s3"
        )

        self.storage_records_total = Gauge(
            "analize_storage_records_total",
            "Total records in storage",
            ["storage_type"],
        )

        # Approval workflow metrics
        self.approval_requests = Counter(
            "analize_approval_requests_total",
            "Total approval requests",
            ["status"],
        )

        self.pending_approvals = Gauge(
            "analize_pending_approvals",
            "Number of pending approval requests",
        )

    def record_ingestion_lag(self, source: str, symbol: str, lag_seconds: float) -> None:
        """Record current ingestion lag."""
        if self._enabled:
            self.ingestion_lag.labels(source=source, symbol=symbol).set(lag_seconds)

    def record_signal_processed(self, symbol: str, mode: str = "LIVE") -> None:
        """Record a processed signal."""
        if self._enabled:
            self.signals_processed.labels(symbol=symbol, mode=mode).inc()

    def record_signal_labeled(self, symbol: str) -> None:
        """Record a labeled signal."""
        if self._enabled:
            self.signals_labeled.labels(symbol=symbol).inc()

    def record_job_started(self, job_type: str) -> None:
        """Record a job starting."""
        if self._enabled:
            self.jobs_total.labels(job_type=job_type, status="started").inc()
            self.jobs_in_progress.labels(job_type=job_type).inc()

    def record_job_completed(self, job_type: str, duration_seconds: float, success: bool = True) -> None:
        """Record a job completing."""
        if self._enabled:
            status = "completed" if success else "failed"
            self.jobs_total.labels(job_type=job_type, status=status).inc()
            self.jobs_in_progress.labels(job_type=job_type).dec()
            self.job_duration.labels(job_type=job_type).observe(duration_seconds)

    def record_optimization_result(
        self,
        objective: str,
        symbol: str,
        best_score: float,
        combinations: int,
    ) -> None:
        """Record optimization results."""
        if self._enabled:
            self.optimization_combinations.labels(objective=objective).set(combinations)
            self.optimization_best_score.labels(objective=objective, symbol=symbol).set(best_score)

    def record_drift_alert(self, severity: str, metric: str, symbol: str) -> None:
        """Record a drift alert."""
        if self._enabled:
            self.drift_alerts.labels(severity=severity, metric=metric, symbol=symbol).inc()

    def record_performance_metrics(self, symbol: str, win_rate: float, profit_factor: float) -> None:
        """Record current performance metrics."""
        if self._enabled:
            self.current_win_rate.labels(symbol=symbol).set(win_rate)
            self.current_profit_factor.labels(symbol=symbol).set(profit_factor)

    def record_api_request(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        latency_seconds: float,
    ) -> None:
        """Record an API request."""
        if self._enabled:
            self.api_requests.labels(
                endpoint=endpoint,
                method=method,
                status_code=str(status_code),
            ).inc()
            self.api_latency.labels(endpoint=endpoint).observe(latency_seconds)

    def record_storage_stats(
        self,
        storage_type: str,
        size_bytes: int,
        records: int,
    ) -> None:
        """Record storage statistics."""
        if self._enabled:
            self.storage_size_bytes.labels(storage_type=storage_type).set(size_bytes)
            self.storage_records_total.labels(storage_type=storage_type).set(records)

    def record_approval_request(self, status: str) -> None:
        """Record an approval request."""
        if self._enabled:
            self.approval_requests.labels(status=status).inc()

    def set_pending_approvals(self, count: int) -> None:
        """Set the number of pending approvals."""
        if self._enabled:
            self.pending_approvals.set(count)

    def get_metrics(self) -> bytes:
        """Get all metrics in Prometheus format."""
        if self._enabled:
            return generate_latest(REGISTRY)
        return b""


# Global metrics instance
metrics = MetricsCollector()
