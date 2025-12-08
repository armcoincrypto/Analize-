"""
Configuration management for Analize.

Uses pydantic-settings for validation and environment variable loading.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, PostgresDsn, RedisDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """Database connection settings."""

    model_config = SettingsConfigDict(env_prefix="DB_")

    # Primary database (Postgres for metadata)
    postgres_url: PostgresDsn | None = None

    # ScalperBot database (can be SQLite or Postgres)
    scalperbot_db_url: str = "sqlite:///data/scalperbot.db"

    # Connection pool settings
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30


class StorageSettings(BaseSettings):
    """Storage settings for raw data and Parquet files."""

    model_config = SettingsConfigDict(env_prefix="STORAGE_")

    # S3-compatible storage
    s3_endpoint: str | None = None
    s3_bucket: str = "analize-data"
    s3_access_key: str | None = None
    s3_secret_key: str | None = None
    s3_region: str = "us-east-1"

    # Local storage fallback
    local_data_path: Path = Path("data")
    raw_data_path: Path = Path("data/raw")
    processed_data_path: Path = Path("data/processed")
    reports_path: Path = Path("data/reports")

    # Retention policies (days)
    raw_data_retention_days: int = 180  # 6 months hot
    parquet_retention_days: int = 730  # 2 years cold


class RedisSettings(BaseSettings):
    """Redis settings for caching and job queues."""

    model_config = SettingsConfigDict(env_prefix="REDIS_")

    url: RedisDsn = Field(default="redis://localhost:6379/0")  # type: ignore
    cache_ttl: int = 3600  # 1 hour default cache TTL
    job_queue_name: str = "analize:jobs"


class APISettings(BaseSettings):
    """API server settings."""

    model_config = SettingsConfigDict(env_prefix="API_")

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    debug: bool = False
    cors_origins: list[str] = ["*"]

    # Rate limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds


class AnalysisSettings(BaseSettings):
    """Analysis and optimization settings."""

    model_config = SettingsConfigDict(env_prefix="ANALYSIS_")

    # Default symbols to analyze
    default_symbols: list[str] = ["BTCUSDT", "ETHUSDT"]

    # Timeframes for analysis
    timeframes: list[str] = ["1m", "5m", "15m", "1h", "4h"]

    # Outcome labeling windows (minutes)
    outcome_windows: list[int] = [1, 5, 15, 60, 240]

    # Optimization settings
    optimization_objective: Literal["profit_factor", "win_rate", "sharpe", "max_drawdown"] = (
        "profit_factor"
    )
    optimization_top_n: int = 5
    optimization_min_trades: int = 30

    # Walk-forward settings
    walk_forward_train_days: int = 30
    walk_forward_test_days: int = 7

    # Simulation settings
    default_slippage_bps: float = 5.0  # basis points
    default_commission_bps: float = 10.0


class NotificationSettings(BaseSettings):
    """Notification settings for alerts."""

    model_config = SettingsConfigDict(env_prefix="NOTIFY_")

    # Telegram
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None

    # Slack
    slack_webhook_url: str | None = None
    slack_channel: str = "#analize-alerts"

    # Email
    smtp_host: str | None = None
    smtp_port: int = 587
    smtp_user: str | None = None
    smtp_password: str | None = None
    email_from: str = "analize@localhost"
    email_to: list[str] = []


class SchedulerSettings(BaseSettings):
    """Scheduler settings for automated jobs."""

    model_config = SettingsConfigDict(env_prefix="SCHEDULER_")

    # Ingestion cadence (cron expressions)
    ingest_cron: str = "*/5 * * * *"  # every 5 minutes

    # Analysis cadence
    daily_analysis_cron: str = "0 1 * * *"  # 1 AM daily
    weekly_analysis_cron: str = "0 2 * * 0"  # 2 AM Sunday

    # Optimization cadence
    optimization_cron: str = "0 3 * * *"  # 3 AM daily

    # Report generation
    daily_report_cron: str = "0 6 * * *"  # 6 AM daily
    weekly_report_cron: str = "0 7 * * 1"  # 7 AM Monday


class Settings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    # Application info
    app_name: str = "Analize"
    app_version: str = "0.1.0"
    environment: Literal["development", "staging", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # Code/data versioning for reproducibility
    git_commit_hash: str | None = None
    strategy_version: str = "v1.0.0"

    # Nested settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    api: APISettings = Field(default_factory=APISettings)
    analysis: AnalysisSettings = Field(default_factory=AnalysisSettings)
    notifications: NotificationSettings = Field(default_factory=NotificationSettings)
    scheduler: SchedulerSettings = Field(default_factory=SchedulerSettings)

    @field_validator("git_commit_hash", mode="before")
    @classmethod
    def get_git_hash(cls, v: str | None) -> str | None:
        """Attempt to get git commit hash if not provided."""
        if v is not None:
            return v
        try:
            import subprocess

            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()[:8]
        except Exception:
            pass
        return None


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
