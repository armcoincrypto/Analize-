"""
Report and analysis result models.

These models define the structure for:
- Daily and weekly analysis reports
- Per-symbol statistics
- Parameter optimization results
- Automated suggestions
"""

from datetime import date, datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from analize.utils.time import utcnow


class OptimizationObjective(str, Enum):
    """Optimization objective function."""

    PROFIT_FACTOR = "profit_factor"
    WIN_RATE = "win_rate"
    SHARPE = "sharpe"
    MAX_DRAWDOWN = "max_drawdown"
    EXPECTANCY = "expectancy"
    TRADES_COUNT = "trades_count"


class ConfidenceLevel(str, Enum):
    """Confidence level for suggestions."""

    HIGH = "HIGH"  # p < 0.01
    MEDIUM = "MEDIUM"  # p < 0.05
    LOW = "LOW"  # p < 0.10
    UNCERTAIN = "UNCERTAIN"  # p >= 0.10


class FilterStats(BaseModel):
    """Statistics for a specific filter or filter combination."""

    filter_name: str
    filter_combination: list[str] | None = None  # For multi-filter analysis

    # Counts
    signals_count: int = 0
    trades_count: int = 0
    wins_count: int = 0
    losses_count: int = 0

    # Metrics
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    avg_rr_ratio: float = 0.0  # Risk:Reward ratio

    # MAE/MFE
    avg_mfe_pct: float = 0.0
    avg_mae_pct: float = 0.0

    # Timing
    avg_time_in_trade_mins: float = 0.0
    avg_time_to_tp_mins: float | None = None
    avg_time_to_sl_mins: float | None = None

    # Sample trades
    sample_trade_ids: list[str] = Field(default_factory=list)


class ThresholdSensitivity(BaseModel):
    """Sensitivity analysis for a threshold parameter."""

    parameter_name: str
    current_value: float
    test_values: list[float] = Field(default_factory=list)

    # Results at each test value
    results: list[dict[str, Any]] = Field(default_factory=list)
    # Each result: {value, win_rate, profit_factor, trade_count, delta_pf, delta_trades}


class TimeHeatmap(BaseModel):
    """Time-based performance heatmap."""

    # Hour of day performance (0-23)
    hour_performance: dict[int, dict[str, float]] = Field(default_factory=dict)
    # Each entry: {win_rate, profit_factor, trade_count}

    # Day of week performance (0=Monday, 6=Sunday)
    day_performance: dict[int, dict[str, float]] = Field(default_factory=dict)


class VolatilityRegimeStats(BaseModel):
    """Performance by volatility regime."""

    regime: str  # "LOW", "MEDIUM", "HIGH"
    atr_range: tuple[float, float]  # ATR range defining regime

    signals_count: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_pnl_pct: float = 0.0


class SymbolSummary(BaseModel):
    """Per-symbol performance summary."""

    symbol: str
    period_start: datetime
    period_end: datetime

    # Counts
    signals_count: int = 0
    trades_count: int = 0
    unique_entries: int = 0
    duplicate_signals: int = 0
    open_positions: int = 0

    # Core metrics
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    avg_rr_ratio: float = 0.0

    # PnL
    total_pnl_usd: float = 0.0
    total_pnl_pct: float = 0.0
    avg_pnl_usd: float = 0.0
    avg_pnl_pct: float = 0.0

    # MAE/MFE
    avg_mfe_pct: float = 0.0
    avg_mae_pct: float = 0.0

    # Execution quality
    avg_slippage_pct: float = 0.0
    median_slippage_pct: float = 0.0
    avg_spread_bps: float = 0.0

    # Timing
    avg_time_in_trade_mins: float = 0.0

    # Risk metrics
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float | None = None
    sortino_ratio: float | None = None

    # Per-filter stats
    filter_stats: list[FilterStats] = Field(default_factory=list)

    # Time analysis
    time_heatmap: TimeHeatmap | None = None

    # Volatility analysis
    volatility_stats: list[VolatilityRegimeStats] = Field(default_factory=list)


class ParameterSuggestion(BaseModel):
    """Automated parameter change suggestion."""

    suggestion_id: UUID = Field(default_factory=uuid4)

    # What to change
    parameter_name: str
    current_value: float | str
    suggested_value: float | str
    change_description: str  # Human-readable description

    # Target (symbol-specific or global)
    symbol: str | None = None  # None = global suggestion
    applies_to: list[str] = Field(default_factory=list)  # List of affected symbols

    # Expected impact
    expected_win_rate_delta: float = 0.0
    expected_pf_delta: float = 0.0
    expected_trade_count_delta: int = 0
    expected_trade_count_delta_pct: float = 0.0

    # Trade-off description
    tradeoff: str | None = None  # e.g., "-20% signals, +12% PF"

    # Statistical confidence
    confidence: ConfidenceLevel = ConfidenceLevel.UNCERTAIN
    p_value: float | None = None
    sample_size: int = 0

    # Supporting evidence
    backtest_pnl_curve: list[float] = Field(default_factory=list)
    sample_trade_ids: list[str] = Field(default_factory=list)

    # Ranking
    rank: int = 0
    objective_score: float = 0.0


class ActionItem(BaseModel):
    """Actionable recommendation from analysis."""

    action_type: str  # "DISABLE_PAIR", "TEST_PARAMETER", "INVESTIGATE", etc.
    priority: str  # "HIGH", "MEDIUM", "LOW"
    description: str
    symbol: str | None = None
    duration: str | None = None  # e.g., "24h", "7 days"
    parameters: dict[str, Any] = Field(default_factory=dict)


class OptimizationResult(BaseModel):
    """Result of parameter optimization."""

    optimization_id: UUID = Field(default_factory=uuid4)
    started_at: datetime
    completed_at: datetime | None = None

    # Configuration
    objective: OptimizationObjective
    symbols: list[str]
    date_range_start: date
    date_range_end: date

    # Parameters searched
    parameters_searched: dict[str, list[Any]] = Field(default_factory=dict)
    total_combinations: int = 0
    combinations_evaluated: int = 0

    # Best results
    best_parameters: dict[str, Any] = Field(default_factory=dict)
    best_score: float = 0.0

    # Top N results
    top_results: list[dict[str, Any]] = Field(default_factory=list)
    # Each: {parameters, score, win_rate, profit_factor, trades_count, pnl_curve}

    # Walk-forward validation
    walk_forward_results: list[dict[str, Any]] = Field(default_factory=list)

    # Reproducibility
    data_hash: str | None = None
    code_tag: str | None = None


class DailyReport(BaseModel):
    """Daily analysis report."""

    report_id: UUID = Field(default_factory=uuid4)
    report_date: date
    generated_at: datetime = Field(default_factory=utcnow)

    # Data versioning
    data_hash: str | None = None
    code_tag: str | None = None

    # Per-symbol summaries
    symbol_summaries: list[SymbolSummary] = Field(default_factory=list)

    # Cross-symbol metrics
    total_signals: int = 0
    total_trades: int = 0
    overall_win_rate: float = 0.0
    overall_profit_factor: float = 0.0
    overall_pnl_usd: float = 0.0

    # Best/worst filter combinations
    top_filter_combos: list[FilterStats] = Field(default_factory=list)
    worst_filter_combos: list[FilterStats] = Field(default_factory=list)

    # Threshold sensitivity analysis
    threshold_sensitivities: list[ThresholdSensitivity] = Field(default_factory=list)

    # Suggestions
    parameter_suggestions: list[ParameterSuggestion] = Field(default_factory=list)

    # Action items
    action_items: list[ActionItem] = Field(default_factory=list)

    # Alerts
    alerts: list[str] = Field(default_factory=list)

    # Notes
    notes: str | None = None


class WeeklyReport(BaseModel):
    """Weekly deep analysis report."""

    report_id: UUID = Field(default_factory=uuid4)
    week_start: date
    week_end: date
    generated_at: datetime = Field(default_factory=utcnow)

    # Data versioning
    data_hash: str | None = None
    code_tag: str | None = None

    # Include daily reports
    daily_reports: list[DailyReport] = Field(default_factory=list)

    # Rolling metrics
    rolling_win_rate_7d: float = 0.0
    rolling_profit_factor_7d: float = 0.0
    rolling_max_drawdown_7d: float = 0.0

    # Symbol rankings
    symbol_rankings: list[dict[str, Any]] = Field(default_factory=list)
    # Each: {symbol, rank, score, win_rate, pf, recommendation}

    # Parameter changes tested
    optimization_results: list[OptimizationResult] = Field(default_factory=list)

    # Cross-validation results
    cross_validation_score: float | None = None
    stability_score: float | None = None  # How stable are results across folds

    # Suggestions with confidence levels
    parameter_suggestions: list[ParameterSuggestion] = Field(default_factory=list)

    # Weekly trends
    trend_analysis: dict[str, Any] = Field(default_factory=dict)

    # Correlation analysis (with BTC, etc.)
    correlation_analysis: dict[str, float] = Field(default_factory=dict)

    # Notes
    notes: str | None = None


class AlertConfig(BaseModel):
    """Configuration for an alert rule."""

    alert_id: str
    name: str
    description: str

    # Trigger conditions
    metric: str  # "win_rate", "profit_factor", "error_rate", etc.
    threshold: float
    comparison: str  # "lt", "gt", "eq", "change_pct"
    window_hours: int = 24

    # Actions
    notify_telegram: bool = True
    notify_slack: bool = True
    notify_email: bool = False

    # State
    is_active: bool = True
    last_triggered: datetime | None = None


class Alert(BaseModel):
    """Generated alert."""

    alert_id: UUID = Field(default_factory=uuid4)
    config_id: str
    triggered_at: datetime = Field(default_factory=utcnow)

    severity: str  # "INFO", "WARNING", "CRITICAL"
    title: str
    message: str

    # Context
    metric_name: str
    metric_value: float
    threshold: float
    symbol: str | None = None

    # Status
    acknowledged: bool = False
    acknowledged_at: datetime | None = None
    acknowledged_by: str | None = None
