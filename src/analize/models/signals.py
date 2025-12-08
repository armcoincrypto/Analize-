"""
Signal data models - core schema for ML-ready per-signal records.

These models define the structure for:
- Raw signal data from ScalperBot
- Processed signal records with all features
- Outcome labels for ML training
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class TradingMode(str, Enum):
    """Trading mode."""

    DRY_RUN = "DRY_RUN"
    LIVE = "LIVE"


class PolicyMode(str, Enum):
    """Policy/filter strictness mode."""

    STRICT = "STRICT"
    MODERATE = "MODERATE"
    PERMISSIVE = "PERMISSIVE"


class ExitReason(str, Enum):
    """Reason for position exit."""

    TP = "TP"  # Take profit hit
    SL = "SL"  # Stop loss hit
    TIMEOUT = "TIMEOUT"  # Time-based exit
    MANUAL = "MANUAL"  # Manual intervention
    SIGNAL = "SIGNAL"  # Opposite signal
    LIQUIDATION = "LIQUIDATION"  # Forced liquidation
    UNKNOWN = "UNKNOWN"


class CandleColor(str, Enum):
    """Candle color/direction."""

    GREEN = "GREEN"
    RED = "RED"
    DOJI = "DOJI"


class CandleData(BaseModel):
    """OHLCV candle data."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: str = "1m"

    @property
    def body_pct(self) -> float:
        """Calculate candle body as percentage of range."""
        range_val = self.high - self.low
        if range_val == 0:
            return 0.0
        body = abs(self.close - self.open)
        return (body / range_val) * 100

    @property
    def color(self) -> CandleColor:
        """Determine candle color."""
        if abs(self.close - self.open) < (self.high - self.low) * 0.1:
            return CandleColor.DOJI
        return CandleColor.GREEN if self.close > self.open else CandleColor.RED


class OrderbookLevel(BaseModel):
    """Single orderbook level."""

    price: float
    quantity: float


class OrderbookSnapshot(BaseModel):
    """Orderbook snapshot at signal time."""

    timestamp: datetime
    symbol: str
    bids: list[OrderbookLevel] = Field(default_factory=list, max_length=10)
    asks: list[OrderbookLevel] = Field(default_factory=list, max_length=10)

    @property
    def spread(self) -> float:
        """Calculate bid-ask spread."""
        if not self.bids or not self.asks:
            return 0.0
        return self.asks[0].price - self.bids[0].price

    @property
    def spread_bps(self) -> float:
        """Calculate spread in basis points."""
        if not self.bids or not self.asks:
            return 0.0
        mid = (self.asks[0].price + self.bids[0].price) / 2
        if mid == 0:
            return 0.0
        return (self.spread / mid) * 10000

    @property
    def imbalance_top5(self) -> float:
        """Calculate order book imbalance for top 5 levels."""
        bid_qty = sum(level.quantity for level in self.bids[:5])
        ask_qty = sum(level.quantity for level in self.asks[:5])
        total = bid_qty + ask_qty
        if total == 0:
            return 0.0
        return (bid_qty - ask_qty) / total


class FilterResult(BaseModel):
    """Result of applying signal filters."""

    filter_name: str
    passed: bool
    value: float | None = None
    threshold: float | None = None
    reason: str | None = None


class HTFIndicators(BaseModel):
    """Higher timeframe (1H/4H) technical indicators."""

    # EMA
    ema_1h_20: float | None = None
    ema_1h_50: float | None = None
    ema_1h_slope: float | None = None  # Slope of EMA
    ema_4h_20: float | None = None

    # RSI
    rsi_1h: float | None = None
    rsi_4h: float | None = None

    # ATR
    atr_1h: float | None = None
    atr_1h_pct: float | None = None  # ATR as % of price
    atr_4h: float | None = None

    # Bollinger Bands
    bb_1h_upper: float | None = None
    bb_1h_lower: float | None = None
    bb_1h_width: float | None = None


class LTFIndicators(BaseModel):
    """Lower timeframe (1m/5m) technical indicators."""

    # RSI
    rsi_1m: float | None = None
    rsi_5m: float | None = None

    # ATR
    atr_1m: float | None = None
    atr_5m: float | None = None

    # Bollinger Bands
    bb_width: float | None = None
    bb_expand_rate: float | None = None  # Rate of BB expansion

    # Volume
    volume_zscore_20: float | None = None  # Z-score vs 20-period average
    volume_ratio: float | None = None  # Current vs average

    # VWAP
    vwap: float | None = None
    vwap_distance_pct: float | None = None


class ExecutionData(BaseModel):
    """Trade execution details."""

    order_id: str | None = None
    exec_price: float | None = None
    exec_qty: float | None = None
    filled: bool = False
    slippage_pct: float | None = None
    commission: float | None = None
    exec_timestamp: datetime | None = None


class OutcomeLabels(BaseModel):
    """Outcome labels for ML training at various time windows."""

    # Price movement outcomes (% change from entry)
    pnl_1m: float | None = None
    pnl_5m: float | None = None
    pnl_15m: float | None = None
    pnl_60m: float | None = None
    pnl_240m: float | None = None
    pnl_eod: float | None = None  # End of day

    # Binary hit labels
    hit_tp_1m: bool | None = None
    hit_tp_5m: bool | None = None
    hit_tp_15m: bool | None = None
    hit_tp_60m: bool | None = None
    hit_tp_240m: bool | None = None

    hit_sl_1m: bool | None = None
    hit_sl_5m: bool | None = None
    hit_sl_15m: bool | None = None
    hit_sl_60m: bool | None = None
    hit_sl_240m: bool | None = None

    # Maximum Favorable/Adverse Excursion
    mfe_pct: float | None = None  # Max profit reached
    mae_pct: float | None = None  # Max drawdown reached
    mfe_time_mins: int | None = None  # Time to reach MFE
    mae_time_mins: int | None = None  # Time to reach MAE

    # Time to hit targets
    time_to_tp_mins: int | None = None
    time_to_sl_mins: int | None = None

    # Final outcome
    exit_price: float | None = None
    exit_time: datetime | None = None
    exit_reason: ExitReason | None = None
    pnl_usd: float | None = None
    pnl_pct: float | None = None


class SignalBase(BaseModel):
    """Base signal model with common fields."""

    timestamp_utc: datetime
    symbol: str
    pair_id: str | None = None
    timeframe: str = "1m"


class SignalCreate(SignalBase):
    """Model for creating a new signal record (input from ScalperBot)."""

    mode: TradingMode = TradingMode.DRY_RUN
    branch: str | None = None
    strategy_version: str | None = None

    # Candle data at signal time
    candle: CandleData

    # Orderbook at signal time
    orderbook: OrderbookSnapshot | None = None

    # Filters applied
    filters_passed: list[FilterResult] = Field(default_factory=list)
    policy_mode: PolicyMode = PolicyMode.MODERATE

    # Thresholds used
    percentile_threshold: float | None = None
    expansion_threshold: float | None = None
    breakout_buffer_bps: float | None = None

    # Position sizing
    expected_tp_pct: float | None = None
    expected_sl_pct: float | None = None
    position_size_usd: float | None = None
    risk_pct_account: float | None = None

    # Execution (if filled)
    execution: ExecutionData | None = None

    # Notes/anomalies
    notes: str | None = None


class Signal(SignalBase):
    """Complete signal model with ID."""

    signal_id: UUID = Field(default_factory=uuid4)
    mode: TradingMode = TradingMode.DRY_RUN
    branch: str | None = None
    strategy_version: str | None = None

    candle: CandleData
    orderbook: OrderbookSnapshot | None = None
    filters_passed: list[FilterResult] = Field(default_factory=list)
    policy_mode: PolicyMode = PolicyMode.MODERATE

    percentile_threshold: float | None = None
    expansion_threshold: float | None = None
    breakout_buffer_bps: float | None = None

    expected_tp_pct: float | None = None
    expected_sl_pct: float | None = None
    position_size_usd: float | None = None
    risk_pct_account: float | None = None

    execution: ExecutionData | None = None
    notes: str | None = None

    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        from_attributes = True


class SignalRecord(BaseModel):
    """
    Complete ML-ready signal record with all features and outcomes.

    This is the primary output schema for the analysis pipeline,
    stored as Parquet daily partitions.
    """

    # Identification
    signal_id: UUID
    timestamp_utc: datetime
    symbol: str
    pair_id: str | None = None
    timeframe: str = "1m"

    # Mode and versioning
    mode: TradingMode
    branch: str | None = None
    strategy_version: str | None = None

    # Price data
    price_open: float
    price_high: float
    price_low: float
    price_close: float
    volume: float

    # Orderbook data
    spread: float | None = None
    spread_bps: float | None = None
    bid: float | None = None
    ask: float | None = None
    orderbook_imbalance_top5: float | None = None

    # Candle features
    candle_body_pct: float | None = None
    candle_color: CandleColor | None = None

    # HTF indicators
    ema_1h_20: float | None = None
    ema_1h_slope: float | None = None
    rsi_1h: float | None = None
    atr_1h: float | None = None
    atr_1h_pct: float | None = None
    rsi_4h: float | None = None
    atr_4h: float | None = None

    # LTF indicators
    rsi_1m: float | None = None
    atr_1m: float | None = None
    bb_width: float | None = None
    bb_expand_rate: float | None = None
    vol_zscore_20: float | None = None

    # Filter results
    filters_passed: list[str] = Field(default_factory=list)  # Names of passed filters
    filters_passed_count: int = 0
    filters_total_count: int = 0
    policy_mode: PolicyMode | None = None

    # Thresholds
    percentile_threshold: float | None = None
    expansion_threshold: float | None = None
    breakout_buffer_bps: float | None = None

    # Position parameters
    expected_tp_pct: float | None = None
    expected_sl_pct: float | None = None
    position_size_usd: float | None = None
    risk_pct_account: float | None = None

    # Execution
    exec_price: float | None = None
    exec_qty: float | None = None
    filled: bool = False
    slippage_pct: float | None = None
    order_id: str | None = None

    # Outcome labels
    mfe_pct: float | None = None
    mae_pct: float | None = None
    pnl_usd: float | None = None
    pnl_pct: float | None = None
    exit_price: float | None = None
    exit_time: datetime | None = None
    exit_reason: ExitReason | None = None

    # Time-windowed outcomes
    hit_tp_1m: bool | None = None
    hit_tp_5m: bool | None = None
    hit_tp_15m: bool | None = None
    hit_tp_60m: bool | None = None
    hit_tp_240m: bool | None = None

    # Metadata
    data_hash: str | None = None  # Hash for reproducibility
    notes: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        from_attributes = True

    def to_flat_dict(self) -> dict[str, Any]:
        """Convert to flat dictionary for Parquet storage."""
        data = self.model_dump()
        # Flatten nested structures if any
        if "filters_passed" in data and isinstance(data["filters_passed"], list):
            data["filters_passed"] = ",".join(data["filters_passed"])
        return data
