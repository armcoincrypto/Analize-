"""
HFT System Configuration
========================
MICROSTRUCTURE TRADING (based on 141K signal analysis)

Key insight from data:
- Orderbook alone: +0.012% at 5m (BEST) - 55,781 signals
- 1/5 condition optimal - more conditions = worse
- Edge is 0.008-0.014% - match targets to edge

Strategy: Orderbook microstructure, NOT scalping
- Hold 10-30 seconds, not minutes
- TP 0.15%, SL 0.1% - match actual edge
- Event-based exit, not time-based
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict
from enum import Enum


# ---------------------------------------------------------------------------
# SMOKE_TEST flag: when True, enables test-trade creation (maker_smoke_test,
# force_paper_trade, manual_close_orphan).  Default OFF in production.
# Set via environment: SMOKE_TEST=1 to enable.
# ---------------------------------------------------------------------------
SMOKE_TEST: bool = os.environ.get("SMOKE_TEST", "0") == "1"

# Exit reasons reserved for test / diagnostic trades
SMOKE_TEST_EXIT_REASONS: frozenset = frozenset({
    "maker_smoke_test",
    "force_paper_trade",
    "manual_close_orphan",
})


# ---------------------------------------------------------------------------
# Adaptive minimum take-profit configuration (Deliverable 4)
# ---------------------------------------------------------------------------
BASE_MIN_PROFIT_PCT: float = float(os.environ.get("BASE_MIN_PROFIT_PCT", "0.08"))
COST_MULTIPLIER: float = float(os.environ.get("COST_MULTIPLIER", "1.0"))
COST_BUFFER_PCT: float = float(os.environ.get("COST_BUFFER_PCT", "0.03"))


class TradingMode(Enum):
    PAPER = "paper"
    LIVE = "live"
    BACKTEST = "backtest"


@dataclass
class AssetConfig:
    """Configuration for each trading asset."""
    symbol: str
    exchange_symbol: str  # e.g., "ATOMUSDT"

    # Entry thresholds
    min_price_drop_pct: float = 0.8      # Minimum price drop to trigger
    max_price_drop_pct: float = 1.2      # Maximum (avoid catching knives)
    min_price_spike_pct: float = 1.0     # For upward impulse
    impulse_window_sec: int = 60         # Time window for impulse detection

    # Volume thresholds
    volume_spike_multiplier: float = 2.5  # Volume must be 2.5x average
    volume_avg_period: int = 20           # 20 candles for average

    # Order book thresholds
    orderbook_imbalance_ratio: float = 0.6  # 60/40 bid/ask ratio

    # RSI thresholds
    rsi_oversold: float = 25.0
    rsi_overbought: float = 75.0

    # Exit settings (STRUCTURE-BASED - let structure decide, not timers)
    # Increased limits to give structure-based exits time to work
    take_profit_pct: float = 0.20         # Slightly higher TP for bigger moves
    stop_loss_pct: float = 0.12           # Slightly wider SL
    time_stop_seconds: int = 60           # Increased to 60s - structure exits first
    min_profit_for_time_check: float = 0.05  # Any profit = exit at time
    use_trailing_stop: bool = False       # Not useful for micro trades
    trailing_stop_activation: float = 0.08  # Lower activation
    trailing_stop_distance: float = 0.05   # Tight trail

    # Position sizing
    max_position_pct: float = 2.0  # 2% of capital per trade


# Asset-specific configurations
ASSETS = {
    "ATOM": AssetConfig(
        symbol="ATOM",
        exchange_symbol="ATOMUSDT",
        min_price_drop_pct=0.8,
        volume_spike_multiplier=2.5,
    ),
    "SUI": AssetConfig(
        symbol="SUI",
        exchange_symbol="SUIUSDT",
        min_price_drop_pct=1.0,  # More volatile
        volume_spike_multiplier=2.0,
    ),
    "XRP": AssetConfig(
        symbol="XRP",
        exchange_symbol="XRPUSDT",
        min_price_drop_pct=0.6,  # Lower threshold - less volatile
        volume_spike_multiplier=3.0,  # Higher volume requirement
    ),
}


@dataclass
class RiskConfig:
    """Risk management configuration."""

    # Position limits
    max_positions_per_symbol: int = 1
    max_total_positions: int = 2
    position_size_pct: float = 1.0  # 1% of capital per trade

    # Daily limits (set high for dry run data collection)
    max_daily_losses: int = 100  # No limit in dry run - collect data fast!
    max_daily_drawdown_pct: float = 50.0  # No limit in dry run

    # Trade limits - microstructure = fast
    min_time_between_trades_sec: int = 2  # Faster for dry run data collection

    # Market filters
    btc_max_volatility_5m: float = 4.0  # Disable if BTC moves >4% in 5min
    max_exchange_latency_ms: int = 500  # Disable if latency >500ms


# ============================================================
# COST MODEL - Realistic trading costs simulation
# ============================================================
# Expert feedback: "Your target edge is 0.01-0.02% which is below
# real trading costs. Must simulate fees + spread + slippage."
# ============================================================
@dataclass
class CostModelConfig:
    """Realistic cost model for paper trading evaluation."""

    # Enable cost model (subtract from PnL)
    enabled: bool = True

    # === EXECUTION MODE ===
    # "taker" = market orders (instant fill, high fees)
    # "maker" = limit orders (may not fill, low fees)
    execution_mode: str = "taker"  # "taker" or "maker"

    # === TAKER COSTS (market orders) ===
    taker_entry_fee_pct: float = 0.10  # 0.10% taker fee
    taker_exit_fee_pct: float = 0.10   # 0.10% taker fee
    taker_spread_cost_pct: float = 0.02  # ~0.02% crossing spread
    taker_slippage_pct: float = 0.02  # Market impact

    # === MAKER COSTS (limit orders) ===
    maker_entry_fee_pct: float = 0.01  # 0.01% maker fee (or rebate)
    maker_exit_fee_pct: float = 0.01   # 0.01% maker fee
    maker_spread_cost_pct: float = 0.005  # Minimal - you set the price
    maker_slippage_pct: float = 0.0  # No slippage - you set the price

    # === MAKER SIMULATION ===
    maker_fill_probability: float = 0.70  # 70% chance of fill
    maker_wait_seconds: float = 1.0  # Wait time for limit order

    # Computed totals (for reference):
    # TAKER: 2*0.10 + 2*0.02 + 0.02 = 0.26% per round-trip
    # MAKER: 2*0.01 + 2*0.005 + 0.0 = 0.03% per round-trip

    @property
    def entry_fee_pct(self) -> float:
        """Get entry fee based on execution mode."""
        return self.maker_entry_fee_pct if self.execution_mode == "maker" else self.taker_entry_fee_pct

    @property
    def exit_fee_pct(self) -> float:
        """Get exit fee based on execution mode."""
        return self.maker_exit_fee_pct if self.execution_mode == "maker" else self.taker_exit_fee_pct

    @property
    def spread_cost_pct(self) -> float:
        """Get spread cost based on execution mode."""
        return self.maker_spread_cost_pct if self.execution_mode == "maker" else self.taker_spread_cost_pct

    @property
    def base_slippage_pct(self) -> float:
        """Get base slippage based on execution mode."""
        return self.maker_slippage_pct if self.execution_mode == "maker" else self.taker_slippage_pct

    # Legacy compatibility
    imbalance_slippage_factor: float = 0.02  # Extra slippage when imbalance is extreme (taker only)


COST_MODEL = CostModelConfig()


# ============================================================
# REGIME HYSTERESIS - Stable regime classification
# ============================================================
# Expert feedback: "Regime flips every ~30s. That means regime
# label is noisy, gate blocks constantly. Add hysteresis."
# ============================================================
@dataclass
class RegimeHysteresisConfig:
    """Stable regime classification with confirmation requirements."""

    enabled: bool = True

    # Confirmations required to ENTER a new regime
    # Regime must be detected N consecutive times to switch
    enter_confirmations: int = 3

    # Failures required to EXIT current regime
    # Current regime must fail M times before switching
    exit_failures: int = 3

    # Minimum time in regime before allowing switch (seconds)
    min_regime_duration_sec: float = 30.0

    # Log both raw and stable regimes for analysis
    log_raw_regime: bool = True


REGIME_HYSTERESIS = RegimeHysteresisConfig()


@dataclass
class SystemConfig:
    """Overall system configuration."""

    mode: TradingMode = TradingMode.PAPER

    # Capital
    initial_capital: float = 10000.0

    # Exchange
    exchange: str = "binance"
    use_testnet: bool = True

    # WebSocket settings
    ws_reconnect_delay_sec: int = 5
    ws_heartbeat_interval_sec: int = 30
    ws_max_reconnect_attempts: int = 10

    # Data settings
    orderbook_depth: int = 20
    kline_interval: str = "1m"

    # Logging
    log_level: str = "INFO"
    log_trades: bool = True
    log_signals: bool = True
    log_orderbook: bool = False  # Very verbose

    # Database
    db_path: str = "hft_trades.db"

    # Assets to trade (data-backed: ATOM had 0% win rate in 19 trades - DISABLED)
    enabled_assets: List[str] = field(default_factory=lambda: ["SUI", "XRP"])

    # Entry conditions - data shows 1/5 is OPTIMAL (more = worse)
    # Orderbook alone: +0.012% at 5m with 55,781 signals
    min_entry_conditions: int = 1


# ============================================================
# WINNER GATE v2 — Dual-Pocket Entry Filter
# ============================================================
# Expert feedback: "Use a 2-layer gate - primary for best regime,
# secondary allows some trades in low_vol_chop with extra confirmation"
#
# POCKET A (Primary): high_vol_trend + HIGH/MEDIUM tier + imb>=0.70
# POCKET B (Secondary): low_vol_chop + imb>=0.80 + tight spread + high depth
# ============================================================
@dataclass
class WinnerGateConfig:
    """Winner Gate v2 - Dual-pocket entry filter."""

    # Master switch
    enabled: bool = True

    # RESEARCH_GATE: Relaxed settings for paper mode data collection
    research_mode: bool = True  # Auto-set based on SYSTEM_CONFIG.mode

    # === POCKET A: PRIMARY (high_vol_trend - the proven edge) ===
    pocket_a_enabled: bool = True
    pocket_a_regimes: List[str] = field(default_factory=lambda: ["high_vol_trend"])
    pocket_a_tiers: List[str] = field(default_factory=lambda: ["high", "medium"])
    pocket_a_min_imbalance: float = 0.70
    pocket_a_max_spread_pct: float = 0.05

    # === POCKET B: SECONDARY (low_vol_chop with extra confirmation) ===
    # Research-friendly but still strict - enough to collect data
    pocket_b_enabled: bool = True  # Enable secondary pocket
    pocket_b_regimes: List[str] = field(default_factory=lambda: ["low_vol_chop"])
    pocket_b_tiers: List[str] = field(default_factory=lambda: ["high", "medium"])  # Allow medium for research
    pocket_b_min_imbalance: float = 0.70  # Match Pocket A for research
    pocket_b_max_spread_pct: float = 0.03  # Slightly relaxed for research (was 0.02)
    pocket_b_min_depth: float = 10000.0  # Minimum OB depth (USD)
    pocket_b_flow_confirm_sec: float = 1.0  # Faster confirm for research (was 2.0)

    # === BLOCKED REGIMES (never trade) ===
    blocked_regimes: List[str] = field(default_factory=lambda: ["mean_reversion", "liquidity_vacuum", "news_spike", "unknown"])

    # === CAUSALITY FILTER (only trade when WHY is clear) ===
    # Only trade when primary_cause is strong, block "unknown" causes
    causality_filter_enabled: bool = True
    allowed_causes_long: List[str] = field(default_factory=lambda: [
        "cvd_buy_pressure",      # CVD shows buy pressure - strong signal
        "ob_bullish_imbalance",  # Orderbook is bullish
        "price_momentum_up"      # Price moving up
    ])
    allowed_causes_short: List[str] = field(default_factory=lambda: [
        "cvd_sell_pressure",     # CVD shows sell pressure - strong signal
        "ob_bearish_imbalance",  # Orderbook is bearish
        "price_momentum_down"    # Price moving down
    ])
    block_unknown_cause: bool = True  # Block trades with no clear cause

    # === LEGACY SETTINGS (for backward compatibility) ===
    strict_mode: bool = True  # If True, only Pocket A in live mode
    min_confidence_tier: str = "high"
    min_imbalance: float = 0.75
    max_spread_pct: float = 0.05
    research_allowed_tiers: List[str] = field(default_factory=lambda: ["high", "medium"])
    research_min_imbalance: float = 0.70

    # === PROBE PROTECTIONS (research mode - higher limits) ===
    max_trades_per_symbol_per_hour: int = 6  # Increased for research (was 2)
    min_seconds_between_trades_per_symbol: int = 120  # 2 minutes


WINNER_GATE = WinnerGateConfig()


# Global configs
RISK_CONFIG = RiskConfig()
SYSTEM_CONFIG = SystemConfig()


def get_asset_config(symbol: str) -> AssetConfig:
    """Get configuration for a specific asset."""
    return ASSETS.get(symbol.upper(), ASSETS["ATOM"])
