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

from dataclasses import dataclass, field
from typing import List, Dict
from enum import Enum


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
    # COST-AWARE: Thresholds must be above round-trip costs (~0.26% taker mode)
    take_profit_pct: float = 0.40         # Must be > costs (0.26%) for net profit
    stop_loss_pct: float = 0.15           # Slightly wider SL to reduce churn
    time_stop_seconds: int = 60           # Increased to 60s - structure exits first
    min_profit_for_time_check: float = 0.28  # Must be above costs to count as profit
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
    # "auto" = choose maker when spread <= auto_maker_spread_threshold
    execution_mode: str = "taker"  # "taker", "maker", or "auto"

    # Auto mode: use maker when spread is tight enough
    auto_maker_spread_threshold: float = 0.02  # Use maker if spread <= 0.02%

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

    # Force maker path in paper mode (for testing maker telemetry)
    # Enable via: HFT_FORCE_MAKER_PAPER=1 environment variable
    force_maker_in_paper: bool = False

    # Computed totals (for reference):
    # TAKER: 2*0.10 + 2*0.02 + 0.02 = 0.26% per round-trip
    # MAKER: 2*0.01 + 2*0.005 + 0.0 = 0.03% per round-trip

    # === ECONOMIC GUARDRAIL ===
    # Prevents exits where profit doesn't cover costs
    # required_profit = max(min_profit_floor_pct, cost * min_profit_multiple)
    min_profit_multiple_of_cost: float = 2.0  # Exit profit must be 2x the expected cost
    min_profit_floor_pct: float = 0.20  # Absolute minimum profit % for any exit

    def get_required_profit_pct(self, execution_mode: str = None) -> float:
        """
        Calculate minimum profit % required for a profitable exit.

        Args:
            execution_mode: "maker" or "taker" (defaults to current execution_mode)

        Returns:
            Minimum profit percentage required to exit
        """
        mode = execution_mode or self.execution_mode
        cost = self.get_total_round_trip_cost(mode)
        return max(self.min_profit_floor_pct, cost * self.min_profit_multiple_of_cost)

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

    def get_effective_mode(self, current_spread_pct: float = None) -> str:
        """
        Get the effective execution mode based on config and market conditions.

        Args:
            current_spread_pct: Current bid-ask spread as percentage (for "auto" mode)

        Returns:
            "maker" or "taker"
        """
        if self.execution_mode == "auto":
            if current_spread_pct is not None and current_spread_pct <= self.auto_maker_spread_threshold:
                return "maker"
            return "taker"
        return self.execution_mode

    def get_total_round_trip_cost(self, mode: str = None) -> float:
        """
        Get total round-trip cost for a given execution mode.

        Args:
            mode: "maker" or "taker" (defaults to current execution_mode)

        Returns:
            Total cost as percentage
        """
        mode = mode or self.execution_mode
        if mode == "maker":
            return (
                self.maker_entry_fee_pct +
                self.maker_exit_fee_pct +
                2 * self.maker_spread_cost_pct +
                self.maker_slippage_pct
            )
        else:
            return (
                self.taker_entry_fee_pct +
                self.taker_exit_fee_pct +
                2 * self.taker_spread_cost_pct +
                self.taker_slippage_pct
            )


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

    # Database schema guardrails
    # Auto-migrate adds missing columns on startup (safe - never removes data)
    # Set False in LIVE mode to fail fast instead of auto-fixing
    auto_migrate_db: bool = True  # True for PAPER, should be False for LIVE

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

    # EXPLORATION MODE: Extremely relaxed gates for data collection
    # Default True in paper mode - collects marginal trades for analysis
    # Should be False in live mode for strict filtering
    exploration_mode: bool = True  # Relax gates for data collection
    exploration_min_size_multiplier: float = 0.10  # 10% size for exploration trades

    # ============================================================
    # RELAXED MODE: Slightly relaxed settings for paper testing
    # ============================================================
    # Enable via: RELAXED_MODE=1 environment variable (paper only)
    # Purpose: Conservative relaxation of blockers to generate more trades
    # while still maintaining some quality filtering.
    # NEVER enable in LIVE mode - this is for paper testing only.
    relaxed_mode: bool = False  # Auto-enable if RELAXED_MODE=1 in paper mode

    # Relaxation amounts (applied as multipliers/additions to thresholds)
    relaxed_imbalance_reduction: float = 0.10  # Reduce min_imbalance by 10%
    relaxed_spread_increase: float = 1.25     # Allow 25% wider spreads
    relaxed_allow_medium_tier: bool = True    # Allow MEDIUM tier in contraction mode
    relaxed_min_depth_reduction: float = 0.50 # Reduce min_depth requirement by 50%

    # ============================================================
    # PROBE MODE: Ultra-relaxed paper mode for data collection
    # ============================================================
    # Enable via: HFT_PROBE_MODE=1 environment variable
    # Purpose: Generate enough trades to validate maker path, telemetry, etc.
    # NEVER enable in LIVE mode - this is for paper testing only
    probe_mode: bool = False  # Auto-enable if HFT_PROBE_MODE=1 in paper mode
    probe_size_multiplier: float = 0.05  # 5% of normal size for probe trades
    probe_min_conditions: int = 2  # Override min_conditions in probe mode
    probe_allow_low_vol_chop: bool = True  # Allow low_vol_chop for probe trades

    # PROBE relaxations: convert hard blocks to probe-allowed
    probe_relax_ob_unstable: bool = True  # Allow ob_unstable unless extreme (>0.95 or <0.05)
    probe_relax_delta_noise: bool = True  # Allow delta_noise unless variance > 1.5
    probe_ob_extreme_threshold: float = 0.95  # Still block if bid_ratio > this or < (1 - this)
    probe_delta_variance_max: float = 1.5  # Still block if delta variance > this

    # Force maker path in paper mode (for testing maker telemetry)
    force_maker_in_paper: bool = False  # Enable via HFT_FORCE_MAKER_PAPER=1

    # === POCKET A: PRIMARY (high_vol_trend - the proven edge) ===
    pocket_a_enabled: bool = True
    pocket_a_regimes: List[str] = field(default_factory=lambda: ["high_vol_trend"])
    pocket_a_tiers: List[str] = field(default_factory=lambda: ["high", "medium"])
    pocket_a_min_imbalance: float = 0.70
    pocket_a_max_spread_pct: float = 0.05

    # === POCKET B: SECONDARY (low_vol_chop with extra confirmation) ===
    # Research-friendly - lowered imbalance to allow data collection
    pocket_b_enabled: bool = True  # Enable secondary pocket
    pocket_b_regimes: List[str] = field(default_factory=lambda: ["low_vol_chop"])
    pocket_b_tiers: List[str] = field(default_factory=lambda: ["high", "medium"])  # Allow medium for research
    pocket_b_min_imbalance: float = 0.58  # Lowered from 0.70 to allow trades (was blocking at 0.61)
    pocket_b_max_spread_pct: float = 0.03  # Slightly relaxed for research (was 0.02)
    pocket_b_min_depth: float = 10000.0  # Minimum OB depth (USD)
    pocket_b_flow_confirm_sec: float = 1.0  # Faster confirm for research (was 2.0)

    # === BLOCKED REGIMES (never trade) ===
    blocked_regimes: List[str] = field(default_factory=lambda: ["mean_reversion", "liquidity_vacuum", "news_spike", "unknown"])

    # === CAUSE POLICY (3-class system to fix trade starvation) ===
    # FULL: Known profitable causes - trade with normal sizing
    # PROBE: Unknown/testing causes - trade with tiny size (0.10x) for evidence collection
    # BLOCK: Everything else
    causality_filter_enabled: bool = True

    # FULL-SIZE CAUSES (evidence: positive expectancy)
    cause_full_allow_long: List[str] = field(default_factory=lambda: [
        "cvd_buy_pressure",      # 16.7% WR in audit - best performer
        "price_momentum_up"      # Price moving up
    ])
    cause_full_allow_short: List[str] = field(default_factory=lambda: [
        "cvd_sell_pressure",     # Strong signal
        "price_momentum_down"    # Price moving down
    ])

    # PROBE-SIZE CAUSES (collecting evidence - 0% WR but need more data)
    # AUDIT 2026-01-22: ob_bullish_imbalance moved to PROBE for data collection
    cause_probe_allow_long: List[str] = field(default_factory=lambda: [
        "ob_bullish_imbalance"   # Was 0% WR with 13 trades - need 30+ for decision
    ])
    cause_probe_allow_short: List[str] = field(default_factory=lambda: [
        "ob_bearish_imbalance"   # Orderbook bearish - needs evidence
    ])

    # PROBE CONFIGURATION
    probe_cause_size_multiplier: float = 0.10  # 10% of normal size (tiny risk)
    probe_requires_pocket_b: bool = True       # Only probe in Pocket B (research)
    probe_requires_maker: bool = True          # Only probe with maker mode (lower costs)

    block_unknown_cause: bool = True  # Block trades with no clear cause

    # PROBE_MODE expanded causes (when probe_mode=True, allow these for data collection)
    probe_mode_allow_causes: List[str] = field(default_factory=lambda: [
        "cvd_buy_pressure", "cvd_sell_pressure",
        "ob_bullish_imbalance", "ob_bearish_imbalance",
        "price_momentum_up", "price_momentum_down",
        "unknown"  # Allow unknown causes in probe mode for data collection
    ])

    # === LEGACY SETTINGS (for backward compatibility) ===
    strict_mode: bool = True  # If True, only Pocket A in live mode
    min_confidence_tier: str = "high"
    min_imbalance: float = 0.75
    max_spread_pct: float = 0.05
    research_allowed_tiers: List[str] = field(default_factory=lambda: ["high", "medium"])
    research_min_imbalance: float = 0.70

    # === PROBE PROTECTIONS (research mode - higher limits) ===
    max_trades_per_symbol_per_hour: int = 6  # Default for LIVE mode (strict)
    paper_max_trades_per_hour: int = 20  # Paper/probe mode (relaxed via PAPER_MAX_TRADES_PER_HOUR env)
    min_seconds_between_trades_per_symbol: int = 120  # 2 minutes
    paper_min_seconds_between_trades: int = 30  # Paper mode: faster cooldown (30s)

    # === PROBE MODE: LOW tier with tiny size ===
    probe_allow_low_tier: bool = True  # Allow LOW tier in PROBE_MODE only
    probe_low_tier_size_mult: float = 0.02  # 0.02x size for LOW tier (tiny)

    # === GRADED BLOCKERS: Size penalties instead of hard blocks ===
    probe_graded_ob_unstable: bool = True  # Apply size penalty instead of block
    probe_graded_delta_noise: bool = True  # Apply size penalty instead of block
    probe_ob_unstable_size_mult: float = 0.5  # 50% size when ob_unstable
    probe_delta_noise_size_mult: float = 0.5  # 50% size when delta_noise


WINNER_GATE = WinnerGateConfig()


# ============================================================
# RUNTIME ALARMS — Monitoring and Safety Controls
# ============================================================
# Automatic monitoring during trading to prevent runaway losses.
# Safe-by-default: alarms are enabled in LIVE mode.
# ============================================================
@dataclass
class RuntimeAlarmsConfig:
    """Runtime monitoring and safety alarms."""

    # Master switch
    enabled: bool = True

    # === LOSS STREAK ALARM ===
    # Pause trading if last N trades have negative PnL
    loss_streak_enabled: bool = True
    loss_streak_window: int = 50  # Check last 50 trades
    loss_streak_threshold_pct: float = -0.50  # Pause if cumulative PnL < -0.50%
    loss_streak_pause_minutes: int = 30  # Pause trading for 30 minutes

    # === SPREAD INSTABILITY ALARM ===
    # Tighten filters if spread_unstable blocks spike
    spread_alarm_enabled: bool = True
    spread_alarm_block_threshold: int = 10  # Alert after 10 spread blocks in window
    spread_alarm_window_minutes: int = 5  # 5-minute window
    spread_alarm_tighten_factor: float = 0.8  # Reduce max_spread by 20%

    # === DRAWDOWN ALARM ===
    # Pause trading if drawdown exceeds threshold
    drawdown_alarm_enabled: bool = True
    drawdown_alarm_threshold_pct: float = 3.0  # Pause if session drawdown > 3%
    drawdown_alarm_pause_minutes: int = 60  # Pause for 1 hour

    # === WIN RATE ALARM ===
    # Alert if win rate drops too low
    winrate_alarm_enabled: bool = True
    winrate_alarm_window: int = 20  # Check last 20 trades
    winrate_alarm_threshold_pct: float = 20.0  # Alert if WR < 20%

    # === LIVE MODE SAFETY ===
    # Extra safety for live trading
    live_mode_fail_closed: bool = True  # Stop trading on any alarm in LIVE mode
    paper_mode_log_only: bool = True  # In PAPER mode, just log warnings (don't pause)


RUNTIME_ALARMS = RuntimeAlarmsConfig()


# Global configs
RISK_CONFIG = RiskConfig()
SYSTEM_CONFIG = SystemConfig()


def get_asset_config(symbol: str) -> AssetConfig:
    """Get configuration for a specific asset."""
    return ASSETS.get(symbol.upper(), ASSETS["ATOM"])
