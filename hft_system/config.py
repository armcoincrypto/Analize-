"""
HFT System Configuration
========================
All thresholds and parameters for the trading system.

Assets: ATOM, SUI, XRP
Strategy: Orderbook + RSI focused (based on 70K signal analysis)
Target: +1.5% TP, -0.8% SL, 180s time stop

Data-driven optimizations (Dec 2024):
- Best combo: Orderbook + RSI (+0.024% at 5m with 4,134 signals)
- Extended time_stop from 90s to 180s (3min)
- Tighter TP/SL ratio for better risk management
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

    # Exit settings (optimized based on 70K signal data)
    take_profit_pct: float = 1.5          # Reduced from 2% - more achievable
    stop_loss_pct: float = 0.8            # Reduced from 1% - tighter risk
    time_stop_seconds: int = 180          # Extended from 90s to 3min
    min_profit_for_time_check: float = 0.3  # Must move 0.3% or exit
    use_trailing_stop: bool = True        # Enable trailing stop
    trailing_stop_activation: float = 0.5  # Activate trailing after 0.5% profit
    trailing_stop_distance: float = 0.3    # Trail by 0.3%

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

    # Daily limits
    max_daily_losses: int = 3
    max_daily_drawdown_pct: float = 3.0

    # Trade limits
    min_time_between_trades_sec: int = 30

    # Market filters
    btc_max_volatility_5m: float = 4.0  # Disable if BTC moves >4% in 5min
    max_exchange_latency_ms: int = 500  # Disable if latency >500ms


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

    # Assets to trade
    enabled_assets: List[str] = field(default_factory=lambda: ["ATOM", "SUI", "XRP"])

    # Entry conditions required (2/5 for data collection, 3/5 for production)
    min_entry_conditions: int = 2


# Global configs
RISK_CONFIG = RiskConfig()
SYSTEM_CONFIG = SystemConfig()


def get_asset_config(symbol: str) -> AssetConfig:
    """Get configuration for a specific asset."""
    return ASSETS.get(symbol.upper(), ASSETS["ATOM"])
