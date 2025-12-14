#!/usr/bin/env python3
"""
Production Trading Configuration

EVIDENCE-BASED RULES - DO NOT MODIFY WITHOUT NEW DATA
=======================================================
UPDATED: Based on Signal Optimizer results (2025-12)

Source: signal_optimizer.py + historical_replay.py
Total Trades Analyzed: 255 (30-day 15m backtest)

OPTIMIZED STRATEGY:
- Timeframe: 15m (NOT 1h - much higher frequency)
- Signal: BB only (79.6% win rate, 5.21 PF)
- Stop Loss: 1.5%
- Take Profit: 2.0%
- Best Asset: ATOMUSDT (only profitable symbol)
- Est. Annual Trades: 3,102
- Est. Annual P&L: +82.5% (backtest - expect 30-50% realistic)

CRITICAL: This is BACKTEST data. Real performance will be lower.
Paper trade for 30+ days before trusting these numbers.

This config enforces ONLY proven profitable conditions.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set
from enum import Enum
import datetime


class TradingMode(Enum):
    """Trading mode determines position sizing."""
    DISABLED = "disabled"       # No trading allowed
    SMALL_SIZE = "small_size"   # 25% of normal size (low confidence)
    NORMAL = "normal"           # 100% position size
    FULL_SIZE = "full_size"     # Can use full allocation


@dataclass
class SymbolConfig:
    """Configuration for a single trading symbol."""
    symbol: str
    enabled: bool
    win_rate: float           # From evidence data
    total_pnl_percent: float  # From evidence data
    trade_count: int          # Sample size
    weight: float             # Position weight (0.0 - 1.0)
    notes: str = ""


@dataclass
class ProductionConfig:
    """
    Production Trading Configuration

    ALL VALUES ARE EVIDENCE-BASED - DO NOT CHANGE WITHOUT NEW DATA
    """

    # =========================================================================
    # SYMBOL CONFIGURATION (Evidence-Based)
    # =========================================================================

    # Only trade symbols with proven edge
    ALLOWED_SYMBOLS: Dict[str, SymbolConfig] = field(default_factory=lambda: {
        "ATOMUSDT": SymbolConfig(
            symbol="ATOMUSDT",
            enabled=True,
            win_rate=84.1,
            total_pnl_percent=308.78,
            trade_count=126,
            weight=1.0,  # Primary focus - best performer
            notes="BEST PERFORMER - Primary trading symbol"
        ),
        "SOLUSDT": SymbolConfig(
            symbol="SOLUSDT",
            enabled=False,  # DISABLED - optimizer found NO profitable configs
            win_rate=45.0,
            total_pnl_percent=0.10,
            trade_count=20,
            weight=0.0,
            notes="DISABLED: 45% win rate - no edge found in optimization"
        ),
        "XRPUSDT": SymbolConfig(
            symbol="XRPUSDT",
            enabled=False,  # Disabled by default - borderline edge
            win_rate=64.5,
            total_pnl_percent=23.00,
            trade_count=93,
            weight=0.25,
            notes="CAUTION: Borderline edge, enable only if needed"
        ),
    })

    # Primary symbol (use this by default)
    PRIMARY_SYMBOL: str = "ATOMUSDT"

    # =========================================================================
    # SIGNAL CONFIGURATION (Evidence-Based)
    # =========================================================================

    # Signal combinations ranked by OPTIMIZER results (2025-12)
    # BB alone on 15m = 79.6% win rate, 5.21 PF, 255 trades/month
    SIGNAL_PRIORITY: List[Dict] = field(default_factory=lambda: [
        {"signals": ["BB"], "win_rate": 79.6, "enabled": True, "min_confidence": 70},  # BEST - high frequency
        {"signals": ["BB", "MACD"], "win_rate": 84.0, "enabled": True, "min_confidence": 75},  # Best edge, fewer trades
        {"signals": ["BB", "RSI"], "win_rate": 75.6, "enabled": True, "min_confidence": 70},
        {"signals": ["RSI"], "win_rate": 74.3, "enabled": False, "min_confidence": 65},  # High frequency but lower edge
        {"signals": ["MACD"], "win_rate": 70.0, "enabled": False, "min_confidence": 65},
    ])

    # Minimum signals required for entry
    MIN_SIGNAL_COUNT: int = 1  # BB alone is sufficient on 15m

    # Required signal for all trades (BB is the core signal now)
    REQUIRED_SIGNAL: str = "BB"  # Bollinger Bands - mean reversion

    # =========================================================================
    # TIME CONFIGURATION (Evidence-Based)
    # =========================================================================

    # Trading hours in UTC (15:00-22:00 are best)
    ALLOWED_HOURS_UTC: List[int] = field(default_factory=lambda: [
        15, 16, 17, 18, 19, 20, 21, 22  # Best performing hours
    ])

    # Extended hours (acceptable but not optimal)
    EXTENDED_HOURS_UTC: List[int] = field(default_factory=lambda: [
        12, 13, 14, 23  # Acceptable with reduced size (removed 00:00 - blocked)
    ])

    # BLOCKED hours - never trade (optimizer found 0% win rate at 00:00)
    BLOCKED_HOURS_UTC: List[int] = field(default_factory=lambda: [
        0,  # 0% win rate in optimizer test
        3, 4, 5, 6  # Statistically worst hours
    ])

    # Day of week configuration (0=Monday, 6=Sunday)
    # Updated from optimizer results
    ALLOWED_DAYS: List[int] = field(default_factory=lambda: [
        0,  # Monday - 100% win rate in optimizer
        1,  # Tuesday - 100% win rate
        4,  # Friday - 75% win rate (good)
        6,  # Sunday - 57-67% win rate (acceptable)
    ])

    # BLOCKED days - never trade
    BLOCKED_DAYS: List[int] = field(default_factory=lambda: [
        2,  # Wednesday - 38-43% win rate (WORST in optimizer)
        3,  # Thursday - 40% win rate (also bad)
    ])

    # Saturday is neutral (65.2%) - trade with caution
    CAUTION_DAYS: List[int] = field(default_factory=lambda: [
        5,  # Saturday - 65.2% (use small size)
    ])

    # =========================================================================
    # REGIME CONFIGURATION (Evidence-Based)
    # =========================================================================

    # All regimes are profitable, but with different edges
    REGIME_CONFIG: Dict[str, Dict] = field(default_factory=lambda: {
        "bullish": {"enabled": True, "win_rate": 85.7, "weight": 1.0},
        "sideways": {"enabled": True, "win_rate": 73.7, "weight": 1.0},
        "bearish": {"enabled": True, "win_rate": 66.7, "weight": 0.75},  # Slightly reduced
    })

    # =========================================================================
    # RISK MANAGEMENT (Expert Recommended)
    # =========================================================================

    # Position Sizing
    BASE_POSITION_SIZE_PERCENT: float = 2.0  # 2% of capital per trade (conservative)
    MAX_POSITION_SIZE_PERCENT: float = 5.0   # Never exceed 5%

    # Confidence-Based Sizing
    CONFIDENCE_SIZING: Dict[str, Dict] = field(default_factory=lambda: {
        "no_trade": {"min": 0, "max": 60, "size_multiplier": 0.0},    # NO TRADE
        "small": {"min": 60, "max": 75, "size_multiplier": 0.5},      # Half size
        "normal": {"min": 75, "max": 85, "size_multiplier": 1.0},     # Full size
        "high": {"min": 85, "max": 100, "size_multiplier": 1.0},      # Full size (no increase)
    })

    # Session Risk Limits (per trading session)
    MAX_SESSION_LOSS_PERCENT: float = 5.0    # Stop trading if session loss > 5%
    MAX_CONSECUTIVE_LOSSES: int = 3          # Stop trading after 3 consecutive losses
    SESSION_LOSS_COOLDOWN_HOURS: int = 4     # Wait 4 hours after hitting loss limit

    # Daily Risk Limits
    MAX_DAILY_LOSS_PERCENT: float = 8.0      # Stop for the day if loss > 8%
    MAX_DAILY_TRADES: int = 10               # Maximum trades per day

    # Drawdown Protection
    MAX_DRAWDOWN_PERCENT: float = 15.0       # Kill switch at 15% drawdown
    DRAWDOWN_WARNING_PERCENT: float = 10.0   # Warning at 10% drawdown

    # Tail Risk Protection (addresses expert concern about -10% to -15% losses)
    MAX_SINGLE_TRADE_LOSS_PERCENT: float = 2.0  # Hard stop at 2% loss per trade
    STOP_LOSS_PERCENT: float = 1.5              # OPTIMIZED: 1.5% stop (from optimizer)

    # =========================================================================
    # VOLATILITY FILTER (Expert Recommended)
    # =========================================================================

    # ATR-based volatility filter
    VOLATILITY_FILTER_ENABLED: bool = True
    MIN_ATR_PERCENT: float = 0.5    # Minimum ATR for trade (avoid dead markets)
    MAX_ATR_PERCENT: float = 5.0    # Maximum ATR (avoid extreme volatility)

    # Reduce size in high volatility
    HIGH_VOLATILITY_THRESHOLD: float = 3.0  # ATR > 3% = high volatility
    HIGH_VOLATILITY_SIZE_MULTIPLIER: float = 0.5  # Half size in high vol

    # =========================================================================
    # TRADE EXECUTION
    # =========================================================================

    # Minimum time between trades (prevent overtrading)
    MIN_TRADE_INTERVAL_MINUTES: int = 30

    # Order settings
    USE_LIMIT_ORDERS: bool = True
    LIMIT_ORDER_OFFSET_PERCENT: float = 0.1  # Place limits 0.1% from market
    LIMIT_ORDER_TIMEOUT_SECONDS: int = 60    # Cancel if not filled in 60s

    # Take profit settings - OPTIMIZED from signal_optimizer.py
    TAKE_PROFIT_PERCENT: float = 2.0         # OPTIMIZED: 2.0% target (from optimizer)
    TRAILING_STOP_ENABLED: bool = False      # Disabled - fixed target works better
    TRAILING_STOP_ACTIVATION_PERCENT: float = 1.5  # Not used
    TRAILING_STOP_DISTANCE_PERCENT: float = 0.75   # Not used


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def is_trading_allowed(config: ProductionConfig = None) -> Dict[str, any]:
    """
    Check if trading is currently allowed based on all rules.

    Returns:
        Dict with 'allowed' (bool), 'reason' (str), 'warnings' (list)
    """
    if config is None:
        config = ProductionConfig()

    now = datetime.datetime.utcnow()
    hour = now.hour
    day = now.weekday()

    result = {
        "allowed": True,
        "reason": "All conditions met",
        "warnings": [],
        "size_multiplier": 1.0
    }

    # Check blocked hours
    if hour in config.BLOCKED_HOURS_UTC:
        result["allowed"] = False
        result["reason"] = f"BLOCKED HOUR: {hour}:00 UTC (worst performing hours)"
        return result

    # Check blocked days
    if day in config.BLOCKED_DAYS:
        day_name = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][day]
        result["allowed"] = False
        result["reason"] = f"BLOCKED DAY: {day_name} (51.2% win rate - near coin flip)"
        return result

    # Check if in optimal hours
    if hour not in config.ALLOWED_HOURS_UTC:
        if hour in config.EXTENDED_HOURS_UTC:
            result["warnings"].append(f"Extended hour {hour}:00 UTC - reduced size recommended")
            result["size_multiplier"] *= 0.75
        else:
            result["warnings"].append(f"Sub-optimal hour {hour}:00 UTC - consider waiting")
            result["size_multiplier"] *= 0.5

    # Check if caution day
    if day in config.CAUTION_DAYS:
        day_name = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][day]
        result["warnings"].append(f"Caution day: {day_name} (65.2% win rate) - reduced size")
        result["size_multiplier"] *= 0.75

    # Check if optimal day
    if day in config.ALLOWED_DAYS and day in [0, 1]:  # Monday or Tuesday
        result["warnings"].append(f"Optimal day for trading")

    return result


def validate_signal(signals: List[str], config: ProductionConfig = None) -> Dict[str, any]:
    """
    Validate if a signal combination should be traded.

    Args:
        signals: List of signal names (e.g., ["BB", "MACD"])
        config: Production config

    Returns:
        Dict with 'valid' (bool), 'reason' (str), 'confidence' (float)
    """
    if config is None:
        config = ProductionConfig()

    result = {
        "valid": False,
        "reason": "",
        "confidence": 0.0,
        "size_multiplier": 1.0
    }

    # Check minimum signal count
    if len(signals) < config.MIN_SIGNAL_COUNT:
        result["reason"] = f"Insufficient signals: {len(signals)} < {config.MIN_SIGNAL_COUNT}"
        return result

    # Check required signal
    if config.REQUIRED_SIGNAL not in signals:
        result["reason"] = f"Missing required signal: {config.REQUIRED_SIGNAL}"
        return result

    # Find matching signal priority
    signals_set = set(s.upper() for s in signals)

    for priority in config.SIGNAL_PRIORITY:
        priority_set = set(s.upper() for s in priority["signals"])
        if priority_set.issubset(signals_set) and priority["enabled"]:
            result["valid"] = True
            result["confidence"] = priority["win_rate"]
            result["reason"] = f"Valid signal: {'+'.join(priority['signals'])} ({priority['win_rate']}% win rate)"

            # Apply confidence-based sizing
            for level, bounds in config.CONFIDENCE_SIZING.items():
                if bounds["min"] <= priority["win_rate"] < bounds["max"]:
                    result["size_multiplier"] = bounds["size_multiplier"]
                    if result["size_multiplier"] == 0:
                        result["valid"] = False
                        result["reason"] = f"Confidence too low: {priority['win_rate']}% (minimum 60%)"
                    break

            return result

    result["reason"] = f"Signal combination not in approved list: {signals}"
    return result


def validate_symbol(symbol: str, config: ProductionConfig = None) -> Dict[str, any]:
    """
    Validate if a symbol should be traded.

    Args:
        symbol: Trading symbol (e.g., "ATOMUSDT")
        config: Production config

    Returns:
        Dict with 'valid' (bool), 'reason' (str), 'weight' (float)
    """
    if config is None:
        config = ProductionConfig()

    result = {
        "valid": False,
        "reason": "",
        "weight": 0.0,
        "win_rate": 0.0
    }

    symbol_upper = symbol.upper()

    if symbol_upper not in config.ALLOWED_SYMBOLS:
        result["reason"] = f"Symbol {symbol} not in approved list"
        return result

    sym_config = config.ALLOWED_SYMBOLS[symbol_upper]

    if not sym_config.enabled:
        result["reason"] = f"Symbol {symbol} is disabled: {sym_config.notes}"
        return result

    result["valid"] = True
    result["weight"] = sym_config.weight
    result["win_rate"] = sym_config.win_rate
    result["reason"] = f"Approved: {sym_config.notes}"

    return result


def get_position_size(
    capital: float,
    confidence: float,
    volatility_atr_percent: float,
    config: ProductionConfig = None
) -> Dict[str, any]:
    """
    Calculate position size based on all factors.

    Args:
        capital: Current capital
        confidence: Signal confidence (0-100)
        volatility_atr_percent: Current ATR as percentage
        config: Production config

    Returns:
        Dict with 'size' (float), 'size_percent' (float), 'adjustments' (list)
    """
    if config is None:
        config = ProductionConfig()

    result = {
        "size": 0.0,
        "size_percent": 0.0,
        "adjustments": []
    }

    # Start with base size
    base_percent = config.BASE_POSITION_SIZE_PERCENT
    result["adjustments"].append(f"Base size: {base_percent}%")

    # Apply confidence multiplier
    for level, bounds in config.CONFIDENCE_SIZING.items():
        if bounds["min"] <= confidence < bounds["max"]:
            multiplier = bounds["size_multiplier"]
            base_percent *= multiplier
            result["adjustments"].append(f"Confidence ({confidence:.1f}%): x{multiplier}")
            break

    # Apply volatility filter
    if config.VOLATILITY_FILTER_ENABLED:
        if volatility_atr_percent < config.MIN_ATR_PERCENT:
            result["adjustments"].append(f"BLOCKED: ATR {volatility_atr_percent:.2f}% < min {config.MIN_ATR_PERCENT}%")
            return result

        if volatility_atr_percent > config.MAX_ATR_PERCENT:
            result["adjustments"].append(f"BLOCKED: ATR {volatility_atr_percent:.2f}% > max {config.MAX_ATR_PERCENT}%")
            return result

        if volatility_atr_percent > config.HIGH_VOLATILITY_THRESHOLD:
            base_percent *= config.HIGH_VOLATILITY_SIZE_MULTIPLIER
            result["adjustments"].append(f"High volatility ({volatility_atr_percent:.2f}%): x{config.HIGH_VOLATILITY_SIZE_MULTIPLIER}")

    # Apply time-based multiplier
    time_check = is_trading_allowed(config)
    if not time_check["allowed"]:
        result["adjustments"].append(f"BLOCKED: {time_check['reason']}")
        return result

    base_percent *= time_check["size_multiplier"]
    if time_check["size_multiplier"] < 1.0:
        result["adjustments"].append(f"Time adjustment: x{time_check['size_multiplier']}")

    # Cap at maximum
    if base_percent > config.MAX_POSITION_SIZE_PERCENT:
        base_percent = config.MAX_POSITION_SIZE_PERCENT
        result["adjustments"].append(f"Capped at max: {config.MAX_POSITION_SIZE_PERCENT}%")

    result["size_percent"] = base_percent
    result["size"] = capital * (base_percent / 100)

    return result


# =============================================================================
# REPORTING
# =============================================================================

def print_config_summary(config: ProductionConfig = None):
    """Print a summary of the production configuration."""
    if config is None:
        config = ProductionConfig()

    print("=" * 70)
    print("PRODUCTION TRADING CONFIGURATION")
    print("=" * 70)

    print("\n[SYMBOLS]")
    print("-" * 40)
    for sym, cfg in config.ALLOWED_SYMBOLS.items():
        status = "ENABLED" if cfg.enabled else "DISABLED"
        print(f"  {sym}: {status} | Win: {cfg.win_rate}% | Weight: {cfg.weight} | {cfg.notes}")
    print(f"  Primary: {config.PRIMARY_SYMBOL}")

    print("\n[SIGNALS]")
    print("-" * 40)
    for sig in config.SIGNAL_PRIORITY:
        status = "ON" if sig["enabled"] else "OFF"
        print(f"  {'+'.join(sig['signals'])}: {status} | Win: {sig['win_rate']}%")
    print(f"  Required signal: {config.REQUIRED_SIGNAL}")
    print(f"  Minimum signals: {config.MIN_SIGNAL_COUNT}")

    print("\n[TRADING HOURS (UTC)]")
    print("-" * 40)
    print(f"  Optimal: {config.ALLOWED_HOURS_UTC}")
    print(f"  Extended: {config.EXTENDED_HOURS_UTC}")
    print(f"  BLOCKED: {config.BLOCKED_HOURS_UTC}")

    print("\n[TRADING DAYS]")
    print("-" * 40)
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    allowed = [days[d] for d in config.ALLOWED_DAYS]
    blocked = [days[d] for d in config.BLOCKED_DAYS]
    caution = [days[d] for d in config.CAUTION_DAYS]
    print(f"  Allowed: {allowed}")
    print(f"  BLOCKED: {blocked}")
    print(f"  Caution: {caution}")

    print("\n[RISK LIMITS]")
    print("-" * 40)
    print(f"  Base position size: {config.BASE_POSITION_SIZE_PERCENT}%")
    print(f"  Max position size: {config.MAX_POSITION_SIZE_PERCENT}%")
    print(f"  Stop loss: {config.STOP_LOSS_PERCENT}%")
    print(f"  Max single trade loss: {config.MAX_SINGLE_TRADE_LOSS_PERCENT}%")
    print(f"  Max session loss: {config.MAX_SESSION_LOSS_PERCENT}%")
    print(f"  Max daily loss: {config.MAX_DAILY_LOSS_PERCENT}%")
    print(f"  Max drawdown (KILL SWITCH): {config.MAX_DRAWDOWN_PERCENT}%")
    print(f"  Max consecutive losses: {config.MAX_CONSECUTIVE_LOSSES}")

    print("\n[CURRENT STATUS]")
    print("-" * 40)
    status = is_trading_allowed(config)
    if status["allowed"]:
        print(f"  Trading: ALLOWED")
    else:
        print(f"  Trading: BLOCKED - {status['reason']}")
    for warn in status["warnings"]:
        print(f"  Warning: {warn}")
    print(f"  Size multiplier: {status['size_multiplier']:.2f}x")

    print("\n" + "=" * 70)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    config = ProductionConfig()
    print_config_summary(config)

    # Test some validations
    print("\n[VALIDATION TESTS]")
    print("-" * 40)

    # Test signal validation
    test_signals = [
        ["BB", "MACD"],
        ["MACD"],
        ["RSI"],
        ["BB"],
    ]

    for signals in test_signals:
        result = validate_signal(signals, config)
        status = "VALID" if result["valid"] else "INVALID"
        print(f"  Signals {signals}: {status} - {result['reason']}")

    # Test symbol validation
    test_symbols = ["ATOMUSDT", "XRPUSDT", "DOGEUSDT"]

    print()
    for symbol in test_symbols:
        result = validate_symbol(symbol, config)
        status = "VALID" if result["valid"] else "INVALID"
        print(f"  Symbol {symbol}: {status} - {result['reason']}")

    # Test position sizing
    print()
    size_result = get_position_size(
        capital=10000,
        confidence=80.0,
        volatility_atr_percent=2.0,
        config=config
    )
    print(f"  Position size ($10k capital, 80% confidence, 2% ATR):")
    print(f"    Size: ${size_result['size']:.2f} ({size_result['size_percent']:.2f}%)")
    for adj in size_result["adjustments"]:
        print(f"    - {adj}")
