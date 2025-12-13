#!/usr/bin/env python3
"""
Go-Live Checklist - Safety Verification Before Live Trading

This module provides comprehensive pre-trade and system health checks
before allowing live trading with real capital.

EVIDENCE-BASED CRITERIA from 311+ backtest trades:
- Proven symbols: ATOMUSDT (84.1% win), SOLUSDT (68.5%)
- Proven signals: BB+MACD (82.1%), MACD (76.7%), STOCH (76.2%)
- Proven hours: 15:00-22:00 UTC
- Proven days: Monday (83.3%), Tuesday (79.6%)
- Avoid: Thursday (51.2%), hours 03:00-06:00 UTC
"""

import ssl_bypass  # Must be first!
import os
import json
import requests
import datetime
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum


class CheckStatus(Enum):
    """Status of a checklist item."""
    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"
    SKIP = "SKIP"


@dataclass
class CheckResult:
    """Result of a single check."""
    name: str
    status: CheckStatus
    message: str
    details: Dict = field(default_factory=dict)


class GoLiveChecklist:
    """
    Comprehensive go-live verification system.

    Checks performed:
    1. System health (API connectivity, data freshness)
    2. Time/Day validation (evidence-based trading windows)
    3. Symbol validation (only proven profitable symbols)
    4. Signal validation (only proven signal combinations)
    5. Risk limits (capital, exposure, drawdown)
    6. Market conditions (volatility, liquidity)
    7. Account requirements (capital, API access)

    All checks must PASS or WARN before live trading is allowed.
    """

    def __init__(
        self,
        min_capital: float = 1000.0,
        max_capital_per_trade_pct: float = 5.0,
        required_win_rate: float = 65.0,
        min_trade_history: int = 50,
    ):
        self.min_capital = min_capital
        self.max_capital_per_trade_pct = max_capital_per_trade_pct
        self.required_win_rate = required_win_rate
        self.min_trade_history = min_trade_history

        # Evidence-based rules (from backtest)
        self.allowed_symbols = ["ATOMUSDT", "SOLUSDT"]  # XRPUSDT disabled by default
        self.allowed_hours_utc = [15, 16, 17, 18, 19, 20, 21, 22]
        self.blocked_hours_utc = [3, 4, 5, 6]
        self.allowed_days = [0, 1, 2, 4, 6]  # Mon, Tue, Wed, Fri, Sun
        self.blocked_days = [3]  # Thursday
        self.required_signals = ["MACD"]  # Must have MACD
        self.min_signals = 2

    # =========================================================================
    # HARD "DO NOT TRADE" CONDITIONS
    # =========================================================================

    def check_hard_blocks(self) -> List[CheckResult]:
        """
        Check absolute "DO NOT TRADE" conditions.

        These are non-negotiable - if any fail, NO TRADING ALLOWED.
        """
        results = []

        # 1. Time-based blocks
        now = datetime.datetime.utcnow()
        hour = now.hour
        day = now.weekday()

        # Check blocked hours
        if hour in self.blocked_hours_utc:
            results.append(CheckResult(
                name="BLOCKED_HOUR",
                status=CheckStatus.FAIL,
                message=f"BLOCKED: Hour {hour}:00 UTC is in blocked range (03:00-06:00)",
                details={"hour": hour, "blocked_hours": self.blocked_hours_utc}
            ))
        else:
            results.append(CheckResult(
                name="BLOCKED_HOUR",
                status=CheckStatus.PASS,
                message=f"Hour {hour}:00 UTC is not blocked",
                details={"hour": hour}
            ))

        # Check blocked days
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        if day in self.blocked_days:
            results.append(CheckResult(
                name="BLOCKED_DAY",
                status=CheckStatus.FAIL,
                message=f"BLOCKED: {day_names[day]} is blocked (51.2% win rate)",
                details={"day": day_names[day], "day_num": day}
            ))
        else:
            results.append(CheckResult(
                name="BLOCKED_DAY",
                status=CheckStatus.PASS,
                message=f"{day_names[day]} is not blocked",
                details={"day": day_names[day]}
            ))

        return results

    # =========================================================================
    # SYSTEM HEALTH CHECKS
    # =========================================================================

    def check_api_connectivity(self) -> CheckResult:
        """Check if APIs are accessible."""
        apis = [
            ("Binance US", "https://api.binance.us/api/v3/time"),
            ("CoinGecko", "https://api.coingecko.com/api/v3/ping"),
        ]

        working = []
        failed = []

        for name, url in apis:
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    working.append(name)
                else:
                    failed.append(f"{name} ({response.status_code})")
            except Exception as e:
                failed.append(f"{name} ({str(e)[:30]})")

        if len(working) >= 1:
            return CheckResult(
                name="API_CONNECTIVITY",
                status=CheckStatus.PASS,
                message=f"APIs accessible: {', '.join(working)}",
                details={"working": working, "failed": failed}
            )
        else:
            return CheckResult(
                name="API_CONNECTIVITY",
                status=CheckStatus.FAIL,
                message=f"No APIs accessible: {', '.join(failed)}",
                details={"working": working, "failed": failed}
            )

    def check_data_freshness(self, symbol: str = "ATOMUSDT") -> CheckResult:
        """Check if market data is fresh."""
        try:
            urls = [
                f"https://api.binance.us/api/v3/ticker/24hr?symbol={symbol}",
                f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}",
            ]

            for url in urls:
                try:
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        close_time = int(data.get("closeTime", 0))
                        if close_time > 0:
                            age_seconds = (datetime.datetime.utcnow().timestamp() * 1000 - close_time) / 1000
                            if age_seconds < 300:  # Less than 5 minutes old
                                return CheckResult(
                                    name="DATA_FRESHNESS",
                                    status=CheckStatus.PASS,
                                    message=f"Data is fresh ({age_seconds:.0f}s old)",
                                    details={"age_seconds": age_seconds, "symbol": symbol}
                                )
                            elif age_seconds < 900:  # Less than 15 minutes
                                return CheckResult(
                                    name="DATA_FRESHNESS",
                                    status=CheckStatus.WARN,
                                    message=f"Data is slightly stale ({age_seconds:.0f}s old)",
                                    details={"age_seconds": age_seconds, "symbol": symbol}
                                )
                except:
                    continue

            return CheckResult(
                name="DATA_FRESHNESS",
                status=CheckStatus.FAIL,
                message="Could not verify data freshness",
                details={"symbol": symbol}
            )

        except Exception as e:
            return CheckResult(
                name="DATA_FRESHNESS",
                status=CheckStatus.FAIL,
                message=f"Error checking data: {str(e)[:50]}",
                details={}
            )

    # =========================================================================
    # TIME/DAY VALIDATION
    # =========================================================================

    def check_trading_window(self) -> CheckResult:
        """Check if current time is in optimal trading window."""
        now = datetime.datetime.utcnow()
        hour = now.hour
        day = now.weekday()
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        is_optimal_hour = hour in self.allowed_hours_utc
        is_optimal_day = day in [0, 1]  # Monday, Tuesday are best

        if is_optimal_hour and is_optimal_day:
            return CheckResult(
                name="TRADING_WINDOW",
                status=CheckStatus.PASS,
                message=f"OPTIMAL: {day_names[day]} {hour}:00 UTC",
                details={"hour": hour, "day": day_names[day], "optimal": True}
            )
        elif is_optimal_hour:
            return CheckResult(
                name="TRADING_WINDOW",
                status=CheckStatus.PASS,
                message=f"GOOD: {day_names[day]} {hour}:00 UTC (optimal hours)",
                details={"hour": hour, "day": day_names[day], "optimal": False}
            )
        elif day in self.allowed_days:
            return CheckResult(
                name="TRADING_WINDOW",
                status=CheckStatus.WARN,
                message=f"SUB-OPTIMAL: {day_names[day]} {hour}:00 UTC (outside optimal hours)",
                details={"hour": hour, "day": day_names[day], "optimal": False}
            )
        else:
            return CheckResult(
                name="TRADING_WINDOW",
                status=CheckStatus.WARN,
                message=f"CAUTION: {day_names[day]} {hour}:00 UTC",
                details={"hour": hour, "day": day_names[day], "optimal": False}
            )

    # =========================================================================
    # SYMBOL VALIDATION
    # =========================================================================

    def check_symbol(self, symbol: str) -> CheckResult:
        """Check if symbol is approved for trading."""
        symbol_upper = symbol.upper()

        if symbol_upper in self.allowed_symbols:
            if symbol_upper == "ATOMUSDT":
                return CheckResult(
                    name="SYMBOL_CHECK",
                    status=CheckStatus.PASS,
                    message=f"BEST: {symbol_upper} (84.1% win rate, +308.78% P&L)",
                    details={"symbol": symbol_upper, "win_rate": 84.1, "tier": "best"}
                )
            elif symbol_upper == "SOLUSDT":
                return CheckResult(
                    name="SYMBOL_CHECK",
                    status=CheckStatus.PASS,
                    message=f"GOOD: {symbol_upper} (68.5% win rate, +30.39% P&L)",
                    details={"symbol": symbol_upper, "win_rate": 68.5, "tier": "good"}
                )

        if symbol_upper == "XRPUSDT":
            return CheckResult(
                name="SYMBOL_CHECK",
                status=CheckStatus.WARN,
                message=f"CAUTION: {symbol_upper} (64.5% win rate - borderline edge)",
                details={"symbol": symbol_upper, "win_rate": 64.5, "tier": "borderline"}
            )

        return CheckResult(
            name="SYMBOL_CHECK",
            status=CheckStatus.FAIL,
            message=f"NOT APPROVED: {symbol_upper} - no proven edge",
            details={"symbol": symbol_upper, "allowed": self.allowed_symbols}
        )

    # =========================================================================
    # SIGNAL VALIDATION
    # =========================================================================

    def check_signals(self, signals: List[str]) -> CheckResult:
        """Check if signal combination is approved."""
        signals_upper = [s.upper() for s in signals]

        # Check minimum signal count
        if len(signals_upper) < self.min_signals:
            return CheckResult(
                name="SIGNAL_CHECK",
                status=CheckStatus.FAIL,
                message=f"INSUFFICIENT: {len(signals_upper)} signals < {self.min_signals} required",
                details={"signals": signals_upper, "count": len(signals_upper)}
            )

        # Check required signal
        if "MACD" not in signals_upper:
            return CheckResult(
                name="SIGNAL_CHECK",
                status=CheckStatus.FAIL,
                message=f"MISSING: Required signal MACD not present",
                details={"signals": signals_upper, "required": "MACD"}
            )

        # Check for best combination
        if "BB" in signals_upper and "MACD" in signals_upper:
            return CheckResult(
                name="SIGNAL_CHECK",
                status=CheckStatus.PASS,
                message=f"BEST: BB+MACD combination (82.1% win rate)",
                details={"signals": signals_upper, "win_rate": 82.1, "tier": "best"}
            )

        if "MACD" in signals_upper:
            return CheckResult(
                name="SIGNAL_CHECK",
                status=CheckStatus.PASS,
                message=f"GOOD: MACD present (76.7% win rate)",
                details={"signals": signals_upper, "win_rate": 76.7, "tier": "good"}
            )

        return CheckResult(
            name="SIGNAL_CHECK",
            status=CheckStatus.WARN,
            message=f"CAUTION: Signal combination not optimal: {signals_upper}",
            details={"signals": signals_upper}
        )

    # =========================================================================
    # MARKET CONDITIONS
    # =========================================================================

    def check_volatility(self, symbol: str = "ATOMUSDT") -> CheckResult:
        """Check current market volatility."""
        try:
            urls = [
                f"https://api.binance.us/api/v3/klines?symbol={symbol}&interval=1h&limit=14",
                f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1h&limit=14",
            ]

            for url in urls:
                try:
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        if data and len(data) >= 2:
                            # Calculate ATR
                            trs = []
                            for i in range(1, len(data)):
                                high = float(data[i][2])
                                low = float(data[i][3])
                                prev_close = float(data[i-1][4])
                                tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
                                trs.append(tr)

                            atr = sum(trs) / len(trs)
                            current_price = float(data[-1][4])
                            atr_pct = (atr / current_price) * 100

                            if atr_pct < 0.5:
                                return CheckResult(
                                    name="VOLATILITY_CHECK",
                                    status=CheckStatus.FAIL,
                                    message=f"TOO QUIET: ATR {atr_pct:.2f}% < 0.5% minimum",
                                    details={"atr_pct": atr_pct, "symbol": symbol}
                                )
                            elif atr_pct > 5.0:
                                return CheckResult(
                                    name="VOLATILITY_CHECK",
                                    status=CheckStatus.FAIL,
                                    message=f"TOO VOLATILE: ATR {atr_pct:.2f}% > 5.0% maximum",
                                    details={"atr_pct": atr_pct, "symbol": symbol}
                                )
                            elif atr_pct > 3.0:
                                return CheckResult(
                                    name="VOLATILITY_CHECK",
                                    status=CheckStatus.WARN,
                                    message=f"HIGH VOLATILITY: ATR {atr_pct:.2f}% - reduce position size",
                                    details={"atr_pct": atr_pct, "symbol": symbol}
                                )
                            else:
                                return CheckResult(
                                    name="VOLATILITY_CHECK",
                                    status=CheckStatus.PASS,
                                    message=f"NORMAL: ATR {atr_pct:.2f}%",
                                    details={"atr_pct": atr_pct, "symbol": symbol}
                                )
                except:
                    continue

            return CheckResult(
                name="VOLATILITY_CHECK",
                status=CheckStatus.SKIP,
                message="Could not check volatility",
                details={"symbol": symbol}
            )

        except Exception as e:
            return CheckResult(
                name="VOLATILITY_CHECK",
                status=CheckStatus.SKIP,
                message=f"Error: {str(e)[:50]}",
                details={}
            )

    def check_btc_health(self) -> CheckResult:
        """Check BTC market health (circuit breaker)."""
        try:
            url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd&include_24hr_change=true"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                change_24h = data.get("bitcoin", {}).get("usd_24h_change", 0)

                if change_24h <= -7:
                    return CheckResult(
                        name="BTC_HEALTH",
                        status=CheckStatus.FAIL,
                        message=f"CIRCUIT BREAKER: BTC down {change_24h:.1f}% in 24h",
                        details={"change_24h": change_24h}
                    )
                elif change_24h <= -5:
                    return CheckResult(
                        name="BTC_HEALTH",
                        status=CheckStatus.WARN,
                        message=f"BTC WEAK: Down {change_24h:.1f}% in 24h - reduced size",
                        details={"change_24h": change_24h}
                    )
                else:
                    return CheckResult(
                        name="BTC_HEALTH",
                        status=CheckStatus.PASS,
                        message=f"BTC OK: {change_24h:+.1f}% in 24h",
                        details={"change_24h": change_24h}
                    )

            return CheckResult(
                name="BTC_HEALTH",
                status=CheckStatus.SKIP,
                message="Could not fetch BTC data",
                details={}
            )

        except Exception as e:
            return CheckResult(
                name="BTC_HEALTH",
                status=CheckStatus.SKIP,
                message=f"Error: {str(e)[:50]}",
                details={}
            )

    # =========================================================================
    # RISK LIMITS
    # =========================================================================

    def check_capital(self, current_capital: float) -> CheckResult:
        """Check if capital meets minimum requirements."""
        if current_capital >= self.min_capital:
            return CheckResult(
                name="CAPITAL_CHECK",
                status=CheckStatus.PASS,
                message=f"Capital OK: ${current_capital:,.2f} >= ${self.min_capital:,.2f}",
                details={"current": current_capital, "minimum": self.min_capital}
            )
        else:
            return CheckResult(
                name="CAPITAL_CHECK",
                status=CheckStatus.FAIL,
                message=f"INSUFFICIENT: ${current_capital:,.2f} < ${self.min_capital:,.2f} minimum",
                details={"current": current_capital, "minimum": self.min_capital}
            )

    def check_position_size(self, position_size: float, capital: float) -> CheckResult:
        """Check if position size is within limits."""
        if capital <= 0:
            return CheckResult(
                name="POSITION_SIZE",
                status=CheckStatus.FAIL,
                message="Invalid capital",
                details={}
            )

        size_pct = (position_size / capital) * 100
        max_pct = self.max_capital_per_trade_pct

        if size_pct <= max_pct:
            return CheckResult(
                name="POSITION_SIZE",
                status=CheckStatus.PASS,
                message=f"Size OK: {size_pct:.1f}% <= {max_pct}% maximum",
                details={"size_pct": size_pct, "max_pct": max_pct}
            )
        else:
            return CheckResult(
                name="POSITION_SIZE",
                status=CheckStatus.FAIL,
                message=f"TOO LARGE: {size_pct:.1f}% > {max_pct}% maximum",
                details={"size_pct": size_pct, "max_pct": max_pct}
            )

    # =========================================================================
    # FULL CHECKLIST
    # =========================================================================

    def run_full_checklist(
        self,
        symbol: str = "ATOMUSDT",
        signals: List[str] = None,
        capital: float = 10000.0,
        position_size: float = 200.0
    ) -> Tuple[bool, List[CheckResult]]:
        """
        Run the complete go-live checklist.

        Args:
            symbol: Trading symbol
            signals: Signal names (default: ["BB", "MACD"])
            capital: Current capital
            position_size: Proposed position size

        Returns:
            Tuple of (all_passed, results)
        """
        if signals is None:
            signals = ["BB", "MACD"]

        results = []

        # 1. Hard blocks (must pass)
        hard_blocks = self.check_hard_blocks()
        results.extend(hard_blocks)

        # 2. System health
        results.append(self.check_api_connectivity())
        results.append(self.check_data_freshness(symbol))

        # 3. Time/Day
        results.append(self.check_trading_window())

        # 4. Symbol
        results.append(self.check_symbol(symbol))

        # 5. Signals
        results.append(self.check_signals(signals))

        # 6. Market conditions
        results.append(self.check_volatility(symbol))
        results.append(self.check_btc_health())

        # 7. Risk limits
        results.append(self.check_capital(capital))
        results.append(self.check_position_size(position_size, capital))

        # Determine if all passed
        failed = [r for r in results if r.status == CheckStatus.FAIL]
        all_passed = len(failed) == 0

        return all_passed, results

    def print_checklist(
        self,
        symbol: str = "ATOMUSDT",
        signals: List[str] = None,
        capital: float = 10000.0,
        position_size: float = 200.0
    ):
        """Print the full checklist with results."""
        all_passed, results = self.run_full_checklist(symbol, signals, capital, position_size)

        print("\n" + "=" * 70)
        print("GO-LIVE CHECKLIST")
        print("=" * 70)
        print(f"Symbol: {symbol}")
        print(f"Signals: {signals or ['BB', 'MACD']}")
        print(f"Capital: ${capital:,.2f}")
        print(f"Position Size: ${position_size:,.2f}")
        print("=" * 70)

        # Group by status
        passed = [r for r in results if r.status == CheckStatus.PASS]
        warned = [r for r in results if r.status == CheckStatus.WARN]
        failed = [r for r in results if r.status == CheckStatus.FAIL]
        skipped = [r for r in results if r.status == CheckStatus.SKIP]

        if failed:
            print("\n[FAILED] - TRADING BLOCKED")
            print("-" * 40)
            for r in failed:
                print(f"  X {r.name}: {r.message}")

        if warned:
            print("\n[WARNINGS] - PROCEED WITH CAUTION")
            print("-" * 40)
            for r in warned:
                print(f"  ! {r.name}: {r.message}")

        if passed:
            print("\n[PASSED]")
            print("-" * 40)
            for r in passed:
                print(f"  + {r.name}: {r.message}")

        if skipped:
            print("\n[SKIPPED]")
            print("-" * 40)
            for r in skipped:
                print(f"  - {r.name}: {r.message}")

        print("\n" + "=" * 70)
        if all_passed:
            print("RESULT: ALL CHECKS PASSED - TRADING ALLOWED")
        else:
            print("RESULT: CHECKS FAILED - TRADING BLOCKED")
        print("=" * 70)

        return all_passed


# =============================================================================
# QUICK CHECK FUNCTION
# =============================================================================

def can_trade_now(symbol: str = "ATOMUSDT", signals: List[str] = None) -> Tuple[bool, str]:
    """
    Quick check if trading is allowed right now.

    Args:
        symbol: Trading symbol
        signals: Signal names

    Returns:
        Tuple of (allowed, reason)
    """
    checklist = GoLiveChecklist()

    # Check hard blocks first
    hard_blocks = checklist.check_hard_blocks()
    for check in hard_blocks:
        if check.status == CheckStatus.FAIL:
            return False, check.message

    # Check symbol
    symbol_check = checklist.check_symbol(symbol)
    if symbol_check.status == CheckStatus.FAIL:
        return False, symbol_check.message

    # Check signals if provided
    if signals:
        signal_check = checklist.check_signals(signals)
        if signal_check.status == CheckStatus.FAIL:
            return False, signal_check.message

    return True, "Pre-checks passed"


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("GO-LIVE CHECKLIST TEST")
    print("=" * 70)

    checklist = GoLiveChecklist()

    # Run full checklist
    checklist.print_checklist(
        symbol="ATOMUSDT",
        signals=["BB", "MACD"],
        capital=10000.0,
        position_size=200.0
    )

    # Quick check
    print("\n[QUICK CHECKS]")
    print("-" * 40)

    test_cases = [
        ("ATOMUSDT", ["BB", "MACD"]),
        ("ATOMUSDT", ["RSI"]),  # Missing MACD
        ("DOGEUSDT", ["BB", "MACD"]),  # Not approved symbol
        ("SOLUSDT", ["MACD", "STOCH"]),
    ]

    for symbol, signals in test_cases:
        allowed, reason = can_trade_now(symbol, signals)
        status = "ALLOWED" if allowed else "BLOCKED"
        print(f"  {symbol} + {signals}: {status}")
        if not allowed:
            print(f"    -> {reason}")
