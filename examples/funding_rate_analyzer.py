#!/usr/bin/env python3
"""
Funding Rate Analyzer - Futures Market Sentiment Analysis

Features:
1. Real-time funding rates from major exchanges
2. Historical funding rate patterns
3. Funding rate divergence detection
4. Long/Short ratio analysis
5. Funding arbitrage opportunity detection
6. Extreme funding rate alerts

Author: Analize Team
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import requests
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class FundingRate:
    """Single funding rate snapshot."""
    symbol: str
    exchange: str
    rate: float  # As percentage
    next_funding_time: datetime
    interval_hours: int
    annualized_rate: float


@dataclass
class FundingAnalysis:
    """Funding rate analysis result."""
    symbol: str
    current_rate: float
    avg_rate_7d: float
    avg_rate_30d: float
    percentile: float  # Where current rate sits historically
    trend: str  # INCREASING, DECREASING, STABLE
    signal: str  # BULLISH, BEARISH, NEUTRAL
    explanation: str


@dataclass
class LongShortRatio:
    """Long/Short ratio data."""
    symbol: str
    exchange: str
    long_ratio: float
    short_ratio: float
    long_short_ratio: float
    timestamp: datetime


@dataclass
class FundingArbitrage:
    """Funding arbitrage opportunity."""
    symbol: str
    long_exchange: str
    short_exchange: str
    long_rate: float
    short_rate: float
    spread: float
    annualized_return: float
    risk_level: str


@dataclass
class FundingAlert:
    """Alert for extreme funding conditions."""
    symbol: str
    alert_type: str
    severity: str
    current_rate: float
    threshold: float
    message: str
    action: str


# =============================================================================
# FUNDING RATE FETCHER
# =============================================================================

class FundingRateFetcher:
    """Fetch funding rates from multiple exchanges."""

    def __init__(self):
        self.cache = {}
        self.symbols = ["XRPUSDT", "SOLUSDT", "ATOMUSDT", "BTCUSDT", "ETHUSDT"]

    def fetch_binance_funding(self, symbol: str) -> Optional[FundingRate]:
        """Fetch current funding rate from Binance Futures."""
        try:
            url = "https://fapi.binance.com/fapi/v1/fundingRate"
            params = {"symbol": symbol, "limit": 1}

            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                if data:
                    rate = float(data[0]["fundingRate"]) * 100  # Convert to percentage

                    return FundingRate(
                        symbol=symbol,
                        exchange="BINANCE",
                        rate=rate,
                        next_funding_time=datetime.fromtimestamp(
                            int(data[0]["fundingTime"]) / 1000
                        ),
                        interval_hours=8,
                        annualized_rate=rate * 3 * 365  # 3 times per day * 365 days
                    )
        except Exception as e:
            print(f"Binance funding fetch error for {symbol}: {e}")

        return None

    def fetch_binance_funding_history(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Fetch historical funding rates from Binance."""
        try:
            url = "https://fapi.binance.com/fapi/v1/fundingRate"
            start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

            params = {
                "symbol": symbol,
                "startTime": start_time,
                "limit": 1000
            }

            response = requests.get(url, params=params, timeout=15)

            if response.status_code == 200:
                data = response.json()

                if data:
                    df = pd.DataFrame(data)
                    df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms")
                    df["fundingRate"] = df["fundingRate"].astype(float) * 100

                    return df[["fundingTime", "fundingRate"]].rename(
                        columns={"fundingTime": "timestamp", "fundingRate": "rate"}
                    )
        except Exception as e:
            print(f"Binance history fetch error for {symbol}: {e}")

        return pd.DataFrame()

    def fetch_bybit_funding(self, symbol: str) -> Optional[FundingRate]:
        """Fetch current funding rate from Bybit."""
        try:
            url = "https://api.bybit.com/v5/market/tickers"
            params = {"category": "linear", "symbol": symbol}

            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                if data.get("result", {}).get("list"):
                    ticker = data["result"]["list"][0]
                    rate = float(ticker.get("fundingRate", 0)) * 100

                    return FundingRate(
                        symbol=symbol,
                        exchange="BYBIT",
                        rate=rate,
                        next_funding_time=datetime.fromtimestamp(
                            int(ticker.get("nextFundingTime", 0)) / 1000
                        ),
                        interval_hours=8,
                        annualized_rate=rate * 3 * 365
                    )
        except Exception as e:
            pass

        return None

    def fetch_all_funding_rates(self) -> Dict[str, List[FundingRate]]:
        """Fetch funding rates from all exchanges."""
        results = {}

        for symbol in self.symbols:
            rates = []

            # Binance
            binance_rate = self.fetch_binance_funding(symbol)
            if binance_rate:
                rates.append(binance_rate)

            # Bybit
            bybit_rate = self.fetch_bybit_funding(symbol)
            if bybit_rate:
                rates.append(bybit_rate)

            if rates:
                results[symbol] = rates

        return results


# =============================================================================
# LONG/SHORT RATIO ANALYZER
# =============================================================================

class LongShortAnalyzer:
    """Analyze long/short ratios."""

    def fetch_binance_long_short_ratio(self, symbol: str) -> Optional[LongShortRatio]:
        """Fetch long/short ratio from Binance."""
        try:
            url = "https://fapi.binance.com/futures/data/globalLongShortAccountRatio"
            params = {"symbol": symbol, "period": "1h", "limit": 1}

            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                if data:
                    ls_ratio = float(data[0]["longShortRatio"])
                    long_account = float(data[0]["longAccount"])
                    short_account = float(data[0]["shortAccount"])

                    return LongShortRatio(
                        symbol=symbol,
                        exchange="BINANCE",
                        long_ratio=long_account,
                        short_ratio=short_account,
                        long_short_ratio=ls_ratio,
                        timestamp=datetime.fromtimestamp(
                            int(data[0]["timestamp"]) / 1000
                        )
                    )
        except Exception as e:
            pass

        return None

    def fetch_long_short_history(self, symbol: str, days: int = 7) -> pd.DataFrame:
        """Fetch historical long/short ratios."""
        try:
            url = "https://fapi.binance.com/futures/data/globalLongShortAccountRatio"
            params = {
                "symbol": symbol,
                "period": "4h",
                "limit": min(days * 6, 500)  # 6 periods per day
            }

            response = requests.get(url, params=params, timeout=15)

            if response.status_code == 200:
                data = response.json()
                if data:
                    df = pd.DataFrame(data)
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                    df["longShortRatio"] = df["longShortRatio"].astype(float)
                    df["longAccount"] = df["longAccount"].astype(float)
                    df["shortAccount"] = df["shortAccount"].astype(float)

                    return df
        except Exception as e:
            pass

        return pd.DataFrame()

    def analyze_sentiment(self, ls_ratio: float, historical_mean: float) -> Dict:
        """Analyze market sentiment from L/S ratio."""
        # Calculate deviation from mean
        deviation = (ls_ratio - historical_mean) / historical_mean * 100

        if ls_ratio > 2.0:
            sentiment = "EXTREME_LONG"
            signal = "BEARISH"  # Contrarian
            explanation = "Market extremely long - potential squeeze risk"
        elif ls_ratio > 1.5:
            sentiment = "BULLISH"
            signal = "NEUTRAL"
            explanation = "Market leaning long - momentum positive"
        elif ls_ratio > 1.0:
            sentiment = "SLIGHTLY_BULLISH"
            signal = "NEUTRAL"
            explanation = "Slight long bias"
        elif ls_ratio > 0.7:
            sentiment = "SLIGHTLY_BEARISH"
            signal = "NEUTRAL"
            explanation = "Slight short bias"
        elif ls_ratio > 0.5:
            sentiment = "BEARISH"
            signal = "NEUTRAL"
            explanation = "Market leaning short"
        else:
            sentiment = "EXTREME_SHORT"
            signal = "BULLISH"  # Contrarian
            explanation = "Market extremely short - potential short squeeze"

        return {
            "sentiment": sentiment,
            "signal": signal,
            "explanation": explanation,
            "deviation_from_mean": deviation
        }


# =============================================================================
# FUNDING RATE ANALYZER
# =============================================================================

class FundingRateAnalyzer:
    """Analyze funding rate patterns and signals."""

    def __init__(self):
        self.fetcher = FundingRateFetcher()

    def analyze_symbol(self, symbol: str) -> FundingAnalysis:
        """Perform full funding rate analysis for a symbol."""
        # Get current rate
        current = self.fetcher.fetch_binance_funding(symbol)
        current_rate = current.rate if current else 0

        # Get historical data
        history = self.fetcher.fetch_binance_funding_history(symbol, days=30)

        if history.empty:
            return FundingAnalysis(
                symbol=symbol,
                current_rate=current_rate,
                avg_rate_7d=0,
                avg_rate_30d=0,
                percentile=50,
                trend="UNKNOWN",
                signal="NEUTRAL",
                explanation="Insufficient data"
            )

        # Calculate averages
        recent_7d = history.tail(21)  # ~7 days * 3 per day
        avg_7d = recent_7d["rate"].mean()
        avg_30d = history["rate"].mean()

        # Calculate percentile
        percentile = (history["rate"] < current_rate).sum() / len(history) * 100

        # Determine trend
        if len(history) >= 6:
            recent_avg = history["rate"].tail(6).mean()
            older_avg = history["rate"].head(6).mean()

            if recent_avg > older_avg + 0.005:
                trend = "INCREASING"
            elif recent_avg < older_avg - 0.005:
                trend = "DECREASING"
            else:
                trend = "STABLE"
        else:
            trend = "UNKNOWN"

        # Generate signal
        if current_rate > 0.1:  # Very positive funding
            signal = "BEARISH"
            explanation = "High positive funding - longs paying shorts. Potential long squeeze."
        elif current_rate > 0.03:  # Positive funding
            signal = "SLIGHTLY_BEARISH"
            explanation = "Positive funding - market leaning long."
        elif current_rate < -0.1:  # Very negative funding
            signal = "BULLISH"
            explanation = "High negative funding - shorts paying longs. Potential short squeeze."
        elif current_rate < -0.03:  # Negative funding
            signal = "SLIGHTLY_BULLISH"
            explanation = "Negative funding - market leaning short."
        else:
            signal = "NEUTRAL"
            explanation = "Funding rate near neutral."

        return FundingAnalysis(
            symbol=symbol,
            current_rate=current_rate,
            avg_rate_7d=avg_7d,
            avg_rate_30d=avg_30d,
            percentile=percentile,
            trend=trend,
            signal=signal,
            explanation=explanation
        )

    def find_arbitrage_opportunities(self) -> List[FundingArbitrage]:
        """Find funding rate arbitrage opportunities."""
        opportunities = []

        all_rates = self.fetcher.fetch_all_funding_rates()

        for symbol, rates in all_rates.items():
            if len(rates) < 2:
                continue

            # Sort by rate
            sorted_rates = sorted(rates, key=lambda x: x.rate)
            lowest = sorted_rates[0]
            highest = sorted_rates[-1]

            spread = highest.rate - lowest.rate

            # Only report if spread is significant
            if abs(spread) > 0.02:  # 0.02% spread
                annualized = spread * 3 * 365  # 3x per day * 365

                # Determine risk level
                if abs(spread) > 0.1:
                    risk = "HIGH"
                elif abs(spread) > 0.05:
                    risk = "MEDIUM"
                else:
                    risk = "LOW"

                opportunities.append(FundingArbitrage(
                    symbol=symbol,
                    long_exchange=lowest.exchange,
                    short_exchange=highest.exchange,
                    long_rate=lowest.rate,
                    short_rate=highest.rate,
                    spread=spread,
                    annualized_return=annualized,
                    risk_level=risk
                ))

        return sorted(opportunities, key=lambda x: x.spread, reverse=True)

    def generate_alerts(self, threshold: float = 0.05) -> List[FundingAlert]:
        """Generate alerts for extreme funding conditions."""
        alerts = []

        for symbol in self.fetcher.symbols:
            analysis = self.analyze_symbol(symbol)

            # Check for extreme positive funding
            if analysis.current_rate > threshold:
                alerts.append(FundingAlert(
                    symbol=symbol,
                    alert_type="HIGH_POSITIVE_FUNDING",
                    severity="HIGH" if analysis.current_rate > 0.1 else "MEDIUM",
                    current_rate=analysis.current_rate,
                    threshold=threshold,
                    message=f"{symbol} has high positive funding ({analysis.current_rate:.3f}%)",
                    action="Consider shorting or closing longs - longs paying premium"
                ))

            # Check for extreme negative funding
            elif analysis.current_rate < -threshold:
                alerts.append(FundingAlert(
                    symbol=symbol,
                    alert_type="HIGH_NEGATIVE_FUNDING",
                    severity="HIGH" if analysis.current_rate < -0.1 else "MEDIUM",
                    current_rate=analysis.current_rate,
                    threshold=-threshold,
                    message=f"{symbol} has high negative funding ({analysis.current_rate:.3f}%)",
                    action="Consider longing - shorts paying premium"
                ))

            # Check for percentile extremes
            if analysis.percentile > 95:
                alerts.append(FundingAlert(
                    symbol=symbol,
                    alert_type="FUNDING_PERCENTILE_HIGH",
                    severity="MEDIUM",
                    current_rate=analysis.current_rate,
                    threshold=95,
                    message=f"{symbol} funding at {analysis.percentile:.0f}th percentile (30d)",
                    action="Historically high funding - potential mean reversion"
                ))
            elif analysis.percentile < 5:
                alerts.append(FundingAlert(
                    symbol=symbol,
                    alert_type="FUNDING_PERCENTILE_LOW",
                    severity="MEDIUM",
                    current_rate=analysis.current_rate,
                    threshold=5,
                    message=f"{symbol} funding at {analysis.percentile:.0f}th percentile (30d)",
                    action="Historically low funding - potential mean reversion"
                ))

        return sorted(alerts, key=lambda x: abs(x.current_rate), reverse=True)


# =============================================================================
# FUNDING CORRELATION ANALYZER
# =============================================================================

class FundingCorrelationAnalyzer:
    """Analyze correlation between funding and price."""

    def __init__(self):
        self.fetcher = FundingRateFetcher()

    def analyze_funding_price_correlation(self, symbol: str, days: int = 30) -> Dict:
        """Analyze correlation between funding rate and price."""
        # Get funding history
        funding = self.fetcher.fetch_binance_funding_history(symbol, days)

        if funding.empty:
            return {"error": "Could not fetch funding data"}

        # Get price history
        price = self._fetch_price_history(symbol, days)

        if price.empty:
            return {"error": "Could not fetch price data"}

        # Resample funding to daily
        funding["date"] = funding["timestamp"].dt.date
        daily_funding = funding.groupby("date")["rate"].mean().reset_index()
        daily_funding["date"] = pd.to_datetime(daily_funding["date"])

        # Merge with price
        price["date"] = price["timestamp"].dt.date
        price["date"] = pd.to_datetime(price["date"])
        price["returns"] = price["close"].pct_change()

        merged = pd.merge(daily_funding, price[["date", "returns"]], on="date", how="inner")

        if len(merged) < 10:
            return {"error": "Insufficient overlapping data"}

        # Calculate correlations
        corr_same_day = merged["rate"].corr(merged["returns"])

        # Lagged correlation (funding predicts next day returns)
        merged["next_returns"] = merged["returns"].shift(-1)
        corr_next_day = merged["rate"].corr(merged["next_returns"])

        # Find extreme funding performance
        high_funding = merged[merged["rate"] > merged["rate"].quantile(0.8)]
        low_funding = merged[merged["rate"] < merged["rate"].quantile(0.2)]

        avg_return_high_funding = high_funding["next_returns"].mean() * 100 if len(high_funding) > 0 else 0
        avg_return_low_funding = low_funding["next_returns"].mean() * 100 if len(low_funding) > 0 else 0

        return {
            "symbol": symbol,
            "correlation_same_day": corr_same_day,
            "correlation_next_day": corr_next_day,
            "avg_return_after_high_funding": avg_return_high_funding,
            "avg_return_after_low_funding": avg_return_low_funding,
            "interpretation": self._interpret_correlation(corr_next_day),
            "predictive_value": abs(corr_next_day) > 0.15
        }

    def _fetch_price_history(self, symbol: str, days: int) -> pd.DataFrame:
        """Fetch price history."""
        try:
            params = {
                "symbol": symbol,
                "interval": "1d",
                "startTime": int((datetime.now() - timedelta(days=days)).timestamp() * 1000),
                "limit": 1000
            }

            response = requests.get(
                "https://api.binance.com/api/v3/klines",
                params=params,
                timeout=15
            )

            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data, columns=[
                    "timestamp", "open", "high", "low", "close", "volume",
                    "close_time", "quote_volume", "trades", "taker_buy_base",
                    "taker_buy_quote", "ignore"
                ])

                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df["close"] = df["close"].astype(float)

                return df[["timestamp", "close"]]

        except Exception:
            pass

        return pd.DataFrame()

    def _interpret_correlation(self, corr: float) -> str:
        """Interpret correlation value."""
        if corr < -0.3:
            return "Strong contrarian signal - high funding often followed by drops"
        elif corr < -0.15:
            return "Moderate contrarian signal"
        elif corr > 0.3:
            return "Strong momentum signal - high funding followed by gains"
        elif corr > 0.15:
            return "Moderate momentum signal"
        else:
            return "Weak predictive relationship"


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_funding_analysis():
    """Run full funding rate analysis."""
    print("=" * 70)
    print("FUNDING RATE ANALYZER - Futures Market Sentiment")
    print("=" * 70)
    print()

    analyzer = FundingRateAnalyzer()
    ls_analyzer = LongShortAnalyzer()
    corr_analyzer = FundingCorrelationAnalyzer()

    # Current Funding Rates
    print("[1] CURRENT FUNDING RATES")
    print("-" * 70)

    all_rates = analyzer.fetcher.fetch_all_funding_rates()

    for symbol in ["XRPUSDT", "SOLUSDT", "ATOMUSDT", "BTCUSDT", "ETHUSDT"]:
        if symbol in all_rates:
            print(f"  {symbol}:")
            for rate in all_rates[symbol]:
                emoji = "🟢" if rate.rate < -0.01 else "🔴" if rate.rate > 0.01 else "⚪"
                print(f"    {rate.exchange}: {rate.rate:+.4f}% {emoji}")
                print(f"      Annualized: {rate.annualized_rate:+.1f}%")
            print()

    # Funding Analysis
    print("[2] FUNDING RATE ANALYSIS")
    print("-" * 70)

    target_coins = ["XRPUSDT", "SOLUSDT", "ATOMUSDT"]

    for symbol in target_coins:
        analysis = analyzer.analyze_symbol(symbol)

        print(f"  {symbol}:")
        print(f"    Current Rate: {analysis.current_rate:+.4f}%")
        print(f"    7-Day Avg: {analysis.avg_rate_7d:+.4f}%")
        print(f"    30-Day Avg: {analysis.avg_rate_30d:+.4f}%")
        print(f"    Percentile: {analysis.percentile:.0f}%")
        print(f"    Trend: {analysis.trend}")
        print(f"    Signal: {analysis.signal}")
        print(f"    {analysis.explanation}")
        print()

    # Long/Short Ratio
    print("[3] LONG/SHORT RATIO ANALYSIS")
    print("-" * 70)

    for symbol in target_coins:
        ls_ratio = ls_analyzer.fetch_binance_long_short_ratio(symbol)

        if ls_ratio:
            # Get historical for context
            history = ls_analyzer.fetch_long_short_history(symbol, days=7)
            historical_mean = history["longShortRatio"].mean() if not history.empty else 1.0

            sentiment = ls_analyzer.analyze_sentiment(ls_ratio.long_short_ratio, historical_mean)

            print(f"  {symbol}:")
            print(f"    Long Accounts: {ls_ratio.long_ratio:.1%}")
            print(f"    Short Accounts: {ls_ratio.short_ratio:.1%}")
            print(f"    L/S Ratio: {ls_ratio.long_short_ratio:.2f}")
            print(f"    7d Avg: {historical_mean:.2f}")
            print(f"    Sentiment: {sentiment['sentiment']}")
            print(f"    Signal: {sentiment['signal']}")
            print(f"    {sentiment['explanation']}")
            print()
        else:
            print(f"  {symbol}: Unable to fetch L/S ratio")
            print()

    # Funding-Price Correlation
    print("[4] FUNDING vs PRICE CORRELATION")
    print("-" * 70)

    for symbol in target_coins:
        corr_result = corr_analyzer.analyze_funding_price_correlation(symbol)

        if "error" not in corr_result:
            print(f"  {symbol}:")
            print(f"    Same-Day Correlation: {corr_result['correlation_same_day']:.3f}")
            print(f"    Next-Day Correlation: {corr_result['correlation_next_day']:.3f}")
            print(f"    Avg Return After High Funding: {corr_result['avg_return_after_high_funding']:.2f}%")
            print(f"    Avg Return After Low Funding: {corr_result['avg_return_after_low_funding']:.2f}%")
            print(f"    {corr_result['interpretation']}")
            print(f"    Predictive Value: {'Yes' if corr_result['predictive_value'] else 'No'}")
            print()
        else:
            print(f"  {symbol}: {corr_result['error']}")
            print()

    # Arbitrage Opportunities
    print("[5] FUNDING ARBITRAGE OPPORTUNITIES")
    print("-" * 70)

    opportunities = analyzer.find_arbitrage_opportunities()

    if opportunities:
        for opp in opportunities[:5]:
            print(f"  {opp.symbol}:")
            print(f"    Long on {opp.long_exchange}: {opp.long_rate:+.4f}%")
            print(f"    Short on {opp.short_exchange}: {opp.short_rate:+.4f}%")
            print(f"    Spread: {opp.spread:.4f}%")
            print(f"    Annualized Return: {opp.annualized_return:.1f}%")
            print(f"    Risk Level: {opp.risk_level}")
            print()
    else:
        print("  No significant arbitrage opportunities found")
    print()

    # Alerts
    print("[6] FUNDING RATE ALERTS")
    print("-" * 70)

    alerts = analyzer.generate_alerts(threshold=0.03)

    if alerts:
        for alert in alerts[:5]:
            emoji = "🔴" if alert.severity == "HIGH" else "🟡"
            print(f"  {emoji} [{alert.severity}] {alert.symbol}")
            print(f"     {alert.message}")
            print(f"     Action: {alert.action}")
            print()
    else:
        print("  No funding rate alerts at this time")
    print()

    # Trading Summary
    print("[7] FUNDING-BASED TRADING SIGNALS")
    print("-" * 70)

    for symbol in target_coins:
        analysis = analyzer.analyze_symbol(symbol)
        ls_ratio = ls_analyzer.fetch_binance_long_short_ratio(symbol)

        # Combine signals
        funding_signal = 1 if "BULLISH" in analysis.signal else (-1 if "BEARISH" in analysis.signal else 0)

        if ls_ratio:
            if ls_ratio.long_short_ratio > 1.5:
                ls_signal = -1  # Contrarian bearish
            elif ls_ratio.long_short_ratio < 0.7:
                ls_signal = 1  # Contrarian bullish
            else:
                ls_signal = 0
        else:
            ls_signal = 0

        combined = funding_signal + ls_signal

        if combined >= 2:
            recommendation = "🟢 STRONG BUY"
        elif combined == 1:
            recommendation = "🟡 MODERATE BUY"
        elif combined <= -2:
            recommendation = "🔴 STRONG SELL"
        elif combined == -1:
            recommendation = "🟠 MODERATE SELL"
        else:
            recommendation = "⚪ NEUTRAL"

        print(f"  {symbol}: {recommendation}")
        print(f"    Funding Signal: {analysis.signal}")
        if ls_ratio:
            print(f"    L/S Ratio Signal: {'Bullish' if ls_signal > 0 else 'Bearish' if ls_signal < 0 else 'Neutral'}")
        print()

    print("=" * 70)
    print("Analysis Complete!")
    print("=" * 70)


if __name__ == "__main__":
    run_funding_analysis()
