#!/usr/bin/env python3
"""
Correlation Analyzer - Cross-Asset Correlation Analysis

Features:
1. Rolling correlation matrices between crypto assets
2. Correlation regime detection (high/low correlation periods)
3. Lead-lag analysis (which coin moves first)
4. Beta calculation vs BTC/ETH
5. Diversification scoring
6. Correlation breakdown alerts

Author: Analize Team
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import requests
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class CorrelationResult:
    """Single correlation measurement."""
    asset1: str
    asset2: str
    correlation: float
    period_days: int
    start_date: datetime
    end_date: datetime
    p_value: float
    is_significant: bool


@dataclass
class LeadLagResult:
    """Lead-lag relationship between assets."""
    leader: str
    follower: str
    lag_periods: int  # Positive = leader leads
    correlation_at_lag: float
    confidence: float


@dataclass
class BetaResult:
    """Beta calculation result."""
    asset: str
    benchmark: str
    beta: float
    alpha: float  # Excess return
    r_squared: float
    volatility_ratio: float


@dataclass
class CorrelationRegime:
    """Correlation regime classification."""
    regime: str  # HIGH_CORR, LOW_CORR, DECORRELATING, CORRELATING
    avg_correlation: float
    trend: str  # INCREASING, DECREASING, STABLE
    confidence: float


@dataclass
class DiversificationScore:
    """Portfolio diversification analysis."""
    score: float  # 0-100
    effective_assets: float  # Number of uncorrelated assets
    concentration_risk: str  # LOW, MEDIUM, HIGH
    recommendations: List[str]


# =============================================================================
# MARKET DATA FETCHER
# =============================================================================

class MultiAssetDataFetcher:
    """Fetch data for multiple assets."""

    def __init__(self):
        self.cache = {}
        self.symbols = [
            "BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "ATOMUSDT",
            "ADAUSDT", "DOTUSDT", "AVAXUSDT", "LINKUSDT", "MATICUSDT"
        ]

    def fetch_all(self, days: int = 365) -> Dict[str, pd.DataFrame]:
        """Fetch data for all tracked assets."""
        data = {}
        for symbol in self.symbols:
            df = self.fetch_ohlcv(symbol, days)
            if not df.empty:
                data[symbol] = df
                print(f"  Fetched {symbol}: {len(df)} days")
            else:
                print(f"  Failed to fetch {symbol}")
        return data

    def fetch_ohlcv(self, symbol: str, days: int = 365) -> pd.DataFrame:
        """Fetch daily OHLCV data with fallbacks."""
        cache_key = f"{symbol}_{days}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Try Binance first
        df = self._fetch_binance(symbol, days)
        if df.empty:
            df = self._fetch_binance_us(symbol, days)
        if df.empty:
            df = self._fetch_coingecko(symbol, days)

        if not df.empty:
            self.cache[cache_key] = df

        return df

    def _fetch_binance(self, symbol: str, days: int) -> pd.DataFrame:
        """Fetch from Binance."""
        try:
            params = {
                "symbol": symbol,
                "interval": "1d",
                "startTime": int((datetime.now() - timedelta(days=days)).timestamp() * 1000),
                "endTime": int(datetime.now().timestamp() * 1000),
                "limit": 1000,
            }

            response = requests.get(
                "https://api.binance.com/api/v3/klines",
                params=params,
                timeout=15
            )
            data = response.json()

            if isinstance(data, list) and len(data) > 0:
                df = pd.DataFrame(data, columns=[
                    "timestamp", "open", "high", "low", "close", "volume",
                    "close_time", "quote_volume", "trades", "taker_buy_base",
                    "taker_buy_quote", "ignore"
                ])

                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                for col in ["open", "high", "low", "close", "volume"]:
                    df[col] = df[col].astype(float)

                return df[["timestamp", "open", "high", "low", "close", "volume"]]
        except Exception as e:
            pass

        return pd.DataFrame()

    def _fetch_binance_us(self, symbol: str, days: int) -> pd.DataFrame:
        """Fetch from Binance US."""
        try:
            params = {
                "symbol": symbol,
                "interval": "1d",
                "startTime": int((datetime.now() - timedelta(days=days)).timestamp() * 1000),
                "endTime": int(datetime.now().timestamp() * 1000),
                "limit": 1000,
            }

            response = requests.get(
                "https://api.binance.us/api/v3/klines",
                params=params,
                timeout=15
            )
            data = response.json()

            if isinstance(data, list) and len(data) > 0:
                df = pd.DataFrame(data, columns=[
                    "timestamp", "open", "high", "low", "close", "volume",
                    "close_time", "quote_volume", "trades", "taker_buy_base",
                    "taker_buy_quote", "ignore"
                ])

                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                for col in ["open", "high", "low", "close", "volume"]:
                    df[col] = df[col].astype(float)

                return df[["timestamp", "open", "high", "low", "close", "volume"]]
        except Exception as e:
            pass

        return pd.DataFrame()

    def _fetch_coingecko(self, symbol: str, days: int) -> pd.DataFrame:
        """Fetch from CoinGecko."""
        coingecko_ids = {
            "BTCUSDT": "bitcoin",
            "ETHUSDT": "ethereum",
            "XRPUSDT": "ripple",
            "SOLUSDT": "solana",
            "ATOMUSDT": "cosmos",
            "ADAUSDT": "cardano",
            "DOTUSDT": "polkadot",
            "AVAXUSDT": "avalanche-2",
            "LINKUSDT": "chainlink",
            "MATICUSDT": "matic-network",
        }

        coin_id = coingecko_ids.get(symbol)
        if not coin_id:
            return pd.DataFrame()

        try:
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
            params = {"vs_currency": "usd", "days": min(days, 365)}

            response = requests.get(url, params=params, timeout=15)
            data = response.json()

            if isinstance(data, list) and len(data) > 0:
                df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df["volume"] = 0.0

                return df[["timestamp", "open", "high", "low", "close", "volume"]]
        except Exception as e:
            pass

        return pd.DataFrame()


# =============================================================================
# CORRELATION ANALYZER
# =============================================================================

class CorrelationAnalyzer:
    """Analyze correlations between crypto assets."""

    def __init__(self, data: Dict[str, pd.DataFrame]):
        self.data = data
        self.returns = self._calculate_returns()

    def _calculate_returns(self) -> pd.DataFrame:
        """Calculate daily returns for all assets."""
        returns_dict = {}

        for symbol, df in self.data.items():
            if len(df) > 1:
                df = df.set_index("timestamp")
                returns_dict[symbol] = df["close"].pct_change()

        returns_df = pd.DataFrame(returns_dict)
        returns_df = returns_df.dropna()

        return returns_df

    def correlation_matrix(self, window: Optional[int] = None) -> pd.DataFrame:
        """Calculate correlation matrix."""
        if window:
            return self.returns.tail(window).corr()
        return self.returns.corr()

    def rolling_correlation(self, asset1: str, asset2: str, window: int = 30) -> pd.Series:
        """Calculate rolling correlation between two assets."""
        if asset1 not in self.returns.columns or asset2 not in self.returns.columns:
            return pd.Series()

        return self.returns[asset1].rolling(window).corr(self.returns[asset2])

    def correlation_percentile(self, asset1: str, asset2: str,
                               current_window: int = 30,
                               history_window: int = 365) -> Dict:
        """Calculate where current correlation sits in historical distribution."""
        rolling_corr = self.rolling_correlation(asset1, asset2, current_window)
        rolling_corr = rolling_corr.dropna()

        if len(rolling_corr) < history_window:
            history_window = len(rolling_corr)

        current_corr = rolling_corr.iloc[-1] if len(rolling_corr) > 0 else 0
        historical = rolling_corr.tail(history_window)

        percentile = (historical < current_corr).sum() / len(historical) * 100

        return {
            "current_correlation": current_corr,
            "percentile": percentile,
            "historical_mean": historical.mean(),
            "historical_std": historical.std(),
            "historical_min": historical.min(),
            "historical_max": historical.max(),
            "z_score": (current_corr - historical.mean()) / historical.std() if historical.std() > 0 else 0
        }

    def detect_correlation_regime(self, window: int = 30) -> CorrelationRegime:
        """Detect current correlation regime across all assets."""
        corr_matrix = self.correlation_matrix(window)

        # Get upper triangle (excluding diagonal)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        correlations = corr_matrix.where(mask).stack().values

        avg_corr = np.mean(correlations)

        # Calculate trend (compare to previous period)
        if len(self.returns) > window * 2:
            prev_returns = self.returns.iloc[-(window*2):-window]
            prev_corr_matrix = prev_returns.corr()
            prev_correlations = prev_corr_matrix.where(mask).stack().values
            prev_avg_corr = np.mean(prev_correlations)

            corr_change = avg_corr - prev_avg_corr

            if corr_change > 0.1:
                trend = "INCREASING"
            elif corr_change < -0.1:
                trend = "DECREASING"
            else:
                trend = "STABLE"
        else:
            trend = "UNKNOWN"

        # Classify regime
        if avg_corr > 0.7:
            regime = "HIGH_CORR"
        elif avg_corr > 0.5:
            regime = "MODERATE_CORR"
        elif avg_corr > 0.3:
            regime = "LOW_CORR"
        else:
            regime = "DECORRELATED"

        confidence = 1 - np.std(correlations)  # Lower variance = higher confidence

        return CorrelationRegime(
            regime=regime,
            avg_correlation=avg_corr,
            trend=trend,
            confidence=max(0, min(1, confidence))
        )


# =============================================================================
# LEAD-LAG ANALYZER
# =============================================================================

class LeadLagAnalyzer:
    """Analyze lead-lag relationships between assets."""

    def __init__(self, returns: pd.DataFrame):
        self.returns = returns

    def cross_correlation(self, asset1: str, asset2: str, max_lag: int = 5) -> Dict[int, float]:
        """Calculate cross-correlation at different lags."""
        if asset1 not in self.returns.columns or asset2 not in self.returns.columns:
            return {}

        r1 = self.returns[asset1].dropna()
        r2 = self.returns[asset2].dropna()

        # Align the series
        common_idx = r1.index.intersection(r2.index)
        r1 = r1[common_idx]
        r2 = r2[common_idx]

        correlations = {}

        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                # asset1 leads (shifted forward)
                corr = r1.iloc[:lag].corr(r2.iloc[-lag:])
            elif lag > 0:
                # asset2 leads (asset1 shifted back)
                corr = r1.iloc[lag:].corr(r2.iloc[:-lag])
            else:
                corr = r1.corr(r2)

            correlations[lag] = corr if not np.isnan(corr) else 0

        return correlations

    def find_lead_lag(self, asset1: str, asset2: str, max_lag: int = 5) -> LeadLagResult:
        """Find the optimal lead-lag relationship."""
        cross_corr = self.cross_correlation(asset1, asset2, max_lag)

        if not cross_corr:
            return LeadLagResult(
                leader=asset1,
                follower=asset2,
                lag_periods=0,
                correlation_at_lag=0,
                confidence=0
            )

        # Find lag with maximum absolute correlation
        best_lag = max(cross_corr, key=lambda x: abs(cross_corr[x]))
        best_corr = cross_corr[best_lag]

        # Determine leader/follower
        if best_lag > 0:
            leader = asset2
            follower = asset1
        else:
            leader = asset1
            follower = asset2

        # Confidence based on how much better the lagged correlation is
        zero_corr = cross_corr.get(0, 0)
        improvement = abs(best_corr) - abs(zero_corr)
        confidence = min(1, max(0, improvement * 5 + 0.5))

        return LeadLagResult(
            leader=leader,
            follower=follower,
            lag_periods=abs(best_lag),
            correlation_at_lag=best_corr,
            confidence=confidence
        )

    def granger_causality_simple(self, asset1: str, asset2: str, lag: int = 1) -> Dict:
        """Simple Granger causality test."""
        if asset1 not in self.returns.columns or asset2 not in self.returns.columns:
            return {"error": "Assets not found"}

        r1 = self.returns[asset1].dropna()
        r2 = self.returns[asset2].dropna()

        # Align
        common_idx = r1.index.intersection(r2.index)
        r1 = r1[common_idx].values
        r2 = r2[common_idx].values

        if len(r1) < lag + 10:
            return {"error": "Not enough data"}

        # Test if asset1 helps predict asset2
        # Baseline: predict r2[t] from r2[t-lag]
        y = r2[lag:]
        X_base = r2[:-lag].reshape(-1, 1)

        # Add r1 lagged
        X_full = np.column_stack([r2[:-lag], r1[:-lag]])

        # Calculate R² for both models
        from numpy.linalg import lstsq

        # Baseline model
        coef_base, _, _, _ = lstsq(X_base, y, rcond=None)
        pred_base = X_base @ coef_base
        ss_res_base = np.sum((y - pred_base) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2_base = 1 - ss_res_base / ss_tot if ss_tot > 0 else 0

        # Full model
        coef_full, _, _, _ = lstsq(X_full, y, rcond=None)
        pred_full = X_full @ coef_full
        ss_res_full = np.sum((y - pred_full) ** 2)
        r2_full = 1 - ss_res_full / ss_tot if ss_tot > 0 else 0

        # Improvement
        r2_improvement = r2_full - r2_base

        return {
            "r2_without_leader": r2_base,
            "r2_with_leader": r2_full,
            "r2_improvement": r2_improvement,
            "granger_causes": r2_improvement > 0.01,  # 1% threshold
            "predictive_power": r2_improvement
        }


# =============================================================================
# BETA CALCULATOR
# =============================================================================

class BetaCalculator:
    """Calculate beta and alpha vs benchmarks."""

    def __init__(self, returns: pd.DataFrame):
        self.returns = returns

    def calculate_beta(self, asset: str, benchmark: str = "BTCUSDT") -> BetaResult:
        """Calculate beta of asset vs benchmark."""
        if asset not in self.returns.columns or benchmark not in self.returns.columns:
            return BetaResult(
                asset=asset,
                benchmark=benchmark,
                beta=0,
                alpha=0,
                r_squared=0,
                volatility_ratio=0
            )

        asset_returns = self.returns[asset].dropna()
        bench_returns = self.returns[benchmark].dropna()

        # Align
        common_idx = asset_returns.index.intersection(bench_returns.index)
        asset_returns = asset_returns[common_idx]
        bench_returns = bench_returns[common_idx]

        if len(asset_returns) < 30:
            return BetaResult(
                asset=asset,
                benchmark=benchmark,
                beta=0,
                alpha=0,
                r_squared=0,
                volatility_ratio=0
            )

        # Calculate beta using covariance method
        covariance = np.cov(asset_returns, bench_returns)[0, 1]
        variance = np.var(bench_returns)

        beta = covariance / variance if variance > 0 else 0

        # Calculate alpha (excess return)
        asset_mean = asset_returns.mean() * 252  # Annualized
        bench_mean = bench_returns.mean() * 252
        alpha = asset_mean - beta * bench_mean

        # Calculate R²
        correlation = np.corrcoef(asset_returns, bench_returns)[0, 1]
        r_squared = correlation ** 2

        # Volatility ratio
        asset_vol = asset_returns.std() * np.sqrt(252)
        bench_vol = bench_returns.std() * np.sqrt(252)
        vol_ratio = asset_vol / bench_vol if bench_vol > 0 else 0

        return BetaResult(
            asset=asset,
            benchmark=benchmark,
            beta=beta,
            alpha=alpha,
            r_squared=r_squared,
            volatility_ratio=vol_ratio
        )

    def beta_stability(self, asset: str, benchmark: str = "BTCUSDT",
                       window: int = 60) -> pd.Series:
        """Calculate rolling beta over time."""
        if asset not in self.returns.columns or benchmark not in self.returns.columns:
            return pd.Series()

        asset_returns = self.returns[asset]
        bench_returns = self.returns[benchmark]

        rolling_cov = asset_returns.rolling(window).cov(bench_returns)
        rolling_var = bench_returns.rolling(window).var()

        rolling_beta = rolling_cov / rolling_var

        return rolling_beta


# =============================================================================
# DIVERSIFICATION SCORER
# =============================================================================

class DiversificationScorer:
    """Score portfolio diversification."""

    def __init__(self, correlation_matrix: pd.DataFrame):
        self.corr_matrix = correlation_matrix

    def calculate_score(self, holdings: List[str]) -> DiversificationScore:
        """Calculate diversification score for given holdings."""
        # Filter correlation matrix to holdings
        available = [h for h in holdings if h in self.corr_matrix.columns]

        if len(available) < 2:
            return DiversificationScore(
                score=0,
                effective_assets=len(available),
                concentration_risk="HIGH",
                recommendations=["Add more assets for diversification"]
            )

        sub_corr = self.corr_matrix.loc[available, available]

        # Calculate average correlation (excluding diagonal)
        mask = np.triu(np.ones_like(sub_corr, dtype=bool), k=1)
        correlations = sub_corr.where(mask).stack().values
        avg_corr = np.mean(correlations)

        # Effective number of assets (based on correlation)
        # Higher correlation = fewer effective assets
        n = len(available)
        effective_assets = n / (1 + (n - 1) * avg_corr) if avg_corr < 1 else 1

        # Diversification score (0-100)
        # Perfect diversification (avg_corr = 0) = 100
        # Perfect correlation (avg_corr = 1) = 0
        score = max(0, min(100, (1 - avg_corr) * 100))

        # Concentration risk
        if avg_corr > 0.7:
            concentration_risk = "HIGH"
        elif avg_corr > 0.5:
            concentration_risk = "MEDIUM"
        else:
            concentration_risk = "LOW"

        # Generate recommendations
        recommendations = []

        if avg_corr > 0.7:
            recommendations.append("Portfolio is highly correlated - consider adding uncorrelated assets")

        if n < 5:
            recommendations.append("Consider adding more assets (minimum 5 recommended)")

        # Find most correlated pair
        if len(correlations) > 0:
            max_corr_idx = np.argmax(correlations)
            i, j = np.triu_indices(n, k=1)
            most_corr_pair = (available[i[max_corr_idx]], available[j[max_corr_idx]])
            if correlations[max_corr_idx] > 0.8:
                recommendations.append(
                    f"Consider removing one of {most_corr_pair[0]}/{most_corr_pair[1]} (correlation: {correlations[max_corr_idx]:.2f})"
                )

        # Find least correlated assets to add
        if len(available) < len(self.corr_matrix.columns):
            other_assets = [a for a in self.corr_matrix.columns if a not in available]
            avg_corr_with_others = {}

            for other in other_assets:
                corrs_with_holdings = [self.corr_matrix.loc[other, h] for h in available]
                avg_corr_with_others[other] = np.mean(corrs_with_holdings)

            if avg_corr_with_others:
                best_add = min(avg_corr_with_others, key=avg_corr_with_others.get)
                recommendations.append(
                    f"Consider adding {best_add} (avg correlation with portfolio: {avg_corr_with_others[best_add]:.2f})"
                )

        return DiversificationScore(
            score=score,
            effective_assets=effective_assets,
            concentration_risk=concentration_risk,
            recommendations=recommendations
        )


# =============================================================================
# CORRELATION ALERTS
# =============================================================================

class CorrelationAlerts:
    """Generate alerts for correlation changes."""

    def __init__(self, analyzer: CorrelationAnalyzer):
        self.analyzer = analyzer

    def check_correlation_breakdown(self, asset1: str, asset2: str,
                                    threshold: float = 0.3) -> Optional[Dict]:
        """Check if correlation has broken down significantly."""
        stats = self.analyzer.correlation_percentile(asset1, asset2)

        current = stats["current_correlation"]
        historical_mean = stats["historical_mean"]
        z_score = stats["z_score"]

        # Alert if correlation changed significantly
        if abs(z_score) > 2:
            direction = "INCREASED" if current > historical_mean else "DECREASED"

            return {
                "alert": f"CORRELATION {direction}",
                "pair": f"{asset1}/{asset2}",
                "current_correlation": current,
                "historical_mean": historical_mean,
                "z_score": z_score,
                "percentile": stats["percentile"],
                "severity": "HIGH" if abs(z_score) > 3 else "MEDIUM"
            }

        return None

    def scan_all_pairs(self) -> List[Dict]:
        """Scan all asset pairs for correlation alerts."""
        alerts = []
        assets = list(self.analyzer.returns.columns)

        for i, asset1 in enumerate(assets):
            for asset2 in assets[i+1:]:
                alert = self.check_correlation_breakdown(asset1, asset2)
                if alert:
                    alerts.append(alert)

        return sorted(alerts, key=lambda x: abs(x["z_score"]), reverse=True)


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_correlation_analysis():
    """Run full correlation analysis."""
    print("=" * 70)
    print("CORRELATION ANALYZER - Cross-Asset Analysis")
    print("=" * 70)
    print()

    # Fetch data
    print("[1] FETCHING MARKET DATA...")
    fetcher = MultiAssetDataFetcher()
    data = fetcher.fetch_all(days=365)

    if len(data) < 3:
        print("Error: Need at least 3 assets for correlation analysis")
        return

    print(f"\nLoaded {len(data)} assets")
    print()

    # Initialize analyzers
    analyzer = CorrelationAnalyzer(data)
    lead_lag = LeadLagAnalyzer(analyzer.returns)
    beta_calc = BetaCalculator(analyzer.returns)

    # Correlation Matrix
    print("[2] CORRELATION MATRIX (Last 90 Days)")
    print("-" * 70)
    corr_matrix = analyzer.correlation_matrix(window=90)
    print(corr_matrix.round(2).to_string())
    print()

    # Correlation Regime
    print("[3] CORRELATION REGIME")
    print("-" * 70)
    regime = analyzer.detect_correlation_regime(window=30)
    print(f"  Regime: {regime.regime}")
    print(f"  Average Correlation: {regime.avg_correlation:.2f}")
    print(f"  Trend: {regime.trend}")
    print(f"  Confidence: {regime.confidence:.1%}")
    print()

    # Lead-Lag Analysis
    print("[4] LEAD-LAG ANALYSIS")
    print("-" * 70)

    # Analyze key pairs
    key_pairs = [
        ("BTCUSDT", "ETHUSDT"),
        ("BTCUSDT", "XRPUSDT"),
        ("BTCUSDT", "SOLUSDT"),
        ("BTCUSDT", "ATOMUSDT"),
        ("ETHUSDT", "SOLUSDT"),
    ]

    for asset1, asset2 in key_pairs:
        if asset1 in analyzer.returns.columns and asset2 in analyzer.returns.columns:
            result = lead_lag.find_lead_lag(asset1, asset2)
            if result.lag_periods > 0:
                print(f"  {result.leader} → {result.follower}")
                print(f"    Lag: {result.lag_periods} day(s)")
                print(f"    Correlation at lag: {result.correlation_at_lag:.2f}")
                print(f"    Confidence: {result.confidence:.1%}")
            else:
                print(f"  {asset1} ↔ {asset2}: No significant lead-lag relationship")
            print()

    # Beta Analysis
    print("[5] BETA ANALYSIS (vs BTC)")
    print("-" * 70)

    for asset in ["ETHUSDT", "XRPUSDT", "SOLUSDT", "ATOMUSDT"]:
        if asset in analyzer.returns.columns:
            beta_result = beta_calc.calculate_beta(asset, "BTCUSDT")
            print(f"  {asset}:")
            print(f"    Beta: {beta_result.beta:.2f}")
            print(f"    Alpha: {beta_result.alpha:.1%} (annualized)")
            print(f"    R²: {beta_result.r_squared:.2f}")
            print(f"    Volatility Ratio: {beta_result.volatility_ratio:.2f}x")
            print()

    # Diversification Score
    print("[6] DIVERSIFICATION ANALYSIS")
    print("-" * 70)

    scorer = DiversificationScorer(corr_matrix)

    # Analyze your portfolio (XRP, SOL, ATOM)
    your_holdings = ["XRPUSDT", "SOLUSDT", "ATOMUSDT"]
    div_score = scorer.calculate_score(your_holdings)

    print(f"  Your Portfolio: {', '.join(your_holdings)}")
    print(f"  Diversification Score: {div_score.score:.0f}/100")
    print(f"  Effective Assets: {div_score.effective_assets:.1f}")
    print(f"  Concentration Risk: {div_score.concentration_risk}")
    print()
    print("  Recommendations:")
    for rec in div_score.recommendations:
        print(f"    • {rec}")
    print()

    # Correlation Alerts
    print("[7] CORRELATION ALERTS")
    print("-" * 70)

    alert_checker = CorrelationAlerts(analyzer)
    alerts = alert_checker.scan_all_pairs()

    if alerts:
        for alert in alerts[:5]:  # Top 5 alerts
            print(f"  [{alert['severity']}] {alert['alert']}")
            print(f"    Pair: {alert['pair']}")
            print(f"    Current: {alert['current_correlation']:.2f}")
            print(f"    Historical Mean: {alert['historical_mean']:.2f}")
            print(f"    Z-Score: {alert['z_score']:.1f}")
            print()
    else:
        print("  No correlation alerts at this time")
    print()

    # Correlation Percentiles for Key Pairs
    print("[8] CORRELATION CONTEXT")
    print("-" * 70)

    for asset in ["XRPUSDT", "SOLUSDT", "ATOMUSDT"]:
        if asset in analyzer.returns.columns and "BTCUSDT" in analyzer.returns.columns:
            stats = analyzer.correlation_percentile(asset, "BTCUSDT")
            print(f"  {asset} vs BTC:")
            print(f"    Current Correlation: {stats['current_correlation']:.2f}")
            print(f"    Percentile (1yr): {stats['percentile']:.0f}%")
            print(f"    Z-Score: {stats['z_score']:.1f}")
            print()

    print("=" * 70)
    print("Analysis Complete!")
    print("=" * 70)


if __name__ == "__main__":
    run_correlation_analysis()
