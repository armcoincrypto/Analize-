#!/usr/bin/env python3
"""
Portfolio Optimizer - Multi-Coin Portfolio Allocation

Features:
1. Mean-Variance Optimization (Markowitz)
2. Risk Parity allocation
3. Maximum Sharpe Ratio portfolio
4. Minimum Volatility portfolio
5. Black-Litterman with views
6. Rebalancing recommendations
7. Risk metrics (VaR, CVaR, Max Drawdown)

Author: Analize Team
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import requests
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class AssetStats:
    """Statistics for a single asset."""
    symbol: str
    expected_return: float  # Annualized
    volatility: float  # Annualized
    sharpe_ratio: float
    max_drawdown: float
    var_95: float  # 95% Value at Risk
    cvar_95: float  # 95% Conditional VaR


@dataclass
class PortfolioAllocation:
    """Portfolio allocation result."""
    name: str
    weights: Dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    var_95: float
    max_drawdown_estimate: float


@dataclass
class RebalanceRecommendation:
    """Rebalancing recommendation."""
    asset: str
    current_weight: float
    target_weight: float
    action: str  # BUY, SELL, HOLD
    amount_change: float  # Percentage change
    priority: str  # HIGH, MEDIUM, LOW


@dataclass
class RiskMetrics:
    """Comprehensive risk metrics."""
    portfolio_var_95: float
    portfolio_cvar_95: float
    max_drawdown: float
    volatility: float
    downside_deviation: float
    sortino_ratio: float
    calmar_ratio: float
    beta_to_btc: float


# =============================================================================
# DATA FETCHER
# =============================================================================

class PortfolioDataFetcher:
    """Fetch data for portfolio optimization."""

    def __init__(self):
        self.cache = {}
        self.symbols = [
            "BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "ATOMUSDT",
            "ADAUSDT", "DOTUSDT", "AVAXUSDT", "LINKUSDT", "MATICUSDT"
        ]

    def fetch_all_prices(self, days: int = 365) -> pd.DataFrame:
        """Fetch price history for all assets."""
        prices = {}

        for symbol in self.symbols:
            df = self._fetch_prices(symbol, days)
            if not df.empty:
                prices[symbol] = df.set_index("timestamp")["close"]
                print(f"  Fetched {symbol}: {len(df)} days")

        if prices:
            return pd.DataFrame(prices)
        return pd.DataFrame()

    def _fetch_prices(self, symbol: str, days: int) -> pd.DataFrame:
        """Fetch prices for single symbol."""
        cache_key = f"{symbol}_{days}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Try Binance US first (works in geo-restricted regions)
        df = self._fetch_binance_us(symbol, days)
        if df.empty:
            df = self._fetch_binance(symbol, days)

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
                "limit": 1000
            }

            response = requests.get(
                "https://api.binance.com/api/v3/klines",
                params=params,
                timeout=15
            )

            if response.status_code == 200:
                data = response.json()
                if data:
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

    def _fetch_binance_us(self, symbol: str, days: int) -> pd.DataFrame:
        """Fetch from Binance US."""
        try:
            params = {
                "symbol": symbol,
                "interval": "1d",
                "startTime": int((datetime.now() - timedelta(days=days)).timestamp() * 1000),
                "limit": 1000
            }

            response = requests.get(
                "https://api.binance.us/api/v3/klines",
                params=params,
                timeout=15
            )

            if response.status_code == 200:
                data = response.json()
                if data:
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


# =============================================================================
# ASSET STATISTICS
# =============================================================================

class AssetStatisticsCalculator:
    """Calculate statistics for assets."""

    def __init__(self, prices: pd.DataFrame, risk_free_rate: float = 0.05):
        self.prices = prices
        self.returns = prices.pct_change().dropna()
        self.risk_free_rate = risk_free_rate

    def calculate_stats(self, symbol: str) -> Optional[AssetStats]:
        """Calculate statistics for a single asset."""
        if symbol not in self.returns.columns:
            return None

        returns = self.returns[symbol].dropna()

        if len(returns) < 30:
            return None

        # Annualized metrics
        expected_return = returns.mean() * 252
        volatility = returns.std() * np.sqrt(252)

        # Sharpe ratio
        sharpe = (expected_return - self.risk_free_rate) / volatility if volatility > 0 else 0

        # Max drawdown
        prices = self.prices[symbol].dropna()
        rolling_max = prices.expanding().max()
        drawdowns = (prices - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()

        # Value at Risk (95%)
        var_95 = np.percentile(returns, 5)

        # Conditional VaR (Expected Shortfall)
        cvar_95 = returns[returns <= var_95].mean()

        return AssetStats(
            symbol=symbol,
            expected_return=expected_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            var_95=var_95,
            cvar_95=cvar_95
        )

    def get_correlation_matrix(self) -> pd.DataFrame:
        """Get correlation matrix."""
        return self.returns.corr()

    def get_covariance_matrix(self) -> pd.DataFrame:
        """Get annualized covariance matrix."""
        return self.returns.cov() * 252


# =============================================================================
# PORTFOLIO OPTIMIZER
# =============================================================================

class PortfolioOptimizer:
    """Optimize portfolio allocations."""

    def __init__(self, returns: pd.DataFrame, risk_free_rate: float = 0.05):
        self.returns = returns
        self.mean_returns = returns.mean() * 252  # Annualized
        self.cov_matrix = returns.cov() * 252  # Annualized
        self.risk_free_rate = risk_free_rate
        self.assets = list(returns.columns)
        self.n_assets = len(self.assets)

    def equal_weight(self) -> PortfolioAllocation:
        """Equal weight allocation."""
        weights = {asset: 1.0 / self.n_assets for asset in self.assets}
        return self._create_allocation("Equal Weight", weights)

    def max_sharpe(self, max_weight: float = 0.20) -> PortfolioAllocation:
        """
        Maximum Sharpe ratio portfolio with per-asset constraints.

        Args:
            max_weight: Maximum weight per asset (default 20% to prevent overconcentration)
        """
        def neg_sharpe(weights):
            port_return = np.dot(weights, self.mean_returns)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            if port_vol == 0:
                return 0
            return -(port_return - self.risk_free_rate) / port_vol

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        # CRITICAL: Cap per-asset allocation to prevent 100% single-asset recommendations
        # In crypto, unconstrained optimization often produces unstable corner solutions
        bounds = tuple((0.01, max_weight) for _ in range(self.n_assets))  # Min 1%, Max 20%
        initial = np.array([1.0 / self.n_assets] * self.n_assets)

        result = minimize(neg_sharpe, initial, method='SLSQP',
                          bounds=bounds, constraints=constraints)

        weights = {asset: max(0.01, min(max_weight, w)) for asset, w in zip(self.assets, result.x)}
        # Renormalize in case of numerical issues
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
        return self._create_allocation(f"Max Sharpe (capped {max_weight:.0%})", weights)

    def min_volatility(self, max_weight: float = 0.30) -> PortfolioAllocation:
        """
        Minimum volatility portfolio with per-asset constraints.

        Args:
            max_weight: Maximum weight per asset (default 30% for min-vol)
        """
        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        # Cap per-asset allocation to prevent 100% single-asset recommendations
        bounds = tuple((0.01, max_weight) for _ in range(self.n_assets))
        initial = np.array([1.0 / self.n_assets] * self.n_assets)

        result = minimize(portfolio_volatility, initial, method='SLSQP',
                          bounds=bounds, constraints=constraints)

        weights = {asset: max(0.01, min(max_weight, w)) for asset, w in zip(self.assets, result.x)}
        # Renormalize in case of numerical issues
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
        return self._create_allocation(f"Min Volatility (capped {max_weight:.0%})", weights)

    def risk_parity(self) -> PortfolioAllocation:
        """Risk parity allocation (equal risk contribution)."""
        def risk_budget_objective(weights):
            port_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            marginal_contrib = np.dot(self.cov_matrix, weights)
            risk_contrib = weights * marginal_contrib / port_vol

            # Target equal risk contribution
            target_risk = port_vol / self.n_assets
            return np.sum((risk_contrib - target_risk) ** 2)

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0.01, 1) for _ in range(self.n_assets))  # Min 1% per asset
        initial = np.array([1.0 / self.n_assets] * self.n_assets)

        result = minimize(risk_budget_objective, initial, method='SLSQP',
                          bounds=bounds, constraints=constraints)

        weights = {asset: w for asset, w in zip(self.assets, result.x)}
        return self._create_allocation("Risk Parity", weights)

    def target_return(self, target: float) -> PortfolioAllocation:
        """Portfolio with target return and minimum volatility."""
        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))

        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: np.dot(x, self.mean_returns) - target}
        ]
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        initial = np.array([1.0 / self.n_assets] * self.n_assets)

        result = minimize(portfolio_volatility, initial, method='SLSQP',
                          bounds=bounds, constraints=constraints)

        weights = {asset: w for asset, w in zip(self.assets, result.x)}
        return self._create_allocation(f"Target {target:.0%} Return", weights)

    def efficient_frontier(self, n_points: int = 20) -> List[PortfolioAllocation]:
        """Generate efficient frontier."""
        # Find min and max returns
        min_ret = self.mean_returns.min()
        max_ret = self.mean_returns.max()

        # Generate target returns
        target_returns = np.linspace(min_ret * 1.1, max_ret * 0.9, n_points)

        frontier = []
        for target in target_returns:
            try:
                allocation = self.target_return(target)
                frontier.append(allocation)
            except Exception:
                pass

        return frontier

    def _create_allocation(self, name: str, weights: Dict[str, float]) -> PortfolioAllocation:
        """Create portfolio allocation with metrics."""
        w = np.array([weights[a] for a in self.assets])

        expected_return = np.dot(w, self.mean_returns)
        volatility = np.sqrt(np.dot(w.T, np.dot(self.cov_matrix, w)))
        sharpe = (expected_return - self.risk_free_rate) / volatility if volatility > 0 else 0

        # Portfolio VaR (parametric)
        var_95 = expected_return / 252 - 1.645 * volatility / np.sqrt(252)

        # Estimate max drawdown from volatility
        max_dd_estimate = -2.5 * volatility  # Rough estimate

        return PortfolioAllocation(
            name=name,
            weights={k: round(v, 4) for k, v in weights.items() if v > 0.001},
            expected_return=expected_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            var_95=var_95,
            max_drawdown_estimate=max_dd_estimate
        )


# =============================================================================
# BLACK-LITTERMAN MODEL
# =============================================================================

class BlackLittermanOptimizer:
    """Black-Litterman portfolio optimization with views."""

    def __init__(self, returns: pd.DataFrame, market_caps: Optional[Dict[str, float]] = None,
                 risk_free_rate: float = 0.05, risk_aversion: float = 2.5):
        self.returns = returns
        self.assets = list(returns.columns)
        self.cov_matrix = returns.cov() * 252
        self.risk_free_rate = risk_free_rate
        self.risk_aversion = risk_aversion

        # Market cap weights (if provided)
        if market_caps:
            total_cap = sum(market_caps.get(a, 1) for a in self.assets)
            self.market_weights = np.array([market_caps.get(a, 1) / total_cap for a in self.assets])
        else:
            self.market_weights = np.array([1.0 / len(self.assets)] * len(self.assets))

        # Calculate equilibrium returns (CAPM implied)
        self.equilibrium_returns = self._calculate_equilibrium_returns()

    def _calculate_equilibrium_returns(self) -> np.ndarray:
        """Calculate equilibrium returns from market weights."""
        return self.risk_aversion * np.dot(self.cov_matrix, self.market_weights)

    def optimize_with_views(self, views: List[Dict]) -> PortfolioAllocation:
        """
        Optimize with investor views.

        views: List of dicts with:
            - 'asset': asset symbol
            - 'return': expected return (absolute)
            - 'confidence': 0-1 (1 = very confident)
        """
        if not views:
            # No views - return equilibrium portfolio
            weights = {a: w for a, w in zip(self.assets, self.market_weights)}
            return self._create_bl_allocation("Black-Litterman (No Views)", weights)

        # Build view matrices
        n_views = len(views)
        P = np.zeros((n_views, len(self.assets)))  # Pick matrix
        Q = np.zeros(n_views)  # View vector
        omega_diag = []  # Uncertainty matrix diagonal

        for i, view in enumerate(views):
            asset_idx = self.assets.index(view['asset']) if view['asset'] in self.assets else None
            if asset_idx is not None:
                P[i, asset_idx] = 1
                Q[i] = view['return']
                # Uncertainty inversely related to confidence
                tau_sigma = self.cov_matrix.iloc[asset_idx, asset_idx] * 0.05  # tau = 0.05
                omega_diag.append(tau_sigma / view['confidence'])

        Omega = np.diag(omega_diag)
        tau = 0.05

        # Black-Litterman formula
        tau_cov = tau * self.cov_matrix.values
        inv_tau_cov = np.linalg.inv(tau_cov)
        inv_omega = np.linalg.inv(Omega)

        # Posterior expected returns
        M1 = np.linalg.inv(inv_tau_cov + P.T @ inv_omega @ P)
        M2 = inv_tau_cov @ self.equilibrium_returns + P.T @ inv_omega @ Q
        posterior_returns = M1 @ M2

        # Posterior covariance
        posterior_cov = tau_cov + M1

        # Optimize using posterior
        def neg_sharpe(weights):
            port_return = np.dot(weights, posterior_returns)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(posterior_cov, weights)))
            return -(port_return - self.risk_free_rate) / port_vol

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(len(self.assets)))
        initial = self.market_weights

        result = minimize(neg_sharpe, initial, method='SLSQP',
                          bounds=bounds, constraints=constraints)

        weights = {a: w for a, w in zip(self.assets, result.x)}
        return self._create_bl_allocation("Black-Litterman (With Views)", weights, posterior_returns)

    def _create_bl_allocation(self, name: str, weights: Dict[str, float],
                              returns: Optional[np.ndarray] = None) -> PortfolioAllocation:
        """Create Black-Litterman allocation."""
        w = np.array([weights[a] for a in self.assets])

        if returns is None:
            returns = self.equilibrium_returns

        expected_return = np.dot(w, returns)
        volatility = np.sqrt(np.dot(w.T, np.dot(self.cov_matrix, w)))
        sharpe = (expected_return - self.risk_free_rate) / volatility if volatility > 0 else 0
        var_95 = expected_return / 252 - 1.645 * volatility / np.sqrt(252)

        return PortfolioAllocation(
            name=name,
            weights={k: round(v, 4) for k, v in weights.items() if v > 0.001},
            expected_return=expected_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            var_95=var_95,
            max_drawdown_estimate=-2.5 * volatility
        )


# =============================================================================
# REBALANCING ADVISOR
# =============================================================================

class RebalancingAdvisor:
    """Generate rebalancing recommendations."""

    def __init__(self, target_weights: Dict[str, float], threshold: float = 0.05):
        self.target_weights = target_weights
        self.threshold = threshold

    def get_recommendations(self, current_weights: Dict[str, float]) -> List[RebalanceRecommendation]:
        """Generate rebalancing recommendations."""
        recommendations = []

        all_assets = set(self.target_weights.keys()) | set(current_weights.keys())

        for asset in all_assets:
            current = current_weights.get(asset, 0)
            target = self.target_weights.get(asset, 0)

            diff = target - current

            # Determine action
            if abs(diff) < 0.01:  # Less than 1% difference
                action = "HOLD"
                priority = "LOW"
            elif diff > self.threshold:
                action = "BUY"
                priority = "HIGH" if diff > self.threshold * 2 else "MEDIUM"
            elif diff < -self.threshold:
                action = "SELL"
                priority = "HIGH" if abs(diff) > self.threshold * 2 else "MEDIUM"
            else:
                action = "HOLD"
                priority = "LOW"

            recommendations.append(RebalanceRecommendation(
                asset=asset,
                current_weight=current,
                target_weight=target,
                action=action,
                amount_change=diff,
                priority=priority
            ))

        # Sort by priority and absolute change
        priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        recommendations.sort(key=lambda x: (priority_order[x.priority], -abs(x.amount_change)))

        return recommendations


# =============================================================================
# RISK METRICS CALCULATOR
# =============================================================================

class RiskMetricsCalculator:
    """Calculate comprehensive risk metrics."""

    def __init__(self, returns: pd.DataFrame, weights: Dict[str, float],
                 risk_free_rate: float = 0.05):
        self.returns = returns
        self.weights = weights
        self.risk_free_rate = risk_free_rate

        # Calculate portfolio returns
        w = np.array([weights.get(col, 0) for col in returns.columns])
        self.portfolio_returns = returns.values @ w

    def calculate_metrics(self) -> RiskMetrics:
        """Calculate all risk metrics."""
        returns = self.portfolio_returns

        # VaR and CVaR
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()

        # Volatility
        volatility = returns.std() * np.sqrt(252)

        # Downside deviation
        negative_returns = returns[returns < 0]
        downside_dev = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0

        # Sortino ratio
        annual_return = returns.mean() * 252
        sortino = (annual_return - self.risk_free_rate) / downside_dev if downside_dev > 0 else 0

        # Max drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()

        # Calmar ratio
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Beta to BTC
        if "BTCUSDT" in self.returns.columns:
            btc_returns = self.returns["BTCUSDT"].values
            cov = np.cov(returns, btc_returns)[0, 1]
            var_btc = np.var(btc_returns)
            beta = cov / var_btc if var_btc > 0 else 1
        else:
            beta = 1

        return RiskMetrics(
            portfolio_var_95=var_95,
            portfolio_cvar_95=cvar_95,
            max_drawdown=max_drawdown,
            volatility=volatility,
            downside_deviation=downside_dev,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            beta_to_btc=beta
        )


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_portfolio_optimization():
    """Run full portfolio optimization."""
    print("=" * 70)
    print("PORTFOLIO OPTIMIZER - Multi-Coin Allocation")
    print("=" * 70)
    print()

    # Fetch data
    print("[1] FETCHING MARKET DATA...")
    print("-" * 70)
    fetcher = PortfolioDataFetcher()
    prices = fetcher.fetch_all_prices(days=365)

    if prices.empty or len(prices.columns) < 3:
        print("Error: Insufficient data for optimization")
        return

    print(f"\nLoaded {len(prices.columns)} assets, {len(prices)} days of data")
    print()

    # Calculate returns
    returns = prices.pct_change().dropna()

    # Asset Statistics
    print("[2] INDIVIDUAL ASSET STATISTICS")
    print("-" * 70)

    stats_calc = AssetStatisticsCalculator(prices)

    # Focus on target coins
    target_coins = ["XRPUSDT", "SOLUSDT", "ATOMUSDT", "BTCUSDT", "ETHUSDT"]

    for symbol in target_coins:
        stats = stats_calc.calculate_stats(symbol)
        if stats:
            print(f"  {symbol}:")
            print(f"    Expected Return: {stats.expected_return:+.1%}")
            print(f"    Volatility: {stats.volatility:.1%}")
            print(f"    Sharpe Ratio: {stats.sharpe_ratio:.2f}")
            print(f"    Max Drawdown: {stats.max_drawdown:.1%}")
            print(f"    VaR (95%): {stats.var_95:.2%}")
            print()

    # Correlation Matrix
    print("[3] CORRELATION MATRIX")
    print("-" * 70)
    corr = stats_calc.get_correlation_matrix()
    # Show for target coins only
    target_corr = corr.loc[
        [c for c in target_coins if c in corr.index],
        [c for c in target_coins if c in corr.columns]
    ]
    print(target_corr.round(2).to_string())
    print()

    # Portfolio Optimization
    print("[4] PORTFOLIO OPTIMIZATION")
    print("-" * 70)

    # Filter to target coins for optimization
    target_returns = returns[[c for c in target_coins if c in returns.columns]]
    optimizer = PortfolioOptimizer(target_returns)

    # Generate different portfolios
    portfolios = [
        optimizer.equal_weight(),
        optimizer.max_sharpe(),
        optimizer.min_volatility(),
        optimizer.risk_parity(),
    ]

    for portfolio in portfolios:
        print(f"\n  {portfolio.name}:")
        print(f"    Expected Return: {portfolio.expected_return:+.1%}")
        print(f"    Volatility: {portfolio.volatility:.1%}")
        print(f"    Sharpe Ratio: {portfolio.sharpe_ratio:.2f}")
        print(f"    VaR (95%): {portfolio.var_95:.2%}")
        print(f"    Weights:")
        for asset, weight in sorted(portfolio.weights.items(), key=lambda x: -x[1]):
            if weight > 0.01:
                bar_len = int(weight * 20)
                print(f"      {asset}: {'█' * bar_len}{'░' * (20-bar_len)} {weight:.1%}")

    # Black-Litterman with Views
    print("\n[5] BLACK-LITTERMAN OPTIMIZATION")
    print("-" * 70)

    # Market caps (approximate, in billions)
    market_caps = {
        "BTCUSDT": 1800,
        "ETHUSDT": 400,
        "XRPUSDT": 130,
        "SOLUSDT": 100,
        "ATOMUSDT": 4,
    }

    bl_optimizer = BlackLittermanOptimizer(target_returns, market_caps)

    # Example views based on analysis
    views = [
        {"asset": "XRPUSDT", "return": 0.50, "confidence": 0.7},  # Bullish on XRP
        {"asset": "SOLUSDT", "return": 0.40, "confidence": 0.6},  # Bullish on SOL
        {"asset": "ATOMUSDT", "return": 0.30, "confidence": 0.5},  # Moderate on ATOM
    ]

    bl_portfolio = bl_optimizer.optimize_with_views(views)

    print(f"\n  {bl_portfolio.name}:")
    print(f"    Views Applied:")
    for view in views:
        print(f"      • {view['asset']}: {view['return']:+.0%} expected (confidence: {view['confidence']:.0%})")

    print(f"\n    Expected Return: {bl_portfolio.expected_return:+.1%}")
    print(f"    Volatility: {bl_portfolio.volatility:.1%}")
    print(f"    Sharpe Ratio: {bl_portfolio.sharpe_ratio:.2f}")
    print(f"    Weights:")
    for asset, weight in sorted(bl_portfolio.weights.items(), key=lambda x: -x[1]):
        if weight > 0.01:
            bar_len = int(weight * 20)
            print(f"      {asset}: {'█' * bar_len}{'░' * (20-bar_len)} {weight:.1%}")

    # Rebalancing Recommendations
    print("\n[6] REBALANCING RECOMMENDATIONS")
    print("-" * 70)

    # Assume current equal weight portfolio
    current_weights = {asset: 1.0 / len(target_coins) for asset in target_coins if asset in returns.columns}

    # Use max sharpe as target
    target_portfolio = optimizer.max_sharpe()
    advisor = RebalancingAdvisor(target_portfolio.weights)

    recommendations = advisor.get_recommendations(current_weights)

    print("  From Equal Weight to Max Sharpe Portfolio:")
    for rec in recommendations:
        if rec.action != "HOLD" or abs(rec.amount_change) > 0.01:
            emoji = "🟢" if rec.action == "BUY" else "🔴" if rec.action == "SELL" else "⚪"
            print(f"    {emoji} {rec.asset}: {rec.action}")
            print(f"       Current: {rec.current_weight:.1%} → Target: {rec.target_weight:.1%}")
            print(f"       Change: {rec.amount_change:+.1%} [{rec.priority}]")

    # Risk Metrics for recommended portfolio
    print("\n[7] RISK METRICS (Max Sharpe Portfolio)")
    print("-" * 70)

    risk_calc = RiskMetricsCalculator(target_returns, target_portfolio.weights)
    risk_metrics = risk_calc.calculate_metrics()

    print(f"  Daily VaR (95%): {risk_metrics.portfolio_var_95:.2%}")
    print(f"  Daily CVaR (95%): {risk_metrics.portfolio_cvar_95:.2%}")
    print(f"  Max Drawdown: {risk_metrics.max_drawdown:.1%}")
    print(f"  Volatility: {risk_metrics.volatility:.1%}")
    print(f"  Downside Deviation: {risk_metrics.downside_deviation:.1%}")
    print(f"  Sortino Ratio: {risk_metrics.sortino_ratio:.2f}")
    print(f"  Calmar Ratio: {risk_metrics.calmar_ratio:.2f}")
    print(f"  Beta to BTC: {risk_metrics.beta_to_btc:.2f}")

    # Summary
    print("\n" + "=" * 70)
    print("RECOMMENDED PORTFOLIO ALLOCATION")
    print("=" * 70)

    # Recommend max sharpe or black-litterman based on views
    best_portfolio = bl_portfolio if bl_portfolio.sharpe_ratio > target_portfolio.sharpe_ratio else target_portfolio

    print(f"\n  Recommended: {best_portfolio.name}")
    print(f"  Expected Return: {best_portfolio.expected_return:+.1%}")
    print(f"  Volatility: {best_portfolio.volatility:.1%}")
    print(f"  Sharpe Ratio: {best_portfolio.sharpe_ratio:.2f}")
    print("\n  Allocation:")

    for asset, weight in sorted(best_portfolio.weights.items(), key=lambda x: -x[1]):
        if weight > 0.01:
            print(f"    {asset}: {weight:.1%}")

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)


if __name__ == "__main__":
    run_portfolio_optimization()
