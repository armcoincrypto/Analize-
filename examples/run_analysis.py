#!/usr/bin/env python3
"""
Master Analysis Runner - Connects All Tools to Quant Database

Runs all 6 analysis tools and stores results in the self-learning database.
This creates a unified view of all signals with weighted consensus.

Usage:
    python examples/run_analysis.py
    python examples/run_analysis.py --symbols BTC ETH XRP SOL
    python examples/run_analysis.py --db-path my_signals.db

Author: Cloud AI Analyzer
"""

import sys
import os
import argparse
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the quant database
from examples.quant_database import (
    QuantDatabase, TechnicalSignal, MLPrediction,
    WhaleActivity, PriceData, Trade
)

# Import analysis tools - using proper class names
HAS_CORRELATION = False
HAS_WHALE = False
HAS_FUNDING = False
HAS_ORDERFLOW = False
HAS_ML = False
HAS_AGGREGATOR = False

try:
    from examples.correlation_analyzer import (
        MultiAssetDataFetcher, CorrelationAnalyzer,
        LeadLagAnalyzer, BetaCalculator, DiversificationScorer
    )
    HAS_CORRELATION = True
except ImportError as e:
    print(f"[WARN] correlation_analyzer not available: {e}")

try:
    from examples.whale_tracker import WhaleTracker, WhaleAlertAggregator
    HAS_WHALE = True
except ImportError as e:
    print(f"[WARN] whale_tracker not available: {e}")

try:
    from examples.funding_rate_analyzer import FundingRateAnalyzer
    HAS_FUNDING = True
except ImportError as e:
    print(f"[WARN] funding_rate_analyzer not available: {e}")

try:
    from examples.order_flow_analyzer import (
        OrderBookFetcher, OrderBookAnalyzer, OrderFlowSignalGenerator
    )
    HAS_ORDERFLOW = True
except ImportError as e:
    print(f"[WARN] order_flow_analyzer not available: {e}")

try:
    from examples.ml_signal_generator import MLSignalGenerator, MLDataFetcher
    HAS_ML = True
except ImportError as e:
    print(f"[WARN] ml_signal_generator not available: {e}")

try:
    from examples.signal_aggregator import (
        SignalAggregator, CircuitBreaker, SignalAuditTable
    )
    HAS_AGGREGATOR = True
except ImportError as e:
    print(f"[WARN] signal_aggregator not available: {e}")

# Portfolio Optimizer
HAS_PORTFOLIO = False
try:
    from examples.portfolio_optimizer import (
        PortfolioDataFetcher, AssetStatisticsCalculator,
        PortfolioOptimizer, BlackLittermanOptimizer, RiskMetricsCalculator
    )
    HAS_PORTFOLIO = True
except ImportError as e:
    print(f"[WARN] portfolio_optimizer not available: {e}")

# Market Analyzer (RSI, MACD, Bollinger)
HAS_MARKET = False
try:
    from examples.market_analyzer import MarketAnalyzer
    HAS_MARKET = True
except ImportError as e:
    print(f"[WARN] market_analyzer not available: {e}")


class MasterAnalyzer:
    """
    Master analyzer that runs all tools and stores results in database.

    This creates a unified trading intelligence system that:
    1. Runs all analysis tools on specified symbols
    2. Stores all signals in the quant database
    3. Calculates weighted consensus signals
    4. Tracks signal performance over time
    5. Detects market regime and adjusts weights
    """

    def __init__(self, db_path: str = "quant_signals.db",
                 symbols: Optional[List[str]] = None):
        """
        Initialize the master analyzer.

        Args:
            db_path: Path to SQLite database
            symbols: List of symbols to analyze (default: BTC, ETH, XRP, SOL, ATOM)
        """
        self.db = QuantDatabase(db_path)
        self.symbols = symbols or ["BTC", "ETH", "XRP", "SOL", "ATOM"]
        self.timestamp = datetime.utcnow().isoformat() + "Z"

        # Results storage
        self.results = {
            "correlation": {},
            "whale": {},
            "funding": {},
            "orderflow": {},
            "ml": {},
            "technical": {},  # RSI, MACD, Bollinger
            "portfolio": {},  # Optimal allocations
            "aggregated": {}
        }

        # Data cache (fetched once, used by multiple analyzers)
        self._price_data = None

        print(f"[MasterAnalyzer] Initialized with {len(self.symbols)} symbols")
        print(f"[MasterAnalyzer] Database: {db_path}")

    def run_correlation_analysis(self) -> Dict[str, Any]:
        """Run correlation analyzer and store results."""
        if not HAS_CORRELATION:
            print("\n[SKIP] Correlation analysis - module not available")
            return {}

        print("\n" + "=" * 60)
        print("CORRELATION ANALYSIS")
        print("=" * 60)

        try:
            # Fetch data using MultiAssetDataFetcher
            print("  Fetching price data...")
            fetcher = MultiAssetDataFetcher()
            data = fetcher.fetch_all(days=90)

            if len(data) < 2:
                print("  [ERROR] Not enough data for correlation analysis")
                return {}

            # Cache for other analyzers
            self._price_data = data

            # Create correlation analyzer with fetched data
            analyzer = CorrelationAnalyzer(data)

            # Get correlation matrix
            corr_matrix = analyzer.correlation_matrix(window=30)

            # Detect regime
            regime = analyzer.detect_correlation_regime(window=30)
            current_regime = regime.regime if regime else "UNKNOWN"

            # Store regime
            self.db.record_market_regime(
                regime=current_regime,
                btc_price=0,
                volatility=0,
                correlation_avg=regime.avg_correlation if regime else 0,
                confidence=regime.confidence if regime else 0.5
            )
            print(f"  Regime: {current_regime} (avg corr: {regime.avg_correlation:.2f})")

            # Store correlation signals for each symbol
            for symbol in self.symbols:
                symbol_key = f"{symbol}USDT"

                # Get BTC correlation for this symbol
                btc_corr = 0
                if symbol_key in corr_matrix.columns and "BTCUSDT" in corr_matrix.index:
                    btc_corr = corr_matrix.loc["BTCUSDT", symbol_key]
                elif symbol_key == "BTCUSDT":
                    btc_corr = 1.0

                # Determine signal based on correlation
                if abs(btc_corr) > 0.8:
                    direction = "NEUTRAL"  # High correlation = follows BTC
                    confidence = 0.6
                elif btc_corr < 0.3:
                    direction = "BUY"  # Low correlation = diversification opportunity
                    confidence = 0.7
                else:
                    direction = "NEUTRAL"
                    confidence = 0.5

                signal = TechnicalSignal(
                    symbol=symbol_key,
                    timestamp=self.timestamp,
                    signal_name="correlation_btc",
                    signal_value=float(btc_corr) if btc_corr else 0,
                    signal_direction=direction,
                    confidence=confidence,
                    timeframe="1d",
                    parameters=json.dumps({"regime": current_regime})
                )
                self.db.insert_technical_signal(signal)

                self.results["correlation"][symbol] = {
                    "btc_correlation": float(btc_corr) if btc_corr else 0,
                    "direction": direction,
                    "confidence": confidence
                }
                print(f"  {symbol}: BTC correlation {btc_corr:.3f} -> {direction}")

            return {"regime": current_regime, "matrix": corr_matrix.to_dict()}
        except Exception as e:
            print(f"  [ERROR] Correlation analysis failed: {e}")
            import traceback
            traceback.print_exc()

        return {}

    def run_whale_analysis(self) -> Dict[str, Any]:
        """Run whale tracker and store results."""
        if not HAS_WHALE:
            print("\n[SKIP] Whale analysis - module not available")
            return {}

        print("\n" + "=" * 60)
        print("WHALE ACTIVITY ANALYSIS")
        print("=" * 60)

        try:
            aggregator = WhaleAlertAggregator()

            for symbol in self.symbols:
                # Use exchange flow analysis
                flow = aggregator._analyze_exchange_flows(symbol)

                if flow:
                    net_flow = flow.net_flow_24h

                    # Determine signal based on net flow
                    # Outflow (negative) = bullish (coins leaving exchanges)
                    # Inflow (positive) = bearish (coins entering exchanges)
                    if net_flow < -1000000:  # Large outflow
                        direction = "BUY"
                        confidence = min(abs(net_flow) / 50000000, 0.85)
                    elif net_flow > 1000000:  # Large inflow
                        direction = "SELL"
                        confidence = min(abs(net_flow) / 50000000, 0.85)
                    else:
                        direction = "NEUTRAL"
                        confidence = 0.5

                    activity = WhaleActivity(
                        symbol=f"{symbol}USDT",
                        timestamp=self.timestamp,
                        activity_type="exchange_flow",
                        amount_usd=abs(net_flow) if net_flow else 0,
                        signal_direction=direction,
                        confidence=confidence,
                        exchange="aggregate"
                    )
                    self.db.insert_whale_activity(activity)

                    flow_dir = "OUTFLOW" if net_flow < 0 else "INFLOW"
                    self.results["whale"][symbol] = {
                        "net_flow": net_flow,
                        "direction": direction,
                        "confidence": confidence,
                        "signal": flow_dir
                    }
                    print(f"  {symbol}: ${abs(net_flow):,.0f} {flow_dir} -> {direction}")
                else:
                    print(f"  {symbol}: No exchange flow data")
        except Exception as e:
            print(f"  [ERROR] Whale analysis failed: {e}")
            import traceback
            traceback.print_exc()

        return self.results["whale"]

    def run_funding_analysis(self) -> Dict[str, Any]:
        """Run funding rate analyzer and store results."""
        if not HAS_FUNDING:
            print("\n[SKIP] Funding analysis - module not available")
            return {}

        print("\n" + "=" * 60)
        print("FUNDING RATE ANALYSIS")
        print("=" * 60)

        try:
            analyzer = FundingRateAnalyzer()

            for symbol in self.symbols:
                # API needs full symbol with USDT suffix
                result = analyzer.analyze_symbol(f"{symbol}USDT")

                if result and result.current_rate is not None:
                    rate = result.current_rate

                    # Negative funding = shorts paying = bullish
                    # Positive funding = longs paying = bearish
                    if rate < -0.01:
                        direction = "STRONG_BUY"
                        confidence = min(abs(rate) * 100, 0.9)
                    elif rate < 0:
                        direction = "BUY"
                        confidence = 0.6
                    elif rate > 0.02:
                        direction = "STRONG_SELL"
                        confidence = min(rate * 100, 0.9)
                    elif rate > 0:
                        direction = "SELL"
                        confidence = 0.6
                    else:
                        direction = "NEUTRAL"
                        confidence = 0.5

                    signal = TechnicalSignal(
                        symbol=f"{symbol}USDT",
                        timestamp=self.timestamp,
                        signal_name="funding_rate",
                        signal_value=rate,
                        signal_direction=direction,
                        confidence=confidence,
                        timeframe="8h",
                        parameters=json.dumps({"signal": result.signal})
                    )
                    self.db.insert_technical_signal(signal)

                    self.results["funding"][symbol] = {
                        "rate": rate,
                        "direction": direction,
                        "confidence": confidence
                    }
                    print(f"  {symbol}: Funding {rate:.4%} -> {direction}")
                else:
                    print(f"  {symbol}: Unable to fetch funding rate")
        except Exception as e:
            print(f"  [ERROR] Funding analysis failed: {e}")
            import traceback
            traceback.print_exc()

        return self.results["funding"]

    def run_orderflow_analysis(self) -> Dict[str, Any]:
        """Run order flow analyzer and store results."""
        if not HAS_ORDERFLOW:
            print("\n[SKIP] Order flow analysis - module not available")
            return {}

        print("\n" + "=" * 60)
        print("ORDER FLOW ANALYSIS")
        print("=" * 60)

        try:
            fetcher = OrderBookFetcher()
            signal_gen = OrderFlowSignalGenerator()

            for symbol in self.symbols:
                # Fetch orderbook
                orderbook = fetcher.fetch_orderbook(f"{symbol}USDT")
                if not orderbook:
                    print(f"  {symbol}: Unable to fetch orderbook")
                    continue

                # Analyze
                analyzer = OrderBookAnalyzer(orderbook)
                imbalance = analyzer.calculate_imbalance()
                signal = signal_gen.generate_signal(orderbook)

                if signal:
                    # OrderFlowSignal has .signal (STRONG_BUY, BUY, etc.), .confidence, .reasons
                    direction = signal.signal  # This is the direction
                    confidence = signal.confidence
                    reasons = signal.reasons

                    tech_signal = TechnicalSignal(
                        symbol=f"{symbol}USDT",
                        timestamp=self.timestamp,
                        signal_name="orderflow",
                        signal_value=imbalance.imbalance_ratio if imbalance else 0,
                        signal_direction=direction,
                        confidence=confidence,
                        timeframe="1h",
                        parameters=json.dumps({"reasons": reasons[:2] if reasons else []})
                    )
                    self.db.insert_technical_signal(tech_signal)

                    reason_str = reasons[0] if reasons else "No specific reason"
                    self.results["orderflow"][symbol] = {
                        "imbalance": imbalance.imbalance_ratio if imbalance else 0,
                        "direction": direction,
                        "confidence": confidence,
                        "signal": direction
                    }
                    print(f"  {symbol}: {direction} (conf: {confidence:.2f}) - {reason_str}")
                else:
                    print(f"  {symbol}: No signal generated")
        except Exception as e:
            print(f"  [ERROR] Order flow analysis failed: {e}")
            import traceback
            traceback.print_exc()

        return self.results["orderflow"]

    def run_ml_analysis(self) -> Dict[str, Any]:
        """Run ML signal generator and store results."""
        if not HAS_ML:
            print("\n[SKIP] ML analysis - module not available")
            return {}

        print("\n" + "=" * 60)
        print("MACHINE LEARNING ANALYSIS")
        print("=" * 60)

        try:
            generator = MLSignalGenerator()

            for symbol in self.symbols:
                # API needs full symbol with USDT suffix
                full_symbol = f"{symbol}USDT"

                # CRITICAL: Must train models before generating signals
                print(f"  {symbol}: Training ML models (365 days)...")
                performance = generator.train_models(full_symbol, lookback_days=365, prediction_horizon=5)

                if not performance:
                    print(f"  {symbol}: Unable to train models (insufficient data)")
                    continue

                # Show model performance
                for model_name, perf in performance.items():
                    print(f"    {model_name}: accuracy={perf.accuracy:.1%}")

                # Now generate signal with trained models
                result = generator.generate_signal(full_symbol)

                if result:
                    # MLSignal has: .prediction (BUY/SELL/HOLD), .confidence,
                    # .probability_up, .probability_down, .model_name
                    direction = result.prediction
                    confidence = result.confidence
                    # Calculate expected return from probabilities
                    predicted_return = (result.probability_up - result.probability_down) * 10

                    pred = MLPrediction(
                        symbol=f"{symbol}USDT",
                        timestamp=self.timestamp,
                        model_name=result.model_name,
                        predicted_direction=direction,
                        predicted_return=predicted_return,
                        confidence=confidence,
                        feature_importance=json.dumps(
                            {"features_used": result.features_used}
                        ),
                        model_version="1.0",
                        validation_score=list(performance.values())[0].accuracy if performance else None
                    )
                    self.db.insert_ml_prediction(pred)

                    self.results["ml"][symbol] = {
                        "direction": direction,
                        "confidence": confidence,
                        "predicted_return": predicted_return,
                        "model": result.model_name
                    }
                    print(f"  {symbol}: {direction} (conf: {confidence:.2f}, "
                          f"prob_up: {result.probability_up:.1%})")
                else:
                    print(f"  {symbol}: Unable to generate ML signal")
        except Exception as e:
            print(f"  [ERROR] ML analysis failed: {e}")
            import traceback
            traceback.print_exc()

        return self.results["ml"]

    def run_technical_analysis(self) -> Dict[str, Any]:
        """Run technical indicators (RSI, MACD, Bollinger) analysis."""
        if not HAS_MARKET:
            print("\n[SKIP] Technical analysis - module not available")
            return {}

        print("\n" + "=" * 60)
        print("TECHNICAL INDICATORS (RSI, Bollinger, Volume)")
        print("=" * 60)

        try:
            analyzer = MarketAnalyzer()

            for symbol in self.symbols:
                result = analyzer.analyze_coin(symbol, days=90)

                if result and result.technical:
                    tech = result.technical

                    # Determine signal direction based on indicators
                    score = tech.score
                    if score >= 3:
                        direction = "STRONG_BUY"
                        confidence = 0.85
                    elif score >= 2:
                        direction = "BUY"
                        confidence = 0.7
                    elif score <= -2:
                        direction = "STRONG_SELL"
                        confidence = 0.85
                    elif score <= -1:
                        direction = "SELL"
                        confidence = 0.7
                    else:
                        direction = "NEUTRAL"
                        confidence = 0.5

                    # Store RSI signal
                    rsi_signal = TechnicalSignal(
                        symbol=f"{symbol}USDT",
                        timestamp=self.timestamp,
                        signal_name="RSI",
                        signal_value=tech.rsi_value,
                        signal_direction="BUY" if tech.rsi_oversold else ("SELL" if tech.rsi_value > 70 else "NEUTRAL"),
                        confidence=0.7 if tech.rsi_oversold or tech.rsi_value > 70 else 0.5,
                        timeframe="1d",
                        parameters=json.dumps({"oversold": tech.rsi_oversold})
                    )
                    self.db.insert_technical_signal(rsi_signal)

                    # Store Bollinger signal
                    bb_signal = TechnicalSignal(
                        symbol=f"{symbol}USDT",
                        timestamp=self.timestamp,
                        signal_name="Bollinger",
                        signal_value=tech.bb_penetration_pct,
                        signal_direction="BUY" if tech.bb_touch else "NEUTRAL",
                        confidence=0.75 if tech.bb_touch else 0.5,
                        timeframe="1d",
                        parameters=json.dumps({"touch": tech.bb_touch, "penetration": tech.bb_penetration_pct})
                    )
                    self.db.insert_technical_signal(bb_signal)

                    self.results["technical"][symbol] = {
                        "rsi": tech.rsi_value,
                        "rsi_oversold": tech.rsi_oversold,
                        "bb_touch": tech.bb_touch,
                        "bb_penetration": tech.bb_penetration_pct,
                        "volume_spike": tech.volume_spike,
                        "above_200ma": tech.above_200ma,
                        "score": score,
                        "direction": direction,
                        "confidence": confidence
                    }

                    # Format output
                    rsi_status = "OVERSOLD" if tech.rsi_oversold else ("OVERBOUGHT" if tech.rsi_value > 70 else "NEUTRAL")
                    bb_status = f"TOUCH ({tech.bb_penetration_pct:.1f}%)" if tech.bb_touch else "NO TOUCH"
                    print(f"  {symbol}: RSI={tech.rsi_value:.1f} ({rsi_status}), BB={bb_status}, Score={score} -> {direction}")
                else:
                    print(f"  {symbol}: Unable to fetch technical data")
        except Exception as e:
            print(f"  [ERROR] Technical analysis failed: {e}")
            import traceback
            traceback.print_exc()

        return self.results["technical"]

    def run_portfolio_analysis(self) -> Dict[str, Any]:
        """Run portfolio optimization analysis."""
        if not HAS_PORTFOLIO:
            print("\n[SKIP] Portfolio analysis - module not available")
            return {}

        print("\n" + "=" * 60)
        print("PORTFOLIO OPTIMIZATION")
        print("=" * 60)

        try:
            # Fetch price data
            print("  Fetching price history...")
            fetcher = PortfolioDataFetcher()
            prices = fetcher.fetch_all_prices(days=365)

            if prices.empty or len(prices.columns) < 2:
                print("  [ERROR] Insufficient price data for optimization")
                return {}

            # Filter to only our symbols
            available_symbols = [f"{s}USDT" for s in self.symbols if f"{s}USDT" in prices.columns]
            if len(available_symbols) < 2:
                print("  [ERROR] Need at least 2 symbols for portfolio optimization")
                return {}

            prices = prices[available_symbols]
            returns = prices.pct_change().dropna()

            # Calculate asset statistics
            print("  Calculating asset statistics...")
            stats_calc = AssetStatisticsCalculator(prices)

            asset_stats = {}
            for symbol in available_symbols:
                stats = stats_calc.calculate_stats(symbol)
                if stats:
                    asset_stats[symbol] = {
                        "expected_return": stats.expected_return,
                        "volatility": stats.volatility,
                        "sharpe_ratio": stats.sharpe_ratio,
                        "max_drawdown": stats.max_drawdown
                    }

            # Run optimizations
            print("  Running portfolio optimizations...")
            optimizer = PortfolioOptimizer(returns)

            # 1. Equal Weight
            equal = optimizer.equal_weight()
            print(f"\n  Equal Weight Portfolio:")
            for asset, weight in equal.weights.items():
                print(f"    {asset}: {weight:.1%}")
            print(f"    Expected Return: {equal.expected_return:.1%}")
            print(f"    Volatility: {equal.volatility:.1%}")
            print(f"    Sharpe Ratio: {equal.sharpe_ratio:.2f}")

            # 2. Max Sharpe (Optimal)
            max_sharpe = optimizer.max_sharpe()
            print(f"\n  Maximum Sharpe Portfolio (OPTIMAL):")
            for asset, weight in sorted(max_sharpe.weights.items(), key=lambda x: x[1], reverse=True):
                if weight > 0.01:
                    print(f"    {asset}: {weight:.1%}")
            print(f"    Expected Return: {max_sharpe.expected_return:.1%}")
            print(f"    Volatility: {max_sharpe.volatility:.1%}")
            print(f"    Sharpe Ratio: {max_sharpe.sharpe_ratio:.2f}")

            # 3. Minimum Volatility (Safe)
            min_vol = optimizer.min_volatility()
            print(f"\n  Minimum Volatility Portfolio (SAFE):")
            for asset, weight in sorted(min_vol.weights.items(), key=lambda x: x[1], reverse=True):
                if weight > 0.01:
                    print(f"    {asset}: {weight:.1%}")
            print(f"    Expected Return: {min_vol.expected_return:.1%}")
            print(f"    Volatility: {min_vol.volatility:.1%}")
            print(f"    Sharpe Ratio: {min_vol.sharpe_ratio:.2f}")

            # Store results
            self.results["portfolio"] = {
                "asset_stats": asset_stats,
                "equal_weight": {
                    "weights": equal.weights,
                    "expected_return": equal.expected_return,
                    "volatility": equal.volatility,
                    "sharpe": equal.sharpe_ratio
                },
                "max_sharpe": {
                    "weights": max_sharpe.weights,
                    "expected_return": max_sharpe.expected_return,
                    "volatility": max_sharpe.volatility,
                    "sharpe": max_sharpe.sharpe_ratio
                },
                "min_volatility": {
                    "weights": min_vol.weights,
                    "expected_return": min_vol.expected_return,
                    "volatility": min_vol.volatility,
                    "sharpe": min_vol.sharpe_ratio
                }
            }

            # Print recommendation
            print(f"\n  RECOMMENDATION: Use Max Sharpe allocation for best risk-adjusted returns")

        except Exception as e:
            print(f"  [ERROR] Portfolio analysis failed: {e}")
            import traceback
            traceback.print_exc()

        return self.results["portfolio"]

    def calculate_aggregated_signals(self) -> Dict[str, Any]:
        """Calculate weighted consensus signals for all symbols."""
        print("\n" + "=" * 60)
        print("AGGREGATED SIGNALS (Weighted Consensus)")
        print("=" * 60)

        for symbol in self.symbols:
            symbol_key = f"{symbol}USDT"
            agg = self.db.get_aggregated_signals(symbol_key, hours_back=24)
            self.results["aggregated"][symbol] = agg

            print(f"  {symbol}: {agg['consensus_direction']} "
                  f"(score: {agg['consensus_score']:.3f}, "
                  f"confidence: {agg['confidence']:.2f}, "
                  f"regime: {agg['regime']})")

        return self.results["aggregated"]

    def run_full_analysis(self) -> Dict[str, Any]:
        """Run all analyses and return comprehensive results."""
        print("\n" + "=" * 70)
        print("MASTER ANALYSIS - Running All Tools")
        print(f"Timestamp: {self.timestamp}")
        print(f"Symbols: {', '.join(self.symbols)}")
        print("=" * 70)

        # Run all analyses
        self.run_correlation_analysis()
        self.run_whale_analysis()
        self.run_funding_analysis()
        self.run_orderflow_analysis()
        self.run_technical_analysis()  # RSI, Bollinger, Volume
        self.run_ml_analysis()
        self.run_portfolio_analysis()  # Optimal allocations

        # Calculate aggregated signals
        self.calculate_aggregated_signals()

        # Print summary
        self.print_summary()

        return self.results

    def print_summary(self):
        """Print comprehensive analysis summary."""
        print("\n" + "=" * 70)
        print("ANALYSIS SUMMARY")
        print("=" * 70)

        # Get current regime
        regime = self.db.get_current_regime()
        if regime:
            print(f"\nMarket Regime: {regime['regime']} "
                  f"(confidence: {regime.get('confidence', 0):.0%})")

        # Signal counts
        stats = self.db.get_database_stats()
        print(f"\nDatabase Records:")
        for name, count in stats["record_counts"].items():
            print(f"  {name}: {count}")

        # Top opportunities
        print("\n" + "-" * 50)
        print("TOP OPPORTUNITIES (by consensus score)")
        print("-" * 50)

        sorted_signals = sorted(
            self.results["aggregated"].items(),
            key=lambda x: x[1].get("consensus_score", 0),
            reverse=True
        )

        for symbol, agg in sorted_signals:
            score = agg.get("consensus_score", 0)
            direction = agg.get("consensus_direction", "NEUTRAL")
            confidence = agg.get("confidence", 0)

            # Emoji indicators
            if direction in ["STRONG_BUY", "BUY"]:
                indicator = "[BULLISH]"
            elif direction in ["STRONG_SELL", "SELL"]:
                indicator = "[BEARISH]"
            else:
                indicator = "[NEUTRAL]"

            print(f"  {symbol:6} {indicator:10} Score: {score:+.3f}  "
                  f"Confidence: {confidence:.0%}")

            # Show signal breakdown
            breakdown = agg.get("signal_breakdown", {})
            if breakdown:
                parts = []
                for sig_type, sig_score in breakdown.items():
                    parts.append(f"{sig_type}={sig_score:+.2f}")
                print(f"         Breakdown: {', '.join(parts)}")

        # Recommendations
        print("\n" + "-" * 50)
        print("RECOMMENDATIONS")
        print("-" * 50)

        strong_buys = [s for s, a in sorted_signals
                      if a.get("consensus_direction") == "STRONG_BUY"]
        buys = [s for s, a in sorted_signals
               if a.get("consensus_direction") == "BUY"]
        sells = [s for s, a in sorted_signals
                if a.get("consensus_direction") in ["SELL", "STRONG_SELL"]]

        if strong_buys:
            print(f"  STRONG BUY: {', '.join(strong_buys)}")
        if buys:
            print(f"  BUY: {', '.join(buys)}")
        if sells:
            print(f"  AVOID: {', '.join(sells)}")
        if not (strong_buys or buys or sells):
            print("  No clear signals - consider waiting")

        print("\n" + "=" * 70)
        print("Analysis complete. Results stored in database.")
        print("=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run all analysis tools and store results in database"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTC", "ETH", "XRP", "SOL", "ATOM"],
        help="Symbols to analyze (default: BTC ETH XRP SOL ATOM)"
    )
    parser.add_argument(
        "--db-path",
        default="quant_signals.db",
        help="Path to SQLite database (default: quant_signals.db)"
    )

    args = parser.parse_args()

    # Run analysis
    analyzer = MasterAnalyzer(
        db_path=args.db_path,
        symbols=args.symbols
    )

    results = analyzer.run_full_analysis()

    # Return results for programmatic use
    return results


if __name__ == "__main__":
    main()
