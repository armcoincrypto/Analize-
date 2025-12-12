#!/usr/bin/env python3
"""
Whale Tracker - Large Wallet Movement Detection

Features:
1. Track large transactions on XRP, SOL, ATOM networks
2. Exchange inflow/outflow monitoring
3. Whale accumulation/distribution detection
4. Alert on significant movements
5. Historical whale activity correlation with price

Author: Analize Team
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import requests
import time
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class WhaleTransaction:
    """Single whale transaction."""
    coin: str
    tx_hash: str
    timestamp: datetime
    amount: float
    amount_usd: float
    from_address: str
    to_address: str
    from_type: str  # EXCHANGE, WHALE, UNKNOWN
    to_type: str
    direction: str  # EXCHANGE_INFLOW, EXCHANGE_OUTFLOW, WHALE_TO_WHALE


@dataclass
class WhaleAlert:
    """Alert for significant whale activity."""
    coin: str
    alert_type: str  # LARGE_INFLOW, LARGE_OUTFLOW, ACCUMULATION, DISTRIBUTION
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    amount: float
    amount_usd: float
    description: str
    timestamp: datetime
    price_impact_estimate: str


@dataclass
class WhaleMetrics:
    """Aggregated whale metrics."""
    coin: str
    period_hours: int
    total_inflow: float
    total_outflow: float
    net_flow: float  # Negative = outflow from exchanges (bullish)
    whale_transactions: int
    avg_transaction_size: float
    largest_transaction: float
    accumulation_score: float  # -100 to +100


@dataclass
class ExchangeFlow:
    """Exchange flow data."""
    exchange: str
    coin: str
    inflow_24h: float
    outflow_24h: float
    net_flow_24h: float
    balance_change_7d: float


# =============================================================================
# BLOCKCHAIN DATA FETCHERS
# =============================================================================

class BlockchainDataFetcher:
    """Fetch blockchain data from various sources."""

    def __init__(self):
        self.cache = {}
        self.known_exchanges = self._load_exchange_addresses()

    def _load_exchange_addresses(self) -> Dict[str, Dict[str, List[str]]]:
        """Load known exchange addresses (simplified)."""
        return {
            "XRP": {
                "binance": ["rEb8TK3gBgk5auZkwc6sHnwrGVJH8DuaLh"],
                "coinbase": ["rw2ciyaNshpHe7bCHo4bRWq6pqqynnWKQg"],
                "kraken": ["rLHzPsX6oXkzU2qL12kHCH8G8cnZv1rBJh"],
                "bitfinex": ["rKwS2TJmX8D5VvYsyxmhSzjzPQJmgXmAoX"],
            },
            "SOL": {
                "binance": ["9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM"],
                "coinbase": ["H8sMJSCQxfKiFTCfDR3DUMLPwcRbM61LGFJ8N4dK3WjS"],
                "kraken": ["4qPLHqPPVBkPPBBGRaPrRMuS6qfnQ7qDwGnKdBnKJ9Pr"],
                "ftx": ["ASTyfSima4LLAdDgoFGkgqoKowG1LZFDr9fAQrg7iaJZ"],
            },
            "ATOM": {
                "binance": ["cosmos1qg5ega6dykkxc307y25pecuufrjkxkaggkkxh7"],
                "coinbase": ["cosmos15r4dkkr4f8f4t39yqcz2l6l34kz8e2h4c2u9v5"],
                "kraken": ["cosmos1c4k24jzduc365kywrsvf5hlyjyqdj628vcqnv8"],
            }
        }

    def is_exchange_address(self, coin: str, address: str) -> Tuple[bool, str]:
        """Check if address belongs to known exchange."""
        if coin not in self.known_exchanges:
            return False, "UNKNOWN"

        for exchange, addresses in self.known_exchanges[coin].items():
            if address in addresses:
                return True, exchange.upper()

        return False, "UNKNOWN"

    def fetch_whale_transactions_xrp(self, min_amount: float = 1000000) -> List[WhaleTransaction]:
        """Fetch large XRP transactions from XRPL."""
        transactions = []

        try:
            # Use XRP Ledger API for recent transactions
            # Note: This is a simplified example - in production use wss://xrplcluster.com
            url = "https://data.ripple.com/v2/transactions"
            params = {
                "type": "Payment",
                "limit": 100,
                "descending": "true"
            }

            response = requests.get(url, params=params, timeout=15)

            if response.status_code == 200:
                data = response.json()

                for tx in data.get("transactions", []):
                    if tx.get("tx", {}).get("TransactionType") == "Payment":
                        amount = float(tx.get("tx", {}).get("Amount", 0)) / 1000000  # drops to XRP

                        if amount >= min_amount:
                            from_addr = tx.get("tx", {}).get("Account", "")
                            to_addr = tx.get("tx", {}).get("Destination", "")

                            from_is_exchange, from_exchange = self.is_exchange_address("XRP", from_addr)
                            to_is_exchange, to_exchange = self.is_exchange_address("XRP", to_addr)

                            if from_is_exchange and not to_is_exchange:
                                direction = "EXCHANGE_OUTFLOW"
                                from_type = f"EXCHANGE_{from_exchange}"
                                to_type = "WHALE"
                            elif not from_is_exchange and to_is_exchange:
                                direction = "EXCHANGE_INFLOW"
                                from_type = "WHALE"
                                to_type = f"EXCHANGE_{to_exchange}"
                            else:
                                direction = "WHALE_TO_WHALE"
                                from_type = "WHALE" if not from_is_exchange else f"EXCHANGE_{from_exchange}"
                                to_type = "WHALE" if not to_is_exchange else f"EXCHANGE_{to_exchange}"

                            transactions.append(WhaleTransaction(
                                coin="XRP",
                                tx_hash=tx.get("hash", ""),
                                timestamp=datetime.fromisoformat(tx.get("date", "").replace("Z", "")),
                                amount=amount,
                                amount_usd=amount * self._get_current_price("XRP"),
                                from_address=from_addr[:20] + "...",
                                to_address=to_addr[:20] + "...",
                                from_type=from_type,
                                to_type=to_type,
                                direction=direction
                            ))

        except Exception as e:
            print(f"XRP fetch error: {e}")

        return transactions

    def fetch_whale_transactions_sol(self, min_amount: float = 10000) -> List[WhaleTransaction]:
        """Fetch large SOL transactions."""
        transactions = []

        try:
            # Use Solana Beach API or similar
            # Note: In production, use Helius, QuickNode, or similar provider
            url = "https://api.solscan.io/transfer/sol"
            params = {
                "limit": 50,
                "offset": 0
            }

            headers = {"Accept": "application/json"}
            response = requests.get(url, params=params, headers=headers, timeout=15)

            if response.status_code == 200:
                data = response.json()

                for tx in data.get("data", []):
                    amount = float(tx.get("lamport", 0)) / 1e9  # lamports to SOL

                    if amount >= min_amount:
                        from_addr = tx.get("src", "")
                        to_addr = tx.get("dst", "")

                        from_is_exchange, from_exchange = self.is_exchange_address("SOL", from_addr)
                        to_is_exchange, to_exchange = self.is_exchange_address("SOL", to_addr)

                        if from_is_exchange and not to_is_exchange:
                            direction = "EXCHANGE_OUTFLOW"
                        elif not from_is_exchange and to_is_exchange:
                            direction = "EXCHANGE_INFLOW"
                        else:
                            direction = "WHALE_TO_WHALE"

                        transactions.append(WhaleTransaction(
                            coin="SOL",
                            tx_hash=tx.get("txHash", "")[:20] + "...",
                            timestamp=datetime.fromtimestamp(tx.get("blockTime", time.time())),
                            amount=amount,
                            amount_usd=amount * self._get_current_price("SOL"),
                            from_address=from_addr[:20] + "...",
                            to_address=to_addr[:20] + "...",
                            from_type="EXCHANGE" if from_is_exchange else "WHALE",
                            to_type="EXCHANGE" if to_is_exchange else "WHALE",
                            direction=direction
                        ))

        except Exception as e:
            print(f"SOL fetch error: {e}")

        return transactions

    def _get_current_price(self, coin: str) -> float:
        """Get current price for USD conversion."""
        prices = {"XRP": 2.30, "SOL": 220.0, "ATOM": 10.0}  # Fallback prices

        try:
            coin_ids = {"XRP": "ripple", "SOL": "solana", "ATOM": "cosmos"}
            coin_id = coin_ids.get(coin)

            if coin_id:
                url = f"https://api.coingecko.com/api/v3/simple/price"
                params = {"ids": coin_id, "vs_currencies": "usd"}
                response = requests.get(url, params=params, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    prices[coin] = data.get(coin_id, {}).get("usd", prices[coin])

        except Exception:
            pass

        return prices.get(coin, 1.0)


# =============================================================================
# WHALE ALERT AGGREGATOR
# =============================================================================

class WhaleAlertAggregator:
    """Aggregate whale alerts from multiple sources."""

    def __init__(self):
        self.api_key = None  # Optional: whale-alert.io API key

    def fetch_whale_alerts(self, min_usd: float = 1000000) -> List[WhaleAlert]:
        """Fetch recent whale alerts."""
        alerts = []

        # Try WhaleAlert API if key available
        if self.api_key:
            alerts.extend(self._fetch_from_whale_alert_api(min_usd))

        # Generate simulated alerts based on market data
        alerts.extend(self._generate_market_based_alerts())

        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)

    def _fetch_from_whale_alert_api(self, min_usd: float) -> List[WhaleAlert]:
        """Fetch from whale-alert.io API."""
        alerts = []

        try:
            url = "https://api.whale-alert.io/v1/transactions"
            params = {
                "api_key": self.api_key,
                "min_value": int(min_usd),
                "limit": 100
            }

            response = requests.get(url, params=params, timeout=15)

            if response.status_code == 200:
                data = response.json()

                for tx in data.get("transactions", []):
                    coin = tx.get("symbol", "").upper()
                    if coin not in ["XRP", "SOL", "ATOM"]:
                        continue

                    amount = tx.get("amount", 0)
                    amount_usd = tx.get("amount_usd", 0)
                    from_type = tx.get("from", {}).get("owner_type", "unknown")
                    to_type = tx.get("to", {}).get("owner_type", "unknown")

                    # Determine alert type
                    if to_type == "exchange":
                        alert_type = "LARGE_INFLOW"
                        severity = self._calculate_severity(amount_usd, "inflow")
                        description = f"{amount:,.0f} {coin} moved to exchange"
                        price_impact = "Potential selling pressure"
                    elif from_type == "exchange":
                        alert_type = "LARGE_OUTFLOW"
                        severity = self._calculate_severity(amount_usd, "outflow")
                        description = f"{amount:,.0f} {coin} withdrawn from exchange"
                        price_impact = "Reduced selling pressure"
                    else:
                        alert_type = "WHALE_TRANSFER"
                        severity = "MEDIUM"
                        description = f"{amount:,.0f} {coin} whale-to-whale transfer"
                        price_impact = "Watch for follow-up movements"

                    alerts.append(WhaleAlert(
                        coin=coin,
                        alert_type=alert_type,
                        severity=severity,
                        amount=amount,
                        amount_usd=amount_usd,
                        description=description,
                        timestamp=datetime.fromtimestamp(tx.get("timestamp", time.time())),
                        price_impact_estimate=price_impact
                    ))

        except Exception as e:
            print(f"Whale Alert API error: {e}")

        return alerts

    def _generate_market_based_alerts(self) -> List[WhaleAlert]:
        """Generate alerts based on market analysis."""
        alerts = []

        # Simulated whale activity based on typical patterns
        coins = ["XRP", "SOL", "ATOM"]

        for coin in coins:
            # Check for significant exchange flow patterns
            flow = self._analyze_exchange_flows(coin)

            if flow and abs(flow.net_flow_24h) > 0:
                if flow.net_flow_24h > 0:
                    alert_type = "ACCUMULATION"
                    severity = "HIGH" if flow.net_flow_24h > flow.inflow_24h * 0.5 else "MEDIUM"
                    description = f"Net inflow of {abs(flow.net_flow_24h):,.0f} {coin} to exchanges (24h)"
                    price_impact = "Potential selling pressure building"
                else:
                    alert_type = "DISTRIBUTION"
                    severity = "HIGH" if abs(flow.net_flow_24h) > flow.outflow_24h * 0.5 else "MEDIUM"
                    description = f"Net outflow of {abs(flow.net_flow_24h):,.0f} {coin} from exchanges (24h)"
                    price_impact = "Bullish - coins moving to cold storage"

                alerts.append(WhaleAlert(
                    coin=coin,
                    alert_type=alert_type,
                    severity=severity,
                    amount=abs(flow.net_flow_24h),
                    amount_usd=abs(flow.net_flow_24h) * self._get_price(coin),
                    description=description,
                    timestamp=datetime.now(),
                    price_impact_estimate=price_impact
                ))

        return alerts

    def _analyze_exchange_flows(self, coin: str) -> Optional[ExchangeFlow]:
        """Analyze exchange flow patterns."""
        # Simulated data - in production, use on-chain data
        simulated_flows = {
            "XRP": ExchangeFlow(
                exchange="ALL",
                coin="XRP",
                inflow_24h=45000000,
                outflow_24h=52000000,
                net_flow_24h=-7000000,  # Net outflow (bullish)
                balance_change_7d=-35000000
            ),
            "SOL": ExchangeFlow(
                exchange="ALL",
                coin="SOL",
                inflow_24h=150000,
                outflow_24h=180000,
                net_flow_24h=-30000,  # Net outflow (bullish)
                balance_change_7d=-120000
            ),
            "ATOM": ExchangeFlow(
                exchange="ALL",
                coin="ATOM",
                inflow_24h=800000,
                outflow_24h=650000,
                net_flow_24h=150000,  # Net inflow (bearish)
                balance_change_7d=500000
            ),
        }

        return simulated_flows.get(coin)

    def _calculate_severity(self, amount_usd: float, flow_type: str) -> str:
        """Calculate alert severity."""
        if amount_usd > 50000000:
            return "CRITICAL"
        elif amount_usd > 10000000:
            return "HIGH"
        elif amount_usd > 1000000:
            return "MEDIUM"
        return "LOW"

    def _get_price(self, coin: str) -> float:
        """Get current price."""
        prices = {"XRP": 2.30, "SOL": 220.0, "ATOM": 10.0}
        return prices.get(coin, 1.0)


# =============================================================================
# WHALE METRICS CALCULATOR
# =============================================================================

class WhaleMetricsCalculator:
    """Calculate aggregated whale metrics."""

    def __init__(self, transactions: List[WhaleTransaction]):
        self.transactions = transactions

    def calculate_metrics(self, coin: str, period_hours: int = 24) -> WhaleMetrics:
        """Calculate whale metrics for a coin."""
        cutoff = datetime.now() - timedelta(hours=period_hours)

        coin_txs = [tx for tx in self.transactions
                    if tx.coin == coin and tx.timestamp > cutoff]

        if not coin_txs:
            return WhaleMetrics(
                coin=coin,
                period_hours=period_hours,
                total_inflow=0,
                total_outflow=0,
                net_flow=0,
                whale_transactions=0,
                avg_transaction_size=0,
                largest_transaction=0,
                accumulation_score=0
            )

        # Calculate flows
        total_inflow = sum(tx.amount for tx in coin_txs if tx.direction == "EXCHANGE_INFLOW")
        total_outflow = sum(tx.amount for tx in coin_txs if tx.direction == "EXCHANGE_OUTFLOW")
        net_flow = total_inflow - total_outflow

        # Transaction stats
        amounts = [tx.amount for tx in coin_txs]
        avg_size = np.mean(amounts) if amounts else 0
        max_size = max(amounts) if amounts else 0

        # Accumulation score (-100 to +100)
        # Negative = more outflows (bullish)
        # Positive = more inflows (bearish)
        total_flow = total_inflow + total_outflow
        if total_flow > 0:
            accumulation_score = (net_flow / total_flow) * 100
        else:
            accumulation_score = 0

        return WhaleMetrics(
            coin=coin,
            period_hours=period_hours,
            total_inflow=total_inflow,
            total_outflow=total_outflow,
            net_flow=net_flow,
            whale_transactions=len(coin_txs),
            avg_transaction_size=avg_size,
            largest_transaction=max_size,
            accumulation_score=accumulation_score
        )


# =============================================================================
# WHALE ACTIVITY CORRELATOR
# =============================================================================

class WhaleActivityCorrelator:
    """Correlate whale activity with price movements."""

    def __init__(self):
        self.cache = {}

    def analyze_whale_price_correlation(self, coin: str,
                                        lookback_days: int = 90) -> Dict:
        """Analyze correlation between whale activity and price."""
        # Fetch historical price data
        price_data = self._fetch_price_history(coin, lookback_days)

        if price_data.empty:
            return {"error": "Could not fetch price data"}

        # Generate simulated whale activity data
        # In production, this would come from on-chain data
        whale_activity = self._simulate_whale_activity(price_data)

        # Calculate correlation
        correlation = price_data["returns"].corr(whale_activity["net_flow"])

        # Find patterns
        patterns = self._find_patterns(price_data, whale_activity)

        return {
            "coin": coin,
            "correlation_with_returns": correlation,
            "patterns": patterns,
            "interpretation": self._interpret_correlation(correlation),
            "signal": self._generate_signal(whale_activity.iloc[-1])
        }

    def _fetch_price_history(self, coin: str, days: int) -> pd.DataFrame:
        """Fetch price history."""
        symbol_map = {"XRP": "XRPUSDT", "SOL": "SOLUSDT", "ATOM": "ATOMUSDT"}
        symbol = symbol_map.get(coin, f"{coin}USDT")

        try:
            params = {
                "symbol": symbol,
                "interval": "1d",
                "startTime": int((datetime.now() - timedelta(days=days)).timestamp() * 1000),
                "limit": 1000
            }

            # Try Binance US first (works in geo-restricted regions)
            response = requests.get(
                "https://api.binance.us/api/v3/klines",
                params=params,
                timeout=15
            )

            # Fallback to Binance Global if US fails
            if response.status_code != 200:
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
                df["volume"] = df["volume"].astype(float)
                df["returns"] = df["close"].pct_change()

                return df[["timestamp", "close", "volume", "returns"]]

        except Exception:
            pass

        return pd.DataFrame()

    def _simulate_whale_activity(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Simulate whale activity based on volume patterns."""
        df = price_data.copy()

        # Simulate net flow based on volume and price changes
        # High volume + price drop = likely selling (positive net flow to exchanges)
        # High volume + price up = likely buying (negative net flow)

        volume_ma = df["volume"].rolling(20).mean()
        volume_ratio = df["volume"] / volume_ma

        # Simulate net flow
        df["net_flow"] = -df["returns"] * volume_ratio * 1000000  # Arbitrary scale

        # Add noise
        df["net_flow"] += np.random.normal(0, df["net_flow"].std() * 0.2, len(df))

        return df[["timestamp", "net_flow"]]

    def _find_patterns(self, price_data: pd.DataFrame,
                       whale_activity: pd.DataFrame) -> List[str]:
        """Find patterns in whale activity."""
        patterns = []

        # Check for recent large outflows
        recent_flow = whale_activity["net_flow"].tail(7).mean()
        if recent_flow < -100000:
            patterns.append("Strong accumulation pattern (large exchange outflows)")
        elif recent_flow > 100000:
            patterns.append("Distribution pattern (large exchange inflows)")

        # Check for divergence
        recent_returns = price_data["returns"].tail(7).sum()
        if recent_returns > 0 and recent_flow > 50000:
            patterns.append("WARNING: Price rising but whales selling")
        elif recent_returns < 0 and recent_flow < -50000:
            patterns.append("BULLISH: Price falling but whales accumulating")

        return patterns

    def _interpret_correlation(self, correlation: float) -> str:
        """Interpret the correlation value."""
        if abs(correlation) < 0.2:
            return "Weak relationship between whale activity and price"
        elif correlation > 0.5:
            return "Strong positive correlation - price moves with exchange inflows"
        elif correlation < -0.5:
            return "Strong negative correlation - price rises when whales withdraw"
        else:
            return "Moderate relationship between whale activity and price"

    def _generate_signal(self, recent_activity: pd.Series) -> Dict:
        """Generate trading signal from whale activity."""
        net_flow = recent_activity.get("net_flow", 0)

        if net_flow < -200000:
            return {
                "signal": "BULLISH",
                "strength": "STRONG",
                "reason": "Large exchange outflows indicate accumulation"
            }
        elif net_flow < -50000:
            return {
                "signal": "BULLISH",
                "strength": "MODERATE",
                "reason": "Moderate exchange outflows"
            }
        elif net_flow > 200000:
            return {
                "signal": "BEARISH",
                "strength": "STRONG",
                "reason": "Large exchange inflows indicate selling pressure"
            }
        elif net_flow > 50000:
            return {
                "signal": "BEARISH",
                "strength": "MODERATE",
                "reason": "Moderate exchange inflows"
            }
        else:
            return {
                "signal": "NEUTRAL",
                "strength": "WEAK",
                "reason": "No significant whale activity"
            }


# =============================================================================
# MAIN WHALE TRACKER
# =============================================================================

class WhaleTracker:
    """Main whale tracking system."""

    def __init__(self):
        self.fetcher = BlockchainDataFetcher()
        self.alert_aggregator = WhaleAlertAggregator()
        self.correlator = WhaleActivityCorrelator()

    def run_analysis(self):
        """Run full whale analysis."""
        print("=" * 70)
        print("WHALE TRACKER - Large Wallet Movement Detection")
        print("=" * 70)
        print()

        coins = ["XRP", "SOL", "ATOM"]

        # Fetch whale alerts
        print("[1] RECENT WHALE ALERTS")
        print("-" * 70)

        alerts = self.alert_aggregator.fetch_whale_alerts(min_usd=500000)

        if alerts:
            for alert in alerts[:10]:
                severity_emoji = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🟠", "CRITICAL": "🔴"}
                emoji = severity_emoji.get(alert.severity, "⚪")

                print(f"  {emoji} [{alert.severity}] {alert.coin}")
                print(f"     Type: {alert.alert_type}")
                print(f"     Amount: {alert.amount:,.0f} (${alert.amount_usd:,.0f})")
                print(f"     {alert.description}")
                print(f"     Impact: {alert.price_impact_estimate}")
                print()
        else:
            print("  No significant whale alerts in the last 24h")
        print()

        # Exchange flow analysis
        print("[2] EXCHANGE FLOW ANALYSIS (24h)")
        print("-" * 70)

        for coin in coins:
            flow = self.alert_aggregator._analyze_exchange_flows(coin)
            if flow:
                flow_direction = "🟢 OUTFLOW" if flow.net_flow_24h < 0 else "🔴 INFLOW"

                print(f"  {coin}:")
                print(f"    Inflow:  {flow.inflow_24h:>15,.0f}")
                print(f"    Outflow: {flow.outflow_24h:>15,.0f}")
                print(f"    Net:     {flow.net_flow_24h:>15,.0f} ({flow_direction})")
                print(f"    7d Change: {flow.balance_change_7d:>12,.0f}")
                print()

        # Whale-Price Correlation
        print("[3] WHALE ACTIVITY vs PRICE CORRELATION")
        print("-" * 70)

        for coin in coins:
            analysis = self.correlator.analyze_whale_price_correlation(coin)

            if "error" not in analysis:
                print(f"  {coin}:")
                print(f"    Correlation: {analysis['correlation_with_returns']:.2f}")
                print(f"    {analysis['interpretation']}")

                if analysis['patterns']:
                    print("    Patterns:")
                    for pattern in analysis['patterns']:
                        print(f"      • {pattern}")

                signal = analysis['signal']
                print(f"    Signal: {signal['signal']} ({signal['strength']})")
                print(f"    Reason: {signal['reason']}")
                print()

        # Accumulation/Distribution Score
        print("[4] ACCUMULATION/DISTRIBUTION SCORE")
        print("-" * 70)

        for coin in coins:
            flow = self.alert_aggregator._analyze_exchange_flows(coin)
            if flow:
                total = flow.inflow_24h + flow.outflow_24h
                if total > 0:
                    score = (flow.net_flow_24h / total) * 100
                else:
                    score = 0

                if score < -20:
                    status = "🟢 ACCUMULATION"
                elif score > 20:
                    status = "🔴 DISTRIBUTION"
                else:
                    status = "⚪ NEUTRAL"

                bar_len = int(abs(score) / 5)
                if score < 0:
                    bar = "█" * bar_len + " " * (20 - bar_len) + "│" + " " * 20
                else:
                    bar = " " * 20 + "│" + "█" * bar_len + " " * (20 - bar_len)

                print(f"  {coin}:")
                print(f"    Score: {score:+.1f}")
                print(f"    [{bar}]")
                print(f"    Status: {status}")
                print()

        # Trading Recommendations
        print("[5] WHALE-BASED TRADING SIGNALS")
        print("-" * 70)

        for coin in coins:
            analysis = self.correlator.analyze_whale_price_correlation(coin)
            flow = self.alert_aggregator._analyze_exchange_flows(coin)

            if "error" not in analysis and flow:
                signal = analysis['signal']

                # Combine signals
                whale_bullish = signal['signal'] == "BULLISH"
                flow_bullish = flow.net_flow_24h < 0

                if whale_bullish and flow_bullish:
                    recommendation = "🟢 STRONG BUY SIGNAL"
                    confidence = "HIGH"
                elif whale_bullish or flow_bullish:
                    recommendation = "🟡 MODERATE BUY SIGNAL"
                    confidence = "MEDIUM"
                elif signal['signal'] == "BEARISH" and flow.net_flow_24h > 0:
                    recommendation = "🔴 CAUTION - Selling Pressure"
                    confidence = "HIGH"
                else:
                    recommendation = "⚪ NEUTRAL - No clear whale signal"
                    confidence = "LOW"

                print(f"  {coin}: {recommendation}")
                print(f"    Confidence: {confidence}")
                print()

        print("=" * 70)
        print("Analysis Complete!")
        print("=" * 70)
        print()
        print("Note: Whale data is simulated. For production, integrate:")
        print("  • whale-alert.io API")
        print("  • Glassnode / CryptoQuant for exchange flows")
        print("  • Direct blockchain node queries")


# =============================================================================
# RUN ANALYSIS
# =============================================================================

def run_whale_tracking():
    """Run whale tracking analysis."""
    tracker = WhaleTracker()
    tracker.run_analysis()


if __name__ == "__main__":
    run_whale_tracking()
