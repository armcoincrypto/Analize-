#!/usr/bin/env python3
"""
Order Flow Analyzer - Order Book Depth Analysis

Features:
1. Real-time order book analysis
2. Bid/Ask imbalance detection
3. Large order detection (whale walls)
4. Support/Resistance from order book
5. Liquidity analysis
6. Order flow momentum signals

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
class OrderBookLevel:
    """Single price level in order book."""
    price: float
    quantity: float
    total_value: float
    side: str  # BID or ASK
    cumulative_quantity: float


@dataclass
class OrderBookSnapshot:
    """Complete order book snapshot."""
    symbol: str
    timestamp: datetime
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    bid_ask_spread: float
    spread_percentage: float
    mid_price: float


@dataclass
class OrderBookImbalance:
    """Bid/Ask imbalance metrics."""
    symbol: str
    imbalance_ratio: float  # > 1 = more bids, < 1 = more asks
    bid_volume: float
    ask_volume: float
    imbalance_percentage: float
    signal: str  # BULLISH, BEARISH, NEUTRAL
    depth_levels: int


@dataclass
class WhaleWall:
    """Large order detection."""
    symbol: str
    price: float
    quantity: float
    value_usd: float
    side: str  # BID or ASK
    distance_from_price: float  # Percentage
    significance: str  # LOW, MEDIUM, HIGH, CRITICAL


@dataclass
class SupportResistance:
    """Support/Resistance level from order book."""
    symbol: str
    price: float
    strength: float  # 0-100
    type: str  # SUPPORT or RESISTANCE
    total_volume: float
    wall_count: int


@dataclass
class LiquidityAnalysis:
    """Liquidity metrics."""
    symbol: str
    bid_liquidity_1pct: float  # Volume within 1% of best bid
    ask_liquidity_1pct: float
    bid_liquidity_5pct: float
    ask_liquidity_5pct: float
    total_book_value: float
    liquidity_score: float  # 0-100
    slippage_estimate_10k: float  # Estimated slippage for $10k order


@dataclass
class OrderFlowSignal:
    """Trading signal from order flow."""
    symbol: str
    signal: str  # STRONG_BUY, BUY, NEUTRAL, SELL, STRONG_SELL
    confidence: float
    reasons: List[str]
    entry_suggestion: Optional[float]
    stop_suggestion: Optional[float]


# =============================================================================
# ORDER BOOK FETCHER
# =============================================================================

class OrderBookFetcher:
    """Fetch order book data from exchanges."""

    def __init__(self):
        self.cache = {}

    def fetch_binance_orderbook(self, symbol: str, limit: int = 100) -> Optional[OrderBookSnapshot]:
        """Fetch order book from Binance."""
        try:
            url = "https://api.binance.com/api/v3/depth"
            params = {"symbol": symbol, "limit": limit}

            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()

                # Process bids
                bids = []
                cumulative_qty = 0
                for price, qty in data["bids"]:
                    price = float(price)
                    qty = float(qty)
                    cumulative_qty += qty
                    bids.append(OrderBookLevel(
                        price=price,
                        quantity=qty,
                        total_value=price * qty,
                        side="BID",
                        cumulative_quantity=cumulative_qty
                    ))

                # Process asks
                asks = []
                cumulative_qty = 0
                for price, qty in data["asks"]:
                    price = float(price)
                    qty = float(qty)
                    cumulative_qty += qty
                    asks.append(OrderBookLevel(
                        price=price,
                        quantity=qty,
                        total_value=price * qty,
                        side="ASK",
                        cumulative_quantity=cumulative_qty
                    ))

                best_bid = bids[0].price if bids else 0
                best_ask = asks[0].price if asks else 0
                spread = best_ask - best_bid
                mid_price = (best_bid + best_ask) / 2

                return OrderBookSnapshot(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    bids=bids,
                    asks=asks,
                    bid_ask_spread=spread,
                    spread_percentage=(spread / mid_price * 100) if mid_price > 0 else 0,
                    mid_price=mid_price
                )

        except Exception as e:
            print(f"Order book fetch error for {symbol}: {e}")

        return None

    def fetch_binance_us_orderbook(self, symbol: str, limit: int = 100) -> Optional[OrderBookSnapshot]:
        """Fetch order book from Binance US."""
        try:
            url = "https://api.binance.us/api/v3/depth"
            params = {"symbol": symbol, "limit": limit}

            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()

                # Same processing as Binance
                bids = []
                cumulative_qty = 0
                for price, qty in data["bids"]:
                    price = float(price)
                    qty = float(qty)
                    cumulative_qty += qty
                    bids.append(OrderBookLevel(
                        price=price,
                        quantity=qty,
                        total_value=price * qty,
                        side="BID",
                        cumulative_quantity=cumulative_qty
                    ))

                asks = []
                cumulative_qty = 0
                for price, qty in data["asks"]:
                    price = float(price)
                    qty = float(qty)
                    cumulative_qty += qty
                    asks.append(OrderBookLevel(
                        price=price,
                        quantity=qty,
                        total_value=price * qty,
                        side="ASK",
                        cumulative_quantity=cumulative_qty
                    ))

                best_bid = bids[0].price if bids else 0
                best_ask = asks[0].price if asks else 0
                spread = best_ask - best_bid
                mid_price = (best_bid + best_ask) / 2

                return OrderBookSnapshot(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    bids=bids,
                    asks=asks,
                    bid_ask_spread=spread,
                    spread_percentage=(spread / mid_price * 100) if mid_price > 0 else 0,
                    mid_price=mid_price
                )

        except Exception:
            pass

        return None

    def fetch_orderbook(self, symbol: str, limit: int = 100) -> Optional[OrderBookSnapshot]:
        """Fetch order book with fallback."""
        # Try Binance first
        ob = self.fetch_binance_orderbook(symbol, limit)
        if ob:
            return ob

        # Try Binance US
        ob = self.fetch_binance_us_orderbook(symbol, limit)
        return ob


# =============================================================================
# ORDER BOOK ANALYZER
# =============================================================================

class OrderBookAnalyzer:
    """Analyze order book for trading signals."""

    def __init__(self, orderbook: OrderBookSnapshot):
        self.ob = orderbook

    def calculate_imbalance(self, depth_levels: int = 10) -> OrderBookImbalance:
        """Calculate bid/ask imbalance."""
        bid_volume = sum(b.quantity for b in self.ob.bids[:depth_levels])
        ask_volume = sum(a.quantity for a in self.ob.asks[:depth_levels])

        total_volume = bid_volume + ask_volume

        if ask_volume > 0:
            imbalance_ratio = bid_volume / ask_volume
        else:
            imbalance_ratio = float('inf')

        if total_volume > 0:
            imbalance_pct = (bid_volume - ask_volume) / total_volume * 100
        else:
            imbalance_pct = 0

        # Generate signal
        if imbalance_ratio > 1.5:
            signal = "BULLISH"
        elif imbalance_ratio > 1.2:
            signal = "SLIGHTLY_BULLISH"
        elif imbalance_ratio < 0.67:
            signal = "BEARISH"
        elif imbalance_ratio < 0.83:
            signal = "SLIGHTLY_BEARISH"
        else:
            signal = "NEUTRAL"

        return OrderBookImbalance(
            symbol=self.ob.symbol,
            imbalance_ratio=imbalance_ratio,
            bid_volume=bid_volume,
            ask_volume=ask_volume,
            imbalance_percentage=imbalance_pct,
            signal=signal,
            depth_levels=depth_levels
        )

    def detect_whale_walls(self, min_multiplier: float = 5.0) -> List[WhaleWall]:
        """Detect large orders (whale walls)."""
        walls = []

        # Calculate average order size
        all_quantities = [b.quantity for b in self.ob.bids] + [a.quantity for a in self.ob.asks]
        avg_size = np.mean(all_quantities) if all_quantities else 0

        if avg_size == 0:
            return walls

        threshold = avg_size * min_multiplier

        # Check bids for walls
        for bid in self.ob.bids:
            if bid.quantity >= threshold:
                distance = (self.ob.mid_price - bid.price) / self.ob.mid_price * 100

                # Determine significance
                multiplier = bid.quantity / avg_size
                if multiplier > 20:
                    significance = "CRITICAL"
                elif multiplier > 10:
                    significance = "HIGH"
                elif multiplier > 5:
                    significance = "MEDIUM"
                else:
                    significance = "LOW"

                walls.append(WhaleWall(
                    symbol=self.ob.symbol,
                    price=bid.price,
                    quantity=bid.quantity,
                    value_usd=bid.total_value,
                    side="BID",
                    distance_from_price=distance,
                    significance=significance
                ))

        # Check asks for walls
        for ask in self.ob.asks:
            if ask.quantity >= threshold:
                distance = (ask.price - self.ob.mid_price) / self.ob.mid_price * 100

                multiplier = ask.quantity / avg_size
                if multiplier > 20:
                    significance = "CRITICAL"
                elif multiplier > 10:
                    significance = "HIGH"
                elif multiplier > 5:
                    significance = "MEDIUM"
                else:
                    significance = "LOW"

                walls.append(WhaleWall(
                    symbol=self.ob.symbol,
                    price=ask.price,
                    quantity=ask.quantity,
                    value_usd=ask.total_value,
                    side="ASK",
                    distance_from_price=distance,
                    significance=significance
                ))

        return sorted(walls, key=lambda x: x.value_usd, reverse=True)

    def find_support_resistance(self, num_levels: int = 5) -> Tuple[List[SupportResistance], List[SupportResistance]]:
        """Find support and resistance levels from order book."""
        supports = []
        resistances = []

        # Group bids by price zones (0.5% buckets)
        bid_zones = {}
        for bid in self.ob.bids:
            zone = round(bid.price / self.ob.mid_price * 200) / 200  # 0.5% zones
            if zone not in bid_zones:
                bid_zones[zone] = {"volume": 0, "count": 0, "price": bid.price}
            bid_zones[zone]["volume"] += bid.quantity
            bid_zones[zone]["count"] += 1

        # Find strongest support zones
        for zone_key in sorted(bid_zones.keys(), key=lambda x: bid_zones[x]["volume"], reverse=True)[:num_levels]:
            zone = bid_zones[zone_key]
            max_volume = max(z["volume"] for z in bid_zones.values())

            supports.append(SupportResistance(
                symbol=self.ob.symbol,
                price=zone["price"],
                strength=(zone["volume"] / max_volume * 100) if max_volume > 0 else 0,
                type="SUPPORT",
                total_volume=zone["volume"],
                wall_count=zone["count"]
            ))

        # Group asks by price zones
        ask_zones = {}
        for ask in self.ob.asks:
            zone = round(ask.price / self.ob.mid_price * 200) / 200
            if zone not in ask_zones:
                ask_zones[zone] = {"volume": 0, "count": 0, "price": ask.price}
            ask_zones[zone]["volume"] += ask.quantity
            ask_zones[zone]["count"] += 1

        # Find strongest resistance zones
        for zone_key in sorted(ask_zones.keys(), key=lambda x: ask_zones[x]["volume"], reverse=True)[:num_levels]:
            zone = ask_zones[zone_key]
            max_volume = max(z["volume"] for z in ask_zones.values())

            resistances.append(SupportResistance(
                symbol=self.ob.symbol,
                price=zone["price"],
                strength=(zone["volume"] / max_volume * 100) if max_volume > 0 else 0,
                type="RESISTANCE",
                total_volume=zone["volume"],
                wall_count=zone["count"]
            ))

        return supports, resistances

    def analyze_liquidity(self) -> LiquidityAnalysis:
        """Analyze order book liquidity."""
        # Calculate liquidity at different depths
        bid_1pct = sum(b.quantity for b in self.ob.bids
                       if b.price >= self.ob.mid_price * 0.99)
        ask_1pct = sum(a.quantity for a in self.ob.asks
                       if a.price <= self.ob.mid_price * 1.01)

        bid_5pct = sum(b.quantity for b in self.ob.bids
                       if b.price >= self.ob.mid_price * 0.95)
        ask_5pct = sum(a.quantity for a in self.ob.asks
                       if a.price <= self.ob.mid_price * 1.05)

        # Total book value
        total_bid_value = sum(b.total_value for b in self.ob.bids)
        total_ask_value = sum(a.total_value for a in self.ob.asks)
        total_book_value = total_bid_value + total_ask_value

        # Liquidity score (based on depth and spread)
        depth_score = min(100, (bid_1pct + ask_1pct) / 1000 * 50)  # Arbitrary scaling
        spread_score = max(0, 100 - self.ob.spread_percentage * 100)
        liquidity_score = (depth_score + spread_score) / 2

        # Estimate slippage for $10k order
        target_value = 10000
        slippage = self._estimate_slippage(target_value)

        return LiquidityAnalysis(
            symbol=self.ob.symbol,
            bid_liquidity_1pct=bid_1pct * self.ob.mid_price,
            ask_liquidity_1pct=ask_1pct * self.ob.mid_price,
            bid_liquidity_5pct=bid_5pct * self.ob.mid_price,
            ask_liquidity_5pct=ask_5pct * self.ob.mid_price,
            total_book_value=total_book_value,
            liquidity_score=liquidity_score,
            slippage_estimate_10k=slippage
        )

    def _estimate_slippage(self, order_value: float) -> float:
        """Estimate slippage for a market order."""
        remaining_value = order_value
        total_cost = 0
        total_quantity = 0

        for ask in self.ob.asks:
            level_value = ask.total_value

            if level_value >= remaining_value:
                # Partial fill at this level
                qty = remaining_value / ask.price
                total_cost += qty * ask.price
                total_quantity += qty
                remaining_value = 0
                break
            else:
                # Full fill at this level
                total_cost += ask.total_value
                total_quantity += ask.quantity
                remaining_value -= level_value

        if total_quantity == 0:
            return 0

        avg_price = total_cost / total_quantity
        slippage = (avg_price - self.ob.asks[0].price) / self.ob.asks[0].price * 100

        return slippage


# =============================================================================
# ORDER FLOW MOMENTUM
# =============================================================================

class OrderFlowMomentum:
    """Calculate order flow momentum signals."""

    def __init__(self):
        self.history = []  # Store historical snapshots

    def add_snapshot(self, imbalance: OrderBookImbalance):
        """Add imbalance snapshot to history."""
        self.history.append({
            "timestamp": datetime.now(),
            "imbalance_ratio": imbalance.imbalance_ratio,
            "bid_volume": imbalance.bid_volume,
            "ask_volume": imbalance.ask_volume
        })

        # Keep last 100 snapshots
        if len(self.history) > 100:
            self.history = self.history[-100:]

    def calculate_momentum(self) -> Dict:
        """Calculate order flow momentum."""
        if len(self.history) < 2:
            return {
                "momentum": 0,
                "trend": "UNKNOWN",
                "strength": 0
            }

        # Calculate change in imbalance ratio
        recent = self.history[-1]["imbalance_ratio"]
        older = self.history[-min(10, len(self.history))]["imbalance_ratio"]

        momentum = recent - older

        # Calculate trend
        if momentum > 0.2:
            trend = "BULLISH_ACCELERATION"
        elif momentum > 0.1:
            trend = "BULLISH"
        elif momentum < -0.2:
            trend = "BEARISH_ACCELERATION"
        elif momentum < -0.1:
            trend = "BEARISH"
        else:
            trend = "NEUTRAL"

        # Strength based on consistency
        changes = []
        for i in range(1, len(self.history)):
            changes.append(
                self.history[i]["imbalance_ratio"] - self.history[i-1]["imbalance_ratio"]
            )

        if changes:
            positive_changes = sum(1 for c in changes if c > 0)
            strength = abs(positive_changes / len(changes) - 0.5) * 2  # 0 to 1
        else:
            strength = 0

        return {
            "momentum": momentum,
            "trend": trend,
            "strength": strength
        }


# =============================================================================
# ORDER FLOW SIGNAL GENERATOR
# =============================================================================

class OrderFlowSignalGenerator:
    """Generate trading signals from order flow analysis."""

    def generate_signal(self, orderbook: OrderBookSnapshot) -> OrderFlowSignal:
        """Generate comprehensive order flow signal."""
        analyzer = OrderBookAnalyzer(orderbook)

        # Get all metrics
        imbalance = analyzer.calculate_imbalance(depth_levels=20)
        walls = analyzer.detect_whale_walls()
        supports, resistances = analyzer.find_support_resistance()
        liquidity = analyzer.analyze_liquidity()

        # Score components
        scores = []
        reasons = []

        # 1. Imbalance signal (weight: 40%)
        if imbalance.imbalance_ratio > 1.5:
            scores.append(2)
            reasons.append(f"Strong bid imbalance ({imbalance.imbalance_ratio:.2f}x)")
        elif imbalance.imbalance_ratio > 1.2:
            scores.append(1)
            reasons.append(f"Moderate bid imbalance ({imbalance.imbalance_ratio:.2f}x)")
        elif imbalance.imbalance_ratio < 0.67:
            scores.append(-2)
            reasons.append(f"Strong ask imbalance ({imbalance.imbalance_ratio:.2f}x)")
        elif imbalance.imbalance_ratio < 0.83:
            scores.append(-1)
            reasons.append(f"Moderate ask imbalance ({imbalance.imbalance_ratio:.2f}x)")
        else:
            scores.append(0)

        # 2. Whale wall signal (weight: 30%)
        bid_walls = [w for w in walls if w.side == "BID" and w.distance_from_price < 2]
        ask_walls = [w for w in walls if w.side == "ASK" and w.distance_from_price < 2]

        if bid_walls and not ask_walls:
            scores.append(1.5)
            reasons.append(f"Strong bid wall at {bid_walls[0].price:.4f}")
        elif ask_walls and not bid_walls:
            scores.append(-1.5)
            reasons.append(f"Strong ask wall at {ask_walls[0].price:.4f}")
        elif bid_walls and ask_walls:
            bid_total = sum(w.value_usd for w in bid_walls)
            ask_total = sum(w.value_usd for w in ask_walls)
            if bid_total > ask_total * 1.5:
                scores.append(1)
                reasons.append("Bid walls stronger than ask walls")
            elif ask_total > bid_total * 1.5:
                scores.append(-1)
                reasons.append("Ask walls stronger than bid walls")
            else:
                scores.append(0)
        else:
            scores.append(0)

        # 3. Support/Resistance proximity (weight: 20%)
        if supports:
            nearest_support = min(supports, key=lambda x: abs(x.price - orderbook.mid_price))
            support_distance = (orderbook.mid_price - nearest_support.price) / orderbook.mid_price * 100

            if support_distance < 0.5 and nearest_support.strength > 50:
                scores.append(1)
                reasons.append(f"Near strong support at {nearest_support.price:.4f}")

        if resistances:
            nearest_resistance = min(resistances, key=lambda x: abs(x.price - orderbook.mid_price))
            resistance_distance = (nearest_resistance.price - orderbook.mid_price) / orderbook.mid_price * 100

            if resistance_distance < 0.5 and nearest_resistance.strength > 50:
                scores.append(-1)
                reasons.append(f"Near strong resistance at {nearest_resistance.price:.4f}")

        # 4. Liquidity signal (weight: 10%)
        if liquidity.liquidity_score > 70:
            scores.append(0.5)
            reasons.append("High liquidity - easy execution")
        elif liquidity.liquidity_score < 30:
            scores.append(-0.5)
            reasons.append("Low liquidity - potential slippage")

        # Calculate final signal
        total_score = sum(scores)

        if total_score >= 3:
            signal = "STRONG_BUY"
            confidence = min(1, total_score / 5)
        elif total_score >= 1.5:
            signal = "BUY"
            confidence = 0.6 + (total_score - 1.5) / 10
        elif total_score <= -3:
            signal = "STRONG_SELL"
            confidence = min(1, abs(total_score) / 5)
        elif total_score <= -1.5:
            signal = "SELL"
            confidence = 0.6 + (abs(total_score) - 1.5) / 10
        else:
            signal = "NEUTRAL"
            confidence = 0.5

        # Entry/Stop suggestions
        if signal in ["BUY", "STRONG_BUY"] and supports:
            entry = orderbook.mid_price
            stop = supports[0].price * 0.99  # Below support
        elif signal in ["SELL", "STRONG_SELL"] and resistances:
            entry = orderbook.mid_price
            stop = resistances[0].price * 1.01  # Above resistance
        else:
            entry = None
            stop = None

        return OrderFlowSignal(
            symbol=orderbook.symbol,
            signal=signal,
            confidence=confidence,
            reasons=reasons,
            entry_suggestion=entry,
            stop_suggestion=stop
        )


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_order_flow_analysis():
    """Run full order flow analysis."""
    print("=" * 70)
    print("ORDER FLOW ANALYZER - Order Book Depth Analysis")
    print("=" * 70)
    print()

    fetcher = OrderBookFetcher()
    signal_gen = OrderFlowSignalGenerator()

    symbols = ["XRPUSDT", "SOLUSDT", "ATOMUSDT"]

    for symbol in symbols:
        print(f"\n{'='*70}")
        print(f"ANALYSIS: {symbol}")
        print("=" * 70)

        # Fetch order book
        ob = fetcher.fetch_orderbook(symbol, limit=100)

        if not ob:
            print(f"  Could not fetch order book for {symbol}")
            continue

        analyzer = OrderBookAnalyzer(ob)

        # Basic Stats
        print("\n[1] ORDER BOOK OVERVIEW")
        print("-" * 50)
        print(f"  Mid Price: ${ob.mid_price:.4f}")
        print(f"  Bid-Ask Spread: ${ob.bid_ask_spread:.6f}")
        print(f"  Spread %: {ob.spread_percentage:.4f}%")
        print(f"  Best Bid: ${ob.bids[0].price:.4f}")
        print(f"  Best Ask: ${ob.asks[0].price:.4f}")

        # Imbalance
        print("\n[2] BID/ASK IMBALANCE")
        print("-" * 50)
        imbalance = analyzer.calculate_imbalance(depth_levels=20)
        print(f"  Imbalance Ratio: {imbalance.imbalance_ratio:.2f}")
        print(f"  Bid Volume: {imbalance.bid_volume:,.2f}")
        print(f"  Ask Volume: {imbalance.ask_volume:,.2f}")
        print(f"  Net Imbalance: {imbalance.imbalance_percentage:+.1f}%")
        print(f"  Signal: {imbalance.signal}")

        # Visualize imbalance
        bid_pct = imbalance.bid_volume / (imbalance.bid_volume + imbalance.ask_volume) * 100
        bid_bar = int(bid_pct / 5)
        ask_bar = 20 - bid_bar
        print(f"\n  BIDS [{'█' * bid_bar}{'░' * ask_bar}] ASKS")
        print(f"       {bid_pct:.0f}%{' ' * 14}{100-bid_pct:.0f}%")

        # Whale Walls
        print("\n[3] WHALE WALLS (Large Orders)")
        print("-" * 50)
        walls = analyzer.detect_whale_walls()

        if walls:
            bid_walls = [w for w in walls if w.side == "BID"][:3]
            ask_walls = [w for w in walls if w.side == "ASK"][:3]

            print("  BID WALLS (Support):")
            for wall in bid_walls:
                emoji = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}
                print(f"    {emoji.get(wall.significance, '⚪')} ${wall.price:.4f} - "
                      f"{wall.quantity:,.0f} (${wall.value_usd:,.0f}) "
                      f"[{wall.significance}]")

            print("\n  ASK WALLS (Resistance):")
            for wall in ask_walls:
                emoji = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}
                print(f"    {emoji.get(wall.significance, '⚪')} ${wall.price:.4f} - "
                      f"{wall.quantity:,.0f} (${wall.value_usd:,.0f}) "
                      f"[{wall.significance}]")
        else:
            print("  No significant whale walls detected")

        # Support/Resistance
        print("\n[4] SUPPORT/RESISTANCE LEVELS")
        print("-" * 50)
        supports, resistances = analyzer.find_support_resistance()

        print("  SUPPORT LEVELS:")
        for s in supports[:3]:
            bar_len = int(s.strength / 10)
            print(f"    ${s.price:.4f} - Strength: {'█' * bar_len}{'░' * (10-bar_len)} {s.strength:.0f}%")

        print("\n  RESISTANCE LEVELS:")
        for r in resistances[:3]:
            bar_len = int(r.strength / 10)
            print(f"    ${r.price:.4f} - Strength: {'█' * bar_len}{'░' * (10-bar_len)} {r.strength:.0f}%")

        # Liquidity Analysis
        print("\n[5] LIQUIDITY ANALYSIS")
        print("-" * 50)
        liquidity = analyzer.analyze_liquidity()

        print(f"  Liquidity Score: {liquidity.liquidity_score:.0f}/100")
        print(f"  Bid Depth (1%): ${liquidity.bid_liquidity_1pct:,.0f}")
        print(f"  Ask Depth (1%): ${liquidity.ask_liquidity_1pct:,.0f}")
        print(f"  Bid Depth (5%): ${liquidity.bid_liquidity_5pct:,.0f}")
        print(f"  Ask Depth (5%): ${liquidity.ask_liquidity_5pct:,.0f}")
        print(f"  Total Book Value: ${liquidity.total_book_value:,.0f}")
        print(f"  Est. Slippage ($10k): {liquidity.slippage_estimate_10k:.4f}%")

        # Trading Signal
        print("\n[6] ORDER FLOW SIGNAL")
        print("-" * 50)
        signal = signal_gen.generate_signal(ob)

        signal_emoji = {
            "STRONG_BUY": "🟢🟢",
            "BUY": "🟢",
            "NEUTRAL": "⚪",
            "SELL": "🔴",
            "STRONG_SELL": "🔴🔴"
        }

        print(f"  Signal: {signal_emoji.get(signal.signal, '')} {signal.signal}")
        print(f"  Confidence: {signal.confidence:.1%}")
        print("\n  Reasons:")
        for reason in signal.reasons:
            print(f"    • {reason}")

        if signal.entry_suggestion:
            print(f"\n  Entry: ${signal.entry_suggestion:.4f}")
        if signal.stop_suggestion:
            print(f"  Stop: ${signal.stop_suggestion:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY - ORDER FLOW SIGNALS")
    print("=" * 70)

    for symbol in symbols:
        ob = fetcher.fetch_orderbook(symbol, limit=100)
        if ob:
            signal = signal_gen.generate_signal(ob)
            emoji = {"STRONG_BUY": "🟢🟢", "BUY": "🟢", "NEUTRAL": "⚪", "SELL": "🔴", "STRONG_SELL": "🔴🔴"}
            print(f"  {symbol}: {emoji.get(signal.signal, '')} {signal.signal} ({signal.confidence:.0%})")

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)


if __name__ == "__main__":
    run_order_flow_analysis()
