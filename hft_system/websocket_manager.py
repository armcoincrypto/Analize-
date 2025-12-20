"""
WebSocket Manager for HFT System
================================
Handles real-time data streams from Binance:
- Trade streams (real-time trades)
- Kline streams (candlestick data)
- Order book depth streams
- Mark price streams (for derivatives signals)

All data pushed to Signal Engine via callbacks.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Callable, Optional, List, Any
from dataclasses import dataclass, field
from collections import deque
import websockets
from websockets.exceptions import ConnectionClosed

from .config import SYSTEM_CONFIG, ASSETS, get_asset_config

logger = logging.getLogger(__name__)


@dataclass
class TradeData:
    """Real-time trade data."""
    symbol: str
    price: float
    quantity: float
    timestamp: int
    is_buyer_maker: bool


@dataclass
class KlineData:
    """Candlestick data."""
    symbol: str
    interval: str
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int
    is_closed: bool


@dataclass
class OrderBookData:
    """Order book snapshot."""
    symbol: str
    bids: List[tuple]  # [(price, qty), ...]
    asks: List[tuple]  # [(price, qty), ...]
    timestamp: int

    @property
    def best_bid(self) -> float:
        return self.bids[0][0] if self.bids else 0

    @property
    def best_ask(self) -> float:
        return self.asks[0][0] if self.asks else 0

    @property
    def spread(self) -> float:
        if self.best_bid and self.best_ask:
            return (self.best_ask - self.best_bid) / self.best_bid * 100
        return 0

    @property
    def bid_volume(self) -> float:
        return sum(qty for _, qty in self.bids)

    @property
    def ask_volume(self) -> float:
        return sum(qty for _, qty in self.asks)

    @property
    def imbalance_ratio(self) -> float:
        """Bid/Ask volume ratio. >0.5 = more buyers."""
        total = self.bid_volume + self.ask_volume
        if total == 0:
            return 0.5
        return self.bid_volume / total


@dataclass
class MarkPriceData:
    """Mark price and funding rate data."""
    symbol: str
    mark_price: float
    index_price: float
    funding_rate: float
    next_funding_time: int
    timestamp: int


class TradeFlowBuffer:
    """
    MICROSTRUCTURE Trade Flow Tracker.

    Tracks aggressive order flow for entry/exit decisions:
    - Aggressive buy volume (market buys hitting asks)
    - Aggressive sell volume (market sells hitting bids)
    - Trades per second
    - Net delta over rolling 1s and 3s windows

    is_buyer_maker=True means seller was aggressor (selling pressure)
    is_buyer_maker=False means buyer was aggressor (buying pressure)
    """

    def __init__(self, max_size: int = 10000):
        self.trades: deque = deque(maxlen=max_size)  # (timestamp_ms, volume, is_buy, price)

    def add_trade(self, timestamp: int, volume: float, price: float, is_buyer_maker: bool):
        """Add a trade. is_buyer_maker=False means buyer was aggressor."""
        is_buy = not is_buyer_maker  # Buyer aggressor = buy
        self.trades.append((timestamp, volume, is_buy, price))

    def get_trade_flow(self, window_ms: int = 1000) -> dict:
        """
        Get trade flow stats for microstructure analysis.

        Returns:
            aggressive_buy_vol: Volume from market buys
            aggressive_sell_vol: Volume from market sells
            trades_per_second: Number of trades in window
            net_delta: buy_vol - sell_vol
            delta_pct: net_delta as % of total volume
        """
        if not self.trades:
            return {
                "aggressive_buy_vol": 0,
                "aggressive_sell_vol": 0,
                "trades_per_second": 0,
                "net_delta": 0,
                "delta_pct": 0,
                "trade_count": 0
            }

        now = self.trades[-1][0]
        cutoff = now - window_ms

        buy_vol = 0
        sell_vol = 0
        trade_count = 0

        for ts, vol, is_buy, price in self.trades:
            if ts >= cutoff:
                trade_count += 1
                if is_buy:
                    buy_vol += vol
                else:
                    sell_vol += vol

        total_vol = buy_vol + sell_vol
        net_delta = buy_vol - sell_vol
        delta_pct = (net_delta / total_vol * 100) if total_vol > 0 else 0
        trades_per_sec = trade_count / (window_ms / 1000)

        return {
            "aggressive_buy_vol": buy_vol,
            "aggressive_sell_vol": sell_vol,
            "trades_per_second": trades_per_sec,
            "net_delta": net_delta,
            "delta_pct": delta_pct,
            "trade_count": trade_count
        }

    def get_delta_1s(self) -> float:
        """Net delta over last 1 second."""
        flow = self.get_trade_flow(1000)
        return flow["net_delta"]

    def get_delta_3s(self) -> float:
        """Net delta over last 3 seconds."""
        flow = self.get_trade_flow(3000)
        return flow["net_delta"]

    def is_delta_negative(self, window_ms: int = 1000) -> bool:
        """Check if trade flow is against buyers."""
        return self.get_trade_flow(window_ms)["net_delta"] < 0

    def is_delta_positive(self, window_ms: int = 1000) -> bool:
        """Check if trade flow favors buyers."""
        return self.get_trade_flow(window_ms)["net_delta"] > 0

    def get_cvd(self, window_sec: int) -> dict:
        """Get CVD stats (backward compatible)."""
        flow = self.get_trade_flow(window_sec * 1000)
        return {
            "cvd": flow["net_delta"],
            "buy_volume": flow["aggressive_buy_vol"],
            "sell_volume": flow["aggressive_sell_vol"]
        }


# Backward compatibility alias
CVDBuffer = TradeFlowBuffer


class PriceBuffer:
    """Circular buffer for tracking price history."""

    def __init__(self, max_size: int = 1000):
        self.prices: deque = deque(maxlen=max_size)
        self.timestamps: deque = deque(maxlen=max_size)

    def add(self, price: float, timestamp: int):
        self.prices.append(price)
        self.timestamps.append(timestamp)

    def get_price_change(self, window_sec: int) -> Optional[float]:
        """Get price change % over the last N seconds."""
        if len(self.prices) < 2:
            return None

        now = self.timestamps[-1]
        cutoff = now - (window_sec * 1000)  # Convert to ms

        # Find oldest price within window
        for i, ts in enumerate(self.timestamps):
            if ts >= cutoff:
                old_price = self.prices[i]
                new_price = self.prices[-1]
                return (new_price - old_price) / old_price * 100

        return None

    @property
    def current_price(self) -> Optional[float]:
        return self.prices[-1] if self.prices else None


class VolumeBuffer:
    """Track volume over time windows."""

    def __init__(self, max_size: int = 1000):
        self.volumes: deque = deque(maxlen=max_size)
        self.timestamps: deque = deque(maxlen=max_size)

    def add(self, volume: float, timestamp: int):
        self.volumes.append(volume)
        self.timestamps.append(timestamp)

    def get_volume_sum(self, window_sec: int) -> float:
        """Sum volume over last N seconds."""
        if not self.volumes:
            return 0

        now = self.timestamps[-1]
        cutoff = now - (window_sec * 1000)

        total = 0
        for i, ts in enumerate(self.timestamps):
            if ts >= cutoff:
                total += self.volumes[i]

        return total

    def get_average_volume(self, window_sec: int) -> float:
        """Average volume per second over window."""
        total = self.get_volume_sum(window_sec)
        return total / window_sec if window_sec > 0 else 0


class WebSocketManager:
    """
    Manages WebSocket connections to Binance.

    Streams:
    - aggTrade: Real-time trades
    - kline_1m: 1-minute candles
    - depth20@100ms: Order book (top 20 levels, 100ms updates)
    - markPrice@1s: Mark price and funding rate
    """

    BASE_URL = "wss://stream.binance.com:9443/stream"
    FUTURES_URL = "wss://fstream.binance.com/stream"

    def __init__(self):
        self.running = False
        self.ws_spot = None
        self.ws_futures = None

        # Data buffers per symbol
        self.price_buffers: Dict[str, PriceBuffer] = {}
        self.volume_buffers: Dict[str, VolumeBuffer] = {}
        self.cvd_buffers: Dict[str, CVDBuffer] = {}  # CVD tracking
        self.orderbooks: Dict[str, OrderBookData] = {}
        self.mark_prices: Dict[str, MarkPriceData] = {}
        self.latest_klines: Dict[str, KlineData] = {}

        # Callbacks
        self.on_trade: Optional[Callable[[TradeData], None]] = None
        self.on_kline: Optional[Callable[[KlineData], None]] = None
        self.on_orderbook: Optional[Callable[[OrderBookData], None]] = None
        self.on_mark_price: Optional[Callable[[MarkPriceData], None]] = None

        # Connection stats
        self.last_message_time: Dict[str, float] = {}
        self.message_count = 0
        self.reconnect_count = 0

        # Initialize buffers for enabled assets
        for symbol in SYSTEM_CONFIG.enabled_assets:
            config = get_asset_config(symbol)
            self.price_buffers[config.exchange_symbol] = PriceBuffer()
            self.volume_buffers[config.exchange_symbol] = VolumeBuffer()
            self.cvd_buffers[config.exchange_symbol] = CVDBuffer()

    def _build_stream_names(self) -> tuple:
        """Build stream names for spot and futures."""
        spot_streams = []
        futures_streams = []

        for symbol in SYSTEM_CONFIG.enabled_assets:
            config = get_asset_config(symbol)
            sym_lower = config.exchange_symbol.lower()

            # Spot streams
            spot_streams.append(f"{sym_lower}@aggTrade")
            spot_streams.append(f"{sym_lower}@kline_1m")
            spot_streams.append(f"{sym_lower}@depth20@100ms")

            # Futures streams (for funding rate)
            futures_streams.append(f"{sym_lower}@markPrice@1s")

        return spot_streams, futures_streams

    async def connect(self):
        """Establish WebSocket connections."""
        self.running = True
        spot_streams, futures_streams = self._build_stream_names()

        # Build URLs with combined streams
        spot_url = f"{self.BASE_URL}?streams={'/'.join(spot_streams)}"
        futures_url = f"{self.FUTURES_URL}?streams={'/'.join(futures_streams)}"

        logger.info(f"Connecting to {len(spot_streams)} spot streams...")
        logger.info(f"Connecting to {len(futures_streams)} futures streams...")

        # Run both connections concurrently
        await asyncio.gather(
            self._run_spot_connection(spot_url),
            self._run_futures_connection(futures_url),
            return_exceptions=True
        )

    async def _run_spot_connection(self, url: str):
        """Manage spot WebSocket connection with auto-reconnect."""
        while self.running:
            try:
                async with websockets.connect(
                    url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5
                ) as ws:
                    self.ws_spot = ws
                    logger.info("Spot WebSocket connected")

                    async for message in ws:
                        await self._handle_spot_message(message)

            except ConnectionClosed as e:
                logger.warning(f"Spot WebSocket closed: {e}")
            except Exception as e:
                logger.error(f"Spot WebSocket error: {e}")

            if self.running:
                self.reconnect_count += 1
                delay = min(SYSTEM_CONFIG.ws_reconnect_delay_sec * (2 ** min(self.reconnect_count, 5)), 60)
                logger.info(f"Reconnecting spot in {delay}s...")
                await asyncio.sleep(delay)

    async def _run_futures_connection(self, url: str):
        """Manage futures WebSocket connection with auto-reconnect."""
        while self.running:
            try:
                async with websockets.connect(
                    url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5
                ) as ws:
                    self.ws_futures = ws
                    logger.info("Futures WebSocket connected")

                    async for message in ws:
                        await self._handle_futures_message(message)

            except ConnectionClosed as e:
                logger.warning(f"Futures WebSocket closed: {e}")
            except Exception as e:
                logger.error(f"Futures WebSocket error: {e}")

            if self.running:
                delay = min(SYSTEM_CONFIG.ws_reconnect_delay_sec * 2, 60)
                logger.info(f"Reconnecting futures in {delay}s...")
                await asyncio.sleep(delay)

    async def _handle_spot_message(self, message: str):
        """Process incoming spot market message."""
        try:
            data = json.loads(message)
            self.message_count += 1

            if "stream" not in data:
                return

            stream = data["stream"]
            payload = data["data"]

            if "@aggTrade" in stream:
                await self._process_trade(payload)
            elif "@kline" in stream:
                await self._process_kline(payload)
            elif "@depth" in stream:
                await self._process_orderbook(stream, payload)

        except json.JSONDecodeError:
            logger.error(f"Invalid JSON: {message[:100]}")
        except Exception as e:
            logger.error(f"Error processing spot message: {e}")

    async def _handle_futures_message(self, message: str):
        """Process incoming futures market message."""
        try:
            data = json.loads(message)

            if "stream" not in data:
                return

            stream = data["stream"]
            payload = data["data"]

            if "@markPrice" in stream:
                await self._process_mark_price(payload)

        except json.JSONDecodeError:
            logger.error(f"Invalid JSON: {message[:100]}")
        except Exception as e:
            logger.error(f"Error processing futures message: {e}")

    async def _process_trade(self, data: dict):
        """Process aggregated trade data."""
        symbol = data["s"]
        trade = TradeData(
            symbol=symbol,
            price=float(data["p"]),
            quantity=float(data["q"]),
            timestamp=data["T"],
            is_buyer_maker=data["m"]
        )

        # Update buffers
        if symbol in self.price_buffers:
            self.price_buffers[symbol].add(trade.price, trade.timestamp)
            self.volume_buffers[symbol].add(
                trade.quantity * trade.price,  # Volume in quote currency
                trade.timestamp
            )

        # Update CVD/TradeFlow buffer
        if symbol in self.cvd_buffers:
            self.cvd_buffers[symbol].add_trade(
                trade.timestamp,
                trade.quantity * trade.price,  # Volume in quote currency
                trade.price,  # Price for weighted analysis
                trade.is_buyer_maker
            )

        self.last_message_time[symbol] = time.time()

        # Callback
        if self.on_trade:
            self.on_trade(trade)

    async def _process_kline(self, data: dict):
        """Process kline/candlestick data."""
        k = data["k"]
        kline = KlineData(
            symbol=data["s"],
            interval=k["i"],
            open_time=k["t"],
            open=float(k["o"]),
            high=float(k["h"]),
            low=float(k["l"]),
            close=float(k["c"]),
            volume=float(k["v"]),
            close_time=k["T"],
            is_closed=k["x"]
        )

        self.latest_klines[kline.symbol] = kline
        self.last_message_time[f"{kline.symbol}_kline"] = time.time()

        # Callback
        if self.on_kline:
            self.on_kline(kline)

    async def _process_orderbook(self, stream: str, data: dict):
        """Process order book depth data."""
        # Extract symbol from stream name (e.g., "atomusdt@depth20@100ms")
        symbol = stream.split("@")[0].upper()

        orderbook = OrderBookData(
            symbol=symbol,
            bids=[(float(p), float(q)) for p, q in data["bids"]],
            asks=[(float(p), float(q)) for p, q in data["asks"]],
            timestamp=int(time.time() * 1000)
        )

        self.orderbooks[symbol] = orderbook
        self.last_message_time[f"{symbol}_book"] = time.time()

        # Callback
        if self.on_orderbook:
            self.on_orderbook(orderbook)

    async def _process_mark_price(self, data: dict):
        """Process mark price and funding rate."""
        mark = MarkPriceData(
            symbol=data["s"],
            mark_price=float(data["p"]),
            index_price=float(data["i"]),
            funding_rate=float(data["r"]),
            next_funding_time=data["T"],
            timestamp=int(time.time() * 1000)
        )

        self.mark_prices[mark.symbol] = mark
        self.last_message_time[f"{mark.symbol}_mark"] = time.time()

        # Callback
        if self.on_mark_price:
            self.on_mark_price(mark)

    def get_price_change(self, symbol: str, window_sec: int) -> Optional[float]:
        """Get price change % for symbol over window."""
        exchange_symbol = get_asset_config(symbol).exchange_symbol
        buffer = self.price_buffers.get(exchange_symbol)
        if buffer:
            return buffer.get_price_change(window_sec)
        return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol."""
        exchange_symbol = get_asset_config(symbol).exchange_symbol
        buffer = self.price_buffers.get(exchange_symbol)
        if buffer:
            return buffer.current_price
        return None

    def get_volume_spike(self, symbol: str, short_window: int = 60, long_window: int = 300) -> Optional[float]:
        """Get volume spike ratio (short/long window average)."""
        exchange_symbol = get_asset_config(symbol).exchange_symbol
        buffer = self.volume_buffers.get(exchange_symbol)
        if buffer:
            short_avg = buffer.get_average_volume(short_window)
            long_avg = buffer.get_average_volume(long_window)
            if long_avg > 0:
                return short_avg / long_avg
        return None

    def get_orderbook(self, symbol: str) -> Optional[OrderBookData]:
        """Get current order book for symbol."""
        exchange_symbol = get_asset_config(symbol).exchange_symbol
        return self.orderbooks.get(exchange_symbol)

    def get_funding_rate(self, symbol: str) -> Optional[float]:
        """Get current funding rate for symbol."""
        exchange_symbol = get_asset_config(symbol).exchange_symbol
        mark = self.mark_prices.get(exchange_symbol)
        if mark:
            return mark.funding_rate
        return None

    def get_cvd(self, symbol: str) -> dict:
        """
        Get CVD (Cumulative Volume Delta) data for symbol.

        Returns dict with cvd_1m, cvd_5m, cvd_15m and buy/sell volumes.
        """
        exchange_symbol = get_asset_config(symbol).exchange_symbol
        buffer = self.cvd_buffers.get(exchange_symbol)

        if buffer:
            cvd_1m = buffer.get_cvd(60)
            cvd_5m = buffer.get_cvd(300)
            cvd_15m = buffer.get_cvd(900)

            return {
                "cvd_1m": cvd_1m["cvd"],
                "cvd_5m": cvd_5m["cvd"],
                "cvd_15m": cvd_15m["cvd"],
                "buy_volume_1m": cvd_1m["buy_volume"],
                "sell_volume_1m": cvd_1m["sell_volume"]
            }

        return {
            "cvd_1m": 0, "cvd_5m": 0, "cvd_15m": 0,
            "buy_volume_1m": 0, "sell_volume_1m": 0
        }

    def get_trade_flow(self, symbol: str, window_ms: int = 1000) -> dict:
        """
        Get MICROSTRUCTURE trade flow data for symbol.

        Returns:
            aggressive_buy_vol: Volume from market buys (hitting asks)
            aggressive_sell_vol: Volume from market sells (hitting bids)
            trades_per_second: Number of trades in window
            net_delta: buy_vol - sell_vol
            delta_pct: net_delta as % of total volume
            delta_1s: Net delta over 1 second
            delta_3s: Net delta over 3 seconds
        """
        exchange_symbol = get_asset_config(symbol).exchange_symbol
        buffer = self.cvd_buffers.get(exchange_symbol)

        if buffer:
            flow = buffer.get_trade_flow(window_ms)
            flow["delta_1s"] = buffer.get_delta_1s()
            flow["delta_3s"] = buffer.get_delta_3s()
            flow["is_delta_negative"] = buffer.is_delta_negative(window_ms)
            flow["is_delta_positive"] = buffer.is_delta_positive(window_ms)
            return flow

        return {
            "aggressive_buy_vol": 0,
            "aggressive_sell_vol": 0,
            "trades_per_second": 0,
            "net_delta": 0,
            "delta_pct": 0,
            "trade_count": 0,
            "delta_1s": 0,
            "delta_3s": 0,
            "is_delta_negative": False,
            "is_delta_positive": False
        }

    def get_price_velocity(self, symbol: str) -> dict:
        """
        TASK 8: Get price velocity (micro moves) for causality analysis.

        Returns price change % over 1s, 3s, 5s windows.
        """
        velocity_1s = self.get_price_change(symbol, 1) or 0
        velocity_3s = self.get_price_change(symbol, 3) or 0
        velocity_5s = self.get_price_change(symbol, 5) or 0

        return {
            "velocity_1s": velocity_1s,
            "velocity_3s": velocity_3s,
            "velocity_5s": velocity_5s
        }

    def get_orderbook_causality_data(self, symbol: str) -> dict:
        """
        TASK 8: Get detailed orderbook data for causality analysis.

        Returns top level prices/sizes and aggregated volumes.
        """
        orderbook = self.get_orderbook(symbol)
        if not orderbook:
            return {
                "bid1_price": 0,
                "bid1_size": 0,
                "ask1_price": 0,
                "ask1_size": 0,
                "top5_bid_volume": 0,
                "top5_ask_volume": 0,
                "imbalance_ratio": 0.5,
                "spread_pct": 0
            }

        # Get top level
        bid1_price = orderbook.bids[0][0] if orderbook.bids else 0
        bid1_size = orderbook.bids[0][1] if orderbook.bids else 0
        ask1_price = orderbook.asks[0][0] if orderbook.asks else 0
        ask1_size = orderbook.asks[0][1] if orderbook.asks else 0

        # Get top 5 levels volume
        top5_bid_volume = sum(qty for _, qty in orderbook.bids[:5])
        top5_ask_volume = sum(qty for _, qty in orderbook.asks[:5])

        return {
            "bid1_price": bid1_price,
            "bid1_size": bid1_size,
            "ask1_price": ask1_price,
            "ask1_size": ask1_size,
            "top5_bid_volume": top5_bid_volume,
            "top5_ask_volume": top5_ask_volume,
            "imbalance_ratio": orderbook.imbalance_ratio,
            "spread_pct": orderbook.spread
        }

    def get_btc_data(self) -> dict:
        """
        TASK 8: Get BTC correlation data for causality analysis.

        Returns BTC price changes and delta.
        """
        # Try to get BTC data (might not be in enabled_assets)
        btc_symbols = ["BTC", "BTCUSDT"]
        btc_price_1s = 0
        btc_price_3s = 0
        btc_delta_1s = 0

        for btc_sym in btc_symbols:
            try:
                exchange_sym = btc_sym.upper() + ("" if btc_sym.endswith("USDT") else "USDT")
                buffer = self.price_buffers.get(exchange_sym)
                if buffer:
                    btc_price_1s = buffer.get_price_change(1) or 0
                    btc_price_3s = buffer.get_price_change(3) or 0
                    break

                cvd_buffer = self.cvd_buffers.get(exchange_sym)
                if cvd_buffer:
                    btc_delta_1s = cvd_buffer.get_delta_1s()
                    break
            except:
                pass

        return {
            "price_change_1s": btc_price_1s,
            "price_change_3s": btc_price_3s,
            "delta_1s": btc_delta_1s
        }

    def classify_market_regime(self, symbol: str) -> tuple:
        """
        TASK 10: Classify market regime based on multiple factors.

        Returns: (regime_type, confidence, metrics_dict)

        Regime types:
        - low_vol_chop: Low volatility, no trend, range-bound
        - high_vol_trend: High volatility with directional move
        - mean_reversion: Overextended price returning to mean
        - liquidity_vacuum: Low depth, wide spreads, choppy
        - news_spike: Sudden volume/volatility spike (>3x normal)
        """
        metrics = {}

        # === 1. VOLATILITY METRICS ===
        velocity_1m = abs(self.get_price_change(symbol, 60) or 0)
        velocity_5m = abs(self.get_price_change(symbol, 300) or 0)
        velocity_1s = abs(self.get_price_change(symbol, 1) or 0)

        metrics["volatility_1m"] = velocity_1m
        metrics["volatility_5m"] = velocity_5m

        # Volatility ratio: recent vs longer term
        if velocity_5m > 0:
            metrics["volatility_ratio"] = velocity_1m / velocity_5m
        else:
            metrics["volatility_ratio"] = 1.0

        # === 2. RANGE METRICS ===
        exchange_symbol = get_asset_config(symbol).exchange_symbol
        kline = self.kline_data.get(exchange_symbol)
        if kline:
            # Range expansion: current range vs typical
            current_range = (kline.high - kline.low) / kline.open * 100 if kline.open > 0 else 0
            metrics["range_expansion"] = current_range
            metrics["range_compression"] = 1.0 / (current_range + 0.01)
        else:
            metrics["range_expansion"] = velocity_1m
            metrics["range_compression"] = 0.5

        # === 3. TREND METRICS ===
        price_change_30s = self.get_price_change(symbol, 30) or 0
        price_change_60s = self.get_price_change(symbol, 60) or 0

        # Trend strength: are moves persistent?
        if abs(price_change_60s) > 0.01:
            if (price_change_30s > 0 and price_change_60s > 0) or \
               (price_change_30s < 0 and price_change_60s < 0):
                metrics["trend_strength"] = min(abs(price_change_60s) * 10, 1.0)
                metrics["trend_direction"] = "up" if price_change_60s > 0 else "down"
            else:
                metrics["trend_strength"] = 0.2
                metrics["trend_direction"] = "choppy"
        else:
            metrics["trend_strength"] = 0.0
            metrics["trend_direction"] = "neutral"

        # === 4. ORDERBOOK PATTERNS ===
        orderbook = self.get_orderbook(symbol)
        if orderbook:
            imbalance = orderbook.imbalance_ratio
            spread = orderbook.spread
            depth = orderbook.bid_volume + orderbook.ask_volume

            # Imbalance stability: how extreme is imbalance?
            metrics["ob_imbalance_stability"] = 1.0 - abs(imbalance - 0.5) * 2
            metrics["ob_depth_ratio"] = min(depth / 10000, 1.0)  # Normalize depth
            metrics["spread_pct"] = spread
        else:
            metrics["ob_imbalance_stability"] = 0.5
            metrics["ob_depth_ratio"] = 0.5
            metrics["spread_pct"] = 0.1

        # === 5. TRADE FLOW PATTERNS ===
        trade_flow = self.get_trade_flow(symbol)
        if trade_flow:
            delta_1s = trade_flow.get("delta_1s", 0)
            delta_3s = trade_flow.get("delta_3s", 0)
            trades_per_sec = trade_flow.get("trades_per_second", 0)

            # Flow consistency: is delta stable or flipping?
            if delta_1s != 0 and delta_3s != 0:
                if (delta_1s > 0 and delta_3s > 0) or (delta_1s < 0 and delta_3s < 0):
                    metrics["flow_consistency"] = 0.8
                else:
                    metrics["flow_consistency"] = 0.2
            else:
                metrics["flow_consistency"] = 0.5

            # Large trade ratio: spike detection
            total_vol = trade_flow.get("aggressive_buy_vol", 0) + trade_flow.get("aggressive_sell_vol", 0)
            metrics["large_trade_ratio"] = min(total_vol / 100000, 1.0)
            metrics["trades_per_sec"] = trades_per_sec
        else:
            metrics["flow_consistency"] = 0.5
            metrics["large_trade_ratio"] = 0.0
            metrics["trades_per_sec"] = 0

        # === REGIME CLASSIFICATION ===
        regime = "unknown"
        confidence = 0.5

        vol_1m = metrics["volatility_1m"]
        vol_ratio = metrics["volatility_ratio"]
        trend = metrics["trend_strength"]
        spread = metrics.get("spread_pct", 0.1)
        depth = metrics["ob_depth_ratio"]
        flow_consistency = metrics["flow_consistency"]

        # NEWS_SPIKE: Sudden volume/volatility spike
        if vol_ratio > 2.5 or metrics["large_trade_ratio"] > 0.7:
            regime = "news_spike"
            confidence = min(vol_ratio / 3, 1.0)

        # HIGH_VOL_TREND: High volatility with directional move
        elif vol_1m > 0.15 and trend > 0.5 and flow_consistency > 0.6:
            regime = "high_vol_trend"
            confidence = min((trend + flow_consistency) / 2, 1.0)

        # LIQUIDITY_VACUUM: Low depth, wide spreads
        elif spread > 0.05 or depth < 0.3:
            regime = "liquidity_vacuum"
            confidence = max(spread / 0.1, 1 - depth)

        # MEAN_REVERSION: Overextended price, flow reversing
        elif vol_1m > 0.1 and flow_consistency < 0.3:
            regime = "mean_reversion"
            confidence = 1.0 - flow_consistency

        # LOW_VOL_CHOP: Low volatility, no trend
        elif vol_1m < 0.08 and trend < 0.3:
            regime = "low_vol_chop"
            confidence = 1.0 - max(vol_1m / 0.08, trend / 0.3)

        # Default: classify by volatility
        else:
            if vol_1m > 0.1:
                regime = "high_vol_trend" if trend > 0.4 else "mean_reversion"
            else:
                regime = "low_vol_chop"
            confidence = 0.4

        return regime, confidence, metrics

    def detect_no_trade_zone(self, symbol: str) -> tuple:
        """
        TASK 11: Detect no-trade zone conditions.

        Returns: (is_blocked, block_reason, block_details, market_conditions)

        Block conditions:
        - spread_unstable: Spread changed >50% in 1s
        - delta_noise: Delta variance too high (flipping rapidly)
        - ob_unstable: Top5 OB volume changed >30% in 1s
        - low_liquidity: Total depth below threshold
        """
        is_blocked = False
        block_reason = None
        block_details = None

        # Thresholds (can be tuned)
        SPREAD_CHANGE_THRESHOLD = 0.5   # 50% change in spread in 1s
        DELTA_VARIANCE_THRESHOLD = 0.8  # High variance = flipping
        OB_INSTABILITY_THRESHOLD = 0.3  # 30% change in top5 volume
        MIN_LIQUIDITY_DEPTH = 5000      # Minimum depth in quote currency

        market_conditions = {
            "spread_pct": 0,
            "spread_change_1s": 0,
            "delta_variance": 0,
            "ob_volume_instability": 0,
            "liquidity_depth": 0
        }

        # === 1. SPREAD STABILITY CHECK ===
        orderbook = self.get_orderbook(symbol)
        if orderbook:
            current_spread = orderbook.spread
            market_conditions["spread_pct"] = current_spread

            # Get spread history from price buffer
            exchange_symbol = get_asset_config(symbol).exchange_symbol
            price_buffer = self.price_buffers.get(exchange_symbol)

            # Check if spread is abnormally wide (>0.1% for major pairs)
            if current_spread > 0.1:
                is_blocked = True
                block_reason = "spread_unstable"
                block_details = f"Spread {current_spread:.4f}% > 0.1% threshold"
                market_conditions["spread_change_1s"] = current_spread

            # Calculate liquidity depth
            total_depth = orderbook.bid_volume + orderbook.ask_volume
            market_conditions["liquidity_depth"] = total_depth

            # Check minimum liquidity
            if total_depth < MIN_LIQUIDITY_DEPTH:
                is_blocked = True
                block_reason = "low_liquidity"
                block_details = f"Depth {total_depth:.0f} < {MIN_LIQUIDITY_DEPTH} threshold"

        # === 2. DELTA VARIANCE CHECK ===
        trade_flow = self.get_trade_flow(symbol)
        if trade_flow and not is_blocked:
            delta_1s = trade_flow.get("delta_1s", 0)
            delta_3s = trade_flow.get("delta_3s", 0)

            # Delta variance: if 1s and 3s deltas have opposite signs, high variance
            if delta_1s != 0 and delta_3s != 0:
                if (delta_1s > 0 and delta_3s < 0) or (delta_1s < 0 and delta_3s > 0):
                    # Deltas are flipping - high variance
                    variance = 1.0
                else:
                    # Deltas aligned - low variance
                    variance = 0.2
            else:
                variance = 0.5

            market_conditions["delta_variance"] = variance

            if variance > DELTA_VARIANCE_THRESHOLD:
                is_blocked = True
                block_reason = "delta_noise"
                block_details = f"Delta variance {variance:.2f} > {DELTA_VARIANCE_THRESHOLD} (flipping rapidly)"

        # === 3. ORDERBOOK VOLUME INSTABILITY CHECK ===
        if orderbook and not is_blocked:
            # Get top 5 levels volume
            top5_bid = sum(qty for _, qty in orderbook.bids[:5])
            top5_ask = sum(qty for _, qty in orderbook.asks[:5])
            total_top5 = top5_bid + top5_ask

            # Check imbalance extremity (one side too thin)
            if total_top5 > 0:
                bid_ratio = top5_bid / total_top5
                # If one side is < 20% of total, it's unstable
                if bid_ratio < 0.2 or bid_ratio > 0.8:
                    instability = abs(bid_ratio - 0.5) * 2
                    market_conditions["ob_volume_instability"] = instability

                    if instability > OB_INSTABILITY_THRESHOLD:
                        is_blocked = True
                        block_reason = "ob_unstable"
                        block_details = f"OB imbalance extreme: bid_ratio={bid_ratio:.2f}"

        return is_blocked, block_reason, block_details, market_conditions

    def get_connection_stats(self) -> dict:
        """Get connection statistics."""
        now = time.time()
        stale_symbols = []

        for symbol in SYSTEM_CONFIG.enabled_assets:
            exchange_symbol = get_asset_config(symbol).exchange_symbol
            last_time = self.last_message_time.get(exchange_symbol, 0)
            if now - last_time > 5:  # No data in 5 seconds
                stale_symbols.append(symbol)

        # Check WebSocket connection status safely
        spot_connected = False
        futures_connected = False
        try:
            if self.ws_spot is not None:
                spot_connected = not self.ws_spot.closed
        except:
            pass
        try:
            if self.ws_futures is not None:
                futures_connected = not self.ws_futures.closed
        except:
            pass

        return {
            "message_count": self.message_count,
            "reconnect_count": self.reconnect_count,
            "stale_symbols": stale_symbols,
            "spot_connected": spot_connected,
            "futures_connected": futures_connected
        }

    async def disconnect(self):
        """Close all WebSocket connections."""
        self.running = False

        if self.ws_spot:
            await self.ws_spot.close()
        if self.ws_futures:
            await self.ws_futures.close()

        logger.info("WebSocket connections closed")
