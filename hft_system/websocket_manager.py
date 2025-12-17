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

    def get_connection_stats(self) -> dict:
        """Get connection statistics."""
        now = time.time()
        stale_symbols = []

        for symbol in SYSTEM_CONFIG.enabled_assets:
            exchange_symbol = get_asset_config(symbol).exchange_symbol
            last_time = self.last_message_time.get(exchange_symbol, 0)
            if now - last_time > 5:  # No data in 5 seconds
                stale_symbols.append(symbol)

        return {
            "message_count": self.message_count,
            "reconnect_count": self.reconnect_count,
            "stale_symbols": stale_symbols,
            "spot_connected": self.ws_spot is not None and self.ws_spot.open,
            "futures_connected": self.ws_futures is not None and self.ws_futures.open
        }

    async def disconnect(self):
        """Close all WebSocket connections."""
        self.running = False

        if self.ws_spot:
            await self.ws_spot.close()
        if self.ws_futures:
            await self.ws_futures.close()

        logger.info("WebSocket connections closed")
