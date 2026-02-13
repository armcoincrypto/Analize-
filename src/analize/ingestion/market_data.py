"""
Market data ingestion.

Handles:
- OHLCV candle data from exchanges or CSV
- Orderbook snapshots
- Exchange metadata (fees, lot sizes)
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from analize.config import get_settings
from analize.models.signals import CandleData, OrderbookLevel, OrderbookSnapshot
from analize.utils.time import utcnow


class MarketDataIngestor:
    """Ingests market data from various sources."""

    # Timeframe to milliseconds mapping
    TIMEFRAME_MS = {
        "1m": 60 * 1000,
        "3m": 3 * 60 * 1000,
        "5m": 5 * 60 * 1000,
        "15m": 15 * 60 * 1000,
        "30m": 30 * 60 * 1000,
        "1h": 60 * 60 * 1000,
        "2h": 2 * 60 * 60 * 1000,
        "4h": 4 * 60 * 60 * 1000,
        "6h": 6 * 60 * 60 * 1000,
        "8h": 8 * 60 * 60 * 1000,
        "12h": 12 * 60 * 60 * 1000,
        "1d": 24 * 60 * 60 * 1000,
        "3d": 3 * 24 * 60 * 60 * 1000,
        "1w": 7 * 24 * 60 * 60 * 1000,
    }

    def __init__(
        self,
        exchange: str = "binance",
        base_url: str | None = None,
    ):
        self.exchange = exchange.lower()
        self.base_url = base_url or self._get_default_base_url()
        self._client: httpx.AsyncClient | None = None

    def _get_default_base_url(self) -> str:
        """Get default API base URL for exchange."""
        urls = {
            "binance": "https://api.binance.com",
            "binance_futures": "https://fapi.binance.com",
            "bybit": "https://api.bybit.com",
        }
        return urls.get(self.exchange, "https://api.binance.com")

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def fetch_klines(
        self,
        symbol: str,
        timeframe: str = "1m",
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 1000,
    ) -> list[CandleData]:
        """
        Fetch OHLCV candles from exchange.

        Args:
            symbol: Trading pair symbol (e.g., BTCUSDT)
            timeframe: Candle timeframe
            start_time: Start time
            end_time: End time
            limit: Maximum candles to fetch

        Returns:
            List of CandleData objects
        """
        client = await self._get_client()

        params: dict[str, Any] = {
            "symbol": symbol.upper(),
            "interval": timeframe,
            "limit": min(limit, 1000),
        }

        if start_time:
            params["startTime"] = int(start_time.timestamp() * 1000)
        if end_time:
            params["endTime"] = int(end_time.timestamp() * 1000)

        # Build endpoint based on exchange
        if self.exchange == "binance":
            endpoint = "/api/v3/klines"
        elif self.exchange == "binance_futures":
            endpoint = "/fapi/v1/klines"
        else:
            endpoint = "/api/v3/klines"

        response = await client.get(f"{self.base_url}{endpoint}", params=params)
        response.raise_for_status()

        data = response.json()
        candles = []

        for row in data:
            # Binance kline format: [open_time, open, high, low, close, volume, close_time, ...]
            candle = CandleData(
                timestamp=datetime.utcfromtimestamp(row[0] / 1000),
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
                timeframe=timeframe,
            )
            candles.append(candle)

        return candles

    async def fetch_all_klines(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime,
    ) -> list[CandleData]:
        """
        Fetch all candles in a date range (handles pagination).

        Args:
            symbol: Trading pair symbol
            timeframe: Candle timeframe
            start_time: Start time
            end_time: End time

        Returns:
            List of all CandleData in range
        """
        all_candles = []
        current_start = start_time
        interval_ms = self.TIMEFRAME_MS.get(timeframe, 60000)

        while current_start < end_time:
            candles = await self.fetch_klines(
                symbol=symbol,
                timeframe=timeframe,
                start_time=current_start,
                end_time=end_time,
                limit=1000,
            )

            if not candles:
                break

            all_candles.extend(candles)

            # Move start to after last candle
            last_time = candles[-1].timestamp
            current_start = last_time + timedelta(milliseconds=interval_ms)

            # Rate limiting
            await asyncio.sleep(0.1)

        return all_candles

    async def fetch_orderbook(
        self,
        symbol: str,
        limit: int = 10,
    ) -> OrderbookSnapshot:
        """
        Fetch current orderbook snapshot.

        Args:
            symbol: Trading pair symbol
            limit: Number of levels to fetch

        Returns:
            OrderbookSnapshot object
        """
        client = await self._get_client()

        params = {
            "symbol": symbol.upper(),
            "limit": min(limit, 100),
        }

        if self.exchange == "binance":
            endpoint = "/api/v3/depth"
        elif self.exchange == "binance_futures":
            endpoint = "/fapi/v1/depth"
        else:
            endpoint = "/api/v3/depth"

        response = await client.get(f"{self.base_url}{endpoint}", params=params)
        response.raise_for_status()

        data = response.json()

        bids = [
            OrderbookLevel(price=float(level[0]), quantity=float(level[1]))
            for level in data.get("bids", [])
        ]

        asks = [
            OrderbookLevel(price=float(level[0]), quantity=float(level[1]))
            for level in data.get("asks", [])
        ]

        return OrderbookSnapshot(
            timestamp=utcnow(),
            symbol=symbol.upper(),
            bids=bids,
            asks=asks,
        )

    async def fetch_exchange_info(self, symbol: str | None = None) -> dict[str, Any]:
        """
        Fetch exchange info (fees, lot sizes, etc.).

        Args:
            symbol: Optional symbol to filter

        Returns:
            Exchange info dictionary
        """
        client = await self._get_client()

        if self.exchange == "binance":
            endpoint = "/api/v3/exchangeInfo"
        elif self.exchange == "binance_futures":
            endpoint = "/fapi/v1/exchangeInfo"
        else:
            endpoint = "/api/v3/exchangeInfo"

        params = {}
        if symbol:
            params["symbol"] = symbol.upper()

        response = await client.get(f"{self.base_url}{endpoint}", params=params)
        response.raise_for_status()

        return response.json()

    async def fetch_ticker(self, symbol: str) -> dict[str, Any]:
        """
        Fetch 24h ticker for symbol.

        Args:
            symbol: Trading pair symbol

        Returns:
            Ticker data dictionary
        """
        client = await self._get_client()

        if self.exchange == "binance":
            endpoint = "/api/v3/ticker/24hr"
        elif self.exchange == "binance_futures":
            endpoint = "/fapi/v1/ticker/24hr"
        else:
            endpoint = "/api/v3/ticker/24hr"

        params = {"symbol": symbol.upper()}

        response = await client.get(f"{self.base_url}{endpoint}", params=params)
        response.raise_for_status()

        return response.json()

    def load_candles_from_csv(
        self,
        file_path: Path | str,
        symbol: str | None = None,
        timeframe: str = "1m",
    ) -> list[CandleData]:
        """
        Load candles from a CSV file.

        Expected columns: timestamp (or date/time), open, high, low, close, volume

        Args:
            file_path: Path to CSV file
            symbol: Symbol name (for metadata)
            timeframe: Timeframe (for metadata)

        Returns:
            List of CandleData objects
        """
        df = pd.read_csv(file_path)

        # Standardize column names
        df.columns = df.columns.str.lower().str.strip()

        # Handle various timestamp column names
        time_cols = ["timestamp", "time", "date", "datetime", "open_time"]
        time_col = None
        for col in time_cols:
            if col in df.columns:
                time_col = col
                break

        if time_col is None:
            raise ValueError(f"No timestamp column found. Expected one of: {time_cols}")

        # Parse timestamps
        df["timestamp"] = pd.to_datetime(df[time_col])

        candles = []
        for _, row in df.iterrows():
            candle = CandleData(
                timestamp=row["timestamp"].to_pydatetime(),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row.get("volume", 0)),
                timeframe=timeframe,
            )
            candles.append(candle)

        return candles

    def candles_to_dataframe(self, candles: list[CandleData]) -> pd.DataFrame:
        """Convert candles to a pandas DataFrame."""
        records = [
            {
                "timestamp": c.timestamp,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
                "timeframe": c.timeframe,
            }
            for c in candles
        ]
        return pd.DataFrame(records)

    def resample_candles(
        self,
        candles: list[CandleData],
        target_timeframe: str,
    ) -> list[CandleData]:
        """
        Resample candles to a different timeframe.

        Args:
            candles: List of candles to resample
            target_timeframe: Target timeframe

        Returns:
            Resampled candles
        """
        if not candles:
            return []

        df = self.candles_to_dataframe(candles)
        df = df.set_index("timestamp")

        # Map timeframe to pandas offset
        tf_map = {
            "1m": "1T",
            "3m": "3T",
            "5m": "5T",
            "15m": "15T",
            "30m": "30T",
            "1h": "1H",
            "2h": "2H",
            "4h": "4H",
            "6h": "6H",
            "12h": "12H",
            "1d": "1D",
            "1w": "1W",
        }

        offset = tf_map.get(target_timeframe, "1H")

        resampled = df.resample(offset).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()

        result = []
        for timestamp, row in resampled.iterrows():
            candle = CandleData(
                timestamp=timestamp.to_pydatetime(),
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
                timeframe=target_timeframe,
            )
            result.append(candle)

        return result
