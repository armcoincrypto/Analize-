"""
Multi-Exchange Data Fetcher

Supports multiple exchanges:
- Bybit (primary)
- MEXC
- Binance
- KuCoin

All exchanges have similar REST APIs for public market data.
Includes retry logic with exponential backoff.
"""

import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum


class Exchange(Enum):
    BINANCE = "binance"
    MEXC = "mexc"
    BYBIT = "bybit"
    KUCOIN = "kucoin"


@dataclass
class ExchangeConfig:
    """Configuration for each exchange."""
    name: str
    base_url: str
    klines_endpoint: str
    symbol_format: str  # How symbols are formatted (e.g., "XRPUSDT" vs "XRP-USDT")
    interval_map: Dict[str, str]  # Map our intervals to exchange format


# Exchange configurations
EXCHANGE_CONFIGS = {
    Exchange.BINANCE: ExchangeConfig(
        name="Binance",
        base_url="https://api.binance.com/api/v3",
        klines_endpoint="/klines",
        symbol_format="{base}{quote}",  # XRPUSDT
        interval_map={"1d": "1d", "1h": "1h", "4h": "4h", "15m": "15m"}
    ),
    Exchange.MEXC: ExchangeConfig(
        name="MEXC",
        base_url="https://api.mexc.com/api/v3",
        klines_endpoint="/klines",
        symbol_format="{base}{quote}",  # XRPUSDT
        interval_map={"1d": "1d", "1h": "1h", "4h": "4h", "15m": "15m"}
    ),
    Exchange.BYBIT: ExchangeConfig(
        name="Bybit",
        base_url="https://api.bybit.com/v5/market",
        klines_endpoint="/kline",
        symbol_format="{base}{quote}",  # XRPUSDT
        interval_map={"1d": "D", "1h": "60", "4h": "240", "15m": "15"}
    ),
    Exchange.KUCOIN: ExchangeConfig(
        name="KuCoin",
        base_url="https://api.kucoin.com/api/v1/market",
        klines_endpoint="/candles",
        symbol_format="{base}-{quote}",  # XRP-USDT
        interval_map={"1d": "1day", "1h": "1hour", "4h": "4hour", "15m": "15min"}
    ),
}


# Retry configuration
MAX_RETRIES = 3
RETRY_DELAYS = [2, 4, 8]  # Exponential backoff: 2s, 4s, 8s


def retry_request(url: str, params: dict, timeout: int = 15) -> Optional[requests.Response]:
    """Make HTTP request with retry logic and exponential backoff."""
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            if response.status_code == 200:
                return response
            elif response.status_code == 429:  # Rate limited
                wait_time = RETRY_DELAYS[attempt] if attempt < len(RETRY_DELAYS) else 16
                time.sleep(wait_time)
                continue
            else:
                return response  # Return non-retryable errors
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                wait_time = RETRY_DELAYS[attempt]
                time.sleep(wait_time)
        except Exception as e:
            last_error = e
            break

    return None


class ExchangeClient:
    """Multi-exchange data fetcher with retry logic."""

    def __init__(self, exchange: Exchange = Exchange.BYBIT, timeout: int = 15):
        self.exchange = exchange
        self.config = EXCHANGE_CONFIGS[exchange]
        self.timeout = timeout

    def format_symbol(self, base: str, quote: str = "USDT") -> str:
        """Format symbol for the exchange."""
        return self.config.symbol_format.format(base=base.upper(), quote=quote.upper())

    def get_klines(
        self,
        symbol: str,
        interval: str = "1d",
        days: int = 250,
    ) -> pd.DataFrame:
        """
        Fetch candlestick data from the exchange.

        Args:
            symbol: Trading pair (e.g., "XRPUSDT" or "XRP")
            interval: Candle interval ("1d", "1h", "4h", "15m")
            days: Number of days of history

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        # Normalize symbol
        if not symbol.endswith("USDT"):
            symbol = self.format_symbol(symbol, "USDT")

        # Get exchange-specific interval
        exchange_interval = self.config.interval_map.get(interval, interval)

        if self.exchange == Exchange.BINANCE:
            return self._fetch_binance(symbol, exchange_interval, days)
        elif self.exchange == Exchange.MEXC:
            return self._fetch_mexc(symbol, exchange_interval, days)
        elif self.exchange == Exchange.BYBIT:
            return self._fetch_bybit(symbol, exchange_interval, days)
        elif self.exchange == Exchange.KUCOIN:
            return self._fetch_kucoin(symbol, exchange_interval, days)
        else:
            return pd.DataFrame()

    def _fetch_binance(self, symbol: str, interval: str, days: int) -> pd.DataFrame:
        """Fetch from Binance with retry."""
        url = f"{self.config.base_url}{self.config.klines_endpoint}"
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": int((datetime.now() - timedelta(days=days)).timestamp() * 1000),
            "limit": 1000
        }

        response = retry_request(url, params, self.timeout)
        if response and response.status_code == 200:
            return self._parse_binance_response(response.json())

        return pd.DataFrame()

    def _fetch_mexc(self, symbol: str, interval: str, days: int) -> pd.DataFrame:
        """Fetch from MEXC with retry."""
        url = f"{self.config.base_url}{self.config.klines_endpoint}"
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": int((datetime.now() - timedelta(days=days)).timestamp() * 1000),
            "limit": 1000
        }

        response = retry_request(url, params, self.timeout)
        if response and response.status_code == 200:
            data = response.json()
            return self._parse_binance_response(data)

        return pd.DataFrame()

    def _fetch_bybit(self, symbol: str, interval: str, days: int) -> pd.DataFrame:
        """Fetch from Bybit with retry."""
        url = f"{self.config.base_url}{self.config.klines_endpoint}"
        params = {
            "category": "spot",
            "symbol": symbol,
            "interval": interval,
            "start": int((datetime.now() - timedelta(days=days)).timestamp() * 1000),
            "limit": 1000
        }

        response = retry_request(url, params, self.timeout)
        if response and response.status_code == 200:
            data = response.json()
            if data.get("retCode") == 0:
                return self._parse_bybit_response(data["result"]["list"])

        return pd.DataFrame()

    def _fetch_kucoin(self, symbol: str, interval: str, days: int) -> pd.DataFrame:
        """Fetch from KuCoin with retry."""
        # KuCoin uses different symbol format
        symbol = symbol.replace("USDT", "-USDT")
        url = f"{self.config.base_url}{self.config.klines_endpoint}"

        start_time = int((datetime.now() - timedelta(days=days)).timestamp())
        end_time = int(datetime.now().timestamp())

        params = {
            "symbol": symbol,
            "type": interval,
            "startAt": start_time,
            "endAt": end_time
        }

        response = retry_request(url, params, self.timeout)
        if response and response.status_code == 200:
            data = response.json()
            if data.get("code") == "200000":
                return self._parse_kucoin_response(data["data"])

        return pd.DataFrame()

    def _parse_binance_response(self, data: List) -> pd.DataFrame:
        """Parse Binance/MEXC response format."""
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        return df[["timestamp", "open", "high", "low", "close", "volume"]]

    def _parse_bybit_response(self, data: List) -> pd.DataFrame:
        """Parse Bybit response format."""
        if not data:
            return pd.DataFrame()

        # Bybit returns: [startTime, openPrice, highPrice, lowPrice, closePrice, volume, turnover]
        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume", "turnover"
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        # Bybit returns newest first, so sort ascending
        return df[["timestamp", "open", "high", "low", "close", "volume"]].sort_values("timestamp").reset_index(drop=True)

    def _parse_kucoin_response(self, data: List) -> pd.DataFrame:
        """Parse KuCoin response format."""
        if not data:
            return pd.DataFrame()

        # KuCoin API returns: [time, open, close, high, low, volume, turnover]
        # Note: KuCoin order is OCLHV not OHLCV!
        rows = []
        for candle in data:
            rows.append({
                "timestamp": int(candle[0]),
                "open": float(candle[1]),
                "high": float(candle[3]),      # Index 3 is high
                "low": float(candle[4]),       # Index 4 is low
                "close": float(candle[2]),     # Index 2 is close
                "volume": float(candle[5]),
            })

        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        # KuCoin returns newest first, so sort ascending
        return df[["timestamp", "open", "high", "low", "close", "volume"]].sort_values("timestamp").reset_index(drop=True)


class MultiExchangeClient:
    """Try multiple exchanges with fallback and retry."""

    def __init__(self, preferred_exchange: Exchange = Exchange.BYBIT):
        self.preferred = preferred_exchange
        self.fallback_order = [
            Exchange.BYBIT,
            Exchange.MEXC,
            Exchange.BINANCE,
            Exchange.KUCOIN,
        ]
        # Move preferred to front
        if preferred_exchange in self.fallback_order:
            self.fallback_order.remove(preferred_exchange)
            self.fallback_order.insert(0, preferred_exchange)

    def get_klines(self, symbol: str, interval: str = "1d", days: int = 250) -> Tuple[pd.DataFrame, str]:
        """
        Fetch data from preferred exchange with fallback.

        Returns:
            (DataFrame, exchange_name) tuple
        """
        for exchange in self.fallback_order:
            try:
                client = ExchangeClient(exchange)
                df = client.get_klines(symbol, interval, days)

                if not df.empty and len(df) > 50:
                    return df, EXCHANGE_CONFIGS[exchange].name
            except Exception as e:
                continue  # Try next exchange

        return pd.DataFrame(), "None"


# Convenience function
def fetch_data(
    symbol: str,
    exchange: str = "bybit",
    interval: str = "1d",
    days: int = 250
) -> pd.DataFrame:
    """
    Fetch market data from specified exchange.

    Args:
        symbol: Trading pair (e.g., "XRP", "XRPUSDT")
        exchange: Exchange name ("bybit", "mexc", "binance", "kucoin")
        interval: Candle interval ("1d", "1h", "4h", "15m")
        days: Number of days of history

    Returns:
        DataFrame with OHLCV data
    """
    exchange_map = {
        "mexc": Exchange.MEXC,
        "binance": Exchange.BINANCE,
        "bybit": Exchange.BYBIT,
        "kucoin": Exchange.KUCOIN,
    }

    exchange_enum = exchange_map.get(exchange.lower(), Exchange.BYBIT)
    client = ExchangeClient(exchange_enum)
    return client.get_klines(symbol, interval, days)


# Test function
def test_exchanges():
    """Test all exchange connections."""
    print("Testing exchange connections...\n")

    symbols = ["XRPUSDT", "SOLUSDT", "ATOMUSDT"]

    for exchange in Exchange:
        print(f"\n{EXCHANGE_CONFIGS[exchange].name}:")
        client = ExchangeClient(exchange)

        for symbol in symbols:
            try:
                df = client.get_klines(symbol, "1d", 30)
                if not df.empty:
                    price = df["close"].iloc[-1]
                    print(f"  {symbol}: ${price:.4f} ({len(df)} candles)")
                else:
                    print(f"  {symbol}: No data")
            except Exception as e:
                print(f"  {symbol}: Error - {e}")


if __name__ == "__main__":
    test_exchanges()
