"""
Multi-Exchange Data Fetcher

Supports multiple exchanges:
- Binance (default)
- MEXC
- Bybit
- KuCoin

All exchanges have similar REST APIs for public market data.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict
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


class ExchangeClient:
    """Multi-exchange data fetcher."""

    def __init__(self, exchange: Exchange = Exchange.MEXC, timeout: int = 15):
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
        """Fetch from Binance."""
        url = f"{self.config.base_url}{self.config.klines_endpoint}"
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": int((datetime.now() - timedelta(days=days)).timestamp() * 1000),
            "limit": 1000
        }

        try:
            response = requests.get(url, params=params, timeout=self.timeout)
            if response.status_code == 200:
                return self._parse_binance_response(response.json())
        except Exception as e:
            print(f"  Binance error: {e}")

        return pd.DataFrame()

    def _fetch_mexc(self, symbol: str, interval: str, days: int) -> pd.DataFrame:
        """Fetch from MEXC."""
        url = f"{self.config.base_url}{self.config.klines_endpoint}"
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": int((datetime.now() - timedelta(days=days)).timestamp() * 1000),
            "limit": 1000
        }

        try:
            response = requests.get(url, params=params, timeout=self.timeout)
            if response.status_code == 200:
                data = response.json()
                # MEXC returns same format as Binance
                return self._parse_binance_response(data)
        except Exception as e:
            print(f"  MEXC error: {e}")

        return pd.DataFrame()

    def _fetch_bybit(self, symbol: str, interval: str, days: int) -> pd.DataFrame:
        """Fetch from Bybit."""
        url = f"{self.config.base_url}{self.config.klines_endpoint}"
        params = {
            "category": "spot",
            "symbol": symbol,
            "interval": interval,
            "start": int((datetime.now() - timedelta(days=days)).timestamp() * 1000),
            "limit": 1000
        }

        try:
            response = requests.get(url, params=params, timeout=self.timeout)
            if response.status_code == 200:
                data = response.json()
                if data.get("retCode") == 0:
                    return self._parse_bybit_response(data["result"]["list"])
        except Exception as e:
            print(f"  Bybit error: {e}")

        return pd.DataFrame()

    def _fetch_kucoin(self, symbol: str, interval: str, days: int) -> pd.DataFrame:
        """Fetch from KuCoin."""
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

        try:
            response = requests.get(url, params=params, timeout=self.timeout)
            if response.status_code == 200:
                data = response.json()
                if data.get("code") == "200000":
                    return self._parse_kucoin_response(data["data"])
        except Exception as e:
            print(f"  KuCoin error: {e}")

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
        return df[["timestamp", "open", "high", "low", "close", "volume"]].sort_values("timestamp")

    def _parse_kucoin_response(self, data: List) -> pd.DataFrame:
        """Parse KuCoin response format."""
        if not data:
            return pd.DataFrame()

        # KuCoin returns: [time, open, close, high, low, volume, amount]
        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "close", "high", "low", "volume", "amount"
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        return df[["timestamp", "open", "high", "low", "close", "volume"]].sort_values("timestamp")


class MultiExchangeClient:
    """Try multiple exchanges with fallback."""

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

    def get_klines(self, symbol: str, interval: str = "1d", days: int = 250) -> tuple:
        """
        Fetch data from preferred exchange with fallback.

        Returns:
            (DataFrame, exchange_name) tuple
        """
        for exchange in self.fallback_order:
            client = ExchangeClient(exchange)
            df = client.get_klines(symbol, interval, days)

            if not df.empty and len(df) > 50:
                return df, EXCHANGE_CONFIGS[exchange].name

        return pd.DataFrame(), "None"


# Convenience function
def fetch_data(
    symbol: str,
    exchange: str = "mexc",
    interval: str = "1d",
    days: int = 250
) -> pd.DataFrame:
    """
    Fetch market data from specified exchange.

    Args:
        symbol: Trading pair (e.g., "XRP", "XRPUSDT")
        exchange: Exchange name ("mexc", "binance", "bybit", "kucoin")
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

    exchange_enum = exchange_map.get(exchange.lower(), Exchange.MEXC)
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
