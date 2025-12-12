#!/usr/bin/env python3
"""
Centralized API Configuration

This file defines API endpoints in priority order. APIs are tried in order,
with working APIs first. Update this file to change API priorities across
all analysis tools.

Current Priority:
1. Binance US (works in most regions)
2. CoinGecko (universal)
3. Kraken (universal)
4. KuCoin (universal)
5. Binance (geo-restricted in some countries)
6. Bybit (geo-restricted in some countries)
"""

# Klines/OHLCV endpoints - tried in order
KLINES_ENDPOINTS = [
    # Binance US - works in most regions including geo-restricted areas
    {
        "name": "Binance US",
        "url": "https://api.binance.us/api/v3/klines",
        "symbol_format": "{}USDT",  # e.g., BTCUSDT
    },
    # CoinGecko - universal but rate limited
    {
        "name": "CoinGecko",
        "url": "https://api.coingecko.com/api/v3/coins/{}/ohlc",
        "symbol_format": "{}",  # Uses coin ID
        "requires_special_handling": True,
    },
    # Kraken - universal
    {
        "name": "Kraken",
        "url": "https://api.kraken.com/0/public/OHLC",
        "symbol_format": "{}USD",
    },
    # KuCoin - universal
    {
        "name": "KuCoin",
        "url": "https://api.kucoin.com/api/v1/market/candles",
        "symbol_format": "{}-USDT",
    },
    # Binance Global - geo-restricted (451 in some countries)
    {
        "name": "Binance",
        "url": "https://api.binance.com/api/v3/klines",
        "symbol_format": "{}USDT",
    },
    # Bybit - geo-restricted (403 in some countries)
    {
        "name": "Bybit",
        "url": "https://api.bybit.com/v5/market/kline",
        "symbol_format": "{}USDT",
    },
]

# Ticker/Price endpoints
TICKER_ENDPOINTS = [
    {
        "name": "Binance US",
        "url": "https://api.binance.us/api/v3/ticker/price",
        "symbol_format": "{}USDT",
    },
    {
        "name": "CoinGecko",
        "url": "https://api.coingecko.com/api/v3/simple/price",
        "symbol_format": "{}",
    },
    {
        "name": "Kraken",
        "url": "https://api.kraken.com/0/public/Ticker",
        "symbol_format": "{}USD",
    },
    {
        "name": "KuCoin",
        "url": "https://api.kucoin.com/api/v1/market/orderbook/level1",
        "symbol_format": "{}-USDT",
    },
    {
        "name": "Binance",
        "url": "https://api.binance.com/api/v3/ticker/price",
        "symbol_format": "{}USDT",
    },
]

# Orderbook endpoints
ORDERBOOK_ENDPOINTS = [
    {
        "name": "Binance US",
        "url": "https://api.binance.us/api/v3/depth",
        "symbol_format": "{}USDT",
    },
    {
        "name": "Kraken",
        "url": "https://api.kraken.com/0/public/Depth",
        "symbol_format": "{}USD",
    },
    {
        "name": "KuCoin",
        "url": "https://api.kucoin.com/api/v1/market/orderbook/level2_20",
        "symbol_format": "{}-USDT",
    },
    {
        "name": "Binance",
        "url": "https://api.binance.com/api/v3/depth",
        "symbol_format": "{}USDT",
    },
]

# Funding rate endpoints (futures only)
FUNDING_ENDPOINTS = [
    {
        "name": "Binance US Futures",
        "url": "https://fapi.binance.us/fapi/v1/fundingRate",
        "symbol_format": "{}USDT",
    },
    {
        "name": "Binance Futures",
        "url": "https://fapi.binance.com/fapi/v1/fundingRate",
        "symbol_format": "{}USDT",
    },
]

# CoinGecko coin ID mapping
COINGECKO_IDS = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "XRP": "ripple",
    "SOL": "solana",
    "ATOM": "cosmos",
    "ADA": "cardano",
    "DOT": "polkadot",
    "AVAX": "avalanche-2",
    "LINK": "chainlink",
    "MATIC": "matic-network",
    "DOGE": "dogecoin",
    "LTC": "litecoin",
}


def get_binance_urls():
    """Get Binance klines URLs in priority order (US first)."""
    return [
        "https://api.binance.us/api/v3/klines",
        "https://api.binance.com/api/v3/klines",
    ]


def get_ticker_urls():
    """Get ticker URLs in priority order."""
    return [
        "https://api.binance.us/api/v3/ticker/price",
        "https://api.binance.com/api/v3/ticker/price",
    ]


def get_orderbook_urls():
    """Get orderbook URLs in priority order."""
    return [
        "https://api.binance.us/api/v3/depth",
        "https://api.binance.com/api/v3/depth",
    ]
