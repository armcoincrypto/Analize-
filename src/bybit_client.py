"""
Bybit API Client with Authentication

Supports:
- Public market data (no auth needed)
- Private account data (auth required)
- Order execution (auth required)
"""

import hashlib
import hmac
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum


class OrderSide(Enum):
    BUY = "Buy"
    SELL = "Sell"


class OrderType(Enum):
    MARKET = "Market"
    LIMIT = "Limit"


class OrderStatus(Enum):
    NEW = "New"
    PARTIALLY_FILLED = "PartiallyFilled"
    FILLED = "Filled"
    CANCELLED = "Cancelled"
    REJECTED = "Rejected"


@dataclass
class Ticker:
    symbol: str
    last_price: float
    bid_price: float
    ask_price: float
    volume_24h: float
    timestamp: datetime


@dataclass
class Balance:
    coin: str
    available: float
    locked: float
    total: float


@dataclass
class Order:
    order_id: str
    symbol: str
    side: str
    order_type: str
    qty: float
    price: Optional[float]
    filled_qty: float
    avg_price: float
    status: str
    created_at: datetime


class BybitClient:
    """
    Bybit Spot API Client.

    Docs: https://bybit-exchange.github.io/docs/v5/intro
    """

    # API endpoints
    MAINNET_URL = "https://api.bybit.com"
    TESTNET_URL = "https://api-testnet.bybit.com"

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        testnet: bool = True,
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = self.TESTNET_URL if testnet else self.MAINNET_URL
        self.testnet = testnet
        self.recv_window = 5000

    def _generate_signature(self, params: Dict[str, Any], timestamp: int) -> str:
        """Generate HMAC-SHA256 signature for authenticated requests."""
        param_str = str(timestamp) + self.api_key + str(self.recv_window)

        # Sort and stringify params
        if params:
            sorted_params = sorted(params.items())
            param_str += "&".join([f"{k}={v}" for k, v in sorted_params])

        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            param_str.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        return signature

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Dict[str, Any] = None,
        auth: bool = False,
    ) -> Dict[str, Any]:
        """Make API request with optional authentication."""
        url = f"{self.base_url}{endpoint}"
        headers = {"Content-Type": "application/json"}

        if auth:
            timestamp = int(time.time() * 1000)
            signature = self._generate_signature(params or {}, timestamp)

            headers.update({
                "X-BAPI-API-KEY": self.api_key,
                "X-BAPI-TIMESTAMP": str(timestamp),
                "X-BAPI-SIGN": signature,
                "X-BAPI-RECV-WINDOW": str(self.recv_window),
            })

        try:
            if method == "GET":
                response = requests.get(url, params=params, headers=headers, timeout=15)
            elif method == "POST":
                response = requests.post(url, json=params, headers=headers, timeout=15)
            else:
                raise ValueError(f"Unsupported method: {method}")

            data = response.json()

            if data.get("retCode") != 0:
                raise Exception(f"Bybit API error: {data.get('retMsg', 'Unknown error')}")

            return data.get("result", {})

        except requests.exceptions.RequestException as e:
            raise Exception(f"Request failed: {e}")

    # =========================================================================
    # PUBLIC ENDPOINTS (No auth required)
    # =========================================================================

    def get_ticker(self, symbol: str) -> Ticker:
        """Get current ticker for a symbol."""
        params = {"category": "spot", "symbol": symbol}
        result = self._request("GET", "/v5/market/tickers", params)

        if not result.get("list"):
            raise Exception(f"No ticker data for {symbol}")

        data = result["list"][0]

        return Ticker(
            symbol=symbol,
            last_price=float(data["lastPrice"]),
            bid_price=float(data["bid1Price"]),
            ask_price=float(data["ask1Price"]),
            volume_24h=float(data["volume24h"]),
            timestamp=datetime.now(),
        )

    def get_klines(
        self,
        symbol: str,
        interval: str = "D",
        limit: int = 200,
    ) -> pd.DataFrame:
        """
        Get candlestick data.

        Intervals: 1,3,5,15,30,60,120,240,360,720,D,W,M
        """
        params = {
            "category": "spot",
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }

        result = self._request("GET", "/v5/market/kline", params)

        if not result.get("list"):
            return pd.DataFrame()

        # Bybit returns: [startTime, open, high, low, close, volume, turnover]
        rows = []
        for candle in result["list"]:
            rows.append({
                "timestamp": datetime.fromtimestamp(int(candle[0]) / 1000),
                "open": float(candle[1]),
                "high": float(candle[2]),
                "low": float(candle[3]),
                "close": float(candle[4]),
                "volume": float(candle[5]),
            })

        df = pd.DataFrame(rows)
        return df.sort_values("timestamp").reset_index(drop=True)

    # =========================================================================
    # PRIVATE ENDPOINTS (Auth required)
    # =========================================================================

    def get_balance(self, coin: str = "USDT") -> Balance:
        """Get account balance for a coin."""
        params = {"accountType": "UNIFIED", "coin": coin}
        result = self._request("GET", "/v5/account/wallet-balance", params, auth=True)

        if not result.get("list"):
            return Balance(coin=coin, available=0, locked=0, total=0)

        account = result["list"][0]
        coins = {c["coin"]: c for c in account.get("coin", [])}

        if coin not in coins:
            return Balance(coin=coin, available=0, locked=0, total=0)

        c = coins[coin]
        return Balance(
            coin=coin,
            available=float(c.get("availableToWithdraw", 0)),
            locked=float(c.get("locked", 0)),
            total=float(c.get("walletBalance", 0)),
        )

    def get_all_balances(self) -> List[Balance]:
        """Get all non-zero balances."""
        params = {"accountType": "UNIFIED"}
        result = self._request("GET", "/v5/account/wallet-balance", params, auth=True)

        balances = []
        if result.get("list"):
            account = result["list"][0]
            for c in account.get("coin", []):
                total = float(c.get("walletBalance", 0))
                if total > 0:
                    balances.append(Balance(
                        coin=c["coin"],
                        available=float(c.get("availableToWithdraw", 0)),
                        locked=float(c.get("locked", 0)),
                        total=total,
                    ))

        return balances

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        qty: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
    ) -> Order:
        """
        Place a spot order.

        Args:
            symbol: Trading pair (e.g., "XRPUSDT")
            side: Buy or Sell
            qty: Order quantity in base currency
            order_type: Market or Limit
            price: Price for limit orders
        """
        params = {
            "category": "spot",
            "symbol": symbol,
            "side": side.value,
            "orderType": order_type.value,
            "qty": str(qty),
        }

        if order_type == OrderType.LIMIT and price:
            params["price"] = str(price)

        result = self._request("POST", "/v5/order/create", params, auth=True)

        return Order(
            order_id=result.get("orderId", ""),
            symbol=symbol,
            side=side.value,
            order_type=order_type.value,
            qty=qty,
            price=price,
            filled_qty=0,
            avg_price=0,
            status="New",
            created_at=datetime.now(),
        )

    def get_order(self, symbol: str, order_id: str) -> Optional[Order]:
        """Get order status."""
        params = {
            "category": "spot",
            "symbol": symbol,
            "orderId": order_id,
        }

        result = self._request("GET", "/v5/order/realtime", params, auth=True)

        if not result.get("list"):
            return None

        o = result["list"][0]
        return Order(
            order_id=o["orderId"],
            symbol=o["symbol"],
            side=o["side"],
            order_type=o["orderType"],
            qty=float(o["qty"]),
            price=float(o["price"]) if o.get("price") else None,
            filled_qty=float(o.get("cumExecQty", 0)),
            avg_price=float(o.get("avgPrice", 0)),
            status=o["orderStatus"],
            created_at=datetime.fromtimestamp(int(o["createdTime"]) / 1000),
        )

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an open order."""
        params = {
            "category": "spot",
            "symbol": symbol,
            "orderId": order_id,
        }

        try:
            self._request("POST", "/v5/order/cancel", params, auth=True)
            return True
        except Exception:
            return False

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all open orders."""
        params = {"category": "spot"}
        if symbol:
            params["symbol"] = symbol

        result = self._request("GET", "/v5/order/realtime", params, auth=True)

        orders = []
        for o in result.get("list", []):
            orders.append(Order(
                order_id=o["orderId"],
                symbol=o["symbol"],
                side=o["side"],
                order_type=o["orderType"],
                qty=float(o["qty"]),
                price=float(o["price"]) if o.get("price") else None,
                filled_qty=float(o.get("cumExecQty", 0)),
                avg_price=float(o.get("avgPrice", 0)),
                status=o["orderStatus"],
                created_at=datetime.fromtimestamp(int(o["createdTime"]) / 1000),
            ))

        return orders

    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================

    def market_buy(self, symbol: str, qty: float) -> Order:
        """Place a market buy order."""
        return self.place_order(symbol, OrderSide.BUY, qty, OrderType.MARKET)

    def market_sell(self, symbol: str, qty: float) -> Order:
        """Place a market sell order."""
        return self.place_order(symbol, OrderSide.SELL, qty, OrderType.MARKET)

    def buy_usdt_amount(self, symbol: str, usdt_amount: float) -> Order:
        """Buy a specific USDT amount of a coin."""
        ticker = self.get_ticker(symbol)
        qty = usdt_amount / ticker.ask_price

        # Round to appropriate precision (varies by symbol)
        qty = round(qty, 4)

        return self.market_buy(symbol, qty)

    def sell_all(self, symbol: str, coin: str) -> Optional[Order]:
        """Sell entire balance of a coin."""
        balance = self.get_balance(coin)

        if balance.available <= 0:
            return None

        return self.market_sell(symbol, balance.available)

    def wait_for_fill(self, symbol: str, order_id: str, timeout: int = 30) -> Order:
        """Wait for an order to be filled."""
        start = time.time()

        while time.time() - start < timeout:
            order = self.get_order(symbol, order_id)

            if order and order.status in ["Filled", "Cancelled", "Rejected"]:
                return order

            time.sleep(1)

        raise TimeoutError(f"Order {order_id} not filled within {timeout}s")


# Test function
def test_client():
    """Test the Bybit client."""
    client = BybitClient(testnet=True)

    print("Testing Bybit Client...")
    print("=" * 50)

    # Test public endpoints
    print("\n1. Get Ticker:")
    ticker = client.get_ticker("XRPUSDT")
    print(f"   XRP/USDT: ${ticker.last_price}")

    print("\n2. Get Klines:")
    df = client.get_klines("XRPUSDT", "D", 10)
    print(f"   Got {len(df)} daily candles")
    if not df.empty:
        print(f"   Latest close: ${df['close'].iloc[-1]}")

    print("\n" + "=" * 50)
    print("Public endpoints working!")
    print("\nTo test private endpoints, add your API keys.")


if __name__ == "__main__":
    test_client()
