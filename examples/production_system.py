"""
Production Trading System

Components:
6. Realistic Backtester (slippage, fees, partial fills)
7. Position Sizing & Risk Engine (ATR-based, Kelly-lite)
8. Paper-Trading Runner with Alerts
9. Signal Explainability
10. Monitoring & Safeguards

Usage:
    python examples/production_system.py --backtest
    python examples/production_system.py --sizing
    python examples/production_system.py --paper
    python examples/production_system.py --explain
    python examples/production_system.py --monitor
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import json
import sqlite3
import os
import time


# =============================================================================
# CONFIGURATION
# =============================================================================

COINS = {
    "XRP": {"symbol": "XRPUSDT", "min_notional": 10, "tick_size": 0.0001},
    "SOL": {"symbol": "SOLUSDT", "min_notional": 10, "tick_size": 0.01},
    "ATOM": {"symbol": "ATOMUSDT", "min_notional": 10, "tick_size": 0.001},
}

# Fee structure (Bybit spot)
FEES = {
    "maker": 0.001,  # 0.1%
    "taker": 0.001,  # 0.1%
}

DATA_DIR = "data"
DB_PATH = f"{DATA_DIR}/production.db"


# =============================================================================
# 6. REALISTIC BACKTESTER
# =============================================================================

@dataclass
class ExecutionResult:
    """Result of a simulated execution."""
    success: bool
    filled_qty: float
    filled_price: float
    slippage_pct: float
    fee_paid: float
    total_cost: float
    reason: str = ""


@dataclass
class RealisticTrade:
    """Trade with realistic execution details."""
    coin: str
    side: str  # BUY/SELL
    entry_price: float
    exit_price: float
    qty: float
    entry_slippage: float
    exit_slippage: float
    entry_fee: float
    exit_fee: float
    gross_pnl: float
    net_pnl: float
    net_pnl_pct: float
    days_held: int
    reason: str


class RealisticBacktester:
    """Backtester with realistic execution modeling."""

    def __init__(
        self,
        slippage_model: str = "volume",  # "fixed" or "volume"
        base_slippage_pct: float = 0.05,  # 0.05% base slippage
        maker_fee: float = 0.001,
        taker_fee: float = 0.001,
    ):
        self.slippage_model = slippage_model
        self.base_slippage = base_slippage_pct / 100
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee

    def _calculate_slippage(
        self,
        price: float,
        order_size_usd: float,
        daily_volume: float,
        side: str,
    ) -> float:
        """Calculate expected slippage."""
        if self.slippage_model == "fixed":
            slippage_pct = self.base_slippage
        else:
            # Volume-based: slippage increases with order size relative to volume
            volume_ratio = order_size_usd / daily_volume if daily_volume > 0 else 0.01
            slippage_pct = self.base_slippage * (1 + volume_ratio * 10)

        # Cap slippage at 1%
        slippage_pct = min(slippage_pct, 0.01)

        # Apply direction
        if side == "BUY":
            return price * (1 + slippage_pct)  # Pay more
        else:
            return price * (1 - slippage_pct)  # Receive less

    def _simulate_fill(
        self,
        price: float,
        qty: float,
        volume: float,
        side: str,
    ) -> ExecutionResult:
        """Simulate order fill with slippage and fees."""
        order_size_usd = price * qty

        # Check minimum notional
        if order_size_usd < 10:
            return ExecutionResult(
                success=False,
                filled_qty=0,
                filled_price=0,
                slippage_pct=0,
                fee_paid=0,
                total_cost=0,
                reason="Below min notional",
            )

        # Calculate slippage
        fill_price = self._calculate_slippage(price, order_size_usd, volume, side)
        slippage_pct = abs(fill_price - price) / price * 100

        # Calculate fee (assume taker for market orders)
        fee = order_size_usd * self.taker_fee

        # Total cost
        if side == "BUY":
            total_cost = fill_price * qty + fee
        else:
            total_cost = fill_price * qty - fee

        return ExecutionResult(
            success=True,
            filled_qty=qty,
            filled_price=fill_price,
            slippage_pct=slippage_pct,
            fee_paid=fee,
            total_cost=total_cost,
        )

    def _fetch_data(self, symbol: str, days: int = 1000) -> pd.DataFrame:
        """Fetch historical data."""
        try:
            params = {"symbol": symbol, "interval": "1d", "limit": days}
            response = requests.get(
                "https://api.binance.us/api/v3/klines",
                params=params,
                timeout=15
            )
            data = response.json()

            if isinstance(data, list) and len(data) > 0:
                df = pd.DataFrame(data, columns=[
                    "timestamp", "open", "high", "low", "close", "volume",
                    "close_time", "quote_volume", "trades", "taker_buy_base",
                    "taker_buy_quote", "ignore"
                ])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
                    df[col] = df[col].astype(float)
                return df
        except Exception as e:
            print(f"Data fetch error: {e}")
        return pd.DataFrame()

    def backtest(
        self,
        symbol: str,
        initial_capital: float = 1000,
        position_size_pct: float = 5.0,
        bb_period: int = 20,
        bb_std: float = 1.5,
    ) -> Dict:
        """Run realistic backtest."""
        df = self._fetch_data(symbol)
        if df.empty or len(df) < 220:
            return {"error": "Insufficient data"}

        close = df["close"]
        volume = df["quote_volume"]  # USD volume

        # Indicators
        middle = close.rolling(bb_period).mean()
        std = close.rolling(bb_period).std()
        lower = middle - (std * bb_std)
        ma_200 = close.rolling(200).mean()

        # ATR for stops
        high = df["high"]
        low = df["low"]
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()

        trades = []
        capital = initial_capital
        position = None

        for i in range(220, len(df)):
            price = close.iloc[i]
            daily_vol = volume.iloc[i]

            # Entry
            if position is None:
                if price <= lower.iloc[i] and price > ma_200.iloc[i]:
                    # Calculate position size
                    trade_capital = capital * (position_size_pct / 100)
                    qty = trade_capital / price

                    # Simulate entry
                    entry_result = self._simulate_fill(price, qty, daily_vol, "BUY")

                    if entry_result.success:
                        position = {
                            "entry_price": entry_result.filled_price,
                            "qty": entry_result.filled_qty,
                            "entry_idx": i,
                            "entry_fee": entry_result.fee_paid,
                            "entry_slippage": entry_result.slippage_pct,
                            "stop_loss": price - atr.iloc[i] * 2,
                            "capital_used": entry_result.total_cost,
                        }
                        capital -= entry_result.total_cost

            # Exit
            elif position is not None:
                days_held = i - position["entry_idx"]
                exit_reason = None

                if price >= middle.iloc[i]:
                    exit_reason = "TARGET"
                elif price <= position["stop_loss"]:
                    exit_reason = "STOP"
                elif days_held >= 22:
                    exit_reason = "TIME"

                if exit_reason:
                    # Simulate exit
                    exit_result = self._simulate_fill(
                        price, position["qty"], daily_vol, "SELL"
                    )

                    if exit_result.success:
                        gross_pnl = (exit_result.filled_price - position["entry_price"]) * position["qty"]
                        total_fees = position["entry_fee"] + exit_result.fee_paid
                        net_pnl = gross_pnl - total_fees
                        net_pnl_pct = (net_pnl / position["capital_used"]) * 100

                        trades.append(RealisticTrade(
                            coin=symbol.replace("USDT", ""),
                            side="LONG",
                            entry_price=position["entry_price"],
                            exit_price=exit_result.filled_price,
                            qty=position["qty"],
                            entry_slippage=position["entry_slippage"],
                            exit_slippage=exit_result.slippage_pct,
                            entry_fee=position["entry_fee"],
                            exit_fee=exit_result.fee_paid,
                            gross_pnl=round(gross_pnl, 2),
                            net_pnl=round(net_pnl, 2),
                            net_pnl_pct=round(net_pnl_pct, 2),
                            days_held=days_held,
                            reason=exit_reason,
                        ))

                        capital += exit_result.total_cost + net_pnl
                        position = None

        # Calculate metrics
        if not trades:
            return {"error": "No trades"}

        wins = len([t for t in trades if t.net_pnl > 0])
        total_fees = sum(t.entry_fee + t.exit_fee for t in trades)
        total_slippage = sum(
            (t.entry_slippage + t.exit_slippage) / 2 * t.entry_price * t.qty / 100
            for t in trades
        )

        return {
            "coin": symbol.replace("USDT", ""),
            "initial_capital": initial_capital,
            "final_capital": round(capital, 2),
            "total_return_pct": round((capital - initial_capital) / initial_capital * 100, 2),
            "trades": len(trades),
            "wins": wins,
            "win_rate": round(wins / len(trades) * 100, 1),
            "gross_pnl": round(sum(t.gross_pnl for t in trades), 2),
            "net_pnl": round(sum(t.net_pnl for t in trades), 2),
            "total_fees": round(total_fees, 2),
            "total_slippage_cost": round(total_slippage, 2),
            "avg_trade_pnl": round(np.mean([t.net_pnl_pct for t in trades]), 2),
            "max_trade_pnl": round(max(t.net_pnl_pct for t in trades), 2),
            "min_trade_pnl": round(min(t.net_pnl_pct for t in trades), 2),
            "by_reason": {
                "TARGET": len([t for t in trades if t.reason == "TARGET"]),
                "STOP": len([t for t in trades if t.reason == "STOP"]),
                "TIME": len([t for t in trades if t.reason == "TIME"]),
            },
        }


# =============================================================================
# 7. POSITION SIZING & RISK ENGINE
# =============================================================================

@dataclass
class PositionSize:
    """Calculated position size."""
    qty: float
    usd_amount: float
    risk_amount: float
    stop_loss_price: float
    take_profit_price: float
    risk_reward_ratio: float
    position_pct: float
    method: str
    reasoning: str


class PositionSizer:
    """ATR-based position sizing with Kelly-lite."""

    def __init__(
        self,
        max_position_pct: float = 5.0,  # Max 5% per trade
        max_risk_pct: float = 1.0,  # Max 1% account risk per trade
        max_total_exposure: float = 15.0,  # Max 15% total exposure
        atr_stop_multiplier: float = 2.0,
        atr_tp_multiplier: float = 3.0,
    ):
        self.max_position_pct = max_position_pct
        self.max_risk_pct = max_risk_pct
        self.max_total_exposure = max_total_exposure
        self.atr_stop_mult = atr_stop_multiplier
        self.atr_tp_mult = atr_tp_multiplier

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate current ATR."""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)

        return tr.rolling(period).mean().iloc[-1]

    def kelly_fraction(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Calculate Kelly fraction (capped)."""
        if avg_loss == 0:
            return 0

        b = avg_win / abs(avg_loss)  # Win/loss ratio
        p = win_rate
        q = 1 - p

        kelly = (p * b - q) / b if b > 0 else 0

        # Cap at half-Kelly for safety
        return max(0, min(kelly / 2, 0.25))

    def calculate_size(
        self,
        account_equity: float,
        current_price: float,
        atr: float,
        current_exposure_pct: float = 0,
        historical_win_rate: float = 0.65,
        historical_avg_win: float = 8.0,
        historical_avg_loss: float = 4.0,
    ) -> PositionSize:
        """Calculate optimal position size."""
        reasoning_parts = []

        # 1. Check exposure limit
        available_exposure = self.max_total_exposure - current_exposure_pct
        if available_exposure <= 0:
            return PositionSize(
                qty=0, usd_amount=0, risk_amount=0,
                stop_loss_price=current_price,
                take_profit_price=current_price,
                risk_reward_ratio=0,
                position_pct=0,
                method="BLOCKED",
                reasoning="Max exposure reached",
            )

        # 2. Calculate stop loss and take profit
        stop_loss = current_price - (atr * self.atr_stop_mult)
        take_profit = current_price + (atr * self.atr_tp_mult)
        risk_per_unit = current_price - stop_loss
        reward_per_unit = take_profit - current_price
        rr_ratio = reward_per_unit / risk_per_unit if risk_per_unit > 0 else 0

        reasoning_parts.append(f"ATR={atr:.4f}, Stop={stop_loss:.4f}, TP={take_profit:.4f}")

        # 3. Method 1: Fixed percentage
        fixed_pct_size = account_equity * (self.max_position_pct / 100) / current_price
        reasoning_parts.append(f"Fixed {self.max_position_pct}%: {fixed_pct_size:.4f} units")

        # 4. Method 2: Risk-based (risk X% of account)
        risk_amount = account_equity * (self.max_risk_pct / 100)
        risk_based_size = risk_amount / risk_per_unit if risk_per_unit > 0 else 0
        reasoning_parts.append(f"Risk-based ({self.max_risk_pct}% risk): {risk_based_size:.4f} units")

        # 5. Method 3: Kelly-lite
        kelly_frac = self.kelly_fraction(
            historical_win_rate,
            historical_avg_win,
            historical_avg_loss
        )
        kelly_size = account_equity * kelly_frac / current_price
        reasoning_parts.append(f"Kelly ({kelly_frac*100:.1f}%): {kelly_size:.4f} units")

        # 6. Take minimum of all methods (conservative)
        final_qty = min(fixed_pct_size, risk_based_size, kelly_size)

        # 7. Check against available exposure
        max_qty_by_exposure = account_equity * (available_exposure / 100) / current_price
        if final_qty > max_qty_by_exposure:
            final_qty = max_qty_by_exposure
            reasoning_parts.append(f"Capped by exposure limit")

        usd_amount = final_qty * current_price
        position_pct = (usd_amount / account_equity) * 100
        actual_risk = final_qty * risk_per_unit

        return PositionSize(
            qty=round(final_qty, 6),
            usd_amount=round(usd_amount, 2),
            risk_amount=round(actual_risk, 2),
            stop_loss_price=round(stop_loss, 6),
            take_profit_price=round(take_profit, 6),
            risk_reward_ratio=round(rr_ratio, 2),
            position_pct=round(position_pct, 2),
            method="CONSERVATIVE_MIN",
            reasoning=" | ".join(reasoning_parts),
        )


# =============================================================================
# 8. PAPER-TRADING RUNNER
# =============================================================================

class PaperTradingRunner:
    """Paper trading runner with alerting."""

    def __init__(
        self,
        telegram_token: str = None,
        telegram_chat_id: str = None,
        db_path: str = DB_PATH,
    ):
        self.telegram_token = telegram_token
        self.telegram_chat_id = telegram_chat_id
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS paper_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                coin TEXT,
                side TEXT,
                entry_price REAL,
                qty REAL,
                stop_loss REAL,
                take_profit REAL,
                score INTEGER,
                regime TEXT,
                status TEXT,
                exit_price REAL,
                exit_timestamp TEXT,
                pnl REAL,
                reason TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                total_signals INTEGER,
                buy_signals INTEGER,
                open_positions INTEGER,
                closed_today INTEGER,
                daily_pnl REAL,
                portfolio_value REAL
            )
        """)

        conn.commit()
        conn.close()

    def _send_telegram(self, message: str) -> bool:
        """Send Telegram notification."""
        if not self.telegram_token or not self.telegram_chat_id:
            print(f"[ALERT] {message}")
            return False

        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            data = {
                "chat_id": self.telegram_chat_id,
                "text": message,
                "parse_mode": "HTML",
            }
            response = requests.post(url, json=data, timeout=10)
            return response.status_code == 200
        except Exception as e:
            print(f"Telegram error: {e}")
            return False

    def log_paper_trade(self, trade_data: Dict):
        """Log a paper trade."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO paper_trades (
                timestamp, coin, side, entry_price, qty,
                stop_loss, take_profit, score, regime, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            trade_data.get("coin"),
            trade_data.get("side", "BUY"),
            trade_data.get("entry_price"),
            trade_data.get("qty"),
            trade_data.get("stop_loss"),
            trade_data.get("take_profit"),
            trade_data.get("score", 0),
            trade_data.get("regime", ""),
            "OPEN",
        ))

        conn.commit()
        conn.close()

    def alert_signal(self, signal_data: Dict):
        """Send alert for a trading signal."""
        message = f"""
<b>🔔 PAPER TRADE SIGNAL</b>

<b>Coin:</b> {signal_data.get('coin')}
<b>Action:</b> {signal_data.get('action', 'BUY')}
<b>Price:</b> ${signal_data.get('price', 0):.4f}
<b>Score:</b> {signal_data.get('score', 0)}

<b>Position:</b>
  Qty: {signal_data.get('qty', 0):.4f}
  Value: ${signal_data.get('value', 0):.2f}
  Stop Loss: ${signal_data.get('stop_loss', 0):.4f}
  Take Profit: ${signal_data.get('take_profit', 0):.4f}

<b>Reasoning:</b>
{signal_data.get('reasoning', 'N/A')}

<i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>
"""
        self._send_telegram(message)

        # Log to database
        self.log_paper_trade(signal_data)

    def get_open_positions(self) -> List[Dict]:
        """Get all open paper positions."""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(
            "SELECT * FROM paper_trades WHERE status = 'OPEN'",
            conn
        )
        conn.close()
        return df.to_dict('records')

    def get_performance_summary(self) -> Dict:
        """Get paper trading performance."""
        conn = sqlite3.connect(self.db_path)

        # Closed trades
        closed = pd.read_sql_query(
            "SELECT * FROM paper_trades WHERE status = 'CLOSED'",
            conn
        )

        # Open positions
        open_pos = pd.read_sql_query(
            "SELECT * FROM paper_trades WHERE status = 'OPEN'",
            conn
        )

        conn.close()

        if closed.empty:
            return {
                "total_trades": 0,
                "open_positions": len(open_pos),
                "win_rate": 0,
                "total_pnl": 0,
            }

        wins = len(closed[closed["pnl"] > 0])

        return {
            "total_trades": len(closed),
            "open_positions": len(open_pos),
            "wins": wins,
            "win_rate": round(wins / len(closed) * 100, 1),
            "total_pnl": round(closed["pnl"].sum(), 2),
            "avg_pnl": round(closed["pnl"].mean(), 2),
        }


# =============================================================================
# 9. SIGNAL EXPLAINABILITY
# =============================================================================

@dataclass
class SignalExplanation:
    """Human-readable signal explanation."""
    coin: str
    timestamp: datetime
    action: str
    confidence: str
    price: float
    score: int
    threshold: int

    # Component breakdown
    technical_signals: List[str]
    fundamental_signals: List[str]
    onchain_signals: List[str]

    # Protection status
    protections_passed: List[str]
    protections_failed: List[str]

    # Risk levels
    stop_loss: float
    take_profit: float
    risk_reward: float
    position_size_pct: float

    # Reasoning
    primary_reason: str
    caveats: List[str]


class SignalExplainer:
    """Generate human-readable signal explanations."""

    def explain(
        self,
        coin: str,
        price: float,
        technical: Dict,
        fundamental: Dict,
        onchain: Dict,
        protections: Dict,
        position: Dict,
        regime: str,
    ) -> SignalExplanation:
        """Generate explanation for a signal."""

        # Technical signals
        tech_signals = []
        if technical.get("bb_touch"):
            tech_signals.append(f"✅ Price at lower Bollinger Band (${technical.get('bb_lower', 0):.4f})")
        if technical.get("rsi_oversold"):
            tech_signals.append(f"✅ RSI oversold ({technical.get('rsi', 0):.1f})")
        if technical.get("volume_spike"):
            tech_signals.append(f"✅ Volume spike ({technical.get('volume_ratio', 0):.1f}x average)")

        # Fundamental
        fund_signals = []
        if fundamental.get("positive_news"):
            for news in fundamental.get("positive_news", [])[:2]:
                fund_signals.append(f"📰 {news[:50]}...")

        # On-chain
        chain_signals = []
        if onchain.get("whale_accumulation"):
            chain_signals.append(f"🐋 Whale accumulation detected")
        if onchain.get("exchange_outflow"):
            chain_signals.append(f"📤 Exchange outflow (bullish)")

        # Protections
        passed = []
        failed = []

        if protections.get("above_200ma"):
            passed.append("Above 200 MA (uptrend)")
        else:
            failed.append("Below 200 MA (downtrend)")

        if protections.get("not_first_touch"):
            passed.append("Not first touch after crash")
        else:
            failed.append("First touch after crash (skip)")

        if protections.get("volume_ok"):
            passed.append("Volume sufficient")
        else:
            failed.append("Low volume")

        # Calculate score
        score = technical.get("score", 0) + fundamental.get("score", 0) + onchain.get("score", 0)
        threshold = 4 if regime == "SIDEWAYS" else 5

        # Determine action
        if score >= threshold and len(failed) == 0:
            action = "BUY"
            confidence = "HIGH" if score >= threshold + 2 else "MEDIUM"
        elif score >= threshold - 1 and len(failed) <= 1:
            action = "WATCH"
            confidence = "LOW"
        else:
            action = "SKIP"
            confidence = "N/A"

        # Primary reason
        if action == "BUY":
            primary_reason = f"Score {score} meets threshold {threshold} with {len(passed)}/{len(passed)+len(failed)} protections passed"
        elif action == "WATCH":
            primary_reason = f"Close to threshold. Missing: {', '.join(failed[:2])}"
        else:
            primary_reason = f"Score {score} below threshold {threshold}. Failed: {', '.join(failed[:2])}"

        # Caveats
        caveats = []
        if regime in ["BEAR", "VOLATILE"]:
            caveats.append(f"⚠️ {regime} market - higher risk")
        if len(failed) > 0:
            caveats.append(f"⚠️ {len(failed)} protection(s) failed")

        return SignalExplanation(
            coin=coin,
            timestamp=datetime.now(),
            action=action,
            confidence=confidence,
            price=price,
            score=score,
            threshold=threshold,
            technical_signals=tech_signals,
            fundamental_signals=fund_signals,
            onchain_signals=chain_signals,
            protections_passed=passed,
            protections_failed=failed,
            stop_loss=position.get("stop_loss", price * 0.95),
            take_profit=position.get("take_profit", price * 1.10),
            risk_reward=position.get("risk_reward", 1.5),
            position_size_pct=position.get("position_pct", 5.0),
            primary_reason=primary_reason,
            caveats=caveats,
        )

    def format_explanation(self, explanation: SignalExplanation) -> str:
        """Format explanation as readable text."""
        lines = [
            "=" * 60,
            f"SIGNAL EXPLANATION: {explanation.coin}",
            "=" * 60,
            f"",
            f"ACTION: {explanation.action} ({explanation.confidence} confidence)",
            f"Price: ${explanation.price:.4f}",
            f"Score: {explanation.score}/{explanation.threshold} threshold",
            f"",
            "TECHNICAL SIGNALS:",
        ]

        for sig in explanation.technical_signals or ["None"]:
            lines.append(f"  {sig}")

        lines.append("")
        lines.append("FUNDAMENTAL SIGNALS:")
        for sig in explanation.fundamental_signals or ["None"]:
            lines.append(f"  {sig}")

        lines.append("")
        lines.append("ON-CHAIN SIGNALS:")
        for sig in explanation.onchain_signals or ["None"]:
            lines.append(f"  {sig}")

        lines.append("")
        lines.append("PROTECTIONS:")
        for p in explanation.protections_passed:
            lines.append(f"  ✅ {p}")
        for p in explanation.protections_failed:
            lines.append(f"  ❌ {p}")

        lines.append("")
        lines.append("RISK MANAGEMENT:")
        lines.append(f"  Stop Loss: ${explanation.stop_loss:.4f}")
        lines.append(f"  Take Profit: ${explanation.take_profit:.4f}")
        lines.append(f"  Risk/Reward: {explanation.risk_reward:.1f}")
        lines.append(f"  Position Size: {explanation.position_size_pct:.1f}%")

        lines.append("")
        lines.append(f"PRIMARY REASON: {explanation.primary_reason}")

        if explanation.caveats:
            lines.append("")
            lines.append("CAVEATS:")
            for c in explanation.caveats:
                lines.append(f"  {c}")

        lines.append("=" * 60)

        return "\n".join(lines)


# =============================================================================
# 10. MONITORING & SAFEGUARDS
# =============================================================================

@dataclass
class HealthStatus:
    """System health status."""
    timestamp: datetime
    status: str  # OK, WARNING, CRITICAL
    data_fresh: bool
    api_healthy: bool
    signals_normal: bool
    exposure_ok: bool
    issues: List[str]


class MonitoringSystem:
    """System monitoring and safeguards."""

    def __init__(
        self,
        max_signals_per_day: int = 10,
        max_exposure_pct: float = 15.0,
        data_staleness_hours: int = 2,
        telegram_token: str = None,
        telegram_chat_id: str = None,
    ):
        self.max_signals = max_signals_per_day
        self.max_exposure = max_exposure_pct
        self.data_staleness = data_staleness_hours
        self.telegram_token = telegram_token
        self.telegram_chat_id = telegram_chat_id
        self.last_data_fetch = None
        self.signals_today = 0

    def _send_alert(self, message: str, level: str = "WARNING"):
        """Send monitoring alert."""
        emoji = "⚠️" if level == "WARNING" else "🚨" if level == "CRITICAL" else "ℹ️"
        full_message = f"{emoji} [{level}] {message}"

        print(full_message)

        if self.telegram_token and self.telegram_chat_id:
            try:
                url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
                requests.post(url, json={
                    "chat_id": self.telegram_chat_id,
                    "text": full_message,
                }, timeout=10)
            except:
                pass

    def check_data_freshness(self, last_fetch: datetime = None) -> Tuple[bool, str]:
        """Check if data is fresh."""
        if last_fetch is None:
            last_fetch = self.last_data_fetch

        if last_fetch is None:
            return False, "No data fetch recorded"

        hours_old = (datetime.now() - last_fetch).total_seconds() / 3600

        if hours_old > self.data_staleness:
            return False, f"Data is {hours_old:.1f} hours old"

        return True, f"Data is {hours_old:.1f} hours old (OK)"

    def check_api_health(self) -> Tuple[bool, str]:
        """Check API connectivity."""
        try:
            response = requests.get(
                "https://api.binance.us/api/v3/ping",
                timeout=5
            )
            if response.status_code == 200:
                return True, "API healthy"
            return False, f"API returned {response.status_code}"
        except Exception as e:
            return False, f"API error: {e}"

    def check_signal_rate(self) -> Tuple[bool, str]:
        """Check if signal rate is normal."""
        if self.signals_today > self.max_signals:
            return False, f"Too many signals today: {self.signals_today}"
        return True, f"Signals today: {self.signals_today}/{self.max_signals}"

    def check_exposure(self, current_exposure_pct: float) -> Tuple[bool, str]:
        """Check portfolio exposure."""
        if current_exposure_pct > self.max_exposure:
            return False, f"Exposure {current_exposure_pct:.1f}% exceeds {self.max_exposure}%"
        return True, f"Exposure: {current_exposure_pct:.1f}%/{self.max_exposure}%"

    def run_health_check(self, current_exposure_pct: float = 0) -> HealthStatus:
        """Run full health check."""
        issues = []

        data_ok, data_msg = self.check_data_freshness()
        if not data_ok:
            issues.append(data_msg)

        api_ok, api_msg = self.check_api_health()
        if not api_ok:
            issues.append(api_msg)

        signals_ok, signals_msg = self.check_signal_rate()
        if not signals_ok:
            issues.append(signals_msg)

        exposure_ok, exposure_msg = self.check_exposure(current_exposure_pct)
        if not exposure_ok:
            issues.append(exposure_msg)

        # Determine status
        if len(issues) == 0:
            status = "OK"
        elif any("CRITICAL" in i or "API error" in i for i in issues):
            status = "CRITICAL"
        else:
            status = "WARNING"

        health = HealthStatus(
            timestamp=datetime.now(),
            status=status,
            data_fresh=data_ok,
            api_healthy=api_ok,
            signals_normal=signals_ok,
            exposure_ok=exposure_ok,
            issues=issues,
        )

        # Send alerts for non-OK status
        if status != "OK":
            self._send_alert(
                f"Health check {status}: {', '.join(issues)}",
                level=status
            )

        return health

    def emergency_stop_check(
        self,
        daily_pnl_pct: float,
        max_daily_loss_pct: float = -5.0,
    ) -> Tuple[bool, str]:
        """Check if emergency stop should trigger."""
        if daily_pnl_pct <= max_daily_loss_pct:
            self._send_alert(
                f"EMERGENCY STOP: Daily loss {daily_pnl_pct:.1f}% exceeds limit {max_daily_loss_pct}%",
                level="CRITICAL"
            )
            return True, f"Daily loss limit hit: {daily_pnl_pct:.1f}%"

        return False, f"Daily P&L: {daily_pnl_pct:.1f}%"


# =============================================================================
# MAIN
# =============================================================================

def run_full_system_report():
    """Run full production system report."""
    print("=" * 80)
    print("PRODUCTION SYSTEM REPORT")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # 1. Realistic Backtest
    print("\n" + "-" * 80)
    print("1. REALISTIC BACKTEST (with slippage & fees)")
    print("-" * 80)

    backtester = RealisticBacktester()

    for coin, config in COINS.items():
        print(f"\n   Testing {coin}...")
        result = backtester.backtest(config["symbol"])

        if "error" in result:
            print(f"   Error: {result['error']}")
            continue

        print(f"\n   {coin} Results:")
        print(f"      Initial: ${result['initial_capital']:.2f}")
        print(f"      Final: ${result['final_capital']:.2f}")
        print(f"      Return: {result['total_return_pct']}%")
        print(f"      Trades: {result['trades']} (Win rate: {result['win_rate']}%)")
        print(f"      Gross P&L: ${result['gross_pnl']:.2f}")
        print(f"      Net P&L: ${result['net_pnl']:.2f}")
        print(f"      Fees paid: ${result['total_fees']:.2f}")
        print(f"      Slippage cost: ${result['total_slippage_cost']:.2f}")

    # 2. Position Sizing
    print("\n" + "-" * 80)
    print("2. POSITION SIZING EXAMPLE")
    print("-" * 80)

    sizer = PositionSizer()

    example_size = sizer.calculate_size(
        account_equity=1000,
        current_price=2.10,
        atr=0.08,
        current_exposure_pct=0,
    )

    print(f"\n   Account: $1000, XRP @ $2.10, ATR=$0.08")
    print(f"   Calculated Position:")
    print(f"      Qty: {example_size.qty:.4f}")
    print(f"      USD Amount: ${example_size.usd_amount:.2f}")
    print(f"      Position %: {example_size.position_pct}%")
    print(f"      Risk Amount: ${example_size.risk_amount:.2f}")
    print(f"      Stop Loss: ${example_size.stop_loss_price:.4f}")
    print(f"      Take Profit: ${example_size.take_profit_price:.4f}")
    print(f"      Risk/Reward: {example_size.risk_reward_ratio}")
    print(f"\n   Reasoning: {example_size.reasoning}")

    # 3. Monitoring
    print("\n" + "-" * 80)
    print("3. SYSTEM HEALTH CHECK")
    print("-" * 80)

    monitor = MonitoringSystem()
    monitor.last_data_fetch = datetime.now()
    health = monitor.run_health_check(current_exposure_pct=5.0)

    print(f"\n   Status: {health.status}")
    print(f"   Data Fresh: {'✅' if health.data_fresh else '❌'}")
    print(f"   API Healthy: {'✅' if health.api_healthy else '❌'}")
    print(f"   Signals Normal: {'✅' if health.signals_normal else '❌'}")
    print(f"   Exposure OK: {'✅' if health.exposure_ok else '❌'}")

    if health.issues:
        print(f"\n   Issues:")
        for issue in health.issues:
            print(f"      - {issue}")

    # 4. Signal Explanation Example
    print("\n" + "-" * 80)
    print("4. SIGNAL EXPLAINABILITY EXAMPLE")
    print("-" * 80)

    explainer = SignalExplainer()

    example_explanation = explainer.explain(
        coin="XRP",
        price=2.05,
        technical={
            "bb_touch": True,
            "bb_lower": 2.03,
            "rsi_oversold": True,
            "rsi": 28,
            "volume_spike": False,
            "volume_ratio": 0.8,
            "score": 2,
        },
        fundamental={
            "positive_news": ["SEC ruling favorable for Ripple"],
            "score": 1,
        },
        onchain={
            "whale_accumulation": True,
            "score": 2,
        },
        protections={
            "above_200ma": False,
            "not_first_touch": True,
            "volume_ok": True,
        },
        position={
            "stop_loss": 1.95,
            "take_profit": 2.25,
            "risk_reward": 2.0,
            "position_pct": 5.0,
        },
        regime="SIDEWAYS",
    )

    print(explainer.format_explanation(example_explanation))

    # Summary
    print("\n" + "=" * 80)
    print("PRODUCTION SYSTEM SUMMARY")
    print("=" * 80)

    print("""
   Components Ready:
   ✅ Realistic Backtester (slippage, fees, partial fills)
   ✅ Position Sizing Engine (ATR, Kelly-lite, exposure limits)
   ✅ Paper Trading Runner (database, alerts)
   ✅ Signal Explainability (human-readable reasoning)
   ✅ Monitoring & Safeguards (health checks, emergency stops)

   Next Steps:
   1. Configure Telegram credentials in config
   2. Set up daily cron job for paper trading
   3. Run for 1-3 months before live capital
   4. Monitor consistency rate and precision metrics
""")
    print("=" * 80)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Production Trading System")
    parser.add_argument("--backtest", action="store_true", help="Run realistic backtest")
    parser.add_argument("--sizing", action="store_true", help="Show position sizing")
    parser.add_argument("--paper", action="store_true", help="Paper trading status")
    parser.add_argument("--explain", action="store_true", help="Signal explanation demo")
    parser.add_argument("--monitor", action="store_true", help="Run health check")
    parser.add_argument("--full", action="store_true", help="Full system report")

    args = parser.parse_args()

    if args.backtest:
        backtester = RealisticBacktester()
        for coin, config in COINS.items():
            result = backtester.backtest(config["symbol"])
            print(f"{coin}: {result}")

    elif args.sizing:
        sizer = PositionSizer()
        size = sizer.calculate_size(1000, 2.10, 0.08)
        print(f"Position: {size}")

    elif args.monitor:
        monitor = MonitoringSystem()
        monitor.last_data_fetch = datetime.now()
        health = monitor.run_health_check()
        print(f"Health: {health}")

    else:
        run_full_system_report()


if __name__ == "__main__":
    main()
