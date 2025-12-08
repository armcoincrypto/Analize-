"""
Trading simulator for backtesting parameter combinations.

Provides realistic execution simulation including:
- Spread and slippage
- Exchange constraints (min notional, lot sizes)
- Commission fees
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from analize.config import get_settings
from analize.models.signals import ExitReason, SignalRecord


@dataclass
class Trade:
    """Represents a simulated trade."""

    signal_id: str
    symbol: str
    entry_time: datetime
    entry_price: float
    position_size: float
    direction: int  # 1 = long, -1 = short

    exit_time: datetime | None = None
    exit_price: float | None = None
    exit_reason: ExitReason | None = None

    tp_price: float | None = None
    sl_price: float | None = None

    slippage_entry: float = 0.0
    slippage_exit: float = 0.0
    commission: float = 0.0

    mfe: float = 0.0
    mae: float = 0.0

    @property
    def is_open(self) -> bool:
        return self.exit_time is None

    @property
    def pnl_pct(self) -> float:
        if self.exit_price is None:
            return 0.0
        raw_pnl = (self.exit_price - self.entry_price) / self.entry_price * 100
        return raw_pnl * self.direction - self.total_costs_pct

    @property
    def pnl_usd(self) -> float:
        return self.pnl_pct / 100 * self.position_size

    @property
    def total_costs_pct(self) -> float:
        return self.slippage_entry + self.slippage_exit + self.commission

    @property
    def duration_minutes(self) -> float | None:
        if self.exit_time is None or self.entry_time is None:
            return None
        return (self.exit_time - self.entry_time).total_seconds() / 60


@dataclass
class SimulationConfig:
    """Configuration for trading simulation."""

    # Execution costs
    slippage_bps: float = 5.0  # Base slippage in basis points
    commission_bps: float = 10.0  # Round-trip commission

    # Slippage model
    slippage_model: str = "fixed"  # "fixed" or "volume_based"
    volume_impact_factor: float = 0.1  # For volume-based model

    # Exchange constraints
    min_notional: float = 10.0  # Minimum trade size in USD
    lot_step: float = 0.001  # Lot size step

    # Position management
    max_position_size_pct: float = 10.0  # Max position as % of account
    max_open_positions: int = 10

    # Timing
    max_trade_duration_mins: int = 1440  # 24 hours


@dataclass
class SimulationResult:
    """Results from a simulation run."""

    config: SimulationConfig
    parameters: dict[str, Any]
    trades: list[Trade] = field(default_factory=list)

    # Aggregate metrics
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    total_pnl_pct: float = 0.0
    total_pnl_usd: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0

    # Equity curve
    equity_curve: list[float] = field(default_factory=list)

    # Execution stats
    avg_slippage_pct: float = 0.0
    total_commission_pct: float = 0.0


class TradingSimulator:
    """Simulates trading with realistic execution model."""

    def __init__(self, config: SimulationConfig | None = None):
        self.config = config or SimulationConfig()
        settings = get_settings()

        if config is None:
            self.config.slippage_bps = settings.analysis.default_slippage_bps
            self.config.commission_bps = settings.analysis.default_commission_bps

    def calculate_slippage(
        self,
        price: float,
        volume: float = 0.0,
        is_entry: bool = True,
    ) -> float:
        """
        Calculate slippage for a trade.

        Returns slippage as percentage of price.
        """
        base_slippage = self.config.slippage_bps / 10000

        if self.config.slippage_model == "volume_based" and volume > 0:
            # Higher volume = lower slippage (more liquidity)
            volume_factor = 1 / (1 + self.config.volume_impact_factor * np.log1p(volume))
            return base_slippage * volume_factor

        return base_slippage

    def simulate_signal(
        self,
        signal: SignalRecord,
        price_data: pd.DataFrame,
        tp_pct: float,
        sl_pct: float,
        position_size_usd: float = 100.0,
    ) -> Trade | None:
        """
        Simulate a single signal.

        Args:
            signal: Signal record
            price_data: DataFrame with timestamp, open, high, low, close, volume
            tp_pct: Take profit percentage
            sl_pct: Stop loss percentage
            position_size_usd: Position size in USD

        Returns:
            Trade object or None if trade couldn't be executed
        """
        # Validate position size
        if position_size_usd < self.config.min_notional:
            return None

        # Calculate entry price with slippage
        entry_price = signal.exec_price or signal.price_close
        slippage_entry = self.calculate_slippage(entry_price, signal.volume)
        adjusted_entry = entry_price * (1 + slippage_entry)  # Assume long

        # Calculate TP/SL prices
        tp_price = adjusted_entry * (1 + tp_pct / 100)
        sl_price = adjusted_entry * (1 - sl_pct / 100)

        # Commission (entry side)
        commission_entry = self.config.commission_bps / 2 / 10000

        # Create trade
        trade = Trade(
            signal_id=str(signal.signal_id),
            symbol=signal.symbol,
            entry_time=signal.timestamp_utc,
            entry_price=adjusted_entry,
            position_size=position_size_usd,
            direction=1,  # Assume long for now
            tp_price=tp_price,
            sl_price=sl_price,
            slippage_entry=slippage_entry * 100,
            commission=self.config.commission_bps / 100,
        )

        # Simulate through price data
        price_data = price_data[price_data.index > signal.timestamp_utc]
        price_data = price_data.head(self.config.max_trade_duration_mins)

        mfe = 0.0
        mae = 0.0

        for timestamp, row in price_data.iterrows():
            high = row["high"]
            low = row["low"]
            close = row["close"]

            # Track MFE/MAE
            high_pnl = (high - adjusted_entry) / adjusted_entry * 100
            low_pnl = (low - adjusted_entry) / adjusted_entry * 100
            mfe = max(mfe, high_pnl)
            mae = min(mae, low_pnl)

            # Check TP hit
            if high >= tp_price:
                slippage_exit = self.calculate_slippage(tp_price)
                trade.exit_time = timestamp
                trade.exit_price = tp_price * (1 - slippage_exit)  # Slippage works against us
                trade.exit_reason = ExitReason.TP
                trade.slippage_exit = slippage_exit * 100
                break

            # Check SL hit
            if low <= sl_price:
                slippage_exit = self.calculate_slippage(sl_price)
                trade.exit_time = timestamp
                trade.exit_price = sl_price * (1 - slippage_exit)
                trade.exit_reason = ExitReason.SL
                trade.slippage_exit = slippage_exit * 100
                break

        # If no exit, timeout
        if trade.exit_time is None and not price_data.empty:
            last_row = price_data.iloc[-1]
            slippage_exit = self.calculate_slippage(last_row["close"])
            trade.exit_time = price_data.index[-1]
            trade.exit_price = last_row["close"] * (1 - slippage_exit)
            trade.exit_reason = ExitReason.TIMEOUT
            trade.slippage_exit = slippage_exit * 100

        trade.mfe = mfe
        trade.mae = mae

        return trade

    def run_simulation(
        self,
        signals: list[SignalRecord],
        price_data: pd.DataFrame,
        parameters: dict[str, Any],
    ) -> SimulationResult:
        """
        Run simulation on a set of signals with given parameters.

        Args:
            signals: List of signal records
            price_data: Price data DataFrame (indexed by timestamp)
            parameters: Parameter dictionary (must include tp_pct, sl_pct)

        Returns:
            SimulationResult with all trades and metrics
        """
        tp_pct = parameters.get("tp_pct", 2.0)
        sl_pct = parameters.get("sl_pct", 1.0)
        position_size = parameters.get("position_size_usd", 100.0)

        result = SimulationResult(
            config=self.config,
            parameters=parameters,
        )

        # Ensure price data is indexed by timestamp
        if "timestamp" in price_data.columns:
            price_data = price_data.set_index("timestamp")

        trades = []
        equity = [0.0]

        for signal in signals:
            trade = self.simulate_signal(
                signal,
                price_data,
                tp_pct,
                sl_pct,
                position_size,
            )

            if trade and trade.exit_time:
                trades.append(trade)
                equity.append(equity[-1] + trade.pnl_pct)

        result.trades = trades
        result.equity_curve = equity

        # Calculate aggregate metrics
        if trades:
            pnl_series = pd.Series([t.pnl_pct for t in trades])
            result.total_trades = len(trades)
            result.wins = sum(1 for t in trades if t.pnl_pct > 0)
            result.losses = result.total_trades - result.wins
            result.win_rate = result.wins / result.total_trades * 100

            gross_profit = sum(t.pnl_pct for t in trades if t.pnl_pct > 0)
            gross_loss = abs(sum(t.pnl_pct for t in trades if t.pnl_pct <= 0))
            result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

            result.total_pnl_pct = sum(t.pnl_pct for t in trades)
            result.total_pnl_usd = sum(t.pnl_usd for t in trades)

            # Expectancy
            if result.total_trades > 0:
                avg_win = gross_profit / result.wins if result.wins > 0 else 0
                avg_loss = gross_loss / result.losses if result.losses > 0 else 0
                result.expectancy = (
                    (result.win_rate / 100 * avg_win) -
                    ((100 - result.win_rate) / 100 * avg_loss)
                )

            # Max drawdown
            equity_series = pd.Series(equity)
            running_max = equity_series.cummax()
            drawdown = equity_series - running_max
            result.max_drawdown_pct = float(drawdown.min())

            # Sharpe ratio (simplified)
            if pnl_series.std() > 0:
                result.sharpe_ratio = float(
                    pnl_series.mean() / pnl_series.std() * np.sqrt(252)
                )

            # Execution stats
            result.avg_slippage_pct = np.mean([t.slippage_entry + t.slippage_exit for t in trades])
            result.total_commission_pct = sum(t.commission for t in trades)

        return result

    def calculate_metrics_from_mfe_mae(
        self,
        signals: list[SignalRecord],
        tp_pct: float,
        sl_pct: float,
    ) -> dict[str, Any]:
        """
        Quick metrics calculation using pre-computed MFE/MAE.

        This is faster than full simulation but less accurate.
        """
        wins = 0
        losses = 0
        timeouts = 0

        for signal in signals:
            mfe = signal.mfe_pct or 0
            mae = signal.mae_pct or 0

            # Simplified: assume TP hit if MFE >= TP
            if mfe >= tp_pct:
                wins += 1
            elif mae <= -sl_pct:
                losses += 1
            else:
                timeouts += 1

        total = wins + losses + timeouts
        if total == 0:
            return {}

        gross_profit = wins * tp_pct
        gross_loss = losses * sl_pct

        return {
            "tp_pct": tp_pct,
            "sl_pct": sl_pct,
            "total_trades": total,
            "wins": wins,
            "losses": losses,
            "timeouts": timeouts,
            "win_rate": round(wins / total * 100, 2),
            "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0,
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
            "net_pnl": round(gross_profit - gross_loss, 2),
        }
