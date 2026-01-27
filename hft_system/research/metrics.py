"""
Performance Metrics
===================
Calculate key performance metrics for strategy evaluation.

Metrics:
- PnL after costs
- Maximum drawdown
- Sharpe ratio (simplified)
- Trade count and win rate
"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class TradeResult:
    """Single trade result for metrics calculation."""
    entry_time: int
    exit_time: int
    entry_price: float
    exit_price: float
    pnl_pct: float
    pnl_after_costs_pct: float
    side: str  # "long" or "short"
    exit_reason: str


@dataclass
class StrategyMetrics:
    """Aggregate metrics for a strategy run."""
    trade_count: int
    win_count: int
    loss_count: int
    win_rate: float
    total_pnl_pct: float
    total_pnl_after_costs_pct: float
    avg_pnl_pct: float
    avg_pnl_after_costs_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    profit_factor: float
    avg_hold_time_sec: float
    # Breakdown
    by_exit_reason: Dict[str, Dict]


def calculate_pnl_after_costs(
    pnl_pct: float,
    entry_fee_pct: float = 0.10,
    exit_fee_pct: float = 0.10,
    spread_cost_pct: float = 0.02,
    slippage_pct: float = 0.02
) -> float:
    """
    Calculate PnL after all trading costs.

    Args:
        pnl_pct: Gross PnL percentage
        entry_fee_pct: Entry fee (e.g., 0.10% for taker)
        exit_fee_pct: Exit fee (e.g., 0.10% for taker)
        spread_cost_pct: Half spread cost (each way)
        slippage_pct: Market impact

    Returns:
        Net PnL percentage after costs
    """
    total_costs = entry_fee_pct + exit_fee_pct + (2 * spread_cost_pct) + slippage_pct
    return pnl_pct - total_costs


def calculate_max_drawdown(equity_curve: List[float]) -> float:
    """
    Calculate maximum drawdown from an equity curve.

    Args:
        equity_curve: List of cumulative equity values

    Returns:
        Maximum drawdown as a percentage (positive number)
    """
    if not equity_curve or len(equity_curve) < 2:
        return 0.0

    equity = np.array(equity_curve)
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak * 100

    return float(np.max(drawdown))


def calculate_sharpe_ratio(
    returns: List[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252 * 24 * 60  # Per-minute for HFT
) -> float:
    """
    Calculate simplified Sharpe ratio.

    For HFT, we use per-trade returns and annualize assuming
    many trades per day.

    Args:
        returns: List of per-trade returns (percentages)
        risk_free_rate: Annual risk-free rate (default 0)
        periods_per_year: Trading periods per year for annualization

    Returns:
        Sharpe ratio (annualized)
    """
    if not returns or len(returns) < 2:
        return 0.0

    returns_array = np.array(returns)
    mean_return = np.mean(returns_array)
    std_return = np.std(returns_array, ddof=1)

    if std_return == 0:
        return 0.0 if mean_return <= 0 else float('inf')

    # Simple Sharpe (not annualized for HFT - trades are too frequent)
    sharpe = mean_return / std_return

    # Optional: annualize
    # Assume ~100 trades per day average for HFT
    # sharpe = sharpe * np.sqrt(100 * 252)

    return float(sharpe)


def calculate_profit_factor(wins: List[float], losses: List[float]) -> float:
    """
    Calculate profit factor (gross wins / gross losses).

    Args:
        wins: List of winning trade PnLs (positive)
        losses: List of losing trade PnLs (negative)

    Returns:
        Profit factor (> 1 is profitable)
    """
    if not losses:
        return float('inf') if wins else 0.0

    total_wins = sum(wins)
    total_losses = abs(sum(losses))

    if total_losses == 0:
        return float('inf') if total_wins > 0 else 0.0

    return total_wins / total_losses


def calculate_metrics(trades: List[TradeResult]) -> StrategyMetrics:
    """
    Calculate all metrics for a list of trades.

    Args:
        trades: List of TradeResult objects

    Returns:
        StrategyMetrics with all calculated metrics
    """
    if not trades:
        return StrategyMetrics(
            trade_count=0,
            win_count=0,
            loss_count=0,
            win_rate=0.0,
            total_pnl_pct=0.0,
            total_pnl_after_costs_pct=0.0,
            avg_pnl_pct=0.0,
            avg_pnl_after_costs_pct=0.0,
            max_drawdown_pct=0.0,
            sharpe_ratio=0.0,
            profit_factor=0.0,
            avg_hold_time_sec=0.0,
            by_exit_reason={}
        )

    # Basic counts
    trade_count = len(trades)
    wins = [t for t in trades if t.pnl_after_costs_pct > 0]
    losses = [t for t in trades if t.pnl_after_costs_pct <= 0]
    win_count = len(wins)
    loss_count = len(losses)
    win_rate = (win_count / trade_count * 100) if trade_count > 0 else 0.0

    # PnL
    pnl_list = [t.pnl_pct for t in trades]
    pnl_after_costs_list = [t.pnl_after_costs_pct for t in trades]

    total_pnl_pct = sum(pnl_list)
    total_pnl_after_costs_pct = sum(pnl_after_costs_list)
    avg_pnl_pct = np.mean(pnl_list)
    avg_pnl_after_costs_pct = np.mean(pnl_after_costs_list)

    # Equity curve for drawdown (starting at 100)
    equity_curve = [100.0]
    for pnl in pnl_after_costs_list:
        equity_curve.append(equity_curve[-1] * (1 + pnl / 100))

    max_drawdown_pct = calculate_max_drawdown(equity_curve)

    # Sharpe ratio
    sharpe_ratio = calculate_sharpe_ratio(pnl_after_costs_list)

    # Profit factor
    win_pnls = [t.pnl_after_costs_pct for t in wins]
    loss_pnls = [t.pnl_after_costs_pct for t in losses]
    profit_factor = calculate_profit_factor(win_pnls, loss_pnls)

    # Hold time
    hold_times = [(t.exit_time - t.entry_time) / 1000 for t in trades]  # ms to sec
    avg_hold_time_sec = np.mean(hold_times) if hold_times else 0.0

    # By exit reason
    by_exit_reason = {}
    for trade in trades:
        reason = trade.exit_reason or "unknown"
        if reason not in by_exit_reason:
            by_exit_reason[reason] = {
                "count": 0,
                "wins": 0,
                "total_pnl_pct": 0.0
            }
        by_exit_reason[reason]["count"] += 1
        by_exit_reason[reason]["total_pnl_pct"] += trade.pnl_after_costs_pct
        if trade.pnl_after_costs_pct > 0:
            by_exit_reason[reason]["wins"] += 1

    # Calculate averages for exit reasons
    for reason in by_exit_reason:
        count = by_exit_reason[reason]["count"]
        by_exit_reason[reason]["avg_pnl_pct"] = by_exit_reason[reason]["total_pnl_pct"] / count if count > 0 else 0
        by_exit_reason[reason]["win_rate"] = by_exit_reason[reason]["wins"] / count * 100 if count > 0 else 0

    return StrategyMetrics(
        trade_count=trade_count,
        win_count=win_count,
        loss_count=loss_count,
        win_rate=win_rate,
        total_pnl_pct=total_pnl_pct,
        total_pnl_after_costs_pct=total_pnl_after_costs_pct,
        avg_pnl_pct=float(avg_pnl_pct),
        avg_pnl_after_costs_pct=float(avg_pnl_after_costs_pct),
        max_drawdown_pct=max_drawdown_pct,
        sharpe_ratio=sharpe_ratio,
        profit_factor=profit_factor,
        avg_hold_time_sec=float(avg_hold_time_sec),
        by_exit_reason=by_exit_reason
    )


def metrics_to_dict(metrics: StrategyMetrics) -> Dict:
    """Convert StrategyMetrics to dictionary for serialization."""
    return {
        "trade_count": metrics.trade_count,
        "win_count": metrics.win_count,
        "loss_count": metrics.loss_count,
        "win_rate": round(metrics.win_rate, 2),
        "total_pnl_pct": round(metrics.total_pnl_pct, 4),
        "total_pnl_after_costs_pct": round(metrics.total_pnl_after_costs_pct, 4),
        "avg_pnl_pct": round(metrics.avg_pnl_pct, 4),
        "avg_pnl_after_costs_pct": round(metrics.avg_pnl_after_costs_pct, 4),
        "max_drawdown_pct": round(metrics.max_drawdown_pct, 4),
        "sharpe_ratio": round(metrics.sharpe_ratio, 4),
        "profit_factor": round(metrics.profit_factor, 4) if metrics.profit_factor != float('inf') else "inf",
        "avg_hold_time_sec": round(metrics.avg_hold_time_sec, 2),
        "by_exit_reason": metrics.by_exit_reason
    }
