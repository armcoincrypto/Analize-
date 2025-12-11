"""
Trading metrics calculations.

Core metric calculations for trading performance analysis.
"""

from typing import Any

import numpy as np
import pandas as pd


class TradingMetrics:
    """Calculates trading performance metrics."""

    @staticmethod
    def win_rate(wins: int, total: int) -> float:
        """Calculate win rate percentage."""
        if total == 0:
            return 0.0
        return (wins / total) * 100

    @staticmethod
    def profit_factor(gross_profit: float, gross_loss: float) -> float:
        """
        Calculate profit factor.

        PF = Gross Profit / Gross Loss
        """
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return abs(gross_profit / gross_loss)

    @staticmethod
    def expectancy(
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """
        Calculate expectancy (expected value per trade).

        E = (Win Rate * Avg Win) - (Loss Rate * Avg Loss)
        """
        loss_rate = 1 - (win_rate / 100)
        return (win_rate / 100 * avg_win) - (loss_rate * abs(avg_loss))

    @staticmethod
    def risk_reward_ratio(avg_win: float, avg_loss: float) -> float:
        """Calculate risk:reward ratio."""
        if avg_loss == 0:
            return float("inf") if avg_win > 0 else 0.0
        return abs(avg_win / avg_loss)

    @staticmethod
    def sharpe_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ) -> float:
        """
        Calculate Sharpe ratio.

        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods in a year

        Returns:
            Annualized Sharpe ratio
        """
        if returns.empty or returns.std() == 0:
            return 0.0

        excess_returns = returns - (risk_free_rate / periods_per_year)
        return float(
            np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()
        )

    @staticmethod
    def sortino_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ) -> float:
        """
        Calculate Sortino ratio (downside risk only).

        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods in a year

        Returns:
            Annualized Sortino ratio
        """
        if returns.empty:
            return 0.0

        excess_returns = returns - (risk_free_rate / periods_per_year)
        downside_returns = returns[returns < 0]

        if downside_returns.empty or downside_returns.std() == 0:
            return float("inf") if excess_returns.mean() > 0 else 0.0

        downside_std = downside_returns.std()
        return float(
            np.sqrt(periods_per_year) * excess_returns.mean() / downside_std
        )

    @staticmethod
    def max_drawdown(equity_curve: pd.Series) -> tuple[float, int, int]:
        """
        Calculate maximum drawdown.

        Args:
            equity_curve: Series of cumulative equity/returns

        Returns:
            Tuple of (max_dd_pct, peak_idx, trough_idx)
        """
        if equity_curve.empty:
            return 0.0, 0, 0

        # Handle both individual returns and cumulative equity
        if equity_curve.iloc[0] != equity_curve.sum():
            cumulative = equity_curve.cumsum()
        else:
            cumulative = equity_curve.copy()

        # Add starting capital (100%) to convert returns to equity
        # This ensures we're calculating drawdown from equity, not just returns
        equity = 100.0 + cumulative  # Start at 100%, add cumulative returns

        # Ensure equity doesn't go negative for drawdown calculation
        # (If equity goes to 0 or negative, that's a -100% drawdown, cap it there)
        running_max = equity.cummax()

        # Protect against division by zero
        running_max = running_max.replace(0, np.nan)
        drawdown = (equity - running_max) / running_max * 100
        drawdown = drawdown.fillna(-100.0)  # If running_max was 0, assume -100% drawdown

        # Cap drawdown at -100% (can't lose more than 100%)
        drawdown = drawdown.clip(lower=-100.0)

        max_dd = drawdown.min()
        trough_idx = int(drawdown.idxmin()) if not pd.isna(drawdown.idxmin()) else 0

        # Find peak before trough
        peak_idx = int(running_max[:trough_idx].idxmax()) if trough_idx > 0 else 0

        return float(max_dd), peak_idx, trough_idx

    @staticmethod
    def calmar_ratio(
        returns: pd.Series,
        periods_per_year: int = 252,
    ) -> float:
        """
        Calculate Calmar ratio (return / max drawdown).

        Args:
            returns: Series of returns
            periods_per_year: Number of periods in a year

        Returns:
            Calmar ratio
        """
        if returns.empty:
            return 0.0

        annual_return = returns.mean() * periods_per_year
        max_dd, _, _ = TradingMetrics.max_drawdown(returns)

        if max_dd == 0:
            return float("inf") if annual_return > 0 else 0.0

        return float(annual_return / abs(max_dd))

    @staticmethod
    def kelly_criterion(win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Calculate Kelly criterion (optimal bet size).

        K = W - (1-W)/R
        where W = win rate, R = win/loss ratio
        """
        if avg_loss == 0:
            return 0.0

        w = win_rate / 100
        r = abs(avg_win / avg_loss)

        kelly = w - (1 - w) / r
        return max(0.0, kelly)  # Never bet negative

    @staticmethod
    def consecutive_stats(results: pd.Series) -> dict[str, int]:
        """
        Calculate consecutive win/loss statistics.

        Args:
            results: Series of boolean (True = win) or numeric (>0 = win)

        Returns:
            Dictionary with consecutive stats
        """
        if results.empty:
            return {
                "max_consecutive_wins": 0,
                "max_consecutive_losses": 0,
                "avg_consecutive_wins": 0,
                "avg_consecutive_losses": 0,
            }

        # Convert to boolean
        wins = results > 0 if results.dtype != bool else results

        # Find consecutive runs
        win_runs = []
        loss_runs = []
        current_run = 0
        is_winning = None

        for w in wins:
            if is_winning is None:
                is_winning = w
                current_run = 1
            elif w == is_winning:
                current_run += 1
            else:
                if is_winning:
                    win_runs.append(current_run)
                else:
                    loss_runs.append(current_run)
                is_winning = w
                current_run = 1

        # Don't forget the last run
        if is_winning is not None:
            if is_winning:
                win_runs.append(current_run)
            else:
                loss_runs.append(current_run)

        return {
            "max_consecutive_wins": max(win_runs) if win_runs else 0,
            "max_consecutive_losses": max(loss_runs) if loss_runs else 0,
            "avg_consecutive_wins": int(np.mean(win_runs)) if win_runs else 0,
            "avg_consecutive_losses": int(np.mean(loss_runs)) if loss_runs else 0,
        }

    @staticmethod
    def trade_duration_stats(durations: pd.Series) -> dict[str, float]:
        """
        Calculate trade duration statistics.

        Args:
            durations: Series of trade durations (in minutes)

        Returns:
            Dictionary with duration stats
        """
        if durations.empty:
            return {
                "avg_duration_mins": 0,
                "median_duration_mins": 0,
                "min_duration_mins": 0,
                "max_duration_mins": 0,
            }

        return {
            "avg_duration_mins": float(durations.mean()),
            "median_duration_mins": float(durations.median()),
            "min_duration_mins": float(durations.min()),
            "max_duration_mins": float(durations.max()),
        }

    @staticmethod
    def calculate_all(
        pnl_series: pd.Series,
        wins: int | None = None,
        total: int | None = None,
    ) -> dict[str, float]:
        """
        Calculate all trading metrics from PnL series.

        Args:
            pnl_series: Series of PnL values
            wins: Number of wins (calculated if None)
            total: Total trades (calculated if None)

        Returns:
            Dictionary of all metrics
        """
        if pnl_series.empty:
            return {}

        # Calculate wins/losses if not provided
        if wins is None:
            wins = int((pnl_series > 0).sum())
        if total is None:
            total = len(pnl_series)

        losses = total - wins

        # Separate winning and losing trades
        winning_trades = pnl_series[pnl_series > 0]
        losing_trades = pnl_series[pnl_series <= 0]

        gross_profit = float(winning_trades.sum()) if not winning_trades.empty else 0
        gross_loss = float(abs(losing_trades.sum())) if not losing_trades.empty else 0

        avg_win = float(winning_trades.mean()) if not winning_trades.empty else 0
        avg_loss = float(abs(losing_trades.mean())) if not losing_trades.empty else 0

        win_rate = TradingMetrics.win_rate(wins, total)
        pf = TradingMetrics.profit_factor(gross_profit, gross_loss)
        exp = TradingMetrics.expectancy(win_rate, avg_win, avg_loss)
        rr = TradingMetrics.risk_reward_ratio(avg_win, avg_loss)
        kelly = TradingMetrics.kelly_criterion(win_rate, avg_win, avg_loss)

        max_dd, _, _ = TradingMetrics.max_drawdown(pnl_series)
        sharpe = TradingMetrics.sharpe_ratio(pnl_series)
        sortino = TradingMetrics.sortino_ratio(pnl_series)

        consecutive = TradingMetrics.consecutive_stats(pnl_series)

        return {
            "total_trades": total,
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate, 2),
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
            "net_profit": round(gross_profit - gross_loss, 2),
            "profit_factor": round(pf, 2) if pf != float("inf") else 999.99,
            "expectancy": round(exp, 4),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "risk_reward_ratio": round(rr, 2) if rr != float("inf") else 999.99,
            "kelly_criterion": round(kelly, 4),
            "max_drawdown_pct": round(max_dd, 2),
            "sharpe_ratio": round(sharpe, 2),
            "sortino_ratio": round(sortino, 2) if sortino != float("inf") else 999.99,
            **consecutive,
        }
