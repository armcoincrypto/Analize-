"""
Outcome labeler for signal records.

Computes outcomes at various time horizons and calculates MFE/MAE.
"""

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from analize.config import get_settings
from analize.models.signals import ExitReason, OutcomeLabels, SignalRecord


class OutcomeLabeler:
    """Labels signal records with outcomes."""

    # Default time windows in minutes
    DEFAULT_WINDOWS = [1, 5, 15, 60, 240]

    def __init__(
        self,
        windows: list[int] | None = None,
        max_horizon_minutes: int = 1440,  # 24 hours
    ):
        """
        Initialize labeler.

        Args:
            windows: Time windows in minutes for outcome calculation
            max_horizon_minutes: Maximum time horizon for MFE/MAE calculation
        """
        settings = get_settings()
        self.windows = windows or settings.analysis.outcome_windows
        self.max_horizon = max_horizon_minutes

    def label_signal(
        self,
        signal: SignalRecord,
        future_prices: pd.DataFrame,
        tp_pct: float | None = None,
        sl_pct: float | None = None,
    ) -> SignalRecord:
        """
        Label a single signal with outcomes.

        Args:
            signal: Signal record to label
            future_prices: DataFrame with timestamp, high, low, close columns
                          for the period after the signal
            tp_pct: Take profit percentage (uses signal's expected_tp_pct if None)
            sl_pct: Stop loss percentage (uses signal's expected_sl_pct if None)

        Returns:
            Signal record with outcome labels
        """
        data = signal.model_dump()

        # Use signal's TP/SL if not provided
        tp_pct = tp_pct or signal.expected_tp_pct
        sl_pct = sl_pct or signal.expected_sl_pct

        entry_price = signal.exec_price or signal.price_close
        entry_time = signal.timestamp_utc

        if future_prices.empty or entry_price is None or entry_price == 0:
            return SignalRecord(**data)

        # Ensure timestamp column is datetime and set as index
        prices = future_prices.copy()
        if "timestamp" in prices.columns:
            prices["timestamp"] = pd.to_datetime(prices["timestamp"])
            prices = prices.set_index("timestamp")

        # Filter to future prices only
        prices = prices[prices.index > entry_time]

        if prices.empty:
            return SignalRecord(**data)

        # Calculate price movements
        prices["pnl_pct"] = (prices["close"] - entry_price) / entry_price * 100
        prices["high_pnl_pct"] = (prices["high"] - entry_price) / entry_price * 100
        prices["low_pnl_pct"] = (prices["low"] - entry_price) / entry_price * 100
        prices["minutes_elapsed"] = (
            (prices.index - entry_time).total_seconds() / 60
        ).astype(int)

        # Calculate outcomes for each time window
        for window in self.windows:
            window_mask = prices["minutes_elapsed"] <= window
            window_prices = prices[window_mask]

            if not window_prices.empty:
                # PnL at window end
                data[f"pnl_{window}m"] = window_prices["pnl_pct"].iloc[-1]

                # TP hit within window
                if tp_pct:
                    tp_hit = (window_prices["high_pnl_pct"] >= tp_pct).any()
                    data[f"hit_tp_{window}m"] = tp_hit

                # SL hit within window
                if sl_pct:
                    sl_hit = (window_prices["low_pnl_pct"] <= -sl_pct).any()
                    data[f"hit_sl_{window}m"] = sl_hit

        # Calculate MFE (Maximum Favorable Excursion)
        max_horizon_mask = prices["minutes_elapsed"] <= self.max_horizon
        horizon_prices = prices[max_horizon_mask]

        if not horizon_prices.empty:
            # For long positions, MFE is max high
            mfe_pct = horizon_prices["high_pnl_pct"].max()
            data["mfe_pct"] = mfe_pct

            # Time to MFE
            mfe_idx = horizon_prices["high_pnl_pct"].idxmax()
            mfe_time = (mfe_idx - entry_time).total_seconds() / 60
            data["mfe_time_mins"] = int(mfe_time)

            # MAE (Maximum Adverse Excursion)
            mae_pct = horizon_prices["low_pnl_pct"].min()
            data["mae_pct"] = mae_pct

            # Time to MAE
            mae_idx = horizon_prices["low_pnl_pct"].idxmin()
            mae_time = (mae_idx - entry_time).total_seconds() / 60
            data["mae_time_mins"] = int(mae_time)

        # Calculate time to TP/SL
        if tp_pct and not horizon_prices.empty:
            tp_hits = horizon_prices[horizon_prices["high_pnl_pct"] >= tp_pct]
            if not tp_hits.empty:
                first_tp_time = tp_hits.index[0]
                data["time_to_tp_mins"] = int(
                    (first_tp_time - entry_time).total_seconds() / 60
                )

        if sl_pct and not horizon_prices.empty:
            sl_hits = horizon_prices[horizon_prices["low_pnl_pct"] <= -sl_pct]
            if not sl_hits.empty:
                first_sl_time = sl_hits.index[0]
                data["time_to_sl_mins"] = int(
                    (first_sl_time - entry_time).total_seconds() / 60
                )

        # Determine exit reason and final PnL
        exit_reason = self._determine_exit_reason(
            signal, horizon_prices, tp_pct, sl_pct
        )
        if exit_reason:
            data["exit_reason"] = exit_reason

            # Calculate final PnL based on exit reason
            if exit_reason == ExitReason.TP and tp_pct:
                data["pnl_pct"] = tp_pct
            elif exit_reason == ExitReason.SL and sl_pct:
                data["pnl_pct"] = -sl_pct
            elif not horizon_prices.empty:
                # Use last price in horizon
                data["pnl_pct"] = horizon_prices["pnl_pct"].iloc[-1]
                data["exit_price"] = horizon_prices["close"].iloc[-1]
                data["exit_time"] = horizon_prices.index[-1]

        return SignalRecord(**data)

    def _determine_exit_reason(
        self,
        signal: SignalRecord,
        prices: pd.DataFrame,
        tp_pct: float | None,
        sl_pct: float | None,
    ) -> ExitReason | None:
        """Determine the exit reason for a signal."""
        if prices.empty:
            return ExitReason.TIMEOUT

        # Check for manual exit first (from signal data)
        if signal.exit_reason:
            return signal.exit_reason

        time_to_tp = None
        time_to_sl = None

        # Check TP hit
        if tp_pct:
            tp_hits = prices[prices["high_pnl_pct"] >= tp_pct]
            if not tp_hits.empty:
                time_to_tp = tp_hits.index[0]

        # Check SL hit
        if sl_pct:
            sl_hits = prices[prices["low_pnl_pct"] <= -sl_pct]
            if not sl_hits.empty:
                time_to_sl = sl_hits.index[0]

        # Determine which hit first
        if time_to_tp and time_to_sl:
            return ExitReason.TP if time_to_tp <= time_to_sl else ExitReason.SL
        elif time_to_tp:
            return ExitReason.TP
        elif time_to_sl:
            return ExitReason.SL
        else:
            return ExitReason.TIMEOUT

    def label_batch(
        self,
        signals: list[SignalRecord],
        price_data: pd.DataFrame,
        symbol_column: str = "symbol",
    ) -> list[SignalRecord]:
        """
        Label a batch of signals.

        Args:
            signals: List of signal records
            price_data: DataFrame with all price data (must include symbol column)
            symbol_column: Column name for symbol

        Returns:
            List of labeled signal records
        """
        labeled = []

        # Group price data by symbol
        if symbol_column in price_data.columns:
            price_groups = {
                symbol: group for symbol, group in price_data.groupby(symbol_column)
            }
        else:
            price_groups = {"_all": price_data}

        for signal in signals:
            # Get price data for this symbol
            symbol_prices = price_groups.get(signal.symbol, price_groups.get("_all"))

            if symbol_prices is None or symbol_prices.empty:
                labeled.append(signal)
                continue

            # Filter to prices after signal time
            future_prices = symbol_prices[symbol_prices.index > signal.timestamp_utc]

            labeled_signal = self.label_signal(signal, future_prices)
            labeled.append(labeled_signal)

        return labeled

    def calculate_outcome_stats(
        self,
        signals: list[SignalRecord],
    ) -> dict[str, Any]:
        """
        Calculate aggregate outcome statistics.

        Args:
            signals: List of labeled signal records

        Returns:
            Dictionary of outcome statistics
        """
        if not signals:
            return {}

        # Convert to DataFrame for easier analysis
        records = [s.model_dump() for s in signals]
        df = pd.DataFrame(records)

        stats: dict[str, Any] = {
            "total_signals": len(signals),
            "windows": {},
            "mfe_mae": {},
            "exit_reasons": {},
        }

        # Per-window stats
        for window in self.windows:
            pnl_col = f"pnl_{window}m"
            tp_col = f"hit_tp_{window}m"
            sl_col = f"hit_sl_{window}m"

            window_stats = {}

            if pnl_col in df.columns:
                valid_pnl = df[pnl_col].dropna()
                if len(valid_pnl) > 0:
                    window_stats["avg_pnl"] = float(valid_pnl.mean())
                    window_stats["median_pnl"] = float(valid_pnl.median())
                    window_stats["std_pnl"] = float(valid_pnl.std())
                    window_stats["positive_pct"] = float((valid_pnl > 0).mean() * 100)

            if tp_col in df.columns:
                tp_hits = df[tp_col].dropna()
                if len(tp_hits) > 0:
                    window_stats["tp_hit_rate"] = float(tp_hits.mean() * 100)

            if sl_col in df.columns:
                sl_hits = df[sl_col].dropna()
                if len(sl_hits) > 0:
                    window_stats["sl_hit_rate"] = float(sl_hits.mean() * 100)

            stats["windows"][f"{window}m"] = window_stats

        # MFE/MAE stats
        if "mfe_pct" in df.columns:
            mfe = df["mfe_pct"].dropna()
            if len(mfe) > 0:
                stats["mfe_mae"]["avg_mfe"] = float(mfe.mean())
                stats["mfe_mae"]["median_mfe"] = float(mfe.median())
                stats["mfe_mae"]["max_mfe"] = float(mfe.max())

        if "mae_pct" in df.columns:
            mae = df["mae_pct"].dropna()
            if len(mae) > 0:
                stats["mfe_mae"]["avg_mae"] = float(mae.mean())
                stats["mfe_mae"]["median_mae"] = float(mae.median())
                stats["mfe_mae"]["min_mae"] = float(mae.min())

        # Exit reason distribution
        if "exit_reason" in df.columns:
            exit_counts = df["exit_reason"].value_counts()
            stats["exit_reasons"] = {
                str(k): int(v) for k, v in exit_counts.items() if k is not None
            }

        return stats

    def find_optimal_tp_sl(
        self,
        signals: list[SignalRecord],
        tp_range: tuple[float, float] = (0.5, 5.0),
        sl_range: tuple[float, float] = (0.5, 3.0),
        step: float = 0.1,
        metric: str = "profit_factor",
    ) -> dict[str, Any]:
        """
        Find optimal TP/SL levels based on historical outcomes.

        Args:
            signals: Labeled signal records with MFE/MAE
            tp_range: Range of TP percentages to test
            sl_range: Range of SL percentages to test
            step: Step size for testing
            metric: Optimization metric (profit_factor, win_rate, expectancy)

        Returns:
            Dictionary with optimal parameters and results
        """
        if not signals:
            return {}

        # Extract MFE/MAE data
        df = pd.DataFrame([s.model_dump() for s in signals])
        df = df.dropna(subset=["mfe_pct", "mae_pct"])

        if df.empty:
            return {}

        results = []

        tp_values = np.arange(tp_range[0], tp_range[1] + step, step)
        sl_values = np.arange(sl_range[0], sl_range[1] + step, step)

        for tp in tp_values:
            for sl in sl_values:
                # Determine outcome for each signal
                # TP hit if MFE >= TP before MAE hits SL
                # This is simplified - ideally we'd use time-series data

                # Approximate: if MFE >= TP, assume TP hit
                # if MAE <= -SL and MFE < TP, assume SL hit
                tp_wins = (df["mfe_pct"] >= tp).sum()
                sl_losses = ((df["mae_pct"] <= -sl) & (df["mfe_pct"] < tp)).sum()
                timeouts = len(df) - tp_wins - sl_losses

                # Calculate metrics
                total_trades = tp_wins + sl_losses + timeouts
                if total_trades == 0:
                    continue

                win_rate = tp_wins / total_trades
                gross_profit = tp_wins * tp
                gross_loss = abs(sl_losses * sl)
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

                # Average timeout PnL (use mean of remaining MFE/MAE)
                timeout_pnl = 0
                if timeouts > 0:
                    timeout_mask = (df["mfe_pct"] < tp) & (df["mae_pct"] > -sl)
                    timeout_df = df[timeout_mask]
                    if not timeout_df.empty:
                        # Assume exit at some middle value
                        timeout_pnl = timeout_df["mfe_pct"].mean() * 0.5

                net_pnl = gross_profit - gross_loss + (timeouts * timeout_pnl)
                expectancy = net_pnl / total_trades if total_trades > 0 else 0

                results.append({
                    "tp_pct": round(tp, 2),
                    "sl_pct": round(sl, 2),
                    "wins": tp_wins,
                    "losses": sl_losses,
                    "timeouts": timeouts,
                    "win_rate": round(win_rate * 100, 2),
                    "profit_factor": round(profit_factor, 2),
                    "expectancy": round(expectancy, 4),
                    "net_pnl": round(net_pnl, 2),
                })

        if not results:
            return {}

        # Sort by metric
        results_df = pd.DataFrame(results)

        if metric == "profit_factor":
            # Filter out infinite PF
            valid = results_df[results_df["profit_factor"] < 100]
            if valid.empty:
                valid = results_df
            best = valid.loc[valid["profit_factor"].idxmax()]
        elif metric == "win_rate":
            best = results_df.loc[results_df["win_rate"].idxmax()]
        else:  # expectancy
            best = results_df.loc[results_df["expectancy"].idxmax()]

        return {
            "optimal_tp": float(best["tp_pct"]),
            "optimal_sl": float(best["sl_pct"]),
            "best_result": best.to_dict(),
            "top_5": results_df.nlargest(5, metric).to_dict("records"),
            "all_results": results,
        }
