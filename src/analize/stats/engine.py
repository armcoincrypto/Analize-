"""
Statistics engine for comprehensive trading analysis.

Orchestrates metric calculations, filter analysis, and performance reports.
"""

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from analize.models.reports import (
    DailyReport,
    FilterStats,
    SymbolSummary,
    ThresholdSensitivity,
    TimeHeatmap,
    VolatilityRegimeStats,
)
from analize.models.signals import SignalRecord
from analize.stats.metrics import TradingMetrics


class StatsEngine:
    """Comprehensive statistics engine for trading analysis."""

    def __init__(self):
        self.metrics = TradingMetrics()

    def analyze_signals(
        self,
        signals: list[SignalRecord],
    ) -> dict[str, Any]:
        """
        Perform comprehensive analysis on a set of signals.

        Args:
            signals: List of labeled signal records

        Returns:
            Dictionary with analysis results
        """
        if not signals:
            return {"error": "No signals provided"}

        df = pd.DataFrame([s.model_dump() for s in signals])

        return {
            "summary": self._calculate_summary(df),
            "by_symbol": self._analyze_by_symbol(df),
            "by_filter": self._analyze_by_filter(df),
            "by_time": self._analyze_by_time(df),
            "by_volatility": self._analyze_by_volatility(df),
        }

    def _calculate_summary(self, df: pd.DataFrame) -> dict[str, Any]:
        """Calculate overall summary statistics."""
        if "pnl_pct" not in df.columns:
            return {}

        pnl = df["pnl_pct"].dropna()
        if pnl.empty:
            return {}

        return TradingMetrics.calculate_all(pnl)

    def _analyze_by_symbol(self, df: pd.DataFrame) -> dict[str, dict[str, Any]]:
        """Analyze performance by symbol."""
        results = {}

        if "symbol" not in df.columns or "pnl_pct" not in df.columns:
            return results

        for symbol, group in df.groupby("symbol"):
            pnl = group["pnl_pct"].dropna()
            if not pnl.empty:
                results[str(symbol)] = TradingMetrics.calculate_all(pnl)

        return results

    def _analyze_by_filter(self, df: pd.DataFrame) -> list[FilterStats]:
        """Analyze performance by filter combinations."""
        results = []

        if "filters_passed" not in df.columns or "pnl_pct" not in df.columns:
            return results

        # Analyze individual filters
        all_filters: set[str] = set()
        for filters in df["filters_passed"].dropna():
            if isinstance(filters, str):
                all_filters.update(filters.split(","))
            elif isinstance(filters, list):
                all_filters.update(filters)

        for filter_name in all_filters:
            if not filter_name:
                continue

            # Find signals that passed this filter
            mask = df["filters_passed"].apply(
                lambda x: filter_name in (x.split(",") if isinstance(x, str) else x or [])
            )
            filter_df = df[mask]

            if filter_df.empty:
                continue

            pnl = filter_df["pnl_pct"].dropna()
            if pnl.empty:
                continue

            metrics = TradingMetrics.calculate_all(pnl)

            # Calculate MFE/MAE if available
            avg_mfe = float(filter_df["mfe_pct"].mean()) if "mfe_pct" in filter_df.columns else 0
            avg_mae = float(filter_df["mae_pct"].mean()) if "mae_pct" in filter_df.columns else 0

            results.append(FilterStats(
                filter_name=filter_name,
                signals_count=len(filter_df),
                trades_count=int(metrics.get("total_trades", 0)),
                wins_count=int(metrics.get("wins", 0)),
                losses_count=int(metrics.get("losses", 0)),
                win_rate=metrics.get("win_rate", 0),
                profit_factor=metrics.get("profit_factor", 0),
                expectancy=metrics.get("expectancy", 0),
                avg_win_pct=metrics.get("avg_win", 0),
                avg_loss_pct=metrics.get("avg_loss", 0),
                avg_rr_ratio=metrics.get("risk_reward_ratio", 0),
                avg_mfe_pct=avg_mfe,
                avg_mae_pct=avg_mae,
            ))

        # Sort by profit factor
        results.sort(key=lambda x: x.profit_factor, reverse=True)

        return results

    def _analyze_by_time(self, df: pd.DataFrame) -> TimeHeatmap | None:
        """Analyze performance by time of day and day of week."""
        if "timestamp_utc" not in df.columns or "pnl_pct" not in df.columns:
            return None

        df = df.copy()
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
        df["hour"] = df["timestamp_utc"].dt.hour
        df["day_of_week"] = df["timestamp_utc"].dt.dayofweek

        hour_perf: dict[int, dict[str, float]] = {}
        day_perf: dict[int, dict[str, float]] = {}

        # Hour performance
        for hour, group in df.groupby("hour"):
            pnl = group["pnl_pct"].dropna()
            if not pnl.empty:
                wins = (pnl > 0).sum()
                hour_perf[int(hour)] = {
                    "win_rate": float(wins / len(pnl) * 100),
                    "profit_factor": float(
                        abs(pnl[pnl > 0].sum() / pnl[pnl <= 0].sum())
                        if pnl[pnl <= 0].sum() != 0
                        else 0
                    ),
                    "trade_count": len(pnl),
                    "avg_pnl": float(pnl.mean()),
                }

        # Day of week performance
        for day, group in df.groupby("day_of_week"):
            pnl = group["pnl_pct"].dropna()
            if not pnl.empty:
                wins = (pnl > 0).sum()
                day_perf[int(day)] = {
                    "win_rate": float(wins / len(pnl) * 100),
                    "profit_factor": float(
                        abs(pnl[pnl > 0].sum() / pnl[pnl <= 0].sum())
                        if pnl[pnl <= 0].sum() != 0
                        else 0
                    ),
                    "trade_count": len(pnl),
                    "avg_pnl": float(pnl.mean()),
                }

        return TimeHeatmap(
            hour_performance=hour_perf,
            day_performance=day_perf,
        )

    def _analyze_by_volatility(self, df: pd.DataFrame) -> list[VolatilityRegimeStats]:
        """Analyze performance by volatility regime."""
        results = []

        # Use ATR or BB width as volatility measure
        vol_col = None
        for col in ["atr_1h_pct", "atr_1h", "bb_width"]:
            if col in df.columns:
                vol_col = col
                break

        if vol_col is None or "pnl_pct" not in df.columns:
            return results

        # Define volatility regimes using percentiles
        vol = df[vol_col].dropna()
        if vol.empty:
            return results

        low_threshold = vol.quantile(0.33)
        high_threshold = vol.quantile(0.67)

        regimes = [
            ("LOW", (0, low_threshold)),
            ("MEDIUM", (low_threshold, high_threshold)),
            ("HIGH", (high_threshold, float("inf"))),
        ]

        for regime_name, (low, high) in regimes:
            mask = (df[vol_col] >= low) & (df[vol_col] < high)
            regime_df = df[mask]

            if regime_df.empty:
                continue

            pnl = regime_df["pnl_pct"].dropna()
            if pnl.empty:
                continue

            wins = (pnl > 0).sum()
            gross_profit = pnl[pnl > 0].sum() if (pnl > 0).any() else 0
            gross_loss = abs(pnl[pnl <= 0].sum()) if (pnl <= 0).any() else 0

            results.append(VolatilityRegimeStats(
                regime=regime_name,
                atr_range=(float(low), float(high)),
                signals_count=len(regime_df),
                win_rate=float(wins / len(pnl) * 100),
                profit_factor=float(gross_profit / gross_loss) if gross_loss > 0 else 0,
                avg_pnl_pct=float(pnl.mean()),
            ))

        return results

    def calculate_threshold_sensitivity(
        self,
        signals: list[SignalRecord],
        parameter: str,
        test_range: tuple[float, float],
        steps: int = 10,
    ) -> ThresholdSensitivity:
        """
        Calculate sensitivity of performance to threshold changes.

        Args:
            signals: Signal records
            parameter: Parameter name to test (e.g., "expected_tp_pct")
            test_range: Range of values to test
            steps: Number of steps in range

        Returns:
            ThresholdSensitivity with results
        """
        df = pd.DataFrame([s.model_dump() for s in signals])

        if parameter not in df.columns:
            return ThresholdSensitivity(
                parameter_name=parameter,
                current_value=0,
                test_values=[],
                results=[],
            )

        current_value = float(df[parameter].median())
        test_values = list(np.linspace(test_range[0], test_range[1], steps))
        results = []

        for test_val in test_values:
            # Simulate with new parameter value
            # This is simplified - actual simulation would re-calculate outcomes
            simulated_results = self._simulate_parameter_change(df, parameter, test_val)
            results.append({
                "value": test_val,
                **simulated_results,
            })

        return ThresholdSensitivity(
            parameter_name=parameter,
            current_value=current_value,
            test_values=test_values,
            results=results,
        )

    def _simulate_parameter_change(
        self,
        df: pd.DataFrame,
        parameter: str,
        new_value: float,
    ) -> dict[str, Any]:
        """Simulate the effect of changing a parameter."""
        # This is a simplified simulation
        # In practice, you'd need to re-calculate outcomes based on MFE/MAE

        if parameter == "expected_tp_pct" and "mfe_pct" in df.columns:
            # Estimate wins based on MFE reaching new TP
            wins = (df["mfe_pct"] >= new_value).sum()
            total = len(df)
            win_rate = wins / total * 100 if total > 0 else 0

            # Estimate PnL
            gross_profit = wins * new_value
            sl_pct = df["expected_sl_pct"].median() if "expected_sl_pct" in df.columns else 1.0
            losses = total - wins
            gross_loss = losses * sl_pct

            pf = gross_profit / gross_loss if gross_loss > 0 else 0

            return {
                "win_rate": round(win_rate, 2),
                "profit_factor": round(pf, 2),
                "trade_count": total,
                "delta_win_rate": 0,  # Would need baseline to calculate
                "delta_pf": 0,
            }

        return {
            "win_rate": 0,
            "profit_factor": 0,
            "trade_count": len(df),
            "delta_win_rate": 0,
            "delta_pf": 0,
        }

    def generate_symbol_summary(
        self,
        signals: list[SignalRecord],
        symbol: str,
        period_start: datetime,
        period_end: datetime,
    ) -> SymbolSummary:
        """Generate a complete summary for a symbol."""
        df = pd.DataFrame([s.model_dump() for s in signals])
        df = df[df["symbol"] == symbol]

        if df.empty:
            return SymbolSummary(
                symbol=symbol,
                period_start=period_start,
                period_end=period_end,
            )

        # Calculate metrics
        pnl = df["pnl_pct"].dropna() if "pnl_pct" in df.columns else pd.Series()
        metrics = TradingMetrics.calculate_all(pnl) if not pnl.empty else {}

        # Slippage stats
        slippage = df["slippage_pct"].dropna() if "slippage_pct" in df.columns else pd.Series()

        # Spread stats
        spread = df["spread_bps"].dropna() if "spread_bps" in df.columns else pd.Series()

        # Filter stats
        filter_stats = self._analyze_by_filter(df)

        # Time heatmap
        time_heatmap = self._analyze_by_time(df)

        # Volatility stats
        volatility_stats = self._analyze_by_volatility(df)

        return SymbolSummary(
            symbol=symbol,
            period_start=period_start,
            period_end=period_end,
            signals_count=len(df),
            trades_count=int(metrics.get("total_trades", len(df))),
            unique_entries=len(df["signal_id"].unique()) if "signal_id" in df.columns else len(df),
            win_rate=metrics.get("win_rate", 0),
            profit_factor=metrics.get("profit_factor", 0),
            expectancy=metrics.get("expectancy", 0),
            avg_rr_ratio=metrics.get("risk_reward_ratio", 0),
            total_pnl_usd=float(df["pnl_usd"].sum()) if "pnl_usd" in df.columns else 0,
            total_pnl_pct=float(pnl.sum()) if not pnl.empty else 0,
            avg_pnl_usd=float(df["pnl_usd"].mean()) if "pnl_usd" in df.columns else 0,
            avg_pnl_pct=float(pnl.mean()) if not pnl.empty else 0,
            avg_mfe_pct=float(df["mfe_pct"].mean()) if "mfe_pct" in df.columns else 0,
            avg_mae_pct=float(df["mae_pct"].mean()) if "mae_pct" in df.columns else 0,
            avg_slippage_pct=float(slippage.mean()) if not slippage.empty else 0,
            median_slippage_pct=float(slippage.median()) if not slippage.empty else 0,
            avg_spread_bps=float(spread.mean()) if not spread.empty else 0,
            max_drawdown_pct=metrics.get("max_drawdown_pct", 0),
            sharpe_ratio=metrics.get("sharpe_ratio"),
            filter_stats=filter_stats,
            time_heatmap=time_heatmap,
            volatility_stats=volatility_stats,
        )

    def generate_daily_report(
        self,
        signals: list[SignalRecord],
        report_date: datetime,
        data_hash: str | None = None,
        code_tag: str | None = None,
    ) -> DailyReport:
        """Generate a daily analysis report."""
        from datetime import date

        df = pd.DataFrame([s.model_dump() for s in signals])

        # Get unique symbols
        symbols = df["symbol"].unique().tolist() if "symbol" in df.columns else []

        # Generate per-symbol summaries
        symbol_summaries = []
        for symbol in symbols:
            symbol_signals = [s for s in signals if s.symbol == symbol]
            summary = self.generate_symbol_summary(
                symbol_signals,
                symbol,
                report_date,
                report_date,
            )
            symbol_summaries.append(summary)

        # Overall metrics
        pnl = df["pnl_pct"].dropna() if "pnl_pct" in df.columns else pd.Series()
        overall_metrics = TradingMetrics.calculate_all(pnl) if not pnl.empty else {}

        # Filter analysis
        filter_stats = self._analyze_by_filter(df)

        return DailyReport(
            report_date=report_date.date() if isinstance(report_date, datetime) else report_date,
            data_hash=data_hash,
            code_tag=code_tag,
            symbol_summaries=symbol_summaries,
            total_signals=len(signals),
            total_trades=int(overall_metrics.get("total_trades", 0)),
            overall_win_rate=overall_metrics.get("win_rate", 0),
            overall_profit_factor=overall_metrics.get("profit_factor", 0),
            overall_pnl_usd=float(df["pnl_usd"].sum()) if "pnl_usd" in df.columns else 0,
            top_filter_combos=filter_stats[:5],
            worst_filter_combos=filter_stats[-5:] if len(filter_stats) > 5 else [],
        )

    def compare_periods(
        self,
        signals_a: list[SignalRecord],
        signals_b: list[SignalRecord],
        period_a_name: str = "Period A",
        period_b_name: str = "Period B",
    ) -> dict[str, Any]:
        """Compare performance between two periods."""
        df_a = pd.DataFrame([s.model_dump() for s in signals_a])
        df_b = pd.DataFrame([s.model_dump() for s in signals_b])

        pnl_a = df_a["pnl_pct"].dropna() if "pnl_pct" in df_a.columns else pd.Series()
        pnl_b = df_b["pnl_pct"].dropna() if "pnl_pct" in df_b.columns else pd.Series()

        metrics_a = TradingMetrics.calculate_all(pnl_a) if not pnl_a.empty else {}
        metrics_b = TradingMetrics.calculate_all(pnl_b) if not pnl_b.empty else {}

        # Calculate deltas
        deltas = {}
        for key in set(metrics_a.keys()) | set(metrics_b.keys()):
            val_a = metrics_a.get(key, 0)
            val_b = metrics_b.get(key, 0)
            if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
                deltas[key] = round(val_b - val_a, 4)

        return {
            period_a_name: metrics_a,
            period_b_name: metrics_b,
            "deltas": deltas,
            "improvement": {
                "win_rate": deltas.get("win_rate", 0) > 0,
                "profit_factor": deltas.get("profit_factor", 0) > 0,
                "expectancy": deltas.get("expectancy", 0) > 0,
            },
        }
