"""
Grid search optimizer for parameter tuning.

Systematically searches parameter combinations to find optimal settings.
"""

import itertools
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable
from uuid import uuid4

import numpy as np
import pandas as pd

from analize.config import get_settings
from analize.models.reports import OptimizationObjective, OptimizationResult, ParameterSuggestion
from analize.models.signals import SignalRecord
from analize.optimizer.simulator import SimulationResult, TradingSimulator


@dataclass
class ParameterSpace:
    """Defines a parameter search space."""

    name: str
    values: list[Any]
    param_type: str = "float"  # "float", "int", "categorical"

    @classmethod
    def from_range(
        cls,
        name: str,
        start: float,
        end: float,
        step: float,
        param_type: str = "float",
    ) -> "ParameterSpace":
        """Create parameter space from range."""
        values = list(np.arange(start, end + step, step))
        if param_type == "int":
            values = [int(v) for v in values]
        return cls(name=name, values=values, param_type=param_type)


class GridSearchOptimizer:
    """Grid search optimizer for trading parameters."""

    def __init__(
        self,
        objective: OptimizationObjective = OptimizationObjective.PROFIT_FACTOR,
        min_trades: int = 30,
    ):
        settings = get_settings()
        self.objective = objective
        self.min_trades = min_trades or settings.analysis.optimization_min_trades
        self.simulator = TradingSimulator()

    def _get_objective_value(self, result: SimulationResult) -> float:
        """Extract objective value from simulation result."""
        if self.objective == OptimizationObjective.PROFIT_FACTOR:
            return result.profit_factor
        elif self.objective == OptimizationObjective.WIN_RATE:
            return result.win_rate
        elif self.objective == OptimizationObjective.SHARPE:
            return result.sharpe_ratio
        elif self.objective == OptimizationObjective.MAX_DRAWDOWN:
            # For drawdown, lower (more negative) is worse, so negate
            return -result.max_drawdown_pct
        elif self.objective == OptimizationObjective.EXPECTANCY:
            return result.expectancy
        elif self.objective == OptimizationObjective.TRADES_COUNT:
            return result.total_trades
        return result.profit_factor

    def generate_combinations(
        self,
        parameter_spaces: list[ParameterSpace],
    ) -> list[dict[str, Any]]:
        """Generate all parameter combinations."""
        names = [p.name for p in parameter_spaces]
        value_lists = [p.values for p in parameter_spaces]

        combinations = []
        for values in itertools.product(*value_lists):
            combo = dict(zip(names, values))
            combinations.append(combo)

        return combinations

    def optimize(
        self,
        signals: list[SignalRecord],
        price_data: pd.DataFrame,
        parameter_spaces: list[ParameterSpace],
        top_n: int = 5,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> OptimizationResult:
        """
        Run grid search optimization.

        Args:
            signals: Signal records with MFE/MAE for simulation
            price_data: Price data for full simulation
            parameter_spaces: List of parameter spaces to search
            top_n: Number of top results to return
            progress_callback: Optional callback(current, total) for progress

        Returns:
            OptimizationResult with best parameters and analysis
        """
        combinations = self.generate_combinations(parameter_spaces)
        total = len(combinations)

        results = []
        started_at = datetime.utcnow()

        for i, params in enumerate(combinations):
            if progress_callback:
                progress_callback(i + 1, total)

            # Run simulation
            sim_result = self.simulator.run_simulation(signals, price_data, params)

            # Filter by minimum trades
            if sim_result.total_trades < self.min_trades:
                continue

            # Get objective value
            score = self._get_objective_value(sim_result)

            results.append({
                "parameters": params,
                "score": score,
                "result": sim_result,
            })

        # Sort by objective
        results.sort(key=lambda x: x["score"], reverse=True)

        # Build result
        best = results[0] if results else None

        return OptimizationResult(
            optimization_id=uuid4(),
            started_at=started_at,
            completed_at=datetime.utcnow(),
            objective=self.objective,
            symbols=list(set(s.symbol for s in signals)),
            date_range_start=min(s.timestamp_utc for s in signals).date(),
            date_range_end=max(s.timestamp_utc for s in signals).date(),
            parameters_searched={p.name: p.values for p in parameter_spaces},
            total_combinations=total,
            combinations_evaluated=len(results),
            best_parameters=best["parameters"] if best else {},
            best_score=best["score"] if best else 0,
            top_results=[
                {
                    "parameters": r["parameters"],
                    "score": round(r["score"], 4),
                    "win_rate": r["result"].win_rate,
                    "profit_factor": r["result"].profit_factor,
                    "total_trades": r["result"].total_trades,
                    "max_drawdown": r["result"].max_drawdown_pct,
                    "equity_curve": r["result"].equity_curve,
                }
                for r in results[:top_n]
            ],
        )

    def optimize_fast(
        self,
        signals: list[SignalRecord],
        parameter_spaces: list[ParameterSpace],
        top_n: int = 5,
    ) -> OptimizationResult:
        """
        Fast optimization using pre-computed MFE/MAE (no price data needed).

        Args:
            signals: Signal records with MFE/MAE
            parameter_spaces: Parameter spaces to search
            top_n: Number of top results

        Returns:
            OptimizationResult
        """
        combinations = self.generate_combinations(parameter_spaces)
        total = len(combinations)

        results = []
        started_at = datetime.utcnow()

        for params in combinations:
            tp_pct = params.get("tp_pct", 2.0)
            sl_pct = params.get("sl_pct", 1.0)

            metrics = self.simulator.calculate_metrics_from_mfe_mae(
                signals, tp_pct, sl_pct
            )

            if not metrics or metrics.get("total_trades", 0) < self.min_trades:
                continue

            # Calculate score based on objective
            if self.objective == OptimizationObjective.PROFIT_FACTOR:
                score = metrics.get("profit_factor", 0)
            elif self.objective == OptimizationObjective.WIN_RATE:
                score = metrics.get("win_rate", 0)
            else:
                score = metrics.get("profit_factor", 0)

            results.append({
                "parameters": params,
                "score": score,
                "metrics": metrics,
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        best = results[0] if results else None

        return OptimizationResult(
            optimization_id=uuid4(),
            started_at=started_at,
            completed_at=datetime.utcnow(),
            objective=self.objective,
            symbols=list(set(s.symbol for s in signals)),
            date_range_start=min(s.timestamp_utc for s in signals).date(),
            date_range_end=max(s.timestamp_utc for s in signals).date(),
            parameters_searched={p.name: p.values for p in parameter_spaces},
            total_combinations=total,
            combinations_evaluated=len(results),
            best_parameters=best["parameters"] if best else {},
            best_score=best["score"] if best else 0,
            top_results=[
                {
                    "parameters": r["parameters"],
                    "score": round(r["score"], 4),
                    **r["metrics"],
                }
                for r in results[:top_n]
            ],
        )

    def generate_suggestions(
        self,
        optimization_result: OptimizationResult,
        current_params: dict[str, Any],
        top_n: int = 3,
    ) -> list[ParameterSuggestion]:
        """
        Generate parameter suggestions from optimization results.

        Args:
            optimization_result: Result from optimization run
            current_params: Current parameter values
            top_n: Number of suggestions to generate

        Returns:
            List of ParameterSuggestion objects
        """
        suggestions = []

        for i, result in enumerate(optimization_result.top_results[:top_n]):
            params = result["parameters"]

            # Find which parameters changed
            for param_name, new_value in params.items():
                current_value = current_params.get(param_name)

                if current_value is None or current_value == new_value:
                    continue

                # Calculate expected impact
                baseline = optimization_result.top_results[-1] if optimization_result.top_results else {}
                baseline_pf = baseline.get("profit_factor", 0)
                baseline_wr = baseline.get("win_rate", 0)
                baseline_trades = baseline.get("total_trades", 0)

                suggestion = ParameterSuggestion(
                    parameter_name=param_name,
                    current_value=current_value,
                    suggested_value=new_value,
                    change_description=f"Change {param_name} from {current_value} to {new_value}",
                    expected_pf_delta=result.get("profit_factor", 0) - baseline_pf,
                    expected_win_rate_delta=result.get("win_rate", 0) - baseline_wr,
                    expected_trade_count_delta=result.get("total_trades", 0) - baseline_trades,
                    rank=i + 1,
                    objective_score=result.get("score", 0),
                    sample_size=result.get("total_trades", 0),
                )

                # Create trade-off description
                pf_change = suggestion.expected_pf_delta
                trade_change_pct = (
                    suggestion.expected_trade_count_delta / baseline_trades * 100
                    if baseline_trades > 0 else 0
                )
                suggestion.tradeoff = (
                    f"{trade_change_pct:+.0f}% trades, {pf_change:+.2f} PF"
                )

                suggestions.append(suggestion)

        return suggestions


class RandomSearchOptimizer(GridSearchOptimizer):
    """Random search optimizer - samples randomly from parameter space."""

    def __init__(
        self,
        objective: OptimizationObjective = OptimizationObjective.PROFIT_FACTOR,
        min_trades: int = 30,
        n_samples: int = 100,
    ):
        super().__init__(objective, min_trades)
        self.n_samples = n_samples

    def generate_combinations(
        self,
        parameter_spaces: list[ParameterSpace],
    ) -> list[dict[str, Any]]:
        """Generate random parameter combinations."""
        combinations = []

        for _ in range(self.n_samples):
            combo = {}
            for space in parameter_spaces:
                value = np.random.choice(space.values)
                if space.param_type == "int":
                    value = int(value)
                combo[space.name] = value
            combinations.append(combo)

        return combinations
