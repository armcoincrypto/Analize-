"""
Walk-forward validation for robust parameter optimization.

Prevents overfitting by using rolling train/test windows.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from analize.config import get_settings
from analize.models.reports import OptimizationResult
from analize.models.signals import SignalRecord
from analize.optimizer.grid_search import GridSearchOptimizer, ParameterSpace


@dataclass
class WalkForwardFold:
    """A single fold in walk-forward validation."""

    fold_number: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime

    train_signals: int = 0
    test_signals: int = 0

    best_params: dict[str, Any] | None = None
    train_score: float = 0.0
    test_score: float = 0.0

    # Detailed metrics
    train_metrics: dict[str, Any] | None = None
    test_metrics: dict[str, Any] | None = None


@dataclass
class WalkForwardResult:
    """Results from walk-forward validation."""

    folds: list[WalkForwardFold]
    total_folds: int = 0

    # Aggregate metrics
    avg_train_score: float = 0.0
    avg_test_score: float = 0.0
    score_degradation: float = 0.0  # How much worse is test vs train

    # Stability metrics
    param_stability: dict[str, float] | None = None  # Variance of chosen params across folds
    score_variance: float = 0.0

    # Best overall parameters (most frequently chosen or best average)
    recommended_params: dict[str, Any] | None = None


class WalkForwardValidator:
    """Performs walk-forward validation."""

    def __init__(
        self,
        train_days: int = 30,
        test_days: int = 7,
        step_days: int | None = None,
    ):
        settings = get_settings()
        self.train_days = train_days or settings.analysis.walk_forward_train_days
        self.test_days = test_days or settings.analysis.walk_forward_test_days
        self.step_days = step_days or self.test_days  # Step by test period by default

        self.optimizer = GridSearchOptimizer()

    def create_folds(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> list[WalkForwardFold]:
        """Create walk-forward folds."""
        folds = []
        fold_num = 0

        current_train_start = start_date
        train_duration = timedelta(days=self.train_days)
        test_duration = timedelta(days=self.test_days)
        step = timedelta(days=self.step_days)

        while current_train_start + train_duration + test_duration <= end_date:
            train_end = current_train_start + train_duration
            test_start = train_end
            test_end = test_start + test_duration

            fold = WalkForwardFold(
                fold_number=fold_num,
                train_start=current_train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
            folds.append(fold)

            current_train_start += step
            fold_num += 1

        return folds

    def filter_signals_by_period(
        self,
        signals: list[SignalRecord],
        start: datetime,
        end: datetime,
    ) -> list[SignalRecord]:
        """Filter signals to a specific time period."""
        return [
            s for s in signals
            if start <= s.timestamp_utc < end
        ]

    def validate(
        self,
        signals: list[SignalRecord],
        parameter_spaces: list[ParameterSpace],
        price_data: pd.DataFrame | None = None,
    ) -> WalkForwardResult:
        """
        Run walk-forward validation.

        Args:
            signals: All signal records
            parameter_spaces: Parameter spaces to optimize
            price_data: Price data (optional, uses fast method if None)

        Returns:
            WalkForwardResult with all folds and analysis
        """
        if not signals:
            return WalkForwardResult(folds=[], total_folds=0)

        # Determine date range
        dates = [s.timestamp_utc for s in signals]
        start_date = min(dates)
        end_date = max(dates)

        # Create folds
        folds = self.create_folds(start_date, end_date)

        if not folds:
            return WalkForwardResult(folds=[], total_folds=0)

        # Process each fold
        for fold in folds:
            # Get train and test signals
            train_signals = self.filter_signals_by_period(
                signals, fold.train_start, fold.train_end
            )
            test_signals = self.filter_signals_by_period(
                signals, fold.test_start, fold.test_end
            )

            fold.train_signals = len(train_signals)
            fold.test_signals = len(test_signals)

            if fold.train_signals < 10:
                continue

            # Optimize on training data
            if price_data is not None:
                opt_result = self.optimizer.optimize(
                    train_signals, price_data, parameter_spaces
                )
            else:
                opt_result = self.optimizer.optimize_fast(
                    train_signals, parameter_spaces
                )

            if not opt_result.best_parameters:
                continue

            fold.best_params = opt_result.best_parameters
            fold.train_score = opt_result.best_score
            fold.train_metrics = opt_result.top_results[0] if opt_result.top_results else None

            # Test on out-of-sample data
            if test_signals:
                if price_data is not None:
                    test_result = self.optimizer.simulator.run_simulation(
                        test_signals, price_data, fold.best_params
                    )
                    fold.test_score = test_result.profit_factor
                    fold.test_metrics = {
                        "win_rate": test_result.win_rate,
                        "profit_factor": test_result.profit_factor,
                        "total_trades": test_result.total_trades,
                    }
                else:
                    test_metrics = self.optimizer.simulator.calculate_metrics_from_mfe_mae(
                        test_signals,
                        fold.best_params.get("tp_pct", 2.0),
                        fold.best_params.get("sl_pct", 1.0),
                    )
                    fold.test_score = test_metrics.get("profit_factor", 0)
                    fold.test_metrics = test_metrics

        # Calculate aggregate metrics
        valid_folds = [f for f in folds if f.best_params is not None]

        if not valid_folds:
            return WalkForwardResult(folds=folds, total_folds=len(folds))

        avg_train = sum(f.train_score for f in valid_folds) / len(valid_folds)
        avg_test = sum(f.test_score for f in valid_folds) / len(valid_folds)
        degradation = (avg_train - avg_test) / avg_train * 100 if avg_train > 0 else 0

        # Calculate parameter stability
        param_values: dict[str, list] = {}
        for fold in valid_folds:
            if fold.best_params:
                for k, v in fold.best_params.items():
                    if k not in param_values:
                        param_values[k] = []
                    param_values[k].append(v)

        param_stability = {}
        for k, values in param_values.items():
            if len(values) > 1 and all(isinstance(v, (int, float)) for v in values):
                import numpy as np
                param_stability[k] = float(np.std(values))

        # Determine recommended parameters (most common or average)
        recommended = {}
        for k, values in param_values.items():
            if all(isinstance(v, (int, float)) for v in values):
                import numpy as np
                recommended[k] = float(np.median(values))
            else:
                # Use mode for categorical
                from collections import Counter
                recommended[k] = Counter(values).most_common(1)[0][0]

        # Score variance
        import numpy as np
        test_scores = [f.test_score for f in valid_folds]
        score_variance = float(np.var(test_scores)) if test_scores else 0

        return WalkForwardResult(
            folds=folds,
            total_folds=len(folds),
            avg_train_score=round(avg_train, 4),
            avg_test_score=round(avg_test, 4),
            score_degradation=round(degradation, 2),
            param_stability=param_stability,
            score_variance=round(score_variance, 4),
            recommended_params=recommended,
        )

    def calculate_robustness_score(self, result: WalkForwardResult) -> float:
        """
        Calculate a robustness score for the optimization.

        Higher score = more robust/reliable results.
        """
        if not result.folds or result.total_folds == 0:
            return 0.0

        # Factors:
        # 1. Low degradation (test vs train)
        # 2. Low score variance
        # 3. High average test score
        # 4. Parameter stability

        degradation_score = max(0, 100 - abs(result.score_degradation)) / 100
        variance_score = max(0, 1 - result.score_variance / 10)  # Normalize
        performance_score = min(result.avg_test_score / 3, 1)  # PF of 3 = max score

        # Parameter stability (lower variance = more stable)
        if result.param_stability:
            import numpy as np
            stability_scores = [
                max(0, 1 - v / 10) for v in result.param_stability.values()
            ]
            stability_score = np.mean(stability_scores)
        else:
            stability_score = 0.5

        # Weighted average
        robustness = (
            0.3 * degradation_score +
            0.2 * variance_score +
            0.3 * performance_score +
            0.2 * stability_score
        )

        return round(robustness * 100, 1)
