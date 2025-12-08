"""
Optimizer module for Analize.

Parameter optimization using:
- Grid search
- Random search
- Walk-forward validation
- Simulation with realistic execution model
- Explainability for suggestions
"""

from analize.optimizer.grid_search import GridSearchOptimizer
from analize.optimizer.simulator import TradingSimulator
from analize.optimizer.walk_forward import WalkForwardValidator
from analize.optimizer.execution_model import RealisticExecutionModel, ExchangeRules, ExecutionResult
from analize.optimizer.explainability import SuggestionExplainer, SuggestionExplanation, FeatureAttribution

__all__ = [
    "GridSearchOptimizer",
    "TradingSimulator",
    "WalkForwardValidator",
    "RealisticExecutionModel",
    "ExchangeRules",
    "ExecutionResult",
    "SuggestionExplainer",
    "SuggestionExplanation",
    "FeatureAttribution",
]
