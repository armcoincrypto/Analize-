"""
HFT Research Package
====================
Walk-forward optimization and strategy discovery tools.

Modules:
- metrics: Performance metrics (PnL, Sharpe, drawdown, etc.)
- walkforward: Walk-forward optimization engine
"""

from .metrics import (
    calculate_pnl_after_costs,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_metrics,
)

from .walkforward import (
    WalkForwardEngine,
    ParameterGrid,
    walk_forward_run,
)

__all__ = [
    # Metrics
    "calculate_pnl_after_costs",
    "calculate_max_drawdown",
    "calculate_sharpe_ratio",
    "calculate_metrics",
    # Walk-forward
    "WalkForwardEngine",
    "ParameterGrid",
    "walk_forward_run",
]
