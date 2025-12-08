"""
Statistics engine module for Analize.

Computes trading metrics and analysis:
- Win rate, profit factor, expectancy
- Per-filter and combination statistics
- Time-based performance analysis
- Volatility regime analysis
- Statistical significance testing
"""

from analize.stats.engine import StatsEngine
from analize.stats.metrics import TradingMetrics
from analize.stats.significance import SignificanceTester, SignificanceResult, BootstrapResult

__all__ = [
    "StatsEngine",
    "TradingMetrics",
    "SignificanceTester",
    "SignificanceResult",
    "BootstrapResult",
]
