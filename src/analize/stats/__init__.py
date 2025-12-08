"""
Statistics engine module for Analize.

Computes trading metrics and analysis:
- Win rate, profit factor, expectancy
- Per-filter and combination statistics
- Time-based performance analysis
- Volatility regime analysis
"""

from analize.stats.engine import StatsEngine
from analize.stats.metrics import TradingMetrics

__all__ = ["StatsEngine", "TradingMetrics"]
