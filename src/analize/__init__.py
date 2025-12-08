"""
Analize - Cloud AI Analyzer for ScalperBot

A comprehensive analysis platform for trading bot data that produces
ML-ready datasets, statistical reports, and parameter optimization suggestions.
"""

__version__ = "0.1.0"
__author__ = "Analize Team"

from analize.analyzer import CloudAIAnalyzer
from analize.config import Settings, get_settings

__all__ = ["CloudAIAnalyzer", "Settings", "get_settings", "__version__"]
