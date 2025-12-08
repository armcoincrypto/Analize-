"""
Data ingestion module for Analize.

Handles:
- ScalperBot database ingestion
- Log file parsing
- Market data (candles, orderbook) ingestion
- CSV/Parquet/JSON file imports
"""

from analize.ingestion.db_ingest import ScalperBotDBIngestor
from analize.ingestion.log_parser import LogParser
from analize.ingestion.market_data import MarketDataIngestor
from analize.ingestion.file_ingest import FileIngestor

__all__ = [
    "ScalperBotDBIngestor",
    "LogParser",
    "MarketDataIngestor",
    "FileIngestor",
]
