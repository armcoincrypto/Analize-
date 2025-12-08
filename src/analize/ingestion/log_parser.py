"""
Log file parser for ScalperBot logs.

Parses structured log lines to extract:
- Signals and their reasons
- Errors and warnings
- Notifications
- Performance metrics
"""

import gzip
import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Generator, TextIO

from analize.models.signals import FilterResult, PolicyMode


@dataclass
class LogEntry:
    """Parsed log entry."""

    timestamp: datetime
    level: str
    logger: str
    message: str
    data: dict[str, Any] = field(default_factory=dict)
    raw_line: str = ""


@dataclass
class SignalLogEntry(LogEntry):
    """Signal-specific log entry."""

    symbol: str = ""
    timeframe: str = "1m"
    signal_type: str = ""  # "entry", "exit", "filter_pass", "filter_fail"
    filters: list[FilterResult] = field(default_factory=list)
    policy_mode: PolicyMode | None = None
    price: float | None = None
    reason: str = ""


@dataclass
class ErrorLogEntry(LogEntry):
    """Error log entry."""

    error_type: str = ""
    error_message: str = ""
    traceback: str = ""
    context: dict[str, Any] = field(default_factory=dict)


class LogParser:
    """Parses ScalperBot log files."""

    # Common log patterns
    PATTERNS = {
        # Standard log format: 2024-01-01 12:00:00.123 | INFO | module:function:line - message
        "standard": re.compile(
            r"^(?P<timestamp>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s*\|\s*"
            r"(?P<level>\w+)\s*\|\s*"
            r"(?P<logger>[\w.:]+)\s*[-:]\s*"
            r"(?P<message>.*)$"
        ),
        # JSON log format
        "json": re.compile(r"^\{.*\}$"),
        # Simple format: [2024-01-01 12:00:00] INFO: message
        "simple": re.compile(
            r"^\[(?P<timestamp>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?)\]\s*"
            r"(?P<level>\w+):\s*"
            r"(?P<message>.*)$"
        ),
        # Signal patterns
        "signal_entry": re.compile(
            r"(?:signal|entry|long|short)\s+(?:for|on)\s+(?P<symbol>\w+)",
            re.IGNORECASE,
        ),
        "signal_exit": re.compile(
            r"(?:exit|close|tp|sl|stop)\s+(?:for|on|hit)\s*(?P<symbol>\w+)?",
            re.IGNORECASE,
        ),
        "filter_result": re.compile(
            r"filter\s+(?P<name>\w+)\s*[:=]\s*(?P<result>pass|fail|true|false)",
            re.IGNORECASE,
        ),
        "price": re.compile(r"price[:\s=]+(?P<price>[\d.]+)"),
    }

    def __init__(self, log_dir: Path | str | None = None):
        self.log_dir = Path(log_dir) if log_dir else None
        self._stats = {
            "files_processed": 0,
            "lines_processed": 0,
            "entries_parsed": 0,
            "errors_found": 0,
            "signals_found": 0,
        }

    def parse_line(self, line: str) -> LogEntry | None:
        """Parse a single log line."""
        line = line.strip()
        if not line:
            return None

        # Try JSON format first
        if self.PATTERNS["json"].match(line):
            try:
                data = json.loads(line)
                return LogEntry(
                    timestamp=datetime.fromisoformat(data.get("timestamp", data.get("ts", ""))),
                    level=data.get("level", data.get("lvl", "INFO")).upper(),
                    logger=data.get("logger", data.get("name", "")),
                    message=data.get("message", data.get("msg", "")),
                    data=data,
                    raw_line=line,
                )
            except (json.JSONDecodeError, ValueError):
                pass

        # Try standard format
        match = self.PATTERNS["standard"].match(line)
        if match:
            groups = match.groupdict()
            try:
                timestamp = datetime.fromisoformat(groups["timestamp"].replace(" ", "T"))
            except ValueError:
                timestamp = datetime.utcnow()

            return LogEntry(
                timestamp=timestamp,
                level=groups["level"].upper(),
                logger=groups["logger"],
                message=groups["message"],
                raw_line=line,
            )

        # Try simple format
        match = self.PATTERNS["simple"].match(line)
        if match:
            groups = match.groupdict()
            try:
                timestamp = datetime.fromisoformat(groups["timestamp"].replace(" ", "T"))
            except ValueError:
                timestamp = datetime.utcnow()

            return LogEntry(
                timestamp=timestamp,
                level=groups["level"].upper(),
                logger="",
                message=groups["message"],
                raw_line=line,
            )

        # Unknown format - still return as entry
        return LogEntry(
            timestamp=datetime.utcnow(),
            level="UNKNOWN",
            logger="",
            message=line,
            raw_line=line,
        )

    def classify_entry(self, entry: LogEntry) -> LogEntry:
        """Classify and enhance a log entry based on content."""
        message_lower = entry.message.lower()

        # Check for error
        if entry.level in ("ERROR", "CRITICAL", "FATAL") or "error" in message_lower:
            error_entry = ErrorLogEntry(
                timestamp=entry.timestamp,
                level=entry.level,
                logger=entry.logger,
                message=entry.message,
                data=entry.data,
                raw_line=entry.raw_line,
                error_message=entry.message,
            )

            # Try to extract error type
            if "exception" in message_lower:
                error_entry.error_type = "exception"
            elif "timeout" in message_lower:
                error_entry.error_type = "timeout"
            elif "connection" in message_lower:
                error_entry.error_type = "connection"

            self._stats["errors_found"] += 1
            return error_entry

        # Check for signal
        is_signal = False
        signal_entry = SignalLogEntry(
            timestamp=entry.timestamp,
            level=entry.level,
            logger=entry.logger,
            message=entry.message,
            data=entry.data,
            raw_line=entry.raw_line,
        )

        # Check for entry signal
        match = self.PATTERNS["signal_entry"].search(entry.message)
        if match:
            is_signal = True
            signal_entry.signal_type = "entry"
            signal_entry.symbol = match.group("symbol").upper()

        # Check for exit signal
        match = self.PATTERNS["signal_exit"].search(entry.message)
        if match:
            is_signal = True
            signal_entry.signal_type = "exit"
            if match.group("symbol"):
                signal_entry.symbol = match.group("symbol").upper()

        # Check for filter results
        for match in self.PATTERNS["filter_result"].finditer(entry.message):
            is_signal = True
            signal_entry.filters.append(
                FilterResult(
                    filter_name=match.group("name"),
                    passed=match.group("result").lower() in ("pass", "true"),
                )
            )
            signal_entry.signal_type = "filter"

        # Extract price if present
        match = self.PATTERNS["price"].search(entry.message)
        if match:
            try:
                signal_entry.price = float(match.group("price"))
            except ValueError:
                pass

        if is_signal:
            self._stats["signals_found"] += 1
            return signal_entry

        return entry

    def parse_file(
        self,
        file_path: Path | str,
        classify: bool = True,
    ) -> Generator[LogEntry, None, None]:
        """
        Parse a log file and yield entries.

        Args:
            file_path: Path to log file
            classify: Whether to classify entries

        Yields:
            LogEntry objects
        """
        file_path = Path(file_path)

        # Handle gzipped files
        if file_path.suffix == ".gz":
            opener = gzip.open
            mode = "rt"
        else:
            opener = open
            mode = "r"

        with opener(file_path, mode, encoding="utf-8", errors="replace") as f:
            self._stats["files_processed"] += 1

            for line in f:
                self._stats["lines_processed"] += 1
                entry = self.parse_line(line)

                if entry:
                    self._stats["entries_parsed"] += 1
                    if classify:
                        entry = self.classify_entry(entry)
                    yield entry

    def parse_directory(
        self,
        dir_path: Path | str | None = None,
        pattern: str = "*.log*",
        classify: bool = True,
        recursive: bool = True,
    ) -> Generator[LogEntry, None, None]:
        """
        Parse all log files in a directory.

        Args:
            dir_path: Directory path (uses self.log_dir if None)
            pattern: Glob pattern for log files
            classify: Whether to classify entries
            recursive: Whether to search recursively

        Yields:
            LogEntry objects
        """
        dir_path = Path(dir_path) if dir_path else self.log_dir
        if not dir_path or not dir_path.exists():
            return

        glob_method = dir_path.rglob if recursive else dir_path.glob

        for file_path in sorted(glob_method(pattern)):
            if file_path.is_file():
                yield from self.parse_file(file_path, classify)

    def extract_signals(
        self,
        entries: Generator[LogEntry, None, None] | list[LogEntry],
    ) -> list[SignalLogEntry]:
        """Extract only signal-related entries."""
        signals = []

        for entry in entries:
            if isinstance(entry, SignalLogEntry):
                signals.append(entry)

        return signals

    def extract_errors(
        self,
        entries: Generator[LogEntry, None, None] | list[LogEntry],
    ) -> list[ErrorLogEntry]:
        """Extract only error entries."""
        errors = []

        for entry in entries:
            if isinstance(entry, ErrorLogEntry):
                errors.append(entry)

        return errors

    def group_by_symbol(
        self,
        entries: list[SignalLogEntry],
    ) -> dict[str, list[SignalLogEntry]]:
        """Group signal entries by symbol."""
        grouped: dict[str, list[SignalLogEntry]] = {}

        for entry in entries:
            if entry.symbol:
                if entry.symbol not in grouped:
                    grouped[entry.symbol] = []
                grouped[entry.symbol].append(entry)

        return grouped

    def get_time_range(
        self,
        entries: list[LogEntry],
    ) -> tuple[datetime | None, datetime | None]:
        """Get time range of entries."""
        if not entries:
            return None, None

        timestamps = [e.timestamp for e in entries if e.timestamp]
        if not timestamps:
            return None, None

        return min(timestamps), max(timestamps)

    def compute_hash(self, file_path: Path | str) -> str:
        """Compute hash of a log file."""
        file_path = Path(file_path)
        hasher = hashlib.sha256()

        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)

        return hasher.hexdigest()[:16]

    def get_stats(self) -> dict[str, int]:
        """Get parsing statistics."""
        return self._stats.copy()

    def reset_stats(self) -> None:
        """Reset parsing statistics."""
        self._stats = {
            "files_processed": 0,
            "lines_processed": 0,
            "entries_parsed": 0,
            "errors_found": 0,
            "signals_found": 0,
        }

    def to_dataframe(self, entries: list[LogEntry]) -> "pd.DataFrame":
        """Convert entries to a pandas DataFrame."""
        import pandas as pd

        records = []
        for entry in entries:
            record = {
                "timestamp": entry.timestamp,
                "level": entry.level,
                "logger": entry.logger,
                "message": entry.message,
            }

            if isinstance(entry, SignalLogEntry):
                record["entry_type"] = "signal"
                record["symbol"] = entry.symbol
                record["signal_type"] = entry.signal_type
                record["price"] = entry.price
                record["filters_count"] = len(entry.filters)
            elif isinstance(entry, ErrorLogEntry):
                record["entry_type"] = "error"
                record["error_type"] = entry.error_type
                record["error_message"] = entry.error_message
            else:
                record["entry_type"] = "log"

            records.append(record)

        return pd.DataFrame(records)
