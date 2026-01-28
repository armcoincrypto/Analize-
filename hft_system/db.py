"""
Shared SQLite Database Utilities
================================
Provides consistent database connection settings across all components.

Key features:
- WAL mode for concurrent read/write access
- Busy timeout to handle lock contention
- Retry logic for transient errors

Usage:
    from hft_system.db import open_sqlite, execute_with_retry

    # Open connection with best practices
    conn = open_sqlite("hft_trades.db")

    # Execute with automatic retry on lock
    execute_with_retry(conn, "INSERT INTO ...", params)
"""

import logging
import sqlite3
import time
from contextlib import contextmanager
from typing import Optional, Any, Tuple, List

logger = logging.getLogger(__name__)

# Default settings
DEFAULT_BUSY_TIMEOUT_MS = 10000  # 10 seconds
DEFAULT_RETRY_ATTEMPTS = 5
DEFAULT_RETRY_BASE_MS = 50  # Start with 50ms, exponential backoff


def open_sqlite(
    path: str,
    busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS,
    check_same_thread: bool = False,
    isolation_level: str = None
) -> sqlite3.Connection:
    """
    Open SQLite connection with optimal settings for concurrent access.

    Settings applied:
    - WAL journal mode (allows concurrent reads during writes)
    - NORMAL synchronous (good balance of safety and speed)
    - Busy timeout (BOTH connect timeout AND pragma for full coverage)
    - Foreign keys enabled

    Args:
        path: Path to SQLite database file
        busy_timeout_ms: How long to wait on lock (default 10s)
        check_same_thread: If False, allow connection use across threads
        isolation_level: Transaction isolation level (None = autocommit)

    Returns:
        sqlite3.Connection with optimal settings
    """
    conn = sqlite3.connect(
        path,
        check_same_thread=check_same_thread,
        timeout=busy_timeout_ms / 1000.0,  # timeout param sets busy_timeout internally
        isolation_level=isolation_level  # None = autocommit mode
    )

    # Apply pragmas for concurrent access
    # Note: We set busy_timeout via PRAGMA too for explicit control
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA synchronous=NORMAL;")
    cursor.execute(f"PRAGMA busy_timeout={busy_timeout_ms};")
    cursor.execute("PRAGMA foreign_keys=ON;")

    # Verify settings were applied
    result = cursor.execute("PRAGMA journal_mode;").fetchone()
    if result and result[0].lower() != 'wal':
        logger.warning(f"Could not set WAL mode, got: {result[0]}")

    bt_result = cursor.execute("PRAGMA busy_timeout;").fetchone()
    if bt_result and int(bt_result[0]) != busy_timeout_ms:
        logger.warning(f"busy_timeout mismatch: expected {busy_timeout_ms}, got {bt_result[0]}")

    cursor.close()
    return conn


def execute_with_retry(
    conn: sqlite3.Connection,
    sql: str,
    params: Tuple = (),
    max_attempts: int = DEFAULT_RETRY_ATTEMPTS,
    base_delay_ms: int = DEFAULT_RETRY_BASE_MS,
    log_retries: bool = True
) -> Optional[sqlite3.Cursor]:
    """
    Execute SQL with exponential backoff retry on lock errors.

    Retries on:
    - database is locked
    - database is busy

    Args:
        conn: SQLite connection
        sql: SQL statement to execute
        params: Parameters for the SQL statement
        max_attempts: Maximum retry attempts
        base_delay_ms: Base delay in milliseconds (doubles each retry)
        log_retries: If True, log each retry attempt

    Returns:
        Cursor if successful, None if all retries failed
    """
    last_error = None
    delay_ms = base_delay_ms
    total_wait_ms = 0

    for attempt in range(1, max_attempts + 1):
        try:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            conn.commit()
            return cursor
        except sqlite3.OperationalError as e:
            error_str = str(e).lower()
            if "locked" in error_str or "busy" in error_str:
                last_error = e
                if attempt < max_attempts:
                    total_wait_ms += delay_ms
                    if log_retries:
                        logger.info(
                            f"SQLite lock retry: attempt {attempt}/{max_attempts}, "
                            f"waiting {delay_ms}ms (total wait: {total_wait_ms}ms)"
                        )
                    time.sleep(delay_ms / 1000.0)
                    delay_ms = min(delay_ms * 2, 2000)  # Cap at 2 seconds
                continue
            else:
                raise  # Re-raise non-lock errors immediately
        except Exception:
            raise

    # All retries exhausted
    logger.warning(
        f"SQLite operation failed after {max_attempts} attempts "
        f"(total wait: {total_wait_ms}ms): {last_error}"
    )
    return None


def insert_with_retry(
    conn: sqlite3.Connection,
    table: str,
    data: dict,
    max_attempts: int = DEFAULT_RETRY_ATTEMPTS
) -> bool:
    """
    Insert a row with retry logic.

    Args:
        conn: SQLite connection
        table: Table name
        data: Dict of column -> value
        max_attempts: Maximum retry attempts

    Returns:
        True if successful, False if all retries failed
    """
    columns = ", ".join(data.keys())
    placeholders = ", ".join(["?" for _ in data])
    sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
    params = tuple(data.values())

    result = execute_with_retry(conn, sql, params, max_attempts)
    return result is not None


def execute_immediate(
    conn: sqlite3.Connection,
    sql: str,
    params: Tuple = (),
    max_attempts: int = DEFAULT_RETRY_ATTEMPTS,
    base_delay_ms: int = DEFAULT_RETRY_BASE_MS
) -> Tuple[bool, str]:
    """
    Execute SQL within a BEGIN IMMEDIATE transaction with retry.

    BEGIN IMMEDIATE acquires a RESERVED lock immediately, failing fast
    if another writer holds the lock (instead of waiting on each statement).

    Args:
        conn: SQLite connection
        sql: SQL statement to execute
        params: Parameters for the SQL statement
        max_attempts: Maximum retry attempts
        base_delay_ms: Base delay in milliseconds

    Returns:
        Tuple of (success: bool, message: str)
    """
    last_error = None
    delay_ms = base_delay_ms
    total_wait_ms = 0

    for attempt in range(1, max_attempts + 1):
        try:
            cursor = conn.cursor()
            # BEGIN IMMEDIATE acquires write lock immediately
            cursor.execute("BEGIN IMMEDIATE;")
            try:
                cursor.execute(sql, params)
                cursor.execute("COMMIT;")
                return (True, f"Success on attempt {attempt}")
            except Exception as inner_e:
                cursor.execute("ROLLBACK;")
                raise inner_e
        except sqlite3.OperationalError as e:
            error_str = str(e).lower()
            if "locked" in error_str or "busy" in error_str:
                last_error = e
                if attempt < max_attempts:
                    total_wait_ms += delay_ms
                    logger.info(
                        f"BEGIN IMMEDIATE lock retry: attempt {attempt}/{max_attempts}, "
                        f"waiting {delay_ms}ms (total wait: {total_wait_ms}ms)"
                    )
                    time.sleep(delay_ms / 1000.0)
                    delay_ms = min(delay_ms * 2, 2000)
                continue
            else:
                return (False, f"SQL error: {e}")
        except Exception as e:
            return (False, f"Unexpected error: {e}")

    # All retries exhausted - another writer is still active
    return (False, f"Writer still active after {max_attempts} attempts ({total_wait_ms}ms total wait): {last_error}")


@contextmanager
def short_lived_connection(path: str, busy_timeout_ms: int = DEFAULT_BUSY_TIMEOUT_MS):
    """
    Context manager for short-lived database operations.

    Opens connection, yields it, then closes - ideal for telemetry writes
    that shouldn't hold connections open.

    Usage:
        with short_lived_connection("hft_trades.db") as conn:
            execute_with_retry(conn, "INSERT INTO ...", params)
    """
    conn = None
    try:
        conn = open_sqlite(path, busy_timeout_ms)
        yield conn
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass


class SafeTelemetryWriter:
    """
    Thread-safe telemetry writer with short-lived connections.

    Each write operation opens a fresh connection, executes, commits,
    and closes - preventing long-held locks.

    Usage:
        writer = SafeTelemetryWriter("hft_trades.db")
        writer.insert("maker_order_telemetry", {"order_id": "123", ...})
    """

    def __init__(self, db_path: str, max_retries: int = DEFAULT_RETRY_ATTEMPTS):
        self.db_path = db_path
        self.max_retries = max_retries

    def insert(self, table: str, data: dict) -> bool:
        """
        Insert a row using a short-lived connection.

        Returns True if successful, False otherwise.
        """
        with short_lived_connection(self.db_path) as conn:
            return insert_with_retry(conn, table, data, self.max_retries)

    def execute(self, sql: str, params: Tuple = ()) -> bool:
        """
        Execute SQL using a short-lived connection.

        Returns True if successful, False otherwise.
        """
        with short_lived_connection(self.db_path) as conn:
            result = execute_with_retry(conn, sql, params, self.max_retries)
            return result is not None

    def query(self, sql: str, params: Tuple = ()) -> Optional[List[Tuple]]:
        """
        Query and return all rows using a short-lived connection.

        Returns list of rows if successful, None otherwise.
        """
        try:
            with short_lived_connection(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(sql, params)
                return cursor.fetchall()
        except Exception as e:
            logger.warning(f"Query failed: {e}")
            return None
