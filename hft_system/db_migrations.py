"""
Database Schema Migrations & Guardrails
=======================================
Ensures the database has all required columns before analytics run.
Prevents silent failures from missing columns.

Usage:
    from hft_system.db_migrations import ensure_schema
    ensure_schema("hft_trades.db", auto_migrate=True)
"""

import sqlite3
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


# ============================================================
# REQUIRED SCHEMA DEFINITION
# ============================================================
# Each table -> list of (column_name, column_type, default_value)
# These are the columns that MUST exist for analytics to work.
# ============================================================

REQUIRED_SCHEMA: Dict[str, List[Tuple[str, str, str]]] = {
    "trades": [
        # Core trade fields
        ("trade_id", "TEXT", None),
        ("symbol", "TEXT", None),
        ("side", "TEXT", None),
        ("entry_price", "REAL", None),
        ("exit_price", "REAL", None),
        ("quantity", "REAL", None),
        ("entry_time", "INTEGER", None),
        ("exit_time", "INTEGER", None),
        ("pnl", "REAL", None),
        ("pnl_pct", "REAL", None),
        ("hold_time_sec", "REAL", None),
        ("exit_reason", "TEXT", None),
        ("status", "TEXT", "'open'"),
        # MFE/MAE - critical for edge analysis
        ("mfe", "REAL", None),
        ("mae", "REAL", None),
        # Cost model fields
        ("entry_fee_pct", "REAL", "0"),
        ("exit_fee_pct", "REAL", "0"),
        ("fees_paid_pct", "REAL", "0"),  # Combined: entry_fee + exit_fee
        ("spread_cost_pct", "REAL", "0"),
        ("slippage_pct", "REAL", "0"),
        ("total_costs_pct", "REAL", "0"),
        ("pnl_after_costs_pct", "REAL", "0"),
        # Pocket tracking
        ("pocket_id", "TEXT", None),
        # Execution mode
        ("execution_mode", "TEXT", "'taker'"),
        # Cause policy (FULL/PROBE)
        ("is_probe_cause", "INTEGER", "0"),
        ("cause_class", "TEXT", "'FULL'"),
        # Regime at entry/exit
        ("regime_at_entry", "TEXT", None),
        ("regime_at_exit", "TEXT", None),
    ],
    "trade_causality": [
        ("trade_id", "TEXT", None),
        ("symbol", "TEXT", None),
        ("timestamp", "INTEGER", None),
        ("primary_cause", "TEXT", None),
        ("cause_strength", "REAL", None),
        ("ob_imbalance_ratio", "REAL", None),
        ("cvd_delta_1s", "REAL", None),
        ("cvd_delta_3s", "REAL", None),
        ("cvd_delta_5s", "REAL", None),
    ],
    "trade_edge": [
        ("trade_id", "TEXT", None),
        ("symbol", "TEXT", None),
        ("side", "TEXT", None),
        ("seconds_to_max_favorable", "REAL", None),
        ("max_favorable_pct", "REAL", None),
        ("edge_duration_sec", "REAL", None),
        ("real_edge_flag", "INTEGER", "0"),
        ("final_pnl_pct", "REAL", None),
        ("exit_reason", "TEXT", None),
    ],
    "market_regime": [
        ("timestamp", "INTEGER", None),
        ("symbol", "TEXT", None),
        ("regime", "TEXT", None),
        ("confidence", "REAL", None),
    ],
    "blocked_signals": [
        ("timestamp", "INTEGER", None),
        ("symbol", "TEXT", None),
        ("signal_type", "TEXT", None),
        ("entry_price", "REAL", None),
        ("block_reason", "TEXT", None),
        ("block_details", "TEXT", None),
        ("regime", "TEXT", None),
        ("would_have_pnl", "REAL", None),
        # Market conditions at block time
        ("spread_pct", "REAL", None),
        ("spread_change_1s", "REAL", None),
        ("delta_variance", "REAL", None),
        ("ob_volume_instability", "REAL", None),
        ("liquidity_depth", "REAL", None),
        # Structured blocker tracking (for 80/20 analysis)
        ("gate_name", "TEXT", None),  # e.g., "WINNER_GATE", "NO_TRADE_ZONE", "CONFIDENCE"
        ("gate_param", "TEXT", None),  # e.g., "pocket_a_min_imbalance", "max_spread_pct"
        ("gate_threshold", "REAL", None),  # threshold value that blocked
        ("actual_value", "REAL", None),  # actual value that failed check
    ],
    "position_sizing": [
        ("timestamp", "INTEGER", None),
        ("trade_id", "TEXT", None),
        ("symbol", "TEXT", None),
        ("confidence_score", "REAL", None),
        ("confidence_tier", "TEXT", None),
        ("size_multiplier", "REAL", None),
        ("regime", "TEXT", None),
        ("primary_cause", "TEXT", None),
        # Multiplier tracking (for single-probe audit)
        ("base_size", "REAL", None),
        ("applied_multipliers", "TEXT", None),  # JSON list: ["cause_probe=0.10", "confidence=1.5"]
        ("final_size", "REAL", None),
        ("is_probe_trade", "INTEGER", "0"),
    ],
    "daily_stats": [
        ("date", "TEXT", None),
        ("total_trades", "INTEGER", None),
        ("wins", "INTEGER", None),
        ("losses", "INTEGER", None),
        ("gross_pnl", "REAL", None),
        ("net_pnl", "REAL", None),
    ],
}


def get_table_columns(conn: sqlite3.Connection, table_name: str) -> List[str]:
    """Get list of column names for a table."""
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    return [row[1] for row in cursor.fetchall()]


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    """Check if a table exists."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,)
    )
    return cursor.fetchone() is not None


def check_schema(db_path: str) -> List[str]:
    """
    Check database schema against REQUIRED_SCHEMA.

    Returns:
        List of human-readable problem descriptions.
        Empty list means schema is valid.
    """
    problems = []

    try:
        conn = sqlite3.connect(db_path)

        for table_name, required_columns in REQUIRED_SCHEMA.items():
            if not table_exists(conn, table_name):
                problems.append(f"Table '{table_name}' does not exist")
                continue

            existing_columns = get_table_columns(conn, table_name)

            for col_name, col_type, default in required_columns:
                if col_name not in existing_columns:
                    problems.append(
                        f"Table '{table_name}' missing column '{col_name}' ({col_type})"
                    )

        conn.close()

    except Exception as e:
        problems.append(f"Database error: {e}")

    return problems


def apply_migrations(db_path: str) -> List[str]:
    """
    Apply migrations to add missing columns.
    Does NOT remove or modify existing columns.

    Returns:
        List of migrations applied.
    """
    applied = []

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        for table_name, required_columns in REQUIRED_SCHEMA.items():
            if not table_exists(conn, table_name):
                logger.warning(f"Table '{table_name}' does not exist - cannot migrate (will be created on first use)")
                continue

            existing_columns = get_table_columns(conn, table_name)

            for col_name, col_type, default in required_columns:
                if col_name not in existing_columns:
                    # Build ALTER TABLE statement
                    if default is not None:
                        sql = f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_type} DEFAULT {default}"
                    else:
                        sql = f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_type}"

                    try:
                        cursor.execute(sql)
                        applied.append(f"Added {table_name}.{col_name} ({col_type})")
                        logger.info(f"Migration: {applied[-1]}")
                    except sqlite3.OperationalError as e:
                        if "duplicate column" in str(e).lower():
                            pass  # Column already exists
                        else:
                            logger.error(f"Failed to add {table_name}.{col_name}: {e}")

        conn.commit()
        conn.close()

    except Exception as e:
        logger.error(f"Migration error: {e}")

    return applied


def ensure_schema(db_path: str, auto_migrate: bool = True) -> None:
    """
    Ensure database schema is valid. Fails fast if not.

    Args:
        db_path: Path to SQLite database
        auto_migrate: If True, attempt to add missing columns

    Raises:
        RuntimeError: If schema is invalid and cannot be fixed
    """
    logger.info(f"Checking database schema: {db_path}")

    # First check
    problems = check_schema(db_path)

    if not problems:
        logger.info("Database schema OK - all required columns present")
        return

    logger.warning(f"Found {len(problems)} schema issues")

    if auto_migrate:
        logger.info("Auto-migrate enabled - attempting to fix...")
        applied = apply_migrations(db_path)

        if applied:
            logger.info(f"Applied {len(applied)} migrations")

        # Re-check after migration
        problems = check_schema(db_path)

        if not problems:
            logger.info("Database schema OK after migration")
            return

    # Still have problems - fail fast
    error_msg = "DATABASE SCHEMA INVALID - Analytics will be unreliable!\n"
    error_msg += "Missing columns:\n"
    for p in problems:
        error_msg += f"  - {p}\n"
    error_msg += "\nFix: Run migrations manually or set AUTO_MIGRATE_DB=True"

    logger.error(error_msg)
    raise RuntimeError(error_msg)


def print_schema_report(db_path: str) -> None:
    """Print a human-readable schema report."""
    print()
    print("=" * 70)
    print(" DATABASE SCHEMA CHECK")
    print("=" * 70)
    print(f" Database: {db_path}")
    print()

    try:
        conn = sqlite3.connect(db_path)

        for table_name, required_columns in REQUIRED_SCHEMA.items():
            print(f"\n--- {table_name.upper()} ---")

            if not table_exists(conn, table_name):
                print("  [MISSING] Table does not exist!")
                continue

            existing_columns = get_table_columns(conn, table_name)

            for col_name, col_type, default in required_columns:
                if col_name in existing_columns:
                    print(f"  [OK] {col_name} ({col_type})")
                else:
                    print(f"  [MISSING] {col_name} ({col_type})")

        conn.close()

    except Exception as e:
        print(f"  ERROR: {e}")

    print()
    print("=" * 70)

    # Summary
    problems = check_schema(db_path)
    if problems:
        print(f" RESULT: {len(problems)} issues found")
        print(" Run with auto_migrate=True to fix")
    else:
        print(" RESULT: Schema OK")
    print("=" * 70)
