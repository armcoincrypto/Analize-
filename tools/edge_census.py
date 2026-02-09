#!/usr/bin/env python3
"""
Edge Census Tool - Expectancy Bucket Analysis
==============================================
Computes expectancy buckets across (symbol, regime, cause_class, confidence_tier)
using AFTER COSTS pnl for realistic edge measurement.

Schema Support:
- trades table: trade_id, symbol, pnl_after_costs_pct, regime_at_entry, cause_class, etc.
- position_sizing table (optional): trade_id, confidence_tier, regime, primary_cause, etc.
- trade_edge table (optional): trade_id, real_edge_flag, final_pnl_pct, etc.

Usage:
    # Inspect database schema
    python tools/edge_census.py --db hft_trades.db --inspect

    # Basic expectancy analysis
    python tools/edge_census.py --db hft_trades.db --min-trades 30

    # Custom dimensions and metric
    python tools/edge_census.py --db hft_trades.db --dimensions symbol,regime,cause_class \\
        --metric pnl_after_costs_pct --min-trades 20

    # Output to CSV
    python tools/edge_census.py --db hft_trades.db --output edge_census.csv --format csv

    # JSON output
    python tools/edge_census.py --db hft_trades.db --output edge_census.json --format json
"""

import argparse
import csv
import json
import sqlite3
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Add parent dir for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# -----------------------------------------------------------------------------
# Schema Detection
# -----------------------------------------------------------------------------

@dataclass
class TableSchema:
    """Schema information for a database table."""
    name: str
    columns: Set[str] = field(default_factory=set)
    has_data: bool = False
    row_count: int = 0


@dataclass
class DbSchema:
    """Complete database schema information."""
    tables: Dict[str, TableSchema] = field(default_factory=dict)

    def has_table(self, name: str) -> bool:
        return name in self.tables and self.tables[name].has_data

    def has_column(self, table: str, column: str) -> bool:
        return table in self.tables and column in self.tables[table].columns

    def get_columns(self, table: str) -> Set[str]:
        return self.tables.get(table, TableSchema(table)).columns


def detect_schema(conn: sqlite3.Connection) -> DbSchema:
    """Detect the database schema and available columns."""
    cursor = conn.cursor()
    schema = DbSchema()

    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    table_names = [row[0] for row in cursor.fetchall()]

    for table_name in table_names:
        table_schema = TableSchema(name=table_name)

        # Get columns
        cursor.execute(f"PRAGMA table_info({table_name})")
        table_schema.columns = {row[1] for row in cursor.fetchall()}

        # Get row count
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            table_schema.row_count = cursor.fetchone()[0]
            table_schema.has_data = table_schema.row_count > 0
        except sqlite3.OperationalError:
            table_schema.row_count = 0
            table_schema.has_data = False

        schema.tables[table_name] = table_schema

    return schema


def print_schema_report(schema: DbSchema):
    """Print a human-readable schema report."""
    print("\n" + "=" * 70)
    print(" DATABASE SCHEMA INSPECTION")
    print("=" * 70)

    for table_name, table in sorted(schema.tables.items()):
        status = f"({table.row_count} rows)" if table.has_data else "(empty)"
        print(f"\n  {table_name} {status}")
        print("  " + "-" * 40)
        for col in sorted(table.columns):
            print(f"    - {col}")

    print("\n" + "=" * 70)

    # Print fast-path compatibility
    print("\n FAST-PATH COMPATIBILITY:")

    # Check trades table
    if schema.has_table("trades"):
        required_trades = {"trade_id", "pnl_after_costs_pct", "status"}
        optional_trades = {"symbol", "regime_at_entry", "cause_class", "signal_confidence",
                         "exit_reason", "mfe", "mae", "pnl_pct", "total_costs_pct"}
        trades_cols = schema.get_columns("trades")

        missing = required_trades - trades_cols
        available = optional_trades & trades_cols

        if missing:
            print(f"   trades: MISSING required columns: {missing}")
        else:
            print(f"   trades: OK - has {len(available)} optional analysis columns")
    else:
        print("   trades: NOT FOUND")

    # Check position_sizing table
    if schema.has_table("position_sizing"):
        ps_cols = schema.get_columns("position_sizing")
        useful = {"trade_id", "confidence_tier", "regime", "primary_cause", "confidence_score"}
        available = useful & ps_cols
        print(f"   position_sizing: OK - has {len(available)} useful columns: {available}")
    else:
        print("   position_sizing: NOT FOUND (confidence_tier will use fallback)")

    # Check trade_edge table
    if schema.has_table("trade_edge"):
        te_cols = schema.get_columns("trade_edge")
        useful = {"trade_id", "real_edge_flag", "final_pnl_pct", "edge_duration_sec"}
        available = useful & te_cols
        print(f"   trade_edge: OK - has {len(available)} useful columns: {available}")
    else:
        print("   trade_edge: NOT FOUND (edge analysis disabled)")

    print()


# -----------------------------------------------------------------------------
# Query Builder
# -----------------------------------------------------------------------------

@dataclass
class QueryPlan:
    """Plan for building the expectancy query."""
    base_table: str = "trades"
    joins: List[str] = field(default_factory=list)
    select_cols: List[str] = field(default_factory=list)
    group_cols: List[str] = field(default_factory=list)
    where_clauses: List[str] = field(default_factory=list)
    metric_col: str = "pnl_after_costs_pct"
    having_clause: str = ""
    order_by: str = "expectancy DESC"


def build_query_plan(
    schema: DbSchema,
    dimensions: List[str],
    metric: str,
    min_trades: int,
    days: Optional[int] = None
) -> QueryPlan:
    """
    Build an optimized query plan based on available schema.

    Handles automatic join detection for confidence_tier and other
    columns that may be in position_sizing or trade_edge tables.
    """
    plan = QueryPlan()
    plan.metric_col = metric

    trades_cols = schema.get_columns("trades")
    ps_cols = schema.get_columns("position_sizing") if schema.has_table("position_sizing") else set()
    te_cols = schema.get_columns("trade_edge") if schema.has_table("trade_edge") else set()

    # Base WHERE clause - only closed trades, exclude paper trades
    plan.where_clauses.append("t.status = 'closed'")
    plan.where_clauses.append("COALESCE(t.exit_reason, '') NOT IN ('force_paper_trade', 'FORCE_PAPER_TRADE')")

    # Optional time filter
    if days:
        plan.where_clauses.append(f"t.entry_time >= (strftime('%s', 'now') - {days * 86400}) * 1000")

    # Track if we need joins
    need_position_sizing = False
    need_trade_edge = False

    # Map dimension names to actual column expressions
    dimension_mapping = {}

    for dim in dimensions:
        dim_lower = dim.lower()

        if dim_lower == "symbol":
            if "symbol" in trades_cols:
                dimension_mapping[dim] = ("t.symbol", "symbol")
            else:
                dimension_mapping[dim] = ("'unknown'", "symbol")

        elif dim_lower == "regime":
            # Prefer trades.regime_at_entry, fallback to position_sizing.regime
            if "regime_at_entry" in trades_cols:
                dimension_mapping[dim] = ("COALESCE(t.regime_at_entry, 'unknown')", "regime")
            elif schema.has_table("position_sizing") and "regime" in ps_cols:
                need_position_sizing = True
                dimension_mapping[dim] = ("COALESCE(ps.regime, 'unknown')", "regime")
            else:
                dimension_mapping[dim] = ("'unknown'", "regime")

        elif dim_lower in ("cause", "cause_class"):
            # Prefer trades.cause_class, fallback to position_sizing.primary_cause
            if "cause_class" in trades_cols:
                dimension_mapping[dim] = ("COALESCE(t.cause_class, 'unknown')", "cause_class")
            elif schema.has_table("position_sizing") and "primary_cause" in ps_cols:
                need_position_sizing = True
                dimension_mapping[dim] = ("COALESCE(ps.primary_cause, 'unknown')", "cause_class")
            else:
                dimension_mapping[dim] = ("'unknown'", "cause_class")

        elif dim_lower == "confidence_tier":
            # From position_sizing table
            if schema.has_table("position_sizing") and "confidence_tier" in ps_cols:
                need_position_sizing = True
                dimension_mapping[dim] = ("COALESCE(ps.confidence_tier, 'unknown')", "confidence_tier")
            elif "signal_confidence" in trades_cols:
                # Fallback: bucket signal_confidence into tiers
                dimension_mapping[dim] = (
                    "CASE "
                    "WHEN t.signal_confidence >= 0.8 THEN 'high' "
                    "WHEN t.signal_confidence >= 0.5 THEN 'medium' "
                    "WHEN t.signal_confidence >= 0.2 THEN 'low' "
                    "ELSE 'unknown' END",
                    "confidence_tier"
                )
            else:
                dimension_mapping[dim] = ("'unknown'", "confidence_tier")

        elif dim_lower == "exit_reason":
            if "exit_reason" in trades_cols:
                dimension_mapping[dim] = ("COALESCE(t.exit_reason, 'unknown')", "exit_reason")
            else:
                dimension_mapping[dim] = ("'unknown'", "exit_reason")

        elif dim_lower == "execution_mode":
            if "execution_mode" in trades_cols:
                dimension_mapping[dim] = ("COALESCE(t.execution_mode, 'taker')", "execution_mode")
            else:
                dimension_mapping[dim] = ("'taker'", "execution_mode")

        elif dim_lower == "real_edge":
            # From trade_edge table
            if schema.has_table("trade_edge") and "real_edge_flag" in te_cols:
                need_trade_edge = True
                dimension_mapping[dim] = ("COALESCE(te.real_edge_flag, 0)", "real_edge")
            else:
                dimension_mapping[dim] = ("0", "real_edge")

        else:
            # Try to find the column directly
            if dim in trades_cols:
                dimension_mapping[dim] = (f"COALESCE(t.{dim}, 'unknown')", dim)
            elif dim in ps_cols and schema.has_table("position_sizing"):
                need_position_sizing = True
                dimension_mapping[dim] = (f"COALESCE(ps.{dim}, 'unknown')", dim)
            elif dim in te_cols and schema.has_table("trade_edge"):
                need_trade_edge = True
                dimension_mapping[dim] = (f"COALESCE(te.{dim}, 'unknown')", dim)
            else:
                dimension_mapping[dim] = ("'unknown'", dim)

    # Build JOIN clauses if needed
    if need_position_sizing and schema.has_table("position_sizing"):
        plan.joins.append("LEFT JOIN position_sizing ps ON t.trade_id = ps.trade_id")

    if need_trade_edge and schema.has_table("trade_edge"):
        plan.joins.append("LEFT JOIN trade_edge te ON t.trade_id = te.trade_id")

    # Build SELECT and GROUP BY
    for dim, (expr, alias) in dimension_mapping.items():
        plan.select_cols.append(f"{expr} AS {alias}")
        plan.group_cols.append(expr)

    # Validate metric column
    metric_expr = f"t.{metric}"
    if metric not in trades_cols:
        # Try position_sizing or trade_edge
        if metric in ps_cols:
            metric_expr = f"ps.{metric}"
            if not need_position_sizing:
                plan.joins.append("LEFT JOIN position_sizing ps ON t.trade_id = ps.trade_id")
        elif metric in te_cols:
            metric_expr = f"te.{metric}"
            if not need_trade_edge:
                plan.joins.append("LEFT JOIN trade_edge te ON t.trade_id = te.trade_id")
        else:
            # Fallback to pnl_after_costs_pct
            print(f"WARNING: metric '{metric}' not found, using pnl_after_costs_pct")
            metric_expr = "t.pnl_after_costs_pct"

    plan.metric_col = metric_expr

    # HAVING clause for min trades
    if min_trades > 0:
        plan.having_clause = f"HAVING COUNT(*) >= {min_trades}"

    return plan


def build_expectancy_sql(plan: QueryPlan) -> str:
    """Build the final SQL query from the plan."""
    # Aggregate expressions
    aggregates = [
        "COUNT(*) AS n_trades",
        f"ROUND(AVG({plan.metric_col}), 6) AS expectancy",
        f"ROUND(SUM({plan.metric_col}), 4) AS total_pnl",
        f"ROUND(100.0 * SUM(CASE WHEN {plan.metric_col} > 0 THEN 1 ELSE 0 END) / COUNT(*), 2) AS win_rate",
        f"ROUND(AVG(CASE WHEN {plan.metric_col} > 0 THEN {plan.metric_col} END), 6) AS avg_win",
        f"ROUND(AVG(CASE WHEN {plan.metric_col} <= 0 THEN {plan.metric_col} END), 6) AS avg_loss",
    ]

    # Add MFE/MAE if available in metric expression table
    if "t." in plan.metric_col or plan.metric_col.startswith("t."):
        aggregates.extend([
            "ROUND(AVG(t.mfe), 6) AS avg_mfe",
            "ROUND(AVG(t.mae), 6) AS avg_mae",
        ])

    # Build SELECT
    select_items = plan.select_cols + aggregates
    select_clause = ",\n        ".join(select_items)

    # Build FROM with JOINs
    from_clause = f"FROM trades t"
    if plan.joins:
        from_clause += "\n        " + "\n        ".join(plan.joins)

    # Build WHERE
    where_clause = "WHERE " + "\n        AND ".join(plan.where_clauses)

    # Build GROUP BY
    group_clause = "GROUP BY " + ", ".join(plan.group_cols)

    # Build final query
    sql = f"""
    SELECT
        {select_clause}
    {from_clause}
    {where_clause}
    {group_clause}
    {plan.having_clause}
    ORDER BY {plan.order_by}
    """

    return sql.strip()


# -----------------------------------------------------------------------------
# Expectancy Analysis
# -----------------------------------------------------------------------------

@dataclass
class ExpectancyBucket:
    """Results for one expectancy bucket."""
    dimensions: Dict[str, str]
    n_trades: int
    expectancy: float
    total_pnl: float
    win_rate: float
    avg_win: Optional[float]
    avg_loss: Optional[float]
    avg_mfe: Optional[float] = None
    avg_mae: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = dict(self.dimensions)
        result.update({
            "n_trades": self.n_trades,
            "expectancy": self.expectancy,
            "total_pnl": self.total_pnl,
            "win_rate": self.win_rate,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
        })
        if self.avg_mfe is not None:
            result["avg_mfe"] = self.avg_mfe
        if self.avg_mae is not None:
            result["avg_mae"] = self.avg_mae
        return result


def run_expectancy_analysis(
    conn: sqlite3.Connection,
    schema: DbSchema,
    dimensions: List[str],
    metric: str,
    min_trades: int,
    days: Optional[int] = None,
    verbose: bool = False
) -> List[ExpectancyBucket]:
    """
    Run expectancy bucket analysis on the database.

    Returns list of ExpectancyBucket sorted by expectancy descending.
    """
    # Build query plan
    plan = build_query_plan(schema, dimensions, metric, min_trades, days)

    # Generate SQL
    sql = build_expectancy_sql(plan)

    if verbose:
        print("\n--- Generated SQL ---")
        print(sql)
        print("--- End SQL ---\n")

    # Execute query
    cursor = conn.cursor()
    cursor.execute(sql)

    # Get column names from description
    col_names = [desc[0] for desc in cursor.description]

    # Parse results
    results = []
    for row in cursor.fetchall():
        row_dict = dict(zip(col_names, row))

        # Extract dimension values
        dim_names = [d.split(" AS ")[-1] if " AS " in d else d for d in
                    [plan.select_cols[i].split(" AS ")[-1] for i in range(len(dimensions))]]

        # Map actual column names to dimension values
        dim_values = {}
        for i, dim in enumerate(dimensions):
            alias = dim_names[i] if i < len(dim_names) else dim.lower()
            dim_values[alias] = str(row_dict.get(alias, "unknown"))

        bucket = ExpectancyBucket(
            dimensions=dim_values,
            n_trades=row_dict.get("n_trades", 0),
            expectancy=row_dict.get("expectancy", 0.0) or 0.0,
            total_pnl=row_dict.get("total_pnl", 0.0) or 0.0,
            win_rate=row_dict.get("win_rate", 0.0) or 0.0,
            avg_win=row_dict.get("avg_win"),
            avg_loss=row_dict.get("avg_loss"),
            avg_mfe=row_dict.get("avg_mfe"),
            avg_mae=row_dict.get("avg_mae"),
        )
        results.append(bucket)

    return results


# -----------------------------------------------------------------------------
# Output Formatters
# -----------------------------------------------------------------------------

def format_table(results: List[ExpectancyBucket], dimensions: List[str]) -> str:
    """Format results as a text table."""
    if not results:
        return "No data found matching criteria."

    lines = []

    # Header
    lines.append("=" * 120)
    lines.append(" EXPECTANCY CENSUS - Bucket Analysis")
    lines.append("=" * 120)
    lines.append("")

    # Build header row
    dim_headers = [d[:12].upper() for d in dimensions]
    header = f"{'RANK':<5} " + " ".join(f"{h:<12}" for h in dim_headers)
    header += f" {'TRADES':>7} {'EXPECT':>10} {'WIN%':>7} {'TOT_PNL':>10} {'AVG_WIN':>10} {'AVG_LOSS':>10}"

    if results[0].avg_mfe is not None:
        header += f" {'AVG_MFE':>10} {'AVG_MAE':>10}"

    lines.append(header)
    lines.append("-" * len(header))

    # Data rows
    for i, bucket in enumerate(results, 1):
        dim_values = [str(bucket.dimensions.get(d.lower(), bucket.dimensions.get(d.split("_")[0], "?")))[:12]
                     for d in dimensions]
        row = f"{i:<5} " + " ".join(f"{v:<12}" for v in dim_values)

        row += f" {bucket.n_trades:>7}"
        row += f" {bucket.expectancy:>+10.4f}%"
        row += f" {bucket.win_rate:>6.1f}%"
        row += f" {bucket.total_pnl:>+10.4f}%"

        avg_win = f"{bucket.avg_win:>+10.4f}%" if bucket.avg_win else f"{'N/A':>11}"
        avg_loss = f"{bucket.avg_loss:>+10.4f}%" if bucket.avg_loss else f"{'N/A':>11}"
        row += f" {avg_win} {avg_loss}"

        if bucket.avg_mfe is not None:
            avg_mfe = f"{bucket.avg_mfe:>+10.4f}%" if bucket.avg_mfe else f"{'N/A':>11}"
            avg_mae = f"{bucket.avg_mae:>+10.4f}%" if bucket.avg_mae else f"{'N/A':>11}"
            row += f" {avg_mfe} {avg_mae}"

        lines.append(row)

    lines.append("-" * len(header))

    # Summary
    total_trades = sum(b.n_trades for b in results)
    total_pnl = sum(b.total_pnl for b in results)
    weighted_expectancy = total_pnl / total_trades if total_trades else 0

    positive_buckets = sum(1 for b in results if b.expectancy > 0)
    negative_buckets = sum(1 for b in results if b.expectancy <= 0)

    lines.append("")
    lines.append(f"SUMMARY: {len(results)} buckets | {total_trades} total trades | "
                f"Weighted Expectancy: {weighted_expectancy:+.4f}%")
    lines.append(f"         Positive expectancy: {positive_buckets} buckets | "
                f"Negative expectancy: {negative_buckets} buckets")
    lines.append("")

    return "\n".join(lines)


def write_csv(results: List[ExpectancyBucket], output_path: str, dimensions: List[str]):
    """Write results to CSV file."""
    if not results:
        print("No data to write.")
        return

    with open(output_path, "w", newline="") as f:
        # Get all keys from first result
        first_dict = results[0].as_dict()
        fieldnames = list(first_dict.keys())

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for bucket in results:
            writer.writerow(bucket.as_dict())

    print(f"Wrote {len(results)} rows to {output_path}")


def write_json(results: List[ExpectancyBucket], output_path: str):
    """Write results to JSON file."""
    data = {
        "generated_at": datetime.now().isoformat(),
        "total_buckets": len(results),
        "total_trades": sum(b.n_trades for b in results),
        "buckets": [b.as_dict() for b in results]
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Wrote {len(results)} buckets to {output_path}")


# -----------------------------------------------------------------------------
# Database Connection
# -----------------------------------------------------------------------------

def open_db(path: str) -> sqlite3.Connection:
    """Open database with optimal settings."""
    try:
        # Try to use the project's db module
        from hft_system.db import open_sqlite
        return open_sqlite(path)
    except ImportError:
        # Fallback to basic connection with WAL mode
        conn = sqlite3.connect(path, timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        return conn


def find_db_path(db_arg: str) -> Path:
    """Find the database file."""
    db_path = Path(db_arg)
    if db_path.exists():
        return db_path

    # Try common locations
    for try_path in [
        Path.cwd() / "hft_trades.db",
        Path.home() / "Analize-" / "hft_trades.db",
        Path("/root/Analize-/hft_trades.db"),
        Path("/home/user/Analize-/hft_trades.db"),
    ]:
        if try_path.exists():
            return try_path

    return db_path  # Return original, let caller handle error


# -----------------------------------------------------------------------------
# Main CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Edge Census - Expectancy Bucket Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inspect database schema
  python tools/edge_census.py --inspect

  # Basic analysis with default dimensions
  python tools/edge_census.py --min-trades 30

  # Custom dimensions
  python tools/edge_census.py --dimensions symbol,regime,cause_class,confidence_tier

  # Output to CSV (new style)
  python tools/edge_census.py --format csv --output edge_census.csv

  # Output to CSV (legacy style)
  python tools/edge_census.py --out-csv edge_census.csv

  # Limit to recent data
  python tools/edge_census.py --days 7 --min-trades 10
        """
    )

    parser.add_argument(
        "--db",
        type=str,
        default="hft_trades.db",
        help="Path to SQLite database (default: hft_trades.db)"
    )

    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Inspect database schema and exit"
    )

    parser.add_argument(
        "--dimensions",
        type=str,
        default="symbol,regime,cause_class",
        help="Comma-separated list of dimensions to group by "
             "(default: symbol,regime,cause_class). "
             "Available: symbol, regime, cause_class, confidence_tier, exit_reason, execution_mode, real_edge"
    )

    parser.add_argument(
        "--metric",
        type=str,
        default="pnl_after_costs_pct",
        help="PnL metric column to use (default: pnl_after_costs_pct). "
             "Common values: pnl_after_costs_pct, pnl_pct, pnl"
    )

    parser.add_argument(
        "--min-trades",
        type=int,
        default=1,
        help="Minimum trades per bucket (default: 1)"
    )

    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Limit to last N days (default: all data)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: stdout)"
    )

    parser.add_argument(
        "--format",
        type=str,
        choices=["table", "csv", "json"],
        default="table",
        help="Output format (default: table)"
    )

    # Legacy output options (backward compatibility)
    parser.add_argument(
        "--out-csv",
        type=str,
        default=None,
        dest="out_csv",
        help="[Legacy] Output CSV file path"
    )

    parser.add_argument(
        "--out-json",
        type=str,
        default=None,
        dest="out_json",
        help="[Legacy] Output JSON file path"
    )

    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Show only top N results"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="[Legacy] Alias for --top"
    )

    parser.add_argument(
        "--sort",
        type=str,
        choices=["expectancy", "trades", "total_pnl", "win_rate"],
        default="expectancy",
        help="Sort by metric (default: expectancy)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show generated SQL and debug info"
    )

    args = parser.parse_args()

    # Handle legacy options
    if args.out_csv:
        args.format = "csv"
        args.output = args.out_csv
    if args.out_json:
        args.format = "json"
        args.output = args.out_json
    if args.limit and not args.top:
        args.top = args.limit

    # Find database
    db_path = find_db_path(args.db)
    if not db_path.exists():
        print(f"ERROR: Database not found at {args.db}")
        print("Try: python tools/edge_census.py --db /path/to/hft_trades.db")
        return 1

    # Connect to database
    try:
        conn = open_db(str(db_path))
    except Exception as e:
        print(f"ERROR: Could not open database: {e}")
        return 1

    # Detect schema
    schema = detect_schema(conn)

    # Inspect mode
    if args.inspect:
        print_schema_report(schema)
        conn.close()
        return 0

    # Validate trades table exists
    if not schema.has_table("trades"):
        print("ERROR: 'trades' table not found in database")
        conn.close()
        return 1

    # Parse dimensions
    dimensions = [d.strip() for d in args.dimensions.split(",")]

    # Print header
    print(f"\nEdge Census Analysis")
    print(f"  Database: {db_path}")
    print(f"  Metric: {args.metric}")
    print(f"  Dimensions: {', '.join(dimensions)}")
    print(f"  Min trades: {args.min_trades}")
    if args.days:
        print(f"  Period: Last {args.days} days")
    print()

    # Run analysis
    try:
        results = run_expectancy_analysis(
            conn=conn,
            schema=schema,
            dimensions=dimensions,
            metric=args.metric,
            min_trades=args.min_trades,
            days=args.days,
            verbose=args.verbose
        )
    except Exception as e:
        print(f"ERROR: Analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        conn.close()
        return 1

    # Sort results
    if args.sort == "trades":
        results.sort(key=lambda x: x.n_trades, reverse=True)
    elif args.sort == "total_pnl":
        results.sort(key=lambda x: x.total_pnl, reverse=True)
    elif args.sort == "win_rate":
        results.sort(key=lambda x: x.win_rate, reverse=True)
    # Default: already sorted by expectancy

    # Limit results
    if args.top:
        results = results[:args.top]

    # Output
    if args.format == "csv":
        if args.output:
            write_csv(results, args.output, dimensions)
        else:
            # Write to stdout
            if results:
                first_dict = results[0].as_dict()
                fieldnames = list(first_dict.keys())
                print(",".join(fieldnames))
                for bucket in results:
                    d = bucket.as_dict()
                    print(",".join(str(d.get(f, "")) for f in fieldnames))

    elif args.format == "json":
        if args.output:
            write_json(results, args.output)
        else:
            # Write to stdout
            data = {
                "generated_at": datetime.now().isoformat(),
                "total_buckets": len(results),
                "total_trades": sum(b.n_trades for b in results),
                "buckets": [b.as_dict() for b in results]
            }
            print(json.dumps(data, indent=2))

    else:  # table format
        output = format_table(results, dimensions)
        if args.output:
            with open(args.output, "w") as f:
                f.write(output)
            print(f"Wrote table to {args.output}")
        else:
            print(output)

    conn.close()
    return 0


if __name__ == "__main__":
    exit(main())
