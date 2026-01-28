#!/usr/bin/env python3
"""
Maker Fill Analysis Report
==========================
Analyzes maker order fill behavior to calibrate fill models and
compare maker vs taker execution performance.

Usage:
    python tools/maker_fill_report.py --db hft_trades.db --days 7
    python tools/maker_fill_report.py --db hft_trades.db --symbol XRP --days 30
    python tools/maker_fill_report.py --db hft_trades.db --calibrate

Reports:
    1. Fill rate by cross depth
    2. Time to fill distribution
    3. Cancel reason analysis
    4. Conservative fill model validation
    5. Maker vs taker cost comparison

Options:
    --symbol    Optional symbol filter (e.g., XRP, XRPUSDT). Filters data if table has symbol column.
"""

import argparse
import sqlite3
import sys
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional
import json

# Add parent dir for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from hft_system.symbol_utils import normalize_db_symbol
from hft_system.db import open_sqlite


@dataclass
class MakerFillStats:
    """Aggregate maker fill statistics."""
    total_orders: int = 0
    filled_orders: int = 0
    partial_orders: int = 0
    cancelled_orders: int = 0
    fill_rate_pct: float = 0
    avg_time_to_fill_ms: float = 0
    avg_fill_ratio: float = 0
    avg_slippage_pct: float = 0


def get_maker_fill_stats(conn: sqlite3.Connection, days: int, symbol: Optional[str] = None,
                         has_symbol_col: bool = False) -> MakerFillStats:
    """Get aggregate maker fill statistics."""
    cursor = conn.cursor()
    cutoff_ms = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

    # Build query with optional symbol filter
    where_clause = "WHERE posted_ts > ?"
    params = [cutoff_ms]
    if symbol and has_symbol_col:
        where_clause += " AND symbol LIKE ?"
        params.append(f"%{symbol}%")

    cursor.execute(f"""
        SELECT
            COUNT(*) as total_orders,
            SUM(CASE WHEN status = 'filled' THEN 1 ELSE 0 END) as filled,
            SUM(CASE WHEN status = 'partial' THEN 1 ELSE 0 END) as partial,
            SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) as cancelled,
            AVG(CASE WHEN status IN ('filled', 'partial') THEN time_to_fill_ms ELSE NULL END) as avg_time,
            AVG(CASE WHEN status IN ('filled', 'partial') THEN fill_ratio ELSE NULL END) as avg_fill_ratio,
            AVG(CASE WHEN status IN ('filled', 'partial') THEN slippage_from_limit_pct ELSE NULL END) as avg_slippage
        FROM maker_order_telemetry
        {where_clause}
    """, params)

    row = cursor.fetchone()

    if not row or row[0] == 0:
        return MakerFillStats()

    total = row[0]
    filled = row[1] or 0
    partial = row[2] or 0

    return MakerFillStats(
        total_orders=total,
        filled_orders=filled,
        partial_orders=partial,
        cancelled_orders=row[3] or 0,
        fill_rate_pct=((filled + partial) / total * 100) if total > 0 else 0,
        avg_time_to_fill_ms=row[4] or 0,
        avg_fill_ratio=row[5] or 0,
        avg_slippage_pct=row[6] or 0
    )


def get_fill_rate_by_cross_depth(conn: sqlite3.Connection, days: int, symbol: Optional[str] = None,
                                  has_symbol_col: bool = False) -> Dict:
    """Get fill rate breakdown by price cross depth."""
    cursor = conn.cursor()
    cutoff_ms = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

    # Build query with optional symbol filter
    where_clause = "WHERE posted_ts > ?"
    params = [cutoff_ms]
    if symbol and has_symbol_col:
        where_clause += " AND symbol LIKE ?"
        params.append(f"%{symbol}%")

    cursor.execute(f"""
        SELECT
            CASE
                WHEN cross_depth_bps >= 5 THEN '5+ bps'
                WHEN cross_depth_bps >= 3 THEN '3-5 bps'
                WHEN cross_depth_bps >= 1 THEN '1-3 bps'
                WHEN cross_depth_bps > 0 THEN '0-1 bps'
                ELSE 'no cross'
            END as cross_bucket,
            COUNT(*) as total,
            SUM(CASE WHEN status = 'filled' THEN 1 ELSE 0 END) as filled,
            AVG(time_to_fill_ms) as avg_time
        FROM maker_order_telemetry
        {where_clause}
        GROUP BY cross_bucket
        ORDER BY
            CASE cross_bucket
                WHEN '5+ bps' THEN 1
                WHEN '3-5 bps' THEN 2
                WHEN '1-3 bps' THEN 3
                WHEN '0-1 bps' THEN 4
                ELSE 5
            END
    """, params)

    return {
        row[0]: {
            "total": row[1],
            "filled": row[2],
            "fill_rate_pct": round(row[2] / row[1] * 100, 1) if row[1] > 0 else 0,
            "avg_time_ms": round(row[3], 0) if row[3] else 0
        }
        for row in cursor.fetchall()
    }


def get_cancel_reason_breakdown(conn: sqlite3.Connection, days: int, symbol: Optional[str] = None,
                                 has_symbol_col: bool = False) -> Dict:
    """Get breakdown of cancel reasons."""
    cursor = conn.cursor()
    cutoff_ms = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

    # Build query with optional symbol filter
    where_clause = "WHERE posted_ts > ? AND status = 'cancelled'"
    params = [cutoff_ms]
    if symbol and has_symbol_col:
        where_clause += " AND symbol LIKE ?"
        params.append(f"%{symbol}%")

    cursor.execute(f"""
        SELECT
            cancel_reason,
            COUNT(*) as count,
            AVG(time_to_fill_ms) as avg_time_alive
        FROM maker_order_telemetry
        {where_clause}
        GROUP BY cancel_reason
        ORDER BY count DESC
    """, params)

    return {
        row[0] or "unknown": {
            "count": row[1],
            "avg_time_alive_ms": round(row[2], 0) if row[2] else 0
        }
        for row in cursor.fetchall()
    }


def get_fill_model_validation(conn: sqlite3.Connection, days: int, symbol: Optional[str] = None,
                               has_symbol_col: bool = False) -> Dict:
    """Validate conservative fill model accuracy."""
    cursor = conn.cursor()
    cutoff_ms = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

    # Build query with optional symbol filter
    where_clause = "WHERE posted_ts > ? AND would_fill_conservative IS NOT NULL"
    params = [cutoff_ms]
    if symbol and has_symbol_col:
        where_clause += " AND symbol LIKE ?"
        params.append(f"%{symbol}%")

    cursor.execute(f"""
        SELECT
            -- True Positives: Model said fill, actually filled
            SUM(CASE WHEN would_fill_conservative = 1 AND status = 'filled' THEN 1 ELSE 0 END) as true_positive,
            -- False Positives: Model said fill, actually cancelled
            SUM(CASE WHEN would_fill_conservative = 1 AND status = 'cancelled' THEN 1 ELSE 0 END) as false_positive,
            -- True Negatives: Model said no fill, actually cancelled
            SUM(CASE WHEN would_fill_conservative = 0 AND status = 'cancelled' THEN 1 ELSE 0 END) as true_negative,
            -- False Negatives: Model said no fill, actually filled
            SUM(CASE WHEN would_fill_conservative = 0 AND status = 'filled' THEN 1 ELSE 0 END) as false_negative,
            COUNT(*) as total
        FROM maker_order_telemetry
        {where_clause}
    """, params)

    row = cursor.fetchone()

    if not row or row[4] == 0:
        return {"message": "No fill model validation data"}

    tp, fp, tn, fn, total = row[0] or 0, row[1] or 0, row[2] or 0, row[3] or 0, row[4]

    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    accuracy = (tp + tn) / total if total > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "total_validated": total,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "accuracy": round(accuracy, 3),
        "f1_score": round(f1, 3)
    }


def get_time_to_fill_distribution(conn: sqlite3.Connection, days: int, symbol: Optional[str] = None,
                                   has_symbol_col: bool = False) -> Dict:
    """Get time to fill distribution."""
    cursor = conn.cursor()
    cutoff_ms = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

    # Build query with optional symbol filter
    where_clause = "WHERE posted_ts > ? AND status IN ('filled', 'partial')"
    params = [cutoff_ms]
    if symbol and has_symbol_col:
        where_clause += " AND symbol LIKE ?"
        params.append(f"%{symbol}%")

    cursor.execute(f"""
        SELECT
            CASE
                WHEN time_to_fill_ms < 100 THEN '<100ms'
                WHEN time_to_fill_ms < 500 THEN '100-500ms'
                WHEN time_to_fill_ms < 1000 THEN '500ms-1s'
                WHEN time_to_fill_ms < 5000 THEN '1-5s'
                WHEN time_to_fill_ms < 10000 THEN '5-10s'
                WHEN time_to_fill_ms < 30000 THEN '10-30s'
                ELSE '30s+'
            END as time_bucket,
            COUNT(*) as count
        FROM maker_order_telemetry
        {where_clause}
        GROUP BY time_bucket
        ORDER BY
            CASE time_bucket
                WHEN '<100ms' THEN 1
                WHEN '100-500ms' THEN 2
                WHEN '500ms-1s' THEN 3
                WHEN '1-5s' THEN 4
                WHEN '5-10s' THEN 5
                WHEN '10-30s' THEN 6
                ELSE 7
            END
    """, params)

    return {row[0]: row[1] for row in cursor.fetchall()}


def get_maker_vs_taker_comparison(conn: sqlite3.Connection, days: int, symbol: Optional[str] = None,
                                   has_trades_symbol: bool = False) -> Dict:
    """Compare maker vs taker execution costs and performance."""
    cursor = conn.cursor()
    cutoff_ms = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

    # Build query with optional symbol filter
    where_clause = "WHERE entry_time > ? AND status = 'closed'"
    params = [cutoff_ms]
    if symbol and has_trades_symbol:
        where_clause += " AND symbol LIKE ?"
        params.append(f"%{symbol}%")

    # Get trades by execution mode
    cursor.execute(f"""
        SELECT
            execution_mode,
            COUNT(*) as trades,
            AVG(fees_paid_pct) as avg_fees,
            AVG(slippage_pct) as avg_slippage,
            AVG(total_costs_pct) as avg_total_costs,
            AVG(pnl_pct) as avg_gross_pnl,
            AVG(pnl_after_costs_pct) as avg_net_pnl,
            SUM(CASE WHEN pnl_after_costs_pct > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as win_rate
        FROM trades
        {where_clause}
        GROUP BY execution_mode
    """, params)

    comparison = {}
    for row in cursor.fetchall():
        mode = row[0] or "unknown"
        comparison[mode] = {
            "trades": row[1],
            "avg_fees_pct": round(row[2], 4) if row[2] else 0,
            "avg_slippage_pct": round(row[3], 4) if row[3] else 0,
            "avg_total_costs_pct": round(row[4], 4) if row[4] else 0,
            "avg_gross_pnl_pct": round(row[5], 4) if row[5] else 0,
            "avg_net_pnl_pct": round(row[6], 4) if row[6] else 0,
            "win_rate_pct": round(row[7], 1) if row[7] else 0
        }

    # Calculate savings
    if "maker" in comparison and "taker" in comparison:
        maker_cost = comparison["maker"]["avg_total_costs_pct"]
        taker_cost = comparison["taker"]["avg_total_costs_pct"]
        savings = taker_cost - maker_cost
        comparison["maker_savings_vs_taker_pct"] = round(savings, 4)

    return comparison


def calibrate_fill_model(conn: sqlite3.Connection, days: int, symbol: Optional[str] = None,
                          has_symbol_col: bool = False) -> Dict:
    """Generate calibrated fill model parameters from data."""
    cursor = conn.cursor()
    cutoff_ms = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

    # Build query with optional symbol filter
    where_clause = "WHERE posted_ts > ? AND cross_depth_bps IS NOT NULL"
    params = [cutoff_ms]
    if symbol and has_symbol_col:
        where_clause += " AND symbol LIKE ?"
        params.append(f"%{symbol}%")

    # Get fill rates by cross depth
    cursor.execute(f"""
        SELECT
            cross_depth_bps,
            CASE WHEN status = 'filled' THEN 1 ELSE 0 END as filled
        FROM maker_order_telemetry
        {where_clause}
        ORDER BY cross_depth_bps
    """, params)

    rows = cursor.fetchall()

    if not rows:
        return {"message": "No data for calibration"}

    # Calculate fill rate at different thresholds
    thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    rates = {}

    for threshold in thresholds:
        above = [r for r in rows if r[0] >= threshold]
        if len(above) >= 10:
            fill_rate = sum(1 for r in above if r[1] == 1) / len(above)
            rates[f"{threshold}bps"] = round(fill_rate, 3)

    # Find minimum cross for 70% fill rate
    min_cross_70 = None
    for threshold in thresholds:
        above = [r for r in rows if r[0] >= threshold]
        if len(above) >= 10:
            fill_rate = sum(1 for r in above if r[1] == 1) / len(above)
            if fill_rate >= 0.7:
                min_cross_70 = threshold
                break

    # Recommended config
    recommended = {
        "mode": "conservative_cross",
        "min_cross_bps": min_cross_70 or 2.0,
        "fill_rate_on_touch": rates.get("0.5bps", 0.30),
        "fill_rate_on_small_cross": rates.get("1.5bps", 0.60),
        "fill_rate_on_large_cross": rates.get("3.0bps", 0.85)
    }

    return {
        "fill_rates_by_threshold": rates,
        "min_cross_for_70pct_fill": min_cross_70,
        "recommended_config": recommended,
        "sample_size": len(rows)
    }


def print_report(
    stats: MakerFillStats,
    cross_breakdown: Dict,
    cancel_reasons: Dict,
    fill_model: Dict,
    time_dist: Dict,
    comparison: Dict,
    days: int,
    symbol: Optional[str] = None
):
    """Print formatted maker fill report."""
    symbol_line = f"| Symbol:  {symbol}" if symbol else "| Symbol:  ALL (global)"
    print(f"""
+================================================================================+
|                      MAKER ORDER FILL ANALYSIS REPORT                          |
+================================================================================+
{symbol_line}
| Period:  Last {days} days
| Orders:  {stats.total_orders}
+================================================================================+

  --- OVERALL FILL STATISTICS ---
  Total Orders:     {stats.total_orders}
  Filled:           {stats.filled_orders} ({stats.fill_rate_pct:.1f}%)
  Partial:          {stats.partial_orders}
  Cancelled:        {stats.cancelled_orders}
  Avg Time to Fill: {stats.avg_time_to_fill_ms:.0f} ms
  Avg Fill Ratio:   {stats.avg_fill_ratio*100:.1f}%
  Avg Slippage:     {stats.avg_slippage_pct:.4f}%

  --- FILL RATE BY PRICE CROSS DEPTH ---
""")

    for bucket, data in cross_breakdown.items():
        print(f"  {bucket:12s}: {data['filled']:4d}/{data['total']:4d} = {data['fill_rate_pct']:5.1f}%  (avg {data['avg_time_ms']:.0f}ms)")

    print("""
  --- TIME TO FILL DISTRIBUTION ---
""")

    for bucket, count in time_dist.items():
        bar = "*" * min(count, 50)
        print(f"  {bucket:12s}: {count:4d} {bar}")

    print("""
  --- CANCEL REASON BREAKDOWN ---
""")

    for reason, data in cancel_reasons.items():
        print(f"  {reason:20s}: {data['count']:4d} orders (avg alive: {data['avg_time_alive_ms']:.0f}ms)")

    print("""
  --- CONSERVATIVE FILL MODEL VALIDATION ---
""")

    if "message" in fill_model:
        print(f"  {fill_model['message']}")
    else:
        print(f"  True Positives:   {fill_model['true_positive']:4d} (model correct: would fill)")
        print(f"  False Positives:  {fill_model['false_positive']:4d} (model wrong: predicted fill, was cancel)")
        print(f"  True Negatives:   {fill_model['true_negative']:4d} (model correct: would not fill)")
        print(f"  False Negatives:  {fill_model['false_negative']:4d} (model wrong: predicted no fill, was filled)")
        print()
        print(f"  Precision:  {fill_model['precision']:.3f}")
        print(f"  Recall:     {fill_model['recall']:.3f}")
        print(f"  Accuracy:   {fill_model['accuracy']:.3f}")
        print(f"  F1 Score:   {fill_model['f1_score']:.3f}")

    print("""
  --- MAKER vs TAKER COMPARISON ---
""")

    if comparison:
        for mode, data in comparison.items():
            if mode == "maker_savings_vs_taker_pct":
                continue
            print(f"  {mode.upper():8s}:")
            print(f"    Trades:      {data['trades']}")
            print(f"    Avg Fees:    {data['avg_fees_pct']:+.4f}%")
            print(f"    Avg Slippage:{data['avg_slippage_pct']:+.4f}%")
            print(f"    Total Costs: {data['avg_total_costs_pct']:+.4f}%")
            print(f"    Gross PnL:   {data['avg_gross_pnl_pct']:+.4f}%")
            print(f"    Net PnL:     {data['avg_net_pnl_pct']:+.4f}%")
            print(f"    Win Rate:    {data['win_rate_pct']:.1f}%")
            print()

        if "maker_savings_vs_taker_pct" in comparison:
            savings = comparison["maker_savings_vs_taker_pct"]
            print(f"  MAKER SAVINGS: {savings:+.4f}% per trade vs taker")

    print()


def print_calibration(calibration: Dict):
    """Print fill model calibration results."""
    print("""
+================================================================================+
|                    FILL MODEL CALIBRATION RESULTS                              |
+================================================================================+

  --- FILL RATES BY CROSS THRESHOLD ---
""")

    if "fill_rates_by_threshold" in calibration:
        for threshold, rate in calibration["fill_rates_by_threshold"].items():
            bar = "*" * int(rate * 50)
            print(f"  >= {threshold:6s}: {rate*100:5.1f}% {bar}")

    print()
    print(f"  Min cross for 70% fill: {calibration.get('min_cross_for_70pct_fill', 'N/A')} bps")
    print(f"  Sample size: {calibration.get('sample_size', 0)} orders")

    print("""
  --- RECOMMENDED CONFIGURATION ---
""")

    if "recommended_config" in calibration:
        config = calibration["recommended_config"]
        print(f"  mode: \"{config['mode']}\"")
        print(f"  min_cross_bps: {config['min_cross_bps']}")
        print(f"  fill_rate_on_touch: {config['fill_rate_on_touch']}")
        print(f"  fill_rate_on_small_cross: {config['fill_rate_on_small_cross']}")
        print(f"  fill_rate_on_large_cross: {config['fill_rate_on_large_cross']}")

    print()


def check_column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    """Check if a column exists in a table."""
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table})")
    columns = [row[1] for row in cursor.fetchall()]
    return column in columns


# normalize_symbol moved to hft_system/symbol_utils.py
# Use normalize_db_symbol from shared module instead


def main():
    parser = argparse.ArgumentParser(
        description="Maker Fill Analysis Report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/maker_fill_report.py --db hft_trades.db --days 7
  python tools/maker_fill_report.py --db hft_trades.db --symbol XRP --days 30
  python tools/maker_fill_report.py --db hft_trades.db --calibrate
  python tools/maker_fill_report.py --db hft_trades.db --json
        """
    )
    parser.add_argument("--db", default="hft_trades.db", help="Database path")
    parser.add_argument("--symbol", type=str, default=None,
                        help="Symbol filter (e.g., XRP, XRPUSDT). Filters if table has symbol column.")
    parser.add_argument("--days", type=int, default=7, help="Analysis period (days)")
    parser.add_argument("--calibrate", action="store_true", help="Show fill model calibration")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    # Find database
    db_path = Path(args.db)
    if not db_path.exists():
        for try_path in [
            Path.cwd() / "hft_trades.db",
            Path.home() / "Analize-" / "hft_trades.db",
            Path("/root/Analize-/hft_trades.db")
        ]:
            if try_path.exists():
                db_path = try_path
                break

    if not db_path.exists():
        print(f"ERROR: Database not found at {args.db}")
        return 1

    # Connect with bot-safe WAL mode and busy_timeout
    conn = open_sqlite(str(db_path))

    # Check if maker_order_telemetry table exists
    cursor = conn.cursor()
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='maker_order_telemetry'
    """)
    if not cursor.fetchone():
        print("ERROR: maker_order_telemetry table not found.")
        print("       Maker telemetry logging must be enabled first.")
        return 1

    # Check for symbol columns in tables
    has_maker_symbol = check_column_exists(conn, "maker_order_telemetry", "symbol")
    has_trades_symbol = check_column_exists(conn, "trades", "symbol")

    # Normalize symbol filter
    symbol = None
    if args.symbol:
        symbol = normalize_db_symbol(args.symbol)
        if not has_maker_symbol:
            print(f"WARNING: --symbol {args.symbol} specified but maker_order_telemetry table has no symbol column.")
            print("         Running global analysis (all symbols).")
            symbol = None
        else:
            print(f"Filtering by symbol: {symbol}")

    if args.calibrate:
        calibration = calibrate_fill_model(conn, args.days, symbol, has_maker_symbol)
        if args.json:
            print(json.dumps(calibration, indent=2))
        else:
            print_calibration(calibration)
    else:
        # Gather all stats
        stats = get_maker_fill_stats(conn, args.days, symbol, has_maker_symbol)
        cross_breakdown = get_fill_rate_by_cross_depth(conn, args.days, symbol, has_maker_symbol)
        cancel_reasons = get_cancel_reason_breakdown(conn, args.days, symbol, has_maker_symbol)
        fill_model = get_fill_model_validation(conn, args.days, symbol, has_maker_symbol)
        time_dist = get_time_to_fill_distribution(conn, args.days, symbol, has_maker_symbol)
        comparison = get_maker_vs_taker_comparison(conn, args.days, symbol, has_trades_symbol)

        if args.json:
            output = {
                "symbol": symbol or "ALL",
                "stats": {
                    "total_orders": stats.total_orders,
                    "filled": stats.filled_orders,
                    "partial": stats.partial_orders,
                    "cancelled": stats.cancelled_orders,
                    "fill_rate_pct": stats.fill_rate_pct,
                    "avg_time_to_fill_ms": stats.avg_time_to_fill_ms,
                    "avg_slippage_pct": stats.avg_slippage_pct
                },
                "cross_breakdown": cross_breakdown,
                "cancel_reasons": cancel_reasons,
                "fill_model_validation": fill_model,
                "time_distribution": time_dist,
                "maker_vs_taker": comparison
            }
            print(json.dumps(output, indent=2))
        else:
            if stats.total_orders == 0:
                print(f"No maker orders found in the last {args.days} days" +
                      (f" for symbol {symbol}" if symbol else ""))
                return 0

            print_report(
                stats, cross_breakdown, cancel_reasons,
                fill_model, time_dist, comparison, args.days, symbol
            )

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
