#!/usr/bin/env python3
"""
Real-Metrics Report Tool
========================
Reads hft_trades.db and computes profitability metrics **excluding** smoke-test
/ diagnostic trades (exit reasons: maker_smoke_test, force_paper_trade,
manual_close_orphan) as well as any row with is_smoke_test=1.

Works stand-alone – no external services required, only sqlite3.

Usage:
    python tools/report_real_metrics.py                 # default: ./hft_trades.db
    python tools/report_real_metrics.py --db /path/to.db
    python tools/report_real_metrics.py --json           # machine-readable output
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

# Exit reasons that mark diagnostic / smoke-test trades
EXCLUDED_EXIT_REASONS = {
    "maker_smoke_test",
    "force_paper_trade",
    "manual_close_orphan",
}


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=5.0)
    conn.row_factory = sqlite3.Row
    return conn


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    cursor = conn.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in cursor.fetchall())


def compute_metrics(db_path: str) -> dict[str, Any]:
    """Return a dict of clean profitability metrics."""
    conn = _connect(db_path)
    cur = conn.cursor()

    # Build WHERE clause – exclude smoke-test exit reasons + is_smoke_test flag
    placeholders = ",".join(["?"] * len(EXCLUDED_EXIT_REASONS))
    has_smoke_col = _has_column(conn, "trades", "is_smoke_test")
    smoke_filter = " AND (is_smoke_test IS NULL OR is_smoke_test = 0)" if has_smoke_col else ""

    base_where = (
        f"status = 'closed' AND "
        f"(exit_reason IS NULL OR exit_reason NOT IN ({placeholders}))"
        f"{smoke_filter}"
    )
    params: list[Any] = list(EXCLUDED_EXIT_REASONS)

    # ── Overall stats ─────────────────────────────────────────────────
    cur.execute(f"""
        SELECT
            COUNT(*)                                              AS trades,
            SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END)            AS wins,
            SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END)           AS losses,
            SUM(pnl)                                              AS total_pnl,
            AVG(pnl_pct)                                          AS avg_pnl_pct,
            SUM(pnl_pct)                                          AS total_pnl_pct,
            AVG(CASE WHEN pnl > 0 THEN pnl ELSE NULL END)       AS avg_win,
            AVG(CASE WHEN pnl <= 0 THEN pnl ELSE NULL END)      AS avg_loss,
            AVG(hold_time_sec)                                    AS avg_hold_sec
        FROM trades WHERE {base_where}
    """, params)
    row = cur.fetchone()

    if not row or row["trades"] == 0:
        conn.close()
        return {"error": "No qualifying trades found"}

    trades = row["trades"]
    wins = row["wins"] or 0
    losses = row["losses"] or 0
    avg_win_abs = abs(row["avg_win"]) if row["avg_win"] else 0
    avg_loss_abs = abs(row["avg_loss"]) if row["avg_loss"] else 0
    win_rate = (wins / trades * 100) if trades > 0 else 0
    profit_factor = (
        (avg_win_abs * wins) / (avg_loss_abs * losses)
        if losses > 0 and avg_loss_abs > 0
        else float("inf") if wins > 0 else 0.0
    )

    overall = {
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": round(win_rate, 2),
        "avg_pnl_pct": round(row["avg_pnl_pct"] or 0, 4),
        "total_pnl_pct": round(row["total_pnl_pct"] or 0, 4),
        "total_pnl_usd": round(row["total_pnl"] or 0, 4),
        "profit_factor": round(profit_factor, 3),
        "avg_hold_sec": round(row["avg_hold_sec"] or 0, 1),
    }

    # ── Breakdown by exit reason ──────────────────────────────────────
    cur.execute(f"""
        SELECT exit_reason,
               COUNT(*)                                        AS cnt,
               AVG(pnl_pct)                                    AS avg_pnl,
               SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS wr
        FROM trades WHERE {base_where}
        GROUP BY exit_reason ORDER BY cnt DESC
    """, params)
    by_exit = {
        r["exit_reason"]: {
            "trades": r["cnt"],
            "avg_pnl_pct": round(r["avg_pnl"], 4) if r["avg_pnl"] else 0,
            "win_rate_pct": round(r["wr"], 1) if r["wr"] else 0,
        }
        for r in cur.fetchall()
    }

    # ── Breakdown by symbol ───────────────────────────────────────────
    cur.execute(f"""
        SELECT symbol,
               COUNT(*)                                        AS cnt,
               AVG(pnl_pct)                                    AS avg_pnl,
               SUM(pnl)                                        AS tot_pnl,
               SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS wr
        FROM trades WHERE {base_where}
        GROUP BY symbol ORDER BY cnt DESC
    """, params)
    by_symbol = {
        r["symbol"]: {
            "trades": r["cnt"],
            "avg_pnl_pct": round(r["avg_pnl"], 4) if r["avg_pnl"] else 0,
            "total_pnl_usd": round(r["tot_pnl"], 4) if r["tot_pnl"] else 0,
            "win_rate_pct": round(r["wr"], 1) if r["wr"] else 0,
        }
        for r in cur.fetchall()
    }

    # ── Breakdown by regime_at_entry ──────────────────────────────────
    has_regime = _has_column(conn, "trades", "regime_at_entry")
    by_regime: dict[str, Any] = {}
    if has_regime:
        cur.execute(f"""
            SELECT regime_at_entry,
                   COUNT(*)                                        AS cnt,
                   AVG(pnl_pct)                                    AS avg_pnl,
                   SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS wr
            FROM trades WHERE {base_where} AND regime_at_entry IS NOT NULL
            GROUP BY regime_at_entry ORDER BY cnt DESC
        """, params)
        by_regime = {
            r["regime_at_entry"]: {
                "trades": r["cnt"],
                "avg_pnl_pct": round(r["avg_pnl"], 4) if r["avg_pnl"] else 0,
                "win_rate_pct": round(r["wr"], 1) if r["wr"] else 0,
            }
            for r in cur.fetchall()
        }

    # ── Excluded trades summary ───────────────────────────────────────
    smoke_where = (
        f"status = 'closed' AND "
        f"(exit_reason IN ({placeholders})"
        f"{' OR is_smoke_test = 1' if has_smoke_col else ''})"
    )
    cur.execute(f"SELECT COUNT(*) AS cnt FROM trades WHERE {smoke_where}", params)
    excluded_count = cur.fetchone()["cnt"]

    conn.close()

    return {
        "overall": overall,
        "by_exit_reason": by_exit,
        "by_symbol": by_symbol,
        "by_regime": by_regime,
        "excluded_smoke_test_trades": excluded_count,
    }


def _print_report(data: dict[str, Any]) -> None:
    """Pretty-print the metrics report to stdout."""
    if "error" in data:
        print(f"ERROR: {data['error']}")
        return

    o = data["overall"]
    print("=" * 60)
    print("  REAL-METRICS REPORT  (smoke-test trades excluded)")
    print("=" * 60)
    print(f"  Excluded trades:  {data['excluded_smoke_test_trades']}")
    print(f"  Real trades:      {o['trades']}")
    print(f"  Wins / Losses:    {o['wins']} / {o['losses']}")
    print(f"  Win rate:         {o['win_rate_pct']:.2f}%")
    print(f"  Avg PnL %:        {o['avg_pnl_pct']:+.4f}%")
    print(f"  Total PnL %:      {o['total_pnl_pct']:+.4f}%")
    print(f"  Total PnL $:      ${o['total_pnl_usd']:+.4f}")
    print(f"  Profit factor:    {o['profit_factor']:.3f}")
    print(f"  Avg hold time:    {o['avg_hold_sec']:.1f}s")

    if data["by_exit_reason"]:
        print()
        print("  BY EXIT REASON:")
        for reason, stats in data["by_exit_reason"].items():
            print(
                f"    {reason:25s} | {stats['trades']:4d} trades | "
                f"avg {stats['avg_pnl_pct']:+.4f}% | WR {stats['win_rate_pct']:.1f}%"
            )

    if data["by_symbol"]:
        print()
        print("  BY SYMBOL:")
        for sym, stats in data["by_symbol"].items():
            print(
                f"    {sym:10s} | {stats['trades']:4d} trades | "
                f"avg {stats['avg_pnl_pct']:+.4f}% | "
                f"total ${stats['total_pnl_usd']:+.4f} | WR {stats['win_rate_pct']:.1f}%"
            )

    if data["by_regime"]:
        print()
        print("  BY REGIME (at entry):")
        for regime, stats in data["by_regime"].items():
            print(
                f"    {regime:25s} | {stats['trades']:4d} trades | "
                f"avg {stats['avg_pnl_pct']:+.4f}% | WR {stats['win_rate_pct']:.1f}%"
            )

    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="HFT real-metrics report")
    parser.add_argument(
        "--db",
        default="hft_trades.db",
        help="Path to hft_trades.db (default: ./hft_trades.db)",
    )
    parser.add_argument(
        "--json", action="store_true", dest="as_json",
        help="Output as JSON instead of human-readable table",
    )
    args = parser.parse_args()

    if not Path(args.db).exists():
        print(f"ERROR: Database not found at {args.db}", file=sys.stderr)
        sys.exit(1)

    data = compute_metrics(args.db)

    if args.as_json:
        print(json.dumps(data, indent=2))
    else:
        _print_report(data)


if __name__ == "__main__":
    main()
