"""Tests for Deliverable 2: real-metrics report tool."""

import sqlite3
import pytest
from pathlib import Path

# We import the module under test directly
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from tools.report_real_metrics import compute_metrics, EXCLUDED_EXIT_REASONS


def _create_test_db(db_path: str) -> None:
    """Create a minimal trades table with a mix of real + smoke-test trades."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id TEXT UNIQUE,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            entry_price REAL NOT NULL,
            exit_price REAL,
            quantity REAL NOT NULL,
            entry_time INTEGER NOT NULL,
            exit_time INTEGER,
            pnl REAL,
            pnl_pct REAL,
            hold_time_sec REAL,
            exit_reason TEXT,
            mfe REAL,
            mae REAL,
            conditions_met INTEGER,
            signal_confidence REAL,
            status TEXT DEFAULT 'open',
            regime_at_entry TEXT,
            is_smoke_test INTEGER DEFAULT 0
        )
    """)

    # Real trades
    trades = [
        ("t1", "SUI", "LONG", 1.0, 1.001, 10, 1000, 1010, 0.01, 0.10, 5, "take_profit", 0.12, -0.01, 1, 0.8, "closed", "high_vol_trend", 0),
        ("t2", "SUI", "LONG", 1.0, 0.999, 10, 2000, 2010, -0.01, -0.10, 8, "stop_loss", 0.02, -0.12, 1, 0.7, "closed", "low_vol_chop", 0),
        ("t3", "XRP", "SHORT", 0.5, 0.498, 20, 3000, 3015, 0.04, 0.40, 15, "micro_profit", 0.50, -0.05, 1, 0.9, "closed", "high_vol_trend", 0),
    ]

    # Smoke-test trades (should be excluded)
    smoke = [
        ("s1", "SUI", "LONG", 1.0, 1.0, 0, 4000, 4000, 0, 0, 0, "maker_smoke_test", 0, 0, 0, 0, "closed", None, 1),
        ("s2", "SUI", "LONG", 1.0, 1.0, 0, 5000, 5000, 0, 0, 0, "force_paper_trade", 0, 0, 0, 0, "closed", None, 0),
        ("s3", "XRP", "LONG", 0.5, 0.5, 0, 6000, 6000, 0, 0, 0, "manual_close_orphan", 0, 0, 0, 0, "closed", None, 0),
    ]

    for t in trades + smoke:
        conn.execute(
            """INSERT INTO trades (trade_id, symbol, side, entry_price, exit_price,
               quantity, entry_time, exit_time, pnl, pnl_pct, hold_time_sec,
               exit_reason, mfe, mae, conditions_met, signal_confidence,
               status, regime_at_entry, is_smoke_test)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            t,
        )
    conn.commit()
    conn.close()


class TestRealMetricsReport:
    def test_excludes_smoke_trades(self, tmp_path):
        db = str(tmp_path / "test.db")
        _create_test_db(db)
        data = compute_metrics(db)

        assert "error" not in data
        assert data["overall"]["trades"] == 3  # only real trades
        assert data["excluded_smoke_test_trades"] == 3  # 3 smoke

    def test_overall_metrics(self, tmp_path):
        db = str(tmp_path / "test.db")
        _create_test_db(db)
        data = compute_metrics(db)

        o = data["overall"]
        assert o["wins"] == 2  # t1 and t3 are wins
        assert o["losses"] == 1  # t2 is a loss
        assert o["win_rate_pct"] == pytest.approx(66.67, abs=0.1)
        assert o["profit_factor"] > 0

    def test_by_exit_reason(self, tmp_path):
        db = str(tmp_path / "test.db")
        _create_test_db(db)
        data = compute_metrics(db)

        reasons = data["by_exit_reason"]
        assert "take_profit" in reasons
        assert "stop_loss" in reasons
        assert "micro_profit" in reasons
        # Smoke-test reasons must NOT appear
        for excluded in EXCLUDED_EXIT_REASONS:
            assert excluded not in reasons

    def test_by_symbol(self, tmp_path):
        db = str(tmp_path / "test.db")
        _create_test_db(db)
        data = compute_metrics(db)

        syms = data["by_symbol"]
        assert "SUI" in syms
        assert "XRP" in syms
        assert syms["SUI"]["trades"] == 2
        assert syms["XRP"]["trades"] == 1

    def test_by_regime(self, tmp_path):
        db = str(tmp_path / "test.db")
        _create_test_db(db)
        data = compute_metrics(db)

        regimes = data["by_regime"]
        assert "high_vol_trend" in regimes
        assert regimes["high_vol_trend"]["trades"] == 2

    def test_empty_db(self, tmp_path):
        db = str(tmp_path / "empty.db")
        conn = sqlite3.connect(db)
        conn.execute("""
            CREATE TABLE trades (
                id INTEGER PRIMARY KEY, trade_id TEXT, symbol TEXT,
                side TEXT, entry_price REAL, exit_price REAL,
                quantity REAL, entry_time INTEGER, exit_time INTEGER,
                pnl REAL, pnl_pct REAL, hold_time_sec REAL,
                exit_reason TEXT, status TEXT DEFAULT 'open',
                is_smoke_test INTEGER DEFAULT 0
            )
        """)
        conn.commit()
        conn.close()

        data = compute_metrics(db)
        assert "error" in data
