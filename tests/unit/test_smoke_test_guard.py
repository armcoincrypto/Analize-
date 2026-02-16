"""Tests for Deliverable 1: SMOKE_TEST env guard."""

import os
import sqlite3
import tempfile

import pytest


def _make_logger(db_path: str):
    """Create a TradeLogger pointing at a temp DB."""
    # Patch SYSTEM_CONFIG to use temp path
    from hft_system.config import SYSTEM_CONFIG
    orig = SYSTEM_CONFIG.db_path
    SYSTEM_CONFIG.db_path = db_path
    from hft_system.trade_logger import TradeLogger
    tl = TradeLogger(db_path=db_path)
    SYSTEM_CONFIG.db_path = orig
    return tl


class TestSmokeTestGuard:
    """Smoke-test trades should only be created when SMOKE_TEST=1."""

    def test_smoke_test_blocked_by_default(self, tmp_path):
        """With default SMOKE_TEST=0, insert_smoke_test_trade returns False."""
        import hft_system.config as cfg
        orig = cfg.SMOKE_TEST
        cfg.SMOKE_TEST = False
        # Re-import to pick up the flag (module-level constant is copied at import)
        import hft_system.trade_logger as tl_mod
        tl_mod.SMOKE_TEST = False

        try:
            db_path = str(tmp_path / "test.db")
            tl = _make_logger(db_path)
            result = tl.insert_smoke_test_trade("SUI", "maker_smoke_test")
            assert result is False

            # Verify no trade was inserted
            cur = tl.conn.cursor()
            cur.execute("SELECT COUNT(*) FROM trades")
            assert cur.fetchone()[0] == 0
            tl.close()
        finally:
            cfg.SMOKE_TEST = orig
            tl_mod.SMOKE_TEST = orig

    def test_smoke_test_allowed_when_enabled(self, tmp_path):
        """With SMOKE_TEST=1, insert_smoke_test_trade works."""
        import hft_system.config as cfg
        import hft_system.trade_logger as tl_mod
        orig = cfg.SMOKE_TEST
        cfg.SMOKE_TEST = True
        tl_mod.SMOKE_TEST = True

        try:
            db_path = str(tmp_path / "test.db")
            tl = _make_logger(db_path)
            result = tl.insert_smoke_test_trade("SUI", "maker_smoke_test")
            assert result is True

            cur = tl.conn.cursor()
            cur.execute("SELECT COUNT(*) FROM trades WHERE is_smoke_test = 1")
            assert cur.fetchone()[0] == 1

            cur.execute("SELECT exit_reason FROM trades WHERE is_smoke_test = 1")
            assert cur.fetchone()[0] == "maker_smoke_test"
            tl.close()
        finally:
            cfg.SMOKE_TEST = orig
            tl_mod.SMOKE_TEST = orig

    def test_smoke_test_rejects_unknown_reason(self, tmp_path):
        """Only known smoke-test exit reasons are accepted."""
        import hft_system.config as cfg
        import hft_system.trade_logger as tl_mod
        orig = cfg.SMOKE_TEST
        cfg.SMOKE_TEST = True
        tl_mod.SMOKE_TEST = True

        try:
            db_path = str(tmp_path / "test.db")
            tl = _make_logger(db_path)
            result = tl.insert_smoke_test_trade("SUI", "totally_invalid_reason")
            assert result is False
            tl.close()
        finally:
            cfg.SMOKE_TEST = orig
            tl_mod.SMOKE_TEST = orig
