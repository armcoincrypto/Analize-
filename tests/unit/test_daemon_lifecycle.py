"""Tests for Deliverable 5: service daemon lifecycle.

Ensures the bot supports --once mode and has proper daemon restart logic.
"""

import inspect
import pytest


class TestDaemonLifecycle:
    """Bot should stay running as daemon unless --once is specified."""

    def test_start_accepts_once_param(self):
        """HFTBot.start() should accept a `once` parameter."""
        from hft_system.hft_bot import HFTBot

        sig = inspect.signature(HFTBot.start)
        assert "once" in sig.parameters
        assert sig.parameters["once"].default is False

    def test_main_parser_has_once_flag(self):
        """The argparse parser in main() should accept --once."""
        import argparse
        from hft_system.hft_bot import main

        # Check source for --once
        source = inspect.getsource(main)
        assert "--once" in source

    def test_daemon_mode_has_backoff_retry(self):
        """In daemon mode, start() should have retry/backoff logic."""
        from hft_system.hft_bot import HFTBot

        source = inspect.getsource(HFTBot.start)
        assert "backoff" in source
        assert "max_backoff" in source
        assert "Restarting" in source

    def test_shutdown_handler_sets_running_false(self):
        """The shutdown handler should set bot.running = False."""
        from hft_system.hft_bot import main

        source = inspect.getsource(main)
        assert "bot.running = False" in source

    def test_run_tasks_is_separate_method(self):
        """_run_tasks should be a separate method for testability."""
        from hft_system.hft_bot import HFTBot

        assert hasattr(HFTBot, "_run_tasks")
        assert callable(getattr(HFTBot, "_run_tasks"))
