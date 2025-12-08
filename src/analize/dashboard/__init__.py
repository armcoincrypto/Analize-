"""
Dashboard module for Analize.

Provides a simple web dashboard for:
- Status overview
- Signal exploration
- Report viewing
- Parameter optimization interface
"""

from analize.dashboard.app import create_dashboard_app

__all__ = ["create_dashboard_app"]
