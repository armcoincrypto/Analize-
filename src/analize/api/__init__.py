"""
REST API module for Analize.

Provides FastAPI endpoints for:
- Status and health checks
- Data ingestion
- Signal queries
- Report generation
- Optimization jobs
- Suggestions
"""

from analize.api.app import create_app
from analize.api.routes import router

__all__ = ["create_app", "router"]
