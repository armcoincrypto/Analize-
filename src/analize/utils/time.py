"""
Timezone-aware UTC datetime utilities.

Replaces deprecated datetime.utcnow() with timezone-aware alternatives.
"""

from datetime import datetime, timezone


def utcnow() -> datetime:
    """Return the current UTC datetime with timezone information."""
    return datetime.now(timezone.utc)


def utcnow_iso() -> str:
    """Return the current UTC datetime as an ISO format string."""
    return utcnow().isoformat()


def utcnow_naive() -> datetime:
    """
    Return the current UTC datetime without timezone information.

    Use only when compatibility with naive datetime is required
    (e.g., SQLAlchemy DateTime columns without timezone support).
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)
