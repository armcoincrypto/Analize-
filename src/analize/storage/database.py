"""
Database connection and session management.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from analize.config import get_settings
from analize.models.database import Base


class DatabaseManager:
    """Manages database connections and sessions."""

    def __init__(self, database_url: str | None = None):
        settings = get_settings()
        self.database_url = database_url or str(settings.database.postgres_url)

        # Convert sync URL to async if needed
        if self.database_url.startswith("postgresql://"):
            self.database_url = self.database_url.replace(
                "postgresql://", "postgresql+asyncpg://", 1
            )
        elif self.database_url.startswith("sqlite://"):
            self.database_url = self.database_url.replace("sqlite://", "sqlite+aiosqlite://", 1)

        # Create async engine
        self.engine = create_async_engine(
            self.database_url,
            echo=settings.api.debug,
            pool_size=settings.database.pool_size if "sqlite" not in self.database_url else 0,
            max_overflow=settings.database.max_overflow if "sqlite" not in self.database_url else 0,
            poolclass=NullPool if "sqlite" in self.database_url else None,
        )

        # Create session factory
        self.async_session = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )

    async def create_tables(self) -> None:
        """Create all database tables."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def drop_tables(self) -> None:
        """Drop all database tables."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get a database session."""
        async with self.async_session() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    async def close(self) -> None:
        """Close the database connection."""
        await self.engine.dispose()


# Global database manager instance
_db_manager: DatabaseManager | None = None


def get_db_manager() -> DatabaseManager:
    """Get or create the global database manager."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting database sessions."""
    db = get_db_manager()
    async with db.session() as session:
        yield session


async def init_database() -> None:
    """Initialize the database (create tables)."""
    db = get_db_manager()
    await db.create_tables()
