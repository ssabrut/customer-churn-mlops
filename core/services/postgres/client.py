from typing import AsyncGenerator, Dict

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from core.config import Settings


class PostgresClient:
    """
    Manages async and sync connections to a PostgreSQL database.

    This class handles the initialization of the async engine,
    session factory, and provides methods for health checks,
    session management, and engine disposal.

    Attributes:
        base_url (str): The asynchronous database connection string
                        (e.g., "postgresql+asyncpg://...").
        engine (AsyncEngine): The SQLAlchemy asynchronous engine instance.
        session_maker (async_sessionmaker[AsyncSession]): The factory
            for creating new asynchronous sessions.
    """

    base_url: str
    engine: AsyncEngine
    session_maker: async_sessionmaker[AsyncSession]

    def __init__(self, settings: Settings) -> None:
        """
        Initializes the PostgresClient and creates the async engine.

        Args:
            settings (Settings): The application configuration object containing
                                 database credentials and details.

        Raises:
            TypeError: If 'settings' is not an instance of the Settings class.
            RuntimeError: If database engine creation fails due to invalid
                          configuration, connection errors, or other
                          SQLAlchemy-related issues.
        """
        if not isinstance(settings, Settings):
            raise TypeError(
                "Argument 'settings' must be an instance of the Settings class"
            )

        self.base_url = f"postgresql+asyncpg://{settings.app_db_user}:{settings.app_db_password}@{settings.db_host}:{settings.db_port}/{settings.app_db_name}"

        try:
            self.engine = create_async_engine(self.base_url, echo=settings.debug)
            self.session_maker = async_sessionmaker(
                self.engine, expire_on_commit=False
            )
        except (SQLAlchemyError, ValueError) as e:
            # Catch specific errors from engine creation (e.g., bad URL, driver issues)
            raise RuntimeError(f"Failed to create database engine: {e}") from e
        except Exception as e:
            # A general catch-all for any other unexpected initialization error
            raise RuntimeError(
                f"An unexpected error occurred during engine creation: {e}"
            ) from e

    def get_sync_engine(self) -> Engine:
        """
        Creates and returns a synchronous SQLAlchemy engine.

        This is useful for operations that do not support async,
        such as database migrations (e.g., Alembic).

        Returns:
            Engine: A synchronous SQLAlchemy Engine instance.

        Raises:
            RuntimeError: If the synchronous engine creation fails due to
                          connection errors or invalid configuration.
        """
        sync_url: str = self.base_url.replace("+asyncpg", "")

        try:
            return create_engine(sync_url)
        except (SQLAlchemyError, ValueError) as e:
            raise RuntimeError(
                f"Failed to create synchronous database engine: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"An unexpected error occurred during sync engine creation: {e}"
            ) from e

    async def health_check(self) -> Dict[str, str]:
        """
        Performs a simple health check against the database.

        Executes a 'SELECT 1' query to verify a valid connection can
        be established and a simple command executed.

        Returns:
            Dict[str, str]: A dictionary containing 'status' ('healthy' or
                            'unhealthy') and a descriptive 'message'.
        """
        try:
            async with self.session_maker() as session:
                await session.execute(text("SELECT 1"))

            return {"status": "healthy", "message": "Postgres service is running"}
        except Exception as e:
            # Any exception during the health check indicates an unhealthy state.
            return {
                "status": "unhealthy",
                "message": f"Connection to Postgres failed: {e}",
            }

    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Provides an asynchronous database session as a context-managed generator.

        This method ensures that the session is rolled back on any exception
        and closed automatically when the 'with' block is exited.

        Yields:
            AsyncGenerator[AsyncSession, None]: A new asynchronous session.

        Raises:
            Exception: Re-raises any exception that occurs within the
                       session block after attempting a rollback.
        """
        async with self.session_maker() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    async def dispose_engine(self) -> None:
        """
        Disposes of the asynchronous engine's connection pool.

        This is intended to be called during a graceful application
        shutdown. Errors during disposal are suppressed to ensure
        shutdown can complete.
        """
        try:
            await self.engine.dispose()
        except SQLAlchemyError:
            # Suppress errors during shutdown/disposal
            pass
