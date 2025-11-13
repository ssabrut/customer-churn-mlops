import sys
from typing import AsyncGenerator, Dict

from loguru import logger
from sqlalchemy import text, create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from core.config import Settings


class PostgresClient:
    base_url: str

    def __init__(self, settings: Settings) -> None:
        if not isinstance(settings, Settings):
            raise TypeError(
                "Argument 'settings' must be an instance of the Settings class"
            )

        self.base_url = f"postgresql+asyncpg://{settings.app_db_user}:{settings.app_db_password}@{settings.db_host}:{settings.db_port}/{settings.app_db_name}"

        try:
            self.engine = create_async_engine(self.base_url, echo=settings.debug)
            self.session_factory = async_sessionmaker(
                self.engine, expire_on_commit=False
            )

            logger.success(
                f"Database engine created for {settings.db_host}:{settings.db_port}"
            )
        except Exception as e:
            logger.error(f"Failed to create database engine: {e}")
            sys.exit(1)

    def get_sync_engine(self):
        sync_url = self.base_url.replace("+asyncpg", "")
        
        try:
            return create_engine(sync_url)
        except Exception as e:
            logger.error(f"Failed to create synchronous database engine: {e}")
            sys.exit(1)

    async def health_check(self) -> Dict[str, str]:
        try:
            async with self.session_factory() as session:
                await session.execute(text("SELECT 1"))

            return {"status": "healthy", "message": "Postgres service is running"}
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"Connection to Postgres failed: {e}",
            }

    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        async with self.session_factory() as session:
            try:

                yield session
            except Exception as e:
                logger.error(f"Database session error: {e}")
                await session.rollback()
                raise
            finally:
                await session.close()

    async def dispose_engine(self):
        logger.info("Shutting down... Disposing database engine.")
        await self.engine.dispose()