import sys
from loguru import logger
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession
from typing import AsyncGenerator

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
            engine = create_async_engine(self.base_url, echo=settings.debug)
            self.session_factory = async_sessionmaker(engine, expire_on_commit=False)
            
            logger.success(f"Database engine created for {settings.db_host}:{settings.db_port}")
        except Exception as e:
            logger.error(f"Failed to create database engine: {e}")
            sys.exit(1)

        logger.info("Shutting down... Disposing database engine.")
        engine.dispose()

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