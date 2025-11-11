from functools import lru_cache

from core.config import get_settings
from core.services.postgres import PostgresClient


@lru_cache(maxsize=1)
def make_postgres_service() -> PostgresClient:
    settings = get_settings()
    return PostgresClient(settings)
