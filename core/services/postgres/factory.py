from functools import lru_cache

from core.config import load_config
from core.services.postgres import PostgresClient


@lru_cache(maxsize=1)
def make_postgres_service() -> PostgresClient:
    settings = load_config()
    return PostgresClient(settings)
