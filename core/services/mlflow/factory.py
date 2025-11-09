from functools import lru_cache

from core.config import get_settings
from core.services.mlflow import MLflowClient


@lru_cache(maxsize=1)
def make_mlflow_service() -> MLflowClient:
    settings = get_settings()
    return MLflowClient(settings)
