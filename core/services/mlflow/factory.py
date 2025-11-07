from functools import lru_cache

from core.services.mlflow import MLflowClient
from core.config import get_settings

@lru_cache(maxsize=1)
def make_mlflow_service() -> MLflowClient:
    settings = get_settings()
    return MLflowClient(settings)