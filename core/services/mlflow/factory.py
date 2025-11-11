from functools import lru_cache

from core.config import load_config
from core.services.mlflow import MLflowClient


@lru_cache(maxsize=1)
def make_mlflow_service() -> MLflowClient:
    settings = load_config()
    return MLflowClient(settings)
