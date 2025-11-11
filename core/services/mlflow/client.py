import os
from typing import Dict, Optional, Tuple, Any

import mlflow
import mlflow.tracking
import mlflow.sklearn
from httpx import AsyncClient, ConnectError, RequestError, TimeoutException
from loguru import logger

from core.config import Settings


class MLflowClient:
    base_url: str
    s3_endpoint_url: str

    def __init__(self, settings: Settings) -> None:
        if not isinstance(settings, Settings):
            raise TypeError(
                "Argument 'settings' must be an instance of the Settings class"
            )

        self.base_url = settings.mlflow_uri
        self.s3_endpoint_url = settings.s3_uri

        logger.info("Initializing MLflowClient...")
        self._configure_client()

        self.model_cache: dict = {}
        self._client = mlflow.tracking.MlflowClient(tracking_uri=self.base_url, registry_uri=self.s3_endpoint_url)

        logger.success("MLflowClient initialized.")

    async def health_check(self) -> Dict[str, str]:
        try:
            async with AsyncClient(timeout=5.0) as client:
                url = self.base_url
                response = await client.get(url)

                if response.status_code == 200:
                    return {"status": "healthy", "message": "MLflow service is running"}
                else:
                    return {
                        "status": "unhealthy",
                        "message": f"HTTP {response.status_code}",
                    }
        except (ConnectError, TimeoutException) as e:
            return {
                "status": "unhealthy",
                "message": f"Connection to MLflow failed: {e}",
            }
        except RequestError as e:
            return {"status": "unhealthy", "message": "Request to MLflow failed: {e}"}

    def _configure_client(self) -> None:
        if os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"):
            logger.info("Using AWS credentials from environment.")
        else:
            logger.warning("AWS credentials not found in environment.")

        # Set the tracking URI for this session
        mlflow.set_tracking_uri(self.base_url)
        logger.success(
            f"MLflow configured: URI=[{self.base_url}], S3=[{self.s3_endpoint_url}]"
        )

    @property
    def client(self) -> mlflow.tracking.MlflowClient:
        return self._client

    def get_model_version(self, name: str, stage: str) -> str:
        try:
            versions = self.client.get_latest_versions(name=name, stages=[stage])
            if not versions:
                logger.warning(f"No model found for '{name}' in stage '{stage}'.")
                return "N/A"
            return versions[0].version  # Return the version number as a string
        except Exception as e:
            logger.error(f"Error getting model version for stage '{stage}': {e}")
            return "N/A"

    def load_model(self, name: str, version: str) -> Tuple[Optional[Any], str]:
        cache_key = f"{name}:{version}"

        if cache_key in self.model_cache:
            logger.info(f"Loading model '{cache_key}' from cache.")
            return self.model_cache[cache_key]

        logger.info(f"Loading model '{cache_key}' from MLflow registry...")
        try:
            model_uri = f"models:/{name}/{version}"
            model = mlflow.sklearn.load_model(model_uri=model_uri)

            self.model_cache[cache_key] = (model, version)

            logger.success(f"Successfully loaded and cached model version: {version}")
            return model, version

        except Exception as e:
            logger.error(f"Model loading failed for '{cache_key}'. Error: {e}")
            return None, "N/A"