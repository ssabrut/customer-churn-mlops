import os
import mlflow
import mlflow.tracking
from loguru import logger
from httpx import AsyncClient, RequestError, ConnectError, TimeoutException
from mlflow.pyfunc import PyFuncModel
from typing import Optional, Tuple, Dict

from core.config import Settings

class MLflowClient:
    base_url: str
    s3_endpoint_url: str
    _instance: Optional["MLflowClient"] = None
    _initialized: bool = False

    def __init__(self, settings: Settings) -> None:
        if not isinstance(settings, Settings):
            raise TypeError("Argument 'settings' must be an instance of the Settings class")

        self.base_url = settings.mlflow_tracking_url
        self.s3_endpoint_url = settings.mlflow_s3_endpoint_url

        if self._initialized:
            return 

        logger.info("Initializing MLflowClient...")
        self._configure_client()
        
        self.model_cache: dict = {}
        self._client = mlflow.tracking.MlflowClient()
        
        self._initialized = True
        logger.success("MLflowClient initialized.")

    async def health_check(self) -> Dict[str, str]:
        try:
            async with AsyncClient(timeout=5.) as client:
                url = self.base_url
                response = await client.get(url)

                if response.status_code == 200:
                    return {
                        "status": "healthy",
                        "message": "MLflow service is running"
                    }
                else:
                    return {
                        "status": "unhealthy",
                        "message": f"HTTP {response.status_code}"
                    }
        except (ConnectError, TimeoutException) as e:
            return {
                "status": "unhealthy",
                "message": f"Connection to MLflow failed: {e}"
            }
        except RequestError as e:
            return {
                "status": "unhealthy",
                "message": "Request to MLflow failed: {e}"
            }

    def _configure_client(self) -> None:
        if os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"):
            logger.info("Using AWS credentials from environment.")
        else:
            logger.warning("AWS credentials not found in environment.")

        # Set the tracking URI for this session
        mlflow.set_tracking_uri(self.base_url)
        logger.success(f"MLflow configured: URI=[{self.base_url}], S3=[{self.s3_endpoint_url}]")

    @property
    def client(self) -> mlflow.tracking.MlflowClient:
        return self._client

    def get_model_version(self, name: str, version: int) -> str:
        try:
            model = self.client.get_model_version(name=name, version=version)
            return model.version
        except IndexError:
            logger.warning(f"No model found for '{name}' in stage '{version}'.")
            return "N/A"
        except Exception as e:
            logger.error(f"Error getting model version: {e}")
            return "N/A"

    def load_model(self, name: str, version: int) -> Tuple[Optional[PyFuncModel], str]:
        """
        Loads a model from the MLflow registry.
        Uses an in-memory cache to avoid re-loading on every call.
        
        Returns a (model, version) tuple or (None, "N/A") on failure.
        """
        cache_key = f"{name}:{version}"
        
        if cache_key in self.model_cache:
            logger.info(f"Loading model '{cache_key}' from cache.")
            return self.model_cache[cache_key]

        logger.info(f"Loading model '{cache_key}' from MLflow registry...")
        try:
            model_uri = f"models:/{name}/{version}"
            model = mlflow.pyfunc.load_model(model_uri=model_uri)
            
            # Get version for the cache
            version = self.get_model_version(name, version)
            
            # Store in cache
            self.model_cache[cache_key] = (model, version)
            
            logger.success(f"Successfully loaded and cached model version: {version}")
            return model, version
        
        except Exception as e:
            logger.error(f"Model loading failed for '{cache_key}'. Error: {e}")
            return None, "N/A"