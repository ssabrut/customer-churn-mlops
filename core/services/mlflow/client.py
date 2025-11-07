import os
import mlflow
import mlflow.tracking
from httpx import AsyncClient, RequestError, ConnectError, TimeoutException
from mlflow.pyfunc import PyFuncModel
from typing import Optional, Tuple, Dict

from core.config import Settings

class MLflowClient:
    base_url: str
    _instance: Optional["MLflowClient"] = None
    _initialized: bool = False

    def __init__(self, settings: Settings) -> None:
        if not isinstance(settings, Settings):
            raise TypeError("Argument 'settings' must be an instance of the Settings class")

        self.base_url = settings.mlflow_s3_endpoint_url

        if self._initialized:
            return 

        print("Initializing MLflowClient singleton...")
        self._configure_client()
        
        self.model_cache: dict = {}
        self._client = mlflow.tracking.MlflowClient()
        
        self._initialized = True
        print("MLflowClient initialized.")

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
        self.tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        self.s3_endpoint_url = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://s3:9000")
        
        os.environ["MLFLOW_TRACKING_URI"] = self.tracking_uri
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = self.s3_endpoint_url
        
        if os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"):
            print("Using AWS credentials from environment.")
        else:
            print("Warning: AWS credentials not found in environment.")

        # Set the tracking URI for this session
        mlflow.set_tracking_uri(self.tracking_uri)
        print(f"MLflow configured: URI=[{self.tracking_uri}], S3=[{self.s3_endpoint_url}]")

    @property
    def client(self) -> mlflow.tracking.MlflowClient:
        return self._client

    def get_model_version(self, model_name: str, model_stage: str) -> str:
        try:
            mv = self.client.get_latest_versions(model_name, stages=[model_stage])[0]
            return mv.version
        except IndexError:
            print(f"Warning: No model found for '{model_name}' in stage '{model_stage}'.")
            return "N/A"
        except Exception as e:
            print(f"Error getting model version: {e}")
            return "N/A"

    def load_model(self, model_name: str, model_stage: str) -> Tuple[Optional[PyFuncModel], str]:
        """
        Loads a model from the MLflow registry.
        Uses an in-memory cache to avoid re-loading on every call.
        
        Returns a (model, version) tuple or (None, "N/A") on failure.
        """
        cache_key = f"{model_name}:{model_stage}"
        
        if cache_key in self.model_cache:
            print(f"Loading model '{cache_key}' from cache.")
            return self.model_cache[cache_key]

        print(f"Loading model '{cache_key}' from MLflow registry...")
        try:
            model_uri = f"models:/{model_name}/{model_stage}"
            model = mlflow.pyfunc.load_model(model_uri=model_uri)
            
            # Get version for the cache
            version = self.get_model_version(model_name, model_stage)
            
            # Store in cache
            self.model_cache[cache_key] = (model, version)
            
            print(f"Successfully loaded and cached model version: {version}")
            return model, version
        
        except Exception as e:
            print(f"FATAL: Model loading failed for '{cache_key}'. Error: {e}")
            return None, "N/A"