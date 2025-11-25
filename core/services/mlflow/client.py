from typing import Any, Dict, Optional, Tuple

import mlflow
import mlflow.sklearn
import mlflow.tracking
from httpx import AsyncClient, ConnectError, RequestError, TimeoutException
from mlflow.exceptions import MlflowException

from core.config import Settings


class MLflowClient:
    """
    Manages client interactions with an MLflow tracking server and registry.

    This class handles the initialization of the MLflow client,
    provides methods for health checks, and loads models from the
    registry with an in-memory cache.

    Attributes:
        base_url (str): The URI for the MLflow tracking server.
        s3_endpoint_url (str): The URI for the S3-compatible artifact
                               registry.
        model_cache (Dict[str, Tuple[Any, str]]): An in-memory cache
            for loaded models.
    """

    base_url: str
    s3_endpoint_url: str
    model_cache: Dict[str, Tuple[Any, str]]
    _client: mlflow.tracking.MlflowClient

    def __init__(self, settings: Settings) -> None:
        """
        Initializes the MLflowClient and configures the tracking URI.

        Args:
            settings (Settings): The application configuration object.

        Raises:
            TypeError: If 'settings' is not an instance of the Settings class.
            RuntimeError: If configuration of the MLflow client fails (e.g.,
                          invalid URI, connection issues).
        """
        if not isinstance(settings, Settings):
            raise TypeError(
                "Argument 'settings' must be an instance of the Settings class"
            )

        self.base_url = settings.mlflow_uri
        self.s3_endpoint_url = settings.s3_uri
        self.model_cache = {}

        try:
            self._configure_client()
            self._client = mlflow.tracking.MlflowClient(
                tracking_uri=self.base_url, registry_uri=self.base_url
            )
        except (MlflowException, Exception) as e:
            raise RuntimeError(f"Failed to initialize MLflowClient: {e}") from e

    async def health_check(self) -> Dict[str, str]:
        """
        Performs a health check against the MLflow tracking server.

        Returns:
            Dict[str, str]: A dictionary containing 'status' ('healthy' or
                            'unhealthy') and a descriptive 'message'.
        """
        try:
            async with AsyncClient(timeout=5.0) as client:
                url: str = self.base_url
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
            return {"status": "unhealthy", "message": f"Request to MLflow failed: {e}"}

    def _configure_client(self) -> None:
        """
        Sets the global MLflow tracking URI.

        Raises:
            RuntimeError: If setting the tracking URI fails.
        """
        try:
            # Set the tracking URI for this session
            mlflow.set_tracking_uri(self.base_url)
        except MlflowException as e:
            raise RuntimeError(f"Failed to set MLflow tracking URI: {e}") from e

    @property
    def client(self) -> mlflow.tracking.MlflowClient:
        """
        Returns the underlying MLflow tracking client instance.

        Returns:
            mlflow.tracking.MlflowClient: The MLflow client.
        """
        return self._client

    def get_model_version(self, name: str, stage: str) -> Optional[str]:
        """
        Retrieves the latest model version string for a given name and stage.

        Args:
            name (str): The registered model name.
            stage (str): The model stage (e.g., 'Production', 'Staging').

        Returns:
            Optional[str]: The version string (e.g., "1") if found,
                           otherwise None.

        Raises:
            RuntimeError: If an error occurs while communicating with the
                          MLflow server.
        """
        try:
            versions = self.client.get_latest_versions(name=name, stages=[stage])
            if not versions:
                return None  # Model or stage not found
            return versions[0].version
        except MlflowException as e:
            raise RuntimeError(
                f"Error getting model version for '{name}' stage '{stage}': {e}"
            ) from e

    def load_model(self, name: str, version: str) -> Tuple[Any, str]:
        """
        Loads a model from the MLflow registry, using an in-memory cache.

        Args:
            name (str): The registered model name.
            version (str): The model version string.

        Returns:
            Tuple[Any, str]: A tuple containing the loaded model object
                             and its version string.

        Raises:
            RuntimeError: If the model fails to load from the MLflow
                          registry (e.g., model not found, artifact error,
                          unpickling error).
        """
        cache_key: str = f"{name}:{version}"

        if cache_key in self.model_cache:
            return self.model_cache[cache_key]

        try:
            model_uri: str = f"models:/{name}/{version}"
            model: Any = mlflow.sklearn.load_model(model_uri=model_uri)

            self.model_cache[cache_key] = (model, version)
            return model, version

        except (MlflowException, Exception) as e:
            # Catch MlflowException and any other potential error
            # (e.g., unpickling, dependency errors)
            raise RuntimeError(
                f"Model loading failed for '{cache_key}'. Error: {e}"
            ) from e
