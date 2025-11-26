import asyncio
from loguru import logger
from typing import Any, Tuple

from core.services.mlflow import MLflowClient
from core.constant import MODEL_NAME

class ModelManager:
    def __init__(self, mlflow_client: MLflowClient):
        self.client = mlflow_client
        self.model = None
        self.version = "N/A"
        self.lock = asyncio.locks() # Prevent reading while writing

    async def load_latest_model(self):
        try:
            latest_models = self.client.client.get_latest_versions(name=MODEL_NAME)

            if not latest_models:
                logger.warning("No production model found.")
                return

            new_version = latest_models[0].version

            if new_version == self.version:
                return

            logger.info(f"New production model detected (v{new_version}). Loading...")

            loop = asyncio.get_running_loop()
            new_model, _ = await loop.run_in_executor(
                None,
                self.client.load_model,
                MODEL_NAME,
                new_version
            )

            # Hot swap model safely
            async with self.lock:
                self.model = new_model
                self.version = new_version

            logger.success(f"Successfully hot-swapped to model v{new_version}")
        except Exception as e:
            logger.error(f"Error during model reload: {e}")

    async def get_model(self) -> Tuple[Any, str]:
        async with self.lock:
            return self.model, self.version