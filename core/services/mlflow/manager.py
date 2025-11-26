import asyncio
from loguru import logger
from typing import Any, Tuple, Optional

from core.services.mlflow import MLflowClient
from core.constant import MODEL_NAME

class ModelManager:
    def __init__(self, mlflow_client: MLflowClient):
        self.client = mlflow_client

        # Prod model state
        self.model = None
        self.version = "N/A"

        # Shadow model state
        self.shadow_model = None
        self.shadow_version = "N/A"
        
        self.lock = asyncio.Lock() # Prevent reading while writing

    async def _load_specific_model(self, stage: str, is_shadow: bool):
        try:
            version = self.client.get_model_version(MODEL_NAME, stage)

            if not version:
                if not is_shadow:
                    logger.warning(f"No model found in {stage}")
                return

            current_version = self.shadow_version if is_shadow else self.version

            if version == current_version:
                return

            logger.info(f"Loading new {stage} model (v{version})...")

            loop = asyncio.get_running_loop()
            model, _ = await loop.run_in_executor(
                None,
                self.client.load_model,
                MODEL_NAME,
                version
            )

            async with self.lock:
                if is_shadow:
                    self.shadow_model = model
                    self.shadow_version = version
                else:
                    self.model = model
                    self.version = version

            logger.success(f"Loaded {stage} model v{version} (Shadow={is_shadow})")
        except Exception as e:
            logger.error(f"Failed to load {stage} model: {e}")

    async def get_production_model(self) -> Tuple[Any, str]:
        async with self.lock:
            return self.model, self.version

    async def get_shadow_model(self) -> Tuple[Optional[Any], str]:
        async with self.lock:
            return self.shadow_model, self.shadow_version

    async def load_models(self):
        await self._load_specific_model("Production", is_shadow=False)
        await self._load_specific_model("Staging", is_shadow=True)