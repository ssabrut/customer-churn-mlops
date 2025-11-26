import asyncio
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import ValidationError
from prometheus_fastapi_instrumentator import Instrumentator

from core.config import load_config
from core.routers.churn import router as churn_router
from core.routers.data import router as data_router
from core.routers.health import router as health_router
from core.services.mlflow import MLflowClient, ModelManager
from core.services.mlflow.factory import make_mlflow_service

try:
    settings = load_config()
except ValidationError as e:
    logger.error(f"Application configuration is invalid.\n{e}")
    sys.exit(1)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.model = None
    app.state.model_version = "N/A"
    app.state.feast_store = None
    app.state.settings = settings

    mlflow_client: MLflowClient = make_mlflow_service()
    model_manager = ModelManager(mlflow_client)

    await model_manager.load_latest_model()

    app.state.model_manager = model_manager

    async def poll_for_updates():
        while True:
            await asyncio.sleep(60)
            await model_manager.load_latest_model()

    poller_task = asyncio.create_task(poll_for_updates())
    logger.info("Started background model poller")

    # try:
    #     latest_model = mlflow_client.client.get_latest_versions("XGBoostChurnModel")

    #     if not latest_model:
    #         logger.warning(
    #             "No Production model found. Falling back to Staging or None."
    #         )
    #         app.state.model = None
    #     else:
    #         prod_version = latest_model[0].version
    #         logger.info(f"Loading Production model version: {prod_version}")

    #         model, version = mlflow_client.load_model(
    #             "XGBoostChurnModel", version=prod_version
    #         )

    #         app.state.model = model
    #         app.state.model_version = version
    #         logger.success(f"Successfully loaded model version '{version}'.")
    # except Exception as e:
    #     logger.critical(f"Failed to load model: {e}")

    try:
        from feast import FeatureStore

        feast_store = FeatureStore(repo_path=settings.feast_repo_path)
        app.state.feast_store = feast_store
        logger.success("Feast FeatureStore successfully initialized.")
    except Exception as e:
        logger.error(f"Feast Store initialization failed: {e}")

    yield

    poller_task.cancel()
    try:
        await poller_task
    except asyncio.CancelledError:
        logger.info("Background poller stopped")


app = FastAPI(
    title=settings.service_name,
    description="A FastAPI service for a customer churn prediction",
    version=settings.app_version,
    root_path="/api/v1",
    lifespan=lifespan,
)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app)


@app.get("/")
def root():
    return {"status": 200, "message": "ok"}


app.include_router(churn_router)
app.include_router(health_router)
app.include_router(data_router)

if __name__ == "__main__":
    import uvicorn

    # Run the application server using uvicorn.
    uvicorn.run(app, port=8000, host="0.0.0.0", timeout_keep_alive=60)
