import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import ValidationError

from core.config import load_config
from core.routers.churn import router as churn_router
from core.routers.health import router as health_router
from core.routers.data import router as data_router
from core.services.mlflow import MLflowClient
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
    logger.info("Polling for 'Production' model version...")
    production_version = "N/A"

    logger.success(
        f"Found 'Production' model version: {production_version}. Loading..."
    )
    try:
        model, version = mlflow_client.load_model("XGBoostChurnModel", version=1)

        app.state.model = model
        app.state.model_version = 1
        logger.success(f"Successfully loaded model version '{version}'.")
    except Exception as e:
        logger.critical(f"Failed to load model version {production_version}: {e}")
        app.state.model = None
        app.state.model_version = "N/A (Load Failed)"

    try:
        from feast import FeatureStore

        feast_store = FeatureStore(repo_path=settings.feast_repo_path)
        app.state.feast_store = feast_store
        logger.success("Feast FeatureStore successfully initialized.")
    except Exception as e:
        logger.error(f"Feast Store initialization failed: {e}")

    yield


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


@app.get("/")
def root():
    return {"status": 200, "message": "ok"}


app.include_router(churn_router, tags=["prediction"])
app.include_router(health_router, tags=["health"])
app.include_router(data_router, tags=["data"])

if __name__ == "__main__":
    import uvicorn

    # Run the application server using uvicorn.
    uvicorn.run(app, port=8000, host="0.0.0.0", timeout_keep_alive=60)
