from dotenv import load_dotenv

load_dotenv()

import sys
from loguru import logger
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import AsyncIterator
from pydantic import ValidationError
from contextlib import asynccontextmanager

from core.config import Settings, get_settings
from core.services.mlflow.factory import make_mlflow_service
from core.services.mlflow import MLflowClient

try:
    settings: Settings = get_settings()
except ValidationError as e:
    logger.error(f"Application configuration is invalid.\n{e}")
    sys.exit(1)

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    mlflow_client: MLflowClient = make_mlflow_service()
    model, _ = mlflow_client.load_model("XGBoostChurnModel", version=1)
    
    app.state.settings = settings
    app.state.model = model
    yield

app = FastAPI(
    title=settings.service_name,
    description="A FastAPI service for a customer churn prediction",
    version=settings.app_version,
    root_path="/api/v1",
    lifespan=lifespan
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
    return {
        "status": 200,
        "message": "ok"
    }

if __name__ == "__main__":
    import uvicorn

    # Run the application server using uvicorn.
    uvicorn.run(app, port=8000, host="0.0.0.0", timeout_keep_alive=60)