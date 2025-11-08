from dotenv import load_dotenv

load_dotenv()

import sys
import pandas as pd
from loguru import logger
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import AsyncIterator
from pydantic import ValidationError
from contextlib import asynccontextmanager
from sklearn.preprocessing import StandardScaler

from core.config import Settings, get_settings
from core.transformer import ChurnFeatureTransformer
from core.services.mlflow.factory import make_mlflow_service
from core.services.mlflow import MLflowClient
from core.routers.churn import router as churn_router

try:
    settings: Settings = get_settings()
except ValidationError as e:
    logger.error(f"Application configuration is invalid.\n{e}")
    sys.exit(1)

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    mlflow_client: MLflowClient = make_mlflow_service()
    model, model_version = mlflow_client.load_model("XGBoostChurnModel", version=11)

    data = pd.read_csv("data/raw/train.csv")
    transformer = ChurnFeatureTransformer()
    scaler = StandardScaler()
    
    transformed_data = transformer.transform(data)
    scaler.fit(transformed_data)
    
    app.state.settings = settings
    app.state.model = model
    app.state.model_version = model_version
    app.state.scaler = scaler
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

app.include_router(churn_router)

if __name__ == "__main__":
    import uvicorn

    # Run the application server using uvicorn.
    uvicorn.run(app, port=8000, host="0.0.0.0", timeout_keep_alive=60)