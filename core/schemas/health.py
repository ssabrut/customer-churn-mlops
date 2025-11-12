from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ServiceStatus(BaseModel):
    status: str = Field(..., description="Service Status", examples=["healthy"])
    message: Optional[str] = Field(
        None, description="Status Message", examples=["connected successfully"]
    )


class HealthResponse(BaseModel):
    status: str = Field(..., description="Overall health status", example="ok")
    version: str = Field(..., description="Application version", example="0.1.0")
    environment: str = Field(
        ..., description="Deployment environment", example="development"
    )
    service_name: str = Field(
        ..., description="Service identifier", example="churn-prediction-api"
    )
    services: Optional[Dict[str, ServiceStatus]] = Field(
        None, description="Individual service statuses"
    )

    class Config:
        json_schema_extra: Dict[str, Any] = {
            "example": {
                "status": "ok",
                "version": "0.1.0",
                "environment": "development",
                "service_name": "churn-prediction-api",
                "services": {
                    "database": {
                        "status": "healthy",
                        "message": "Connected successfully",
                    }
                },
            }
        }
