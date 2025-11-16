from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ServiceStatus(BaseModel):
    """
    Represents the health status of an individual downstream service.

    Attributes:
        status (str): The operational status of the service (e.g.,
                      "healthy", "unhealthy").
        message (Optional[str]): An optional message providing additional
                                 details about the service's status.
    """

    status: str = Field(..., description="Service Status", examples=["healthy"])
    message: Optional[str] = Field(
        None, description="Status Message", examples=["connected successfully"]
    )


class HealthResponse(BaseModel):
    """
    Defines the structured response for the application's health check endpoint.

    This model provides overall application status, versioning information,
    and a breakdown of individual downstream service statuses.

    Attributes:
        status (str): The overall health status of the application
                      (e.g., "ok", "degraded").
        version (str): The current deployed version of the application.
        environment (str): The environment in which the application is
                           running (e.g., "development", "production").
        service_name (str): The unique identifier for this service.
        services (Optional[Dict[str, ServiceStatus]]): A dictionary mapping
            service names (e.g., "database", "mlflow") to their
            individual ServiceStatus.
    """

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
