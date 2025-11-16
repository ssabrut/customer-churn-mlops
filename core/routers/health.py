from typing import Dict

from fastapi import APIRouter, Depends

from core.config import Settings
from core.dependencies import SettingsDependencies
from core.schemas import HealthResponse, ServiceStatus
from core.services.mlflow import MLflowClient
from core.services.mlflow.factory import make_mlflow_service
from core.services.postgres import PostgresClient
from core.services.postgres.factory import make_postgres_service

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check the health and status of the API service and its dependencies.",
    response_description="Service health information",
    tags=["Health"],
)
async def health_check(
    settings: Settings = Depends(SettingsDependencies),
    postgres_client: PostgresClient = Depends(make_postgres_service),
    mlflow_client: MLflowClient = Depends(make_mlflow_service),
) -> HealthResponse:
    """
    Checks the health and status of the API service and its dependencies.

    This endpoint consumes singleton service clients (PostgreSQL and MLflow)
    provided by the dependency injection system and reports their status.

    Args:
        settings (Settings): The application configuration settings.
        postgres_client (PostgresClient): The singleton PostgreSQL client.
        mlflow_client (MLflowClient): The singleton MLflow client.

    Returns:
        HealthResponse: An object containing the overall application status,
                        version, and individual service health statuses.
    """
    services: Dict[str, ServiceStatus] = {}
    overall_status: str = "ok"

    # Check MLflow health
    try:
        mlflow_health: Dict[str, str] = await mlflow_client.health_check()
        services["mlflow"] = ServiceStatus(
            status=mlflow_health["status"], message=mlflow_health["message"]
        )
        if mlflow_health["status"] != "healthy":
            overall_status = "degraded"
    except Exception as e:
        services["mlflow"] = ServiceStatus(
            status="unhealthy", message=f"MLflow client check failed: {str(e)}"
        )
        overall_status = "degraded"

    # Check Postgres health
    try:
        postgres_health: Dict[str, str] = await postgres_client.health_check()
        services["postgres"] = ServiceStatus(
            status=postgres_health["status"], message=postgres_health["message"]
        )
        if postgres_health["status"] != "healthy":
            overall_status = "degraded"
    except Exception as e:
        services["postgres"] = ServiceStatus(
            status="unhealthy", message=f"Postgres client check failed: {str(e)}"
        )
        overall_status = "degraded"

    return HealthResponse(
        status=overall_status,
        version=settings.app_version,
        environment=settings.environment,
        service_name=settings.service_name,
        services=services,
    )
