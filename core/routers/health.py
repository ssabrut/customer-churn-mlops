from typing import Dict

from fastapi import APIRouter

from core.dependencies import SettingsDependencies
from core.schemas import HealthResponse, ServiceStatus
from core.services.mlflow import MLflowClient
from core.services.postgres import PostgresClient

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check the health and status of the API service and its dependencies.",
    response_description="Service health information",
    tags=["Health"],
)
async def health_check(settings: SettingsDependencies) -> HealthResponse:
    services: Dict[str, ServiceStatus] = {}
    overall_status: str = "ok"

    try:
        mlflow_client: MLflowClient = MLflowClient(settings)
        postgres_health: Dict[str, str] = await mlflow_client.health_check()
        services["mlflow"] = ServiceStatus(
            status=postgres_health["status"], message=postgres_health["message"]
        )
        if postgres_health["status"] != "healthy":
            overall_status = "degraded"
    except Exception as e:
        services["mlflow"] = ServiceStatus(
            status="unhealthy", message=f"MLflow client check failed: {str(e)}"
        )
        overall_status = "degraded"

    try:
        postgres_client: PostgresClient = PostgresClient(settings)
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
