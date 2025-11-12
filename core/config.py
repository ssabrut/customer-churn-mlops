import os
from typing import ClassVar

from loguru import logger
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DefaultSettings(BaseSettings):
    model_config: SettingsConfigDict = SettingsConfigDict(
        env_file=".env", extra="ignore", env_nested_delimiter="__"
    )


class Settings(DefaultSettings):
    app_version: str = "0.1.0"
    debug: bool = True
    environment: str = "development"
    service_name: str = "churn-api"

    is_docker: bool = Field(..., env="IS_DOCKER")

    mlflow_s3_endpoint_url: str = Field(..., env="MLFLOW_S3_ENDPOINT_URL")
    mlflow_tracking_uri: str = Field(..., env="MLFLOW_TRACKING_URI")

    mlflow_s3_endpoint_url_local: str = Field(..., env="MLFLOW_S3_ENDPOINT_URL_LOCAL")
    mlflow_tracking_uri_local: str = Field(..., env="MLFLOW_TRACKING_URI_LOCAL")

    app_db_user: str = Field(..., env="APP_DB_USER")
    app_db_password: str = Field(..., env="APP_DB_PASSWORD")
    app_db_name: str = Field(..., env="APP_DB_NAME")

    db_host: str = "localhost"
    db_port: str = "5435"
    mlflow_uri: str = "http://localhost:5050"
    s3_uri: str = "http://localhost:9002"

    feast_repo_name: ClassVar[str] = "feature_repo"


def get_settings() -> Settings:
    """
    Standard function for FastAPI to load the configuration.
    """
    return Settings()


def load_config(project_root: str = os.getcwd()) -> Settings:
    """
    Loads configuration for isolated scripts, resolving hosts by creating
    a new, updated Settings object.
    """

    settings = get_settings()
    logger.info(f"IS_DOCKER={settings.is_docker}")

    if settings.is_docker:
        new_mlflow_uri = settings.mlflow_tracking_uri
        new_s3_uri = settings.mlflow_s3_endpoint_url
        new_db_host = os.environ.get("APP_DB_HOST", "app_postgres")
        new_db_port = os.environ.get("APP_DB_PORT", "5432")
        logger.debug("Config: Using internal Docker hosts.")
    else:
        new_mlflow_uri = settings.mlflow_tracking_uri_local
        new_s3_uri = settings.mlflow_s3_endpoint_url_local
        new_db_host = os.environ.get("APP_DB_HOST_LOCAL", "localhost")
        new_db_port = os.environ.get("APP_DB_PORT_LOCAL", "5435")
        logger.debug("Config: Using local host ports.")

    updated_settings = settings.model_copy(
        update={
            "mlflow_uri": new_mlflow_uri,
            "s3_uri": new_s3_uri,
            "db_host": new_db_host,
            "db_port": new_db_port,
            "feast_repo_path": os.path.join(project_root, settings.feast_repo_name),
        }
    )

    return updated_settings
