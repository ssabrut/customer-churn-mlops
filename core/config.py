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

    # MLflow config
    aws_access_key_id: str = Field(..., env="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str = Field(..., env="AWS_SECRET_ACCESS_KEY")
    mlflow_s3_endpoint_url: str = Field(..., env="MLFLOW_S3_ENDPOINT_URL")
    mlflow_tracking_uri: str = Field(..., env="MLFLOW_TRACKING_URI")

    # App Postgres config
    app_db_user: str = Field(..., env="APP_DB_USER")
    app_db_password: str = Field(..., env="APP_DB_PASSWORD")
    app_db_name: str = Field(..., env="APP_DB_NAME")
    app_db_host: str = Field(..., env="APP_DB_HOST")
    app_db_port: int = Field(..., env="APP_DB_PORT")

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
        new_db_host = settings.app_db_host
        new_db_port = settings.app_db_port
        logger.debug("Config: Using internal Docker hosts.")
    else:
        new_mlflow_uri = "http://127.0.0.1:5050"
        new_s3_uri = "http://127.0.0.1:9002"
        new_db_host = "localhost"
        new_db_port = 5435
        logger.debug("Config: Using local host ports.")

    feast_repo_path: str = os.path.join(project_root, settings.feast_repo_name)

    updated_settings = settings.model_copy(
        update={
            "mlflow_uri": new_mlflow_uri,
            "s3_uri": new_s3_uri,
            "db_host": new_db_host,
            "db_port": new_db_port,
            "feast_repo_path": feast_repo_path,
        }
    )

    return updated_settings
