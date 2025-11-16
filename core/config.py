import os
from typing import ClassVar

from loguru import logger
from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class DefaultSettings(BaseSettings):
    """
    Base configuration class.

    Defines the Pydantic model configuration, such as the .env file name
    and the delimiter for nested environment variables.
    """

    model_config: SettingsConfigDict = SettingsConfigDict(
        env_file=".env", extra="ignore", env_nested_delimiter="__"
    )


class Settings(DefaultSettings):
    """
    Application settings class.

    Defines all configuration variables for the application, loading them
    from environment variables or .env files. Variables marked with '...'
    are required, and their absence will cause a validation error.
    """

    # Application metadata
    app_version: str = "0.1.0"
    debug: bool = True
    environment: str = "development"
    service_name: str = "churn-api"

    # Environment flag
    is_docker: bool = Field(..., env="IS_DOCKER")

    # MLflow config (required)
    aws_access_key_id: str = Field(..., env="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str = Field(..., env="AWS_SECRET_ACCESS_KEY")
    mlflow_s3_endpoint_url: str = Field(..., env="MLFLOW_S3_ENDPOINT_URL")
    mlflow_tracking_uri: str = Field(..., env="MLFLOW_TRACKING_URI")

    # App Postgres config (required)
    app_db_user: str = Field(..., env="APP_DB_USER")
    app_db_password: str = Field(..., env="APP_DB_PASSWORD")
    app_db_name: str = Field(..., env="APP_DB_NAME")
    app_db_host: str = Field(..., env="APP_DB_HOST")
    app_db_port: int = Field(..., env="APP_DB_PORT")

    # Gradio config
    fastapi_url: str = Field(..., env="FASTAPI_URL")
    
    # --- Modular variables (overwritten by load_config) ---
    # These have defaults for non-Docker, but are updated by load_config
    # based on the IS_DOCKER flag.
    db_host: str = "localhost"
    db_port: str = "5435"
    mlflow_uri: str = "http://localhost:5050"
    s3_uri: str = "http://localhost:9002"

    # --- Static variables ---
    feast_repo_name: ClassVar[str] = "feature_repo"

    # --- Dynamic variables (set by load_config) ---
    # Excluded from .env loading, set at runtime by load_config
    feast_repo_path: str = Field("", exclude=True)


def get_settings() -> Settings:
    """
    Standard function for FastAPI to load the configuration.

    Instantiates the Settings object, which triggers Pydantic's
    validation and loading of environment variables.

    Returns:
        Settings: The application settings object.

    Raises:
        EnvironmentError: If a required environment variable is missing,
                          wrapping Pydantic's 'ValidationError'.
    """
    try:
        return Settings()
    except ValidationError as e:
        logger.error(
            "Failed to load required settings. "
            "Ensure all required .env variables are set."
        )
        raise EnvironmentError(f"Missing required environment variables: {e}")


def load_config(project_root: str = os.getcwd()) -> Settings:
    """
    Loads configuration for isolated scripts, resolving hosts by creating
    a new, updated Settings object based on the 'IS_DOCKER' flag.

    This function first loads the base settings, then intelligently
    overwrites specific fields (like 'db_host' or 'mlflow_uri')
    to point to Docker internal hosts or localhost, depending on
    the execution environment. It also constructs the absolute path
    to the Feast repository.

    Args:
        project_root (str): The absolute path to the project's root
                            directory. Defaults to the current working
                            directory.

    Returns:
        Settings: A new, updated Settings object with environment-specific
                  hosts and paths.

    Raises:
        EnvironmentError: If the initial 'get_settings()' call fails
                          due to missing environment variables.
    """

    # get_settings() may raise EnvironmentError if required vars are missing
    settings = get_settings()
    logger.info(f"IS_DOCKER={settings.is_docker}")

    # Declare variables for new values
    new_mlflow_uri: str
    new_s3_uri: str
    new_db_host: str
    new_db_port: str
    new_fastapi_url: str

    if settings.is_docker:
        new_mlflow_uri = settings.mlflow_tracking_uri
        new_s3_uri = settings.mlflow_s3_endpoint_url
        new_db_host = settings.app_db_host
        new_db_port = str(settings.app_db_port)  # Ensure type match
        new_fastapi_url = settings.fastapi_url
        logger.debug("Config: Using internal Docker hosts.")
    else:
        new_mlflow_uri = "http://127.0.0.1:5050"
        new_s3_uri = "http://127.0.0.1:9002"
        new_db_host = "localhost"
        new_db_port = "5435"  # Ensure type match
        new_fastapi_url = "http://localhost:8000"
        logger.debug("Config: Using local host ports.")

    feast_repo_path: str = os.path.join(project_root, settings.feast_repo_name)

    # Use model_copy to create an updated, validated instance
    updated_settings = settings.model_copy(
        update={
            "mlflow_uri": new_mlflow_uri,
            "s3_uri": new_s3_uri,
            "db_host": new_db_host,
            "db_port": new_db_port,
            "feast_repo_path": feast_repo_path,
            "fastapi_url": new_fastapi_url
        }
    )

    return updated_settings
