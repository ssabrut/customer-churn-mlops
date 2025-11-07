from pydantic_settings import BaseSettings, SettingsConfigDict


class DefaultSettings(BaseSettings):
    model_config: SettingsConfigDict = SettingsConfigDict(
        env_file=".env", extra="ignore", frozen=True, env_nested_delimiter="__"
    )


class Settings(DefaultSettings):
    app_version: str = "0.1.0"
    debug: bool = True
    environment: str = "development"
    service_name: str = "churn-api"

    # MLflow config
    mlflow_s3_endpoint_url: str = "http://s3:9000"
    mlflow_s3_ignore_tls: bool = "true"


def get_settings() -> Settings:
    return Settings()


def get_settings() -> Settings:
    return Settings()