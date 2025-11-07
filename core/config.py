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
    mlflow_s3_endpoint_url: str = "http://127.0.0.1:9002"


def get_settings() -> Settings:
    return Settings()