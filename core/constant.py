from core.config import load_config

config = load_config()

# Features
TARGET = "Churn"

# App DB
APP_TABlE_NAME = "raw_churn_data"

# MLflow
EXPERIMENT_NAME = "churn_prediction"
MODEL_NAME = "XGBoostChurnModel"
MLFLOW_TRACKING_URI = config.mlflow_uri
MLFLOW_S3_ENDPOINT_URL = config.s3_uri

# Feast feature store
FEAST_REPO_PATH = config.feast_repo_path