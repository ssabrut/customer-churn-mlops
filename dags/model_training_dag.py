"""
Airflow DAG for the complete Churn Model Training Pipeline.

This DAG orchestrates the data preprocessing, feature materialization,
and model training tasks as Docker containers.

Environment variables are NOT hardcoded. They are loaded at DAG parse time
from a .env file located in the project root by using python-dotenv.
A validation step ensures all required variables are present, otherwise,
the DAG will fail to parse.
"""

import logging
import os
from datetime import datetime
from typing import Dict, List

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from dotenv import load_dotenv

# Set up logging
logger = logging.getLogger(__name__)

# --- Environment Variable Loading and Validation ---
try:
    dag_file_path: str = os.path.abspath(__file__)
    project_root: str = os.path.dirname(os.path.dirname(dag_file_path))
    env_path: str = os.path.join(project_root, ".env")
except NameError:
    logger.warning("__file__ not defined. Assuming .env is in a standard location.")
    project_root = "/opt/airflow"
    env_path = os.path.join(project_root, ".env")

# 2. Load the .env file
if not os.path.exists(env_path):
    logger.warning(
        f".env file not found at {env_path}. "
        "Relying on environment variables set by other means."
    )
else:
    load_dotenv(env_path)
    logger.info(f"Loaded environment variables from {env_path}")

# 3. Define REQUIRED keys for each task
DB_ENV_KEYS: List[str] = [
    "APP_DB_USER",
    "APP_DB_PASSWORD",
    "APP_DB_NAME",
    "APP_DB_HOST",
    "APP_DB_PORT",
]

MLFLOW_ENV_KEYS: List[str] = [
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "MLFLOW_TRACKING_URI",
    "MLFLOW_S3_ENDPOINT_URL",
]

FEAST_ENV_KEYS: List[str] = ["FEAST_REDIS_URL"]

# 4. Build environment dictionaries and validate
try:
    db_env_vars: Dict[str, str] = {key: os.environ[key] for key in DB_ENV_KEYS}
    mlflow_env_vars: Dict[str, str] = {key: os.environ[key] for key in MLFLOW_ENV_KEYS}
    feast_env_vars: Dict[str, str] = {key: os.environ[key] for key in FEAST_ENV_KEYS}
except KeyError as e:
    logger.error(
        f"Fatal Error: Missing required environment variable: {e}. "
        f"Ensure '{e.args[0]}' is in the .env file or set in the environment."
    )
    raise ValueError(f"Missing required environment variable: {e}")


# --- DAG Definition ---
with DAG(
    dag_id="churn_model_training_pipeline",
    start_date=datetime.now(),
    schedule_interval="@daily",
    catchup=False,
    tags=["preprocess-data", "feast-materialize", "model-training"],
) as dag:
    task_preprocess_data = DockerOperator(
        task_id="preprocess_data",
        image="churn-mlops-image:latest",
        command="python scripts/preprocess_data.py",
        network_mode="customer-churn-mlops_internal",
        environment=db_env_vars,
        auto_remove=True,
        mount_tmp_dir=False,
        tty=True,
    )

    task_feast_materialize = DockerOperator(
        task_id="feast_materialize",
        image="churn-mlops-image:latest",
        command="python scripts/feast_materialize.py",
        network_mode="customer-churn-mlops_internal",
        environment={
            **db_env_vars,
            **feast_env_vars,
        },
        auto_remove=True,
        tty=True,
        mount_tmp_dir=False,
    )

    task_train_model = DockerOperator(
        task_id="train_model",
        image="churn-mlops-image:latest",
        command="python scripts/train.py",
        network_mode="customer-churn-mlops_internal",
        environment=mlflow_env_vars,
        docker_url="unix://var/run/docker.sock",
        auto_remove=True,
        mount_tmp_dir=False,
        tty=True,
    )

    # Define task dependencies
    task_preprocess_data >> task_feast_materialize >> task_train_model
