import os
from datetime import timedelta
from typing import Any, Dict

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago

# --- Configuration Constants ---
DAG_ID: str = "churn_drift_detection_pipeline"
DOCKER_IMAGE: str = "churn-mlops-image:latest"
DOCKER_NETWORK: str = "customer-churn-mlops_internal"
TASK_ID: str = "detect_drift"

DEFAULT_ARGS: Dict[str, Any] = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def get_db_environment() -> Dict[str, str]:
    """
    Retrieves and validates database environment variables.

    This function ensures that the Airflow worker environment has the
    necessary credentials to pass to the Docker container.

    Args:
        None

    Returns:
        Dict[str, str]: A dictionary of environment variables.

    Raises:
        EnvironmentError: If critical environment variables are missing.
    """
    required_vars = [
        "APP_DB_USER",
        "APP_DB_PASSWORD",
        "APP_DB_NAME",
        "APP_DB_HOST",
        "APP_DB_PORT",
    ]
    env_vars: Dict[str, str] = {}
    missing_vars = []

    for var in required_vars:
        value = os.environ.get(var)
        if not value:
            missing_vars.append(var)
        else:
            env_vars[var] = value

    if missing_vars:
        # Note: Raising an error here will cause the DAG to show as 'Broken'
        # in the Airflow UI, which is intended for critical config failures.
        raise EnvironmentError(
            f"Missing required environment variables for DAG: {missing_vars}"
        )

    return env_vars


# --- DAG Definition ---
with DAG(
    dag_id=DAG_ID,
    default_args=DEFAULT_ARGS,
    description="Orchestrates the data drift detection job via Docker",
    schedule_interval="@daily",  # Explicit schedule interval
    start_date=days_ago(1),  # Fixed start date for scheduler stability
    catchup=False,
    tags=["drift-detection", "mlops"],
) as dag:

    # 1. Fetch Environment
    # We execute this outside the operator to fail fast if config is missing
    container_env: Dict[str, str] = get_db_environment()

    # 2. Define Task
    detect_drift_task = DockerOperator(
        task_id=TASK_ID,
        image=DOCKER_IMAGE,
        command="python scripts/detect_drift.py",
        environment=container_env,
        network_mode=DOCKER_NETWORK,
        auto_remove=True,  # Clean up container after exit
        mount_tmp_dir=False,  # Prevent mounting temporary dirs if not needed
        tty=True,  # Allocation of pseudo-TTY
        # docker_url="unix://var/run/docker.sock" # Uncomment if needed for local socket mapping
    )

    detect_drift_task
