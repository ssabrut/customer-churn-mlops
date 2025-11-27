import os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

env = {
    "APP_DB_USER": os.environ.get("APP_DB_USER"),
    "APP_DB_PASSWORD": os.environ.get("APP_DB_PASSWORD"),
    "APP_DB_NAME": os.environ.get("APP_DB_NAME"),
    "APP_DB_HOST": os.environ.get("APP_DB_HOST"),
    "APP_DB_PORT": os.environ.get("APP_DB_PORT")
}

with DAG(
    "churn_drift_detection_pipeline",
    default_args=default_args,
    description="Check for data drift between Prod and Train data",
    schedule="@daily",
    start_date=datetime.now(),
    catchup=False,
    tags=["drift-detection"]
) as dag:
    detect_drift = DockerOperator(
        task_id="detect_drift",
        image="churn-mlops-image:latest",
        command="python scripts/detect_drift.py",
        environment=env,
        network_mode="customer-churn-mlops_internal",
        auto_remove=True,
        mount_tmp_dir=False,
        tty=True
    )

    detect_drift