from datetime import datetime

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator

db_env_vars = {
    "APP_DB_USER": "${APP_DB_USER}",
    "APP_DB_PASSWORD": "${APP_DB_PASSWORD}",
    "APP_DB_NAME": "${APP_DB_NAME}",
    "APP_DB_HOST": "localhost",
    "APP_DB_PORT": "5435",
}

mlflow_env_vars = {
    "MLFLOW_TRACKING_URI": "${MLFLOW_TRACKING_URI}",
    "MLFLOW_S3_ENDPOINT_URL": "${MLFLOW_S3_ENDPOINT_URL}",
    "AWS_ACCESS_KEY_ID": "${AWS_ACCESS_KEY_ID}",
    "AWS_SECRET_ACCESS_KEY": "${AWS_SECRET_ACCESS_KEY}",
}

with DAG(
    dag_id="churn_model_training_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["mlops", "model-training", "churn"],
) as dag:
    task_populate_db = DockerOperator(
        task_id="populate_database",
        image="churn-mlops-image:latest",
        command="python scripts/populate_db.py",
        network_mode="customer-churn-mlops_internal",
        environment=db_env_vars,
        auto_remove=True,
        tty=True,
    )

    task_preprocess_data = DockerOperator(
        task_id="preprocess_data",
        image="churn-mlops-image:latest",
        command="python scripts/preprocess_data.py",
        network_mode="customer-churn-mlops_internal",
        environment=db_env_vars,
        auto_remove=True,
        tty=True,
    )

    task_train_model = DockerOperator(
        task_id="train_model",
        image="churn-mlops-image:latest",
        command="python scripts/train.py",
        network_mode="customer-churn-mlops_internal",
        environment=mlflow_env_vars,
        auto_remove=True,
        tty=True,
    )

    task_promote_model = DockerOperator(
        task_id="promote_model",
        image="churn-mlops-image:latest",
        command="python scripts/promote_model.py",
        network_mode="customer-churn-mlops_internal",
        environment=mlflow_env_vars,
        auto_remove=True,
        tty=True,
    )

    task_populate_db >> task_preprocess_data >> task_train_model >> task_promote_model
