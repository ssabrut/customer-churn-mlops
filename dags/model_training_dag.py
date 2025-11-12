from datetime import datetime

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator

db_env_vars = {
    "APP_DB_USER": "admin",
    "APP_DB_PASSWORD": "admin",
    "APP_DB_NAME": "churn",
    "APP_DB_HOST": "app_postgres",
    "APP_DB_PORT": "5432",
}

mlflow_env_vars = {
    "MLFLOW_TRACKING_URI": "http://mlflow:5000",
    "MLFLOW_S3_ENDPOINT_URL": "http://s3:9000",
    "AWS_ACCESS_KEY_ID": "churnadmin",
    "AWS_SECRET_ACCESS_KEY": "churnadmin",
}

with DAG(
    dag_id="churn_model_training_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    tags=["mlops", "model-training", "churn"],
) as dag:
    task_preprocess_data = DockerOperator(
        task_id="preprocess_data",
        image="churn-mlops-image:latest",
        command="python scripts/preprocess_data.py",
        network_mode="customer-churn-mlops_internal",
        environment=db_env_vars,
        do_xcom_push=True,
        auto_remove=True,
        tty=True,
    )

    task_feast_materialize = DockerOperator(
        task_id="feast_materialize",
        image="churn-mlops-image:latest",
        command="python scripts/feast_materialize.py",
        network_mode="customer-churn-mlops_internal",
        environment={
            **db_env_vars,
            "FEAST_REDIS_URL": "redis:6379",
        },
        auto_remove=True,
        tty=True,
        mount_tmp_dir=False
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

    task_preprocess_data >> task_feast_materialize >> task_train_model
