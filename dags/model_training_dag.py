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
    schedule_interval=None,
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

    task_materialize_features = DockerOperator(
        task_id="materialize_features",
        image="churn-mlops-image:latest",
        command="""
            /bin/bash -c '
            # 1. Pull the JSON payload as a string
            # We wrap the Jinja template in single quotes
            PAYLOAD='{{ ti.xcom_pull(task_ids='preprocess_data') }}'

            # 2. Use jq to safely parse the JSON strings
            START_TS=$(echo "$PAYLOAD" | jq -r ".start")
            END_TS=$(echo "$PAYLOAD" | jq -r ".end")

            echo "Materializing features from $START_TS to $END_TS"
            
            # 3. Run the Feast Materialize command (using positional arguments)
            cd feature_repo && feast materialize "$START_TS" "$END_TS"
            '
        """,
        network_mode='customer-churn-mlops_internal',
        environment={
            'AWS_ACCESS_KEY_ID': '${AWS_ACCESS_KEY_ID}',
            'AWS_SECRET_ACCESS_KEY': '${AWS_SECRET_ACCESS_KEY}'
        },
        mount_tmp_dir=False,
        auto_remove=True,
        tty=True
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

    task_preprocess_data >> task_materialize_features >> task_train_model >> task_promote_model
