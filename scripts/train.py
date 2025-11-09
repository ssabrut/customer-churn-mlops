import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import mlflow
import mlflow.sklearn
import pandas as pd
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from core import constant
from core.utils.converting import DataFrameConverter

MLFLOW_S3_ENDPOINT_URL = os.environ.get(
    "MLFLOW_S3_ENDPOINT_URL", "http://127.0.0.1:9002"
)
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5050")
EXPERIMENT_NAME = "churn_prediction"
MODEL_NAME = "XGBoostChurnModel"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    logger.info(f"Creating new MLflow experiment: {EXPERIMENT_NAME}")
    mlflow.create_experiment(EXPERIMENT_NAME)

mlflow.set_experiment(EXPERIMENT_NAME)
logger.info(f"MLflow configured. Experiment: {EXPERIMENT_NAME}")

logger.info("Loading data...")
df = pd.read_parquet("data/preprocessed/train.parquet")
cols_to_drop = ["event_timestamp", "created_timestamp"]
if all(col in df.columns for col in cols_to_drop):
    df = df.drop(cols_to_drop, axis=1)

X = df.drop(constant.TARGET, axis=1)
y = df[constant.TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

original_columns = X.columns

logger.info("Building model...")
params = {"objective": "binary:logistic", "random_state": 42}
pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("to_dataframe", DataFrameConverter(column_names=original_columns)),
        ("model", XGBClassifier(**params)),
    ]
)

with mlflow.start_run() as run:
    run_id = run.info.run_id
    logger.info(f"Starting run: {run_id}")

    mlflow.log_params(params)

    logger.info("Training model...")
    pipeline.fit(X_train, y_train)

    logger.info("Evaluating model...")
    yhat = pipeline.predict(X_test)
    f1 = f1_score(y_test, yhat)
    accuracy = accuracy_score(y_test, yhat)
    auc_score = roc_auc_score(y_test, yhat)

    mlflow.log_metrics(
        {"f1_score": f1, "accuracy": accuracy, "roc_auc_curve": auc_score}
    )

    mlflow.sklearn.log_model(
        sk_model=pipeline,
        name=MODEL_NAME,
        registered_model_name=MODEL_NAME,
        input_example=X_train.head(5),
        model_type="json",
    )

    logger.info("Transitioning new model to 'Staging'...")
    client = mlflow.MlflowClient()

    new_version = client.search_model_versions(f"run_id='{run_id}'")[0]
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=new_version.version,
        stage="Staging",
        archive_existing_versions=False,
    )

    logger.success(f"Transitioned model version {new_version.version} to 'Staging'.")
    logger.info(f"Run ID: {run.info.run_id}")
    logger.success("Training complete.")
