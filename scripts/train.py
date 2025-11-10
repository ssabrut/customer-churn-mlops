import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import mlflow
import mlflow.sklearn
import pandas as pd
from feast import FeatureStore
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from core import constant
from core.config import load_mlops_config
from core.utils.converting import DataFrameConverter

try:
    config = load_mlops_config(project_root)
except EnvironmentError as e:
    logger.error(f"Configuration failed to load: {e}")
    sys.exit(1)

MLFLOW_TRACKING_URI = config.mlflow_uri
MLFLOW_S3_ENDPOINT_URL = config.s3_uri
FEAST_REPO_PATH = config.feast_repo_path
EXPERIMENT_NAME = "churn_prediction"
MODEL_NAME = "XGBoostChurnModel"

store = FeatureStore(repo_path=FEAST_REPO_PATH)

if not MLFLOW_TRACKING_URI or not MLFLOW_S3_ENDPOINT_URL:
    logger.error("MLFLOW_TRACKING_URI or MLFLOW_S3_ENDPOINT_URL is missing.")
    raise EnvironmentError(
        "MLflow environment variables must be set by the Airflow DockerOperator."
    )

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    logger.info(f"Creating new MLflow experiment: {EXPERIMENT_NAME}")
    mlflow.create_experiment(EXPERIMENT_NAME)

mlflow.set_experiment(EXPERIMENT_NAME)
logger.info(f"MLflow configured. Experiment: {EXPERIMENT_NAME}")


logger.info("Loading data...")
df = pd.read_parquet(
    f"{project_root}/data/preprocessed/train.parquet",
    columns=["customer_id", "event_timestamp"],
)
target_df = pd.read_parquet(
    f"{project_root}/data/preprocessed/train.parquet", columns=[constant.TARGET]
)

training_data = store.get_historical_features(
    entity_df=df,
    features=[
        "customer_features:Age",
        "customer_features:Support Calls",
        "customer_features:Payment Delay",
        "customer_features:Total Spend",
        "customer_features:Last Interaction",
        "customer_features:Churn",
        "customer_features:Male",
        "customer_features:Age_Group",
        "customer_features:Interaction_Frequency",
    ],
).to_df()

X_full = training_data.drop(["customer_id", constant.TARGET], axis=1)
y_full = target_df[constant.TARGET]

combined_df = pd.concat([X_full, y_full], axis=1)

cols_to_drop = ["event_timestamp", "created_timestamp", "customer_id"]
for col in cols_to_drop:
    if col in combined_df.columns:
        combined_df = combined_df.drop([col], axis=1)

X = combined_df.drop(constant.TARGET, axis=1)
y = combined_df[constant.TARGET]

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

    logger.info("Logging model...")
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
