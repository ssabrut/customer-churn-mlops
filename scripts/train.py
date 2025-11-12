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
from core.config import load_config
from core.utils.converting import DataFrameConverter

try:
    config = load_config()
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


logger.info("Loading entity dataframe spine...")
entity_df = pd.read_parquet(
    f"{project_root}/data/preprocessed/train.parquet",
    columns=["customer_id", "event_timestamp"],
)

logger.info("Getting historical features from Feast...")
training_data = store.get_historical_features(
    entity_df=entity_df,
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
)

training_df = training_data.to_df()
logger.info(f"Successfully retrieved {len(training_df)} feature rows.")
print(training_df.head())

cols_to_drop = ["event_timestamp", "customer_id"]
if "created_timestamp" in training_df.columns:
    cols_to_drop.append("created_timestamp")

training_df = training_df.drop(columns=cols_to_drop)

X = training_df.drop(constant.TARGET, axis=1)
y = training_df[constant.TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

original_columns = X.columns

logger.info("Building model...")
params = {"objective": "binary:logistic", "random_state": 42}
pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=False)),
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
    
    try:
        auc_score = roc_auc_score(y_test, yhat)
    except ValueError:
        auc_score = float("nan")


    mlflow.log_metrics(
        {"f1_score": f1, "accuracy": accuracy, "roc_auc_curve": auc_score}
    )

    logger.info("Logging model...")
    mlflow.sklearn.log_model(
        sk_model=pipeline,
        name=MODEL_NAME,
        registered_model_name=MODEL_NAME,
        input_example=X_train.iloc[:10],
        model_type="json",
    )
    
    logger.info(f"Run ID: {run.info.run_id}")
    logger.success("Training complete.")
