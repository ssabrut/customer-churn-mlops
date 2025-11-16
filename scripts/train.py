import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Any, Dict, List

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from feast import FeatureStore
from loguru import logger
from mlflow.entities import Experiment
from mlflow.exceptions import MlflowException
from pandas import DataFrame, Index, Series
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from core import constant
from core.config import load_config
from core.constant import (EXPERIMENT_NAME, FEAST_REPO_PATH,
                           MLFLOW_S3_ENDPOINT_URL, MLFLOW_TRACKING_URI,
                           MODEL_NAME)
from core.utils.converting import DataFrameConverter


def main() -> None:
    """
    Executes the end-to-end model training pipeline.

    This function performs several key steps:
    1.  Loads configuration and validates essential MLflow environment variables.
    2.  Initializes and configures the MLflow tracking server and experiment.
    3.  Initializes the Feast FeatureStore.
    4.  Loads the entity 'spine' (customer IDs and timestamps) from a
        Parquet file.
    5.  Retrieves historical features from Feast to build the training dataset.
    6.  Performs data preprocessing, feature/target splitting, and a
        train-test split.
    7.  Defines a scikit-learn pipeline (Scaler, DataFrameConverter,
        XGBClassifier).
    8.  Starts an MLflow run, trains the model, logs parameters, calculates
        and logs metrics (F1, Accuracy, ROC-AUC).
    9.  Logs the trained pipeline to MLflow and registers it as a new
        model version.

    The script exits with a non-zero status code if any critical
    step fails, such as configuration loading, database connection,
    data retrieval, or model training.

    Returns:
        None
    """

    # --- 1. Load Configuration ---
    try:
        config: Any = load_config()
    except EnvironmentError as e:
        logger.error(f"Configuration failed to load: {e}")
        sys.exit(1)

    # --- 2. MLflow Setup ---
    logger.info("Configuring MLflow...")
    if not MLFLOW_TRACKING_URI or not MLFLOW_S3_ENDPOINT_URL:
        logger.error("MLFLOW_TRACKING_URI or MLFLOW_S3_ENDPOINT_URL is missing.")
        logger.error(
            "MLflow environment variables must be set by the Airflow DockerOperator."
        )
        sys.exit(1)

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        experiment: Experiment | None = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            logger.info(f"Creating new MLflow experiment: {EXPERIMENT_NAME}")
            mlflow.create_experiment(EXPERIMENT_NAME)
        mlflow.set_experiment(EXPERIMENT_NAME)
    except MlflowException as e:
        logger.error(f"Fatal Error: Failed to configure MLflow: {e}")
        sys.exit(1)

    logger.info(f"MLflow configured. Experiment: {EXPERIMENT_NAME}")

    # --- 3. Feast Store Initialization ---
    try:
        store: FeatureStore = FeatureStore(repo_path=FEAST_REPO_PATH)
    except Exception as e:
        logger.error(f"Fatal Error: Failed to initialize Feast FeatureStore: {e}")
        sys.exit(1)

    # --- 4. Load Entity Spine ---
    logger.info("Loading entity dataframe spine...")
    entity_df_path: str = f"{project_root}/data/preprocessed/train.parquet"
    try:
        if not os.path.exists(entity_df_path):
            raise FileNotFoundError(
                f"Entity Parquet file not found at: {entity_df_path}"
            )
        entity_df: DataFrame = pd.read_parquet(
            entity_df_path,
            columns=["customer_id", "event_timestamp"],
        )
    except (FileNotFoundError, Exception) as e:
        logger.error(f"Fatal Error: Failed to read entity Parquet file: {e}")
        sys.exit(1)

    # --- 5. Entity Spine Validation ---
    if entity_df.empty:
        logger.error("Fatal Error: Entity dataframe spine is empty.")
        logger.error("Cannot retrieve historical features.")
        sys.exit(1)

    # --- 6. Get Historical Features ---
    logger.info("Getting historical features from Feast...")
    try:
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
        training_df: DataFrame = training_data.to_df()
    except Exception as e:
        logger.error(f"Fatal Error: Failed to get historical features from Feast: {e}")
        sys.exit(1)

    logger.info(f"Successfully retrieved {len(training_df)} feature rows.")
    print(training_df.head())

    # --- 7. Feature/Target Preparation ---
    if training_df.empty:
        logger.error("Fatal Error: Training dataframe is empty after Feast retrieval.")
        sys.exit(1)

    cols_to_drop: List[str] = ["event_timestamp", "customer_id"]
    if "created_timestamp" in training_df.columns:
        cols_to_drop.append("created_timestamp")

    training_df = training_df.drop(columns=cols_to_drop)

    X: DataFrame = training_df.drop(constant.TARGET, axis=1)
    y: Series = training_df[constant.TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    # --- 8. Train Split Validation ---
    if X_train.empty or y_train.empty:
        logger.error("Fatal Error: Training set is empty after train-test split.")
        logger.error("Check test_size or original dataset size.")
        sys.exit(1)

    original_columns: Index = X.columns

    # --- 9. Model Pipeline & Training ---
    logger.info("Building model...")
    params: Dict[str, Any] = {"objective": "binary:logistic", "random_state": 42}
    pipeline: Pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=False)),
            ("to_dataframe", DataFrameConverter(column_names=original_columns)),
            ("model", XGBClassifier(**params)),
        ]
    )

    logger.info("Starting MLflow run...")
    try:
        with mlflow.start_run() as run:
            run_id: str = run.info.run_id
            logger.info(f"Starting run: {run_id}")

            mlflow.log_params(params)

            logger.info("Training model...")
            pipeline.fit(X_train, y_train)

            logger.info("Evaluating model...")
            yhat: np.ndarray = pipeline.predict(X_test)
            f1: float = f1_score(y_test, yhat)
            accuracy: float = accuracy_score(y_test, yhat)

            try:
                auc_score: float = roc_auc_score(y_test, yhat)
            except ValueError:
                auc_score: float = float("nan")

            mlflow.log_metrics(
                {"f1_score": f1, "accuracy": accuracy, "roc_auc_curve": auc_score}
            )

            logger.info("Logging model...")
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                name=MODEL_NAME,
                registered_model_name=MODEL_NAME,
                input_example=X_train.iloc[:10],
            )

            logger.info(f"Run ID: {run.info.run_id}")
            logger.success("Training complete.")

    except (MlflowException, Exception) as e:
        logger.error(f"Fatal Error: MLflow run failed during training or logging: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
