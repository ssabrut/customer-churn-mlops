import os
import sys
from typing import Any, Dict, List, Optional

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from feast import FeatureStore
from loguru import logger
from mlflow.entities import Experiment
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from pandas import DataFrame, Index, Series
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Ensure project root is in sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.utils.converting import DataFrameConverter

# Constants
EXPERIMENT_NAME: str = "churn_prediction"
TARGET_COLUMN: str = "Churn"
MODEL_NAME: str = "XGBoostChurnModel"
FEAST_REPO_PATH: str = "feature_repo"
ENTITY_DF_PATH: str = os.path.join(project_root, "data/preprocessed/train.parquet")


def validate_dataframe(df: DataFrame, required_columns: List[str], step_name: str) -> None:
    """
    Validates a DataFrame for emptiness and missing columns.

    Args:
        df: The pandas DataFrame to validate.
        required_columns: A list of column names that must exist in the DataFrame.
        step_name: A string descriptor of the current pipeline step for logging purposes.

    Returns:
        None

    Raises:
        ValueError: If the DataFrame is empty or missing required columns.
    """
    if df.empty:
        raise ValueError(f"Fatal Error: DataFrame is empty at step: {step_name}.")
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Fatal Error: Missing required columns {missing_cols} at step: {step_name}."
        )


def main() -> None:
    """
    Executes the end-to-end churn prediction model training pipeline.

    This function orchestrates the following workflow:
    1.  **Configuration Validation**: Verifies the existence of critical MLflow
        environment variables (`MLFLOW_TRACKING_URI`, `MLFLOW_S3_ENDPOINT_URL`).
    2.  **MLflow Initialization**: Connects to the tracking server and sets up
        or retrieves the specified experiment.
    3.  **Feature Store Connection**: Initializes the Feast FeatureStore from
        the local repository configuration.
    4.  **Entity Loading**: Reads the 'spine' dataset (customer IDs and timestamps)
        from a Parquet file and validates its schema.
    5.  **Feature Retrieval**: Queries the Feature Store to join historical feature
        data onto the entity spine (Point-in-Time correctness).
    6.  **Preprocessing**: cleans the dataset, drops non-feature columns, and
        performs a train-test split.
    7.  **Pipeline Construction**: Builds a Scikit-Learn pipeline consisting of
        StandardScaler, DataFrameConverter, and XGBClassifier.
    8.  **Training & Logging**: executes an MLflow run to train the model, log
        parameters, and record performance metrics (F1, Accuracy, ROC-AUC).
    9.  **Model Registration**: Logs the model artifact and transitions the
        latest version to the 'Staging' stage in the MLflow Model Registry.

    Returns:
        None

    Raises:
        EnvironmentError: If required environment variables are missing.
        FileNotFoundError: If the entity dataset or Feast repository is missing.
        ValueError: If datasets are empty or missing required columns.
        MlflowException: If communication with the MLflow tracking server fails.
        RuntimeError: If model training or metric calculation fails.
    """
    # --- 1. Environment Validation ---
    mlflow_tracking_uri: Optional[str] = os.environ.get("MLFLOW_TRACKING_URI")
    mlflow_s3_endpoint: Optional[str] = os.environ.get("MLFLOW_S3_ENDPOINT_URL")

    if not mlflow_tracking_uri or not mlflow_s3_endpoint:
        logger.error("Missing required environment variables.")
        raise EnvironmentError(
            "MLFLOW_TRACKING_URI and MLFLOW_S3_ENDPOINT_URL must be set."
        )

    # --- 2. MLflow Setup ---
    logger.info("Configuring MLflow connection...")
    logger.info(f"Tracking URI: {mlflow_tracking_uri}")
    logger.info(f"S3 Endpoint: {mlflow_s3_endpoint}")

    try:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        client = MlflowClient(tracking_uri=mlflow_tracking_uri, registry_uri=mlflow_tracking_uri)
        
        experiment: Optional[Experiment] = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            logger.info(f"Creating new MLflow experiment: {EXPERIMENT_NAME}")
            mlflow.create_experiment(EXPERIMENT_NAME)
        
        mlflow.set_experiment(EXPERIMENT_NAME)
    except (MlflowException, ConnectionError) as e:
        logger.error(f"Fatal Error: Failed to configure MLflow: {e}")
        sys.exit(1)

    # --- 3. Feast Store Initialization ---
    logger.info(f"Initializing Feast FeatureStore from: {FEAST_REPO_PATH}")
    if not os.path.exists(FEAST_REPO_PATH):
        logger.error(f"Feast repository path not found: {FEAST_REPO_PATH}")
        sys.exit(1)

    try:
        store = FeatureStore(repo_path=FEAST_REPO_PATH)
    except Exception as e:
        logger.error(f"Fatal Error: Failed to initialize Feast FeatureStore: {e}")
        sys.exit(1)

    # --- 4. Load Entity Spine ---
    logger.info(f"Loading entity dataframe from: {ENTITY_DF_PATH}")
    try:
        if not os.path.exists(ENTITY_DF_PATH):
            raise FileNotFoundError(f"Entity Parquet file not found at: {ENTITY_DF_PATH}")
        
        entity_df: DataFrame = pd.read_parquet(
            ENTITY_DF_PATH,
            columns=["customer_id", "event_timestamp"],
        )
        
        # Validate Entity DF
        validate_dataframe(entity_df, ["customer_id", "event_timestamp"], "Entity Loading")
        
    except (FileNotFoundError, ValueError, Exception) as e:
        logger.error(f"Fatal Error during entity loading: {e}")
        sys.exit(1)

    # --- 5. Get Historical Features ---
    logger.info("Retrieving historical features from Feast...")
    feature_refs: List[str] = [
        "customer_features:Age",
        "customer_features:Support Calls",
        "customer_features:Payment Delay",
        "customer_features:Total Spend",
        "customer_features:Last Interaction",
        "customer_features:Churn",
        "customer_features:Male",
        "customer_features:Age_Group",
        "customer_features:Interaction_Frequency",
    ]

    try:
        training_data = store.get_historical_features(
            entity_df=entity_df,
            features=feature_refs,
        )
        training_df: DataFrame = training_data.to_df()
        
        # Validate Training DF
        validate_dataframe(training_df, [TARGET_COLUMN], "Feature Retrieval")
        
    except Exception as e:
        logger.error(f"Fatal Error: Failed to get historical features from Feast: {e}")
        sys.exit(1)

    logger.info(f"Successfully retrieved {len(training_df)} rows.")

    # --- 6. Feature/Target Preparation ---
    cols_to_drop: List[str] = ["event_timestamp", "customer_id"]
    if "created_timestamp" in training_df.columns:
        cols_to_drop.append("created_timestamp")

    try:
        training_df = training_df.drop(columns=cols_to_drop, errors='ignore')
        
        # Check for NaN values which might break XGBoost or Scaler if not handled
        if training_df.isnull().any().any():
            logger.warning("NaN values found in training data. Ensure XGBoost handles them or impute upstream.")

        X: DataFrame = training_df.drop(TARGET_COLUMN, axis=1)
        y: Series = training_df[TARGET_COLUMN]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=42
        )
        
        # Validate Split
        validate_dataframe(X_train, [], "Train Split - X")
        if y_train.empty:
            raise ValueError("Training target vector is empty.")
            
    except ValueError as e:
        logger.error(f"Fatal Error during data preparation: {e}")
        sys.exit(1)

    original_columns: Index = X.columns

    # --- 7. Model Pipeline & Training ---
    logger.info("Initializing model pipeline...")
    params: Dict[str, Any] = {
        "objective": "binary:logistic",
        "random_state": 42,
    }
    
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
            logger.info(f"Active Run ID: {run_id}")

            mlflow.log_params(params)

            logger.info("Fitting model...")
            pipeline.fit(X_train, y_train)

            logger.info("Evaluating model...")
            yhat: np.ndarray = pipeline.predict(X_test)
            
            f1: float = f1_score(y_test, yhat)
            accuracy: float = accuracy_score(y_test, yhat)
            
            try:
                auc_score: float = roc_auc_score(y_test, yhat)
            except ValueError:
                logger.warning("Only one class present in y_test. ROC AUC score is not defined.")
                auc_score = float("nan")

            metrics: Dict[str, float] = {
                "f1_score": f1,
                "accuracy": accuracy,
                "roc_auc_curve": auc_score
            }
            mlflow.log_metrics(metrics)
            logger.info(f"Metrics: {metrics}")

            logger.info(f"Logging model artifact as '{MODEL_NAME}'...")
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                name=MODEL_NAME,
                registered_model_name=MODEL_NAME,
                input_example=X_train.iloc[:5],
            )

            # --- 8. Model Registry Transition ---
            logger.info("Checking model registry for transition...")
            latest_versions = client.get_latest_versions(MODEL_NAME, stages=["None"])
            
            if latest_versions:
                target_version = latest_versions[0].version
                logger.info(f"Transitioning version {target_version} to 'Staging'...")
                
                client.transition_model_version_stage(
                    name=MODEL_NAME,
                    version=target_version,
                    stage="Staging",
                    archive_existing_versions=False
                )
                logger.success(f"Successfully transitioned model version {target_version} to 'Staging'.")
            else:
                logger.warning("No new model version found in 'None' stage to transition.")

            logger.success("Pipeline execution completed successfully.")

    except (MlflowException, ValueError, RuntimeError) as e:
        logger.error(f"Fatal Error: MLflow run failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected Fatal Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()