import os
import sys

# Ensures the project root is in the Python path for core imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
from datetime import datetime, timedelta

import mlflow
import pandas as pd
from loguru import logger
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError


def main(args: argparse.Namespace) -> None:
    """
    Fetches daily model performance data from the database, calculates key
    metrics (F1, Accuracy, AUC), and logs these metrics back to the
    existing MLflow run associated with the production model.

    Args:
        args (argparse.Namespace): Command-line arguments containing 'date'
                                   (str, ISO format) and 'days_ago' (int).

    Raises:
        ConnectionError: If the MLflow service or database connection fails.
        MlflowException: If fetching the model version or logging metrics fails.
        RuntimeError: If the production model run_id cannot be found, or if
                      metric calculation fails.
        SQLAlchemyError: If the database query for performance data fails.
        ValueError: If the provided 'date' is not a valid ISO format or
                    if metric calculation fails due to invalid data.
    """
    APP_DB_NAME: str = os.environ.get("APP_DB_NAME")
    APP_DB_PASSWORD: str = os.environ.get("APP_DB_PASSWORD")
    APP_DB_USER: str = os.environ.get("APP_DB_USER")
    APP_DB_HOST: str = os.environ.get("APP_DB_HOST")
    APP_DB_PORT: str = os.environ.get("APP_DB_PORT")
    MLFLOW_TRACKING_URI: str = os.environ.get("MLFLOW_TRACKING_URI")
    MODEL_NAME: str = "XGBoostChurnModel"

    try:
        client: MlflowClient = MlflowClient(
            tracking_uri=MLFLOW_TRACKING_URI, registry_uri=MLFLOW_TRACKING_URI
        )
    except MlflowException as e:
        logger.critical(f"Error: Failed to initialize MLflow service: {e}")
        raise ConnectionError("MLflow service connection failed") from e
    except Exception as e:
        logger.critical(f"Error: Failed to load configuration: {e}")
        raise

    try:
        # Get the run ID for the *current* production model
        prod_version = client.get_latest_versions(MODEL_NAME)[0]
        run_id: str = prod_version.run_id
        logger.success(f"Logging metrics to production model run: {run_id}")
    except (MlflowException, IndexError) as e:
        logger.error(f"Error: Could not find production model run_id. {e}")
        # Fail fast: we should not create a new run
        raise RuntimeError(f"Could not find production model '{MODEL_NAME}'") from e

    # 3. Fetch performance data for the day
    try:
        db_url = (
            f"postgresql://{APP_DB_USER}:{APP_DB_PASSWORD}@"
            f"{APP_DB_HOST}:{APP_DB_PORT}/{APP_DB_NAME}"
        )
        engine: Engine = create_engine(db_url)
        # Test connection
        with engine.connect() as conn:
            pass
    except SQLAlchemyError as e:
        logger.critical(f"Error: Failed to create database engine or connection: {e}")
        raise ConnectionError("Database connection failed") from e

    try:
        process_date_dt = datetime.fromisoformat(args.date) - timedelta(
            days=args.days_ago
        )
        process_date: str = process_date_dt.strftime("%Y-%m-%d")
    except ValueError as e:
        logger.error(f"Error: Invalid date format '{args.date}'. Must be YYYY-MM-DD.")
        raise

    # Use parameterized query to prevent SQL injection
    sql = text(
        """
        SELECT prediction, actual_churn
        FROM model_performance
        WHERE process_date > :process_date
    """
    )
    try:
        df: pd.DataFrame = pd.read_sql(
            sql, engine, params={"process_date": process_date}
        )
    except SQLAlchemyError as e:
        logger.error(f"Error: Failed to fetch performance data: {e}")
        raise

    if df.empty:
        logger.warning("No performance data found for this date. Exiting.")
        return

    # 4. Calculate metrics
    try:
        f1: float = f1_score(df["actual_churn"], df["prediction"])
        acc: float = accuracy_score(df["actual_churn"], df["prediction"])
        auc: float = roc_auc_score(df["actual_churn"], df["prediction"])
    except ValueError as e:
        logger.error(f"Error: Failed to calculate metrics. {e}")
        logger.error("This can happen if 'actual_churn' contains only one class.")
        raise RuntimeError("Metric calculation failed") from e

    logger.success(
        f"Date: {process_date} | F1: {f1:.4f} | Accuracy: {acc:.4f} | AUC: {auc:.4f}"
    )

    try:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metric(f"prod_f1_score", f1)
            mlflow.log_metric(f"prod_accuracy", acc)
            mlflow.log_metric(f"prod_roc_auc", auc)
        logger.success("Successfully logged metrics to MLflow.")
    except MlflowException as e:
        logger.error(f"Error: Failed to log metrics to MLflow run {run_id}: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate and log model performance metrics to MLflow."
    )
    parser.add_argument(
        "--date",
        required=True,
        help="Airflow execution date (ds) in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--days_ago",
        type=int,
        default=30,
        help="Performance window in days (default: 30).",
    )

    try:
        parsed_args = parser.parse_args()
        main(parsed_args)
    except (
        ValueError,
        ConnectionError,
        SQLAlchemyError,
        MlflowException,
        RuntimeError,
    ) as e:
        # Catch expected, handled errors from main()
        logger.error(f"\nScript terminated due to an error: {e}")
        sys.exit(1)
    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f"\nScript terminated due to an unexpected error: {e}")
        sys.exit(1)
