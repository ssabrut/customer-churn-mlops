import os
import sys

# Ensures the project root is in the Python path for core imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
import mlflow
import argparse
from datetime import datetime, timedelta
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

from core.constant import MODEL_NAME, APP_DB_USER, APP_DB_PASSWORD, APP_DB_NAME
from core.services.mlflow.factory import make_mlflow_service
from core.config import load_config, Config

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
    try:
        client: MlflowClient = make_mlflow_service()
        config: Config = load_config()
    except MlflowException as e:
        print(f"Error: Failed to initialize MLflow service: {e}", file=sys.stderr)
        raise ConnectionError("MLflow service connection failed") from e
    except Exception as e:
        print(f"Error: Failed to load configuration: {e}", file=sys.stderr)
        raise

    try:
        # Get the run ID for the *current* production model
        prod_version = client._client.get_latest_versions(MODEL_NAME)[0]
        run_id: str = prod_version.run_id
        print(f"Logging metrics to production model run: {run_id}")
    except (MlflowException, IndexError) as e:
        print(f"Error: Could not find production model run_id. {e}", file=sys.stderr)
        # Fail fast: we should not create a new run
        raise RuntimeError(f"Could not find production model '{MODEL_NAME}'") from e

    # 3. Fetch performance data for the day
    try:
        db_url = (
            f"postgresql://{APP_DB_USER}:{APP_DB_PASSWORD}@"
            f"{config.db_host}:{config.db_port}/{APP_DB_NAME}"
        )
        engine: Engine = create_engine(db_url)
        # Test connection
        with engine.connect() as conn:
            pass
    except SQLAlchemyError as e:
        print(f"Error: Failed to create database engine or connection: {e}", file=sys.stderr)
        raise ConnectionError("Database connection failed") from e

    try:
        process_date_dt = (datetime.fromisoformat(args.date) -
                           timedelta(days=args.days_ago))
        process_date: str = process_date_dt.strftime('%Y-%m-%d')
    except ValueError as e:
        print(f"Error: Invalid date format '{args.date}'. Must be YYYY-MM-DD.",
              file=sys.stderr)
        raise

    # Use parameterized query to prevent SQL injection
    sql: str = """
        SELECT prediction, actual_churn
        FROM model_performance
        WHERE process_date > :process_date
    """
    try:
        df: pd.DataFrame = pd.read_sql(
            sql,
            engine,
            params={"process_date": process_date}
        )
    except SQLAlchemyError as e:
        print(f"Error: Failed to fetch performance data: {e}", file=sys.stderr)
        raise

    if df.empty:
        print("No performance data found for this date. Exiting.")
        return

    # 4. Calculate metrics
    try:
        f1: float = f1_score(df["actual_churn"], df["prediction"])
        acc: float = accuracy_score(df["actual_churn"], df["prediction"])
        auc: float = roc_auc_score(df["actual_churn"], df["prediction"])
    except ValueError as e:
        print(f"Error: Failed to calculate metrics. {e}", file=sys.stderr)
        print("This can happen if 'actual_churn' contains only one class.", file=sys.stderr)
        raise RuntimeError("Metric calculation failed") from e

    print(f"Date: {process_date} | F1: {f1:.4f} | Accuracy: {acc:.4f} | AUC: {auc:.4f}")

    try:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metric(f"prod_f1_score", f1)
            mlflow.log_metric(f"prod_accuracy", acc)
            mlflow.log_metric(f"prod_roc_auc", auc)
        print("Successfully logged metrics to MLflow.")
    except MlflowException as e:
        print(f"Error: Failed to log metrics to MLflow run {run_id}: {e}", file=sys.stderr)
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate and log model performance metrics to MLflow."
    )
    parser.add_argument(
        "--date",
        required=True,
        help="Airflow execution date (ds) in YYYY-MM-DD format."
    )
    parser.add_argument(
        "--days_ago",
        type=int,
        default=30,
        help="Performance window in days (default: 30)."
    )

    try:
        parsed_args = parser.parse_args()
        main(parsed_args)
    except (
        ValueError,
        ConnectionError,
        SQLAlchemyError,
        MlflowException,
        RuntimeError
    ) as e:
        # Catch expected, handled errors from main()
        print(f"\nScript terminated due to an error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        # Catch any other unexpected errors
        print(f"\nScript terminated due to an unexpected error: {e}", file=sys.stderr)
        sys.exit(1)