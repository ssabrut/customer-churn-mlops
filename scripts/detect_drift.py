import os
import sys
from typing import Any, Dict, List, Optional

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

# Ensure project root is in sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Constants
REFERENCE_DATA_PATH: str = os.path.join(project_root, "data/preprocessed/train.parquet")


def get_db_connection_string() -> str:
    """
    Constructs and validates the database connection string from environment variables.

    Args:
        None

    Returns:
        str: The formatted PostgreSQL connection string.

    Raises:
        EnvironmentError: If any required environment variable is missing.
    """
    required_vars = [
        "APP_DB_USER",
        "APP_DB_PASSWORD",
        "APP_DB_NAME",
        "APP_DB_HOST",
        "APP_DB_PORT",
    ]
    env_values = {}
    missing_vars = []

    for var in required_vars:
        value = os.environ.get(var)
        if not value:
            missing_vars.append(var)
        env_values[var] = value

    if missing_vars:
        raise EnvironmentError(f"Missing required environment variables: {missing_vars}")

    return (
        f"postgresql://{env_values['APP_DB_USER']}:{env_values['APP_DB_PASSWORD']}"
        f"@{env_values['APP_DB_HOST']}:{env_values['APP_DB_PORT']}"
        f"/{env_values['APP_DB_NAME']}"
    )


def fetch_production_data(connection_str: str, limit: int = 5000) -> pd.DataFrame:
    """
    Retrieves the most recent prediction logs from the production database.

    Args:
        connection_str: The database connection string.
        limit: The maximum number of recent rows to fetch.

    Returns:
        pd.DataFrame: A DataFrame containing production data.

    Raises:
        SQLAlchemyError: If the database connection or query fails.
    """
    query = """
        SELECT 
            age, support_calls, payment_delay, total_spend, 
            last_interaction, gender as "Male", age_group, 
            interaction_frequency, ground_truth as "Churn"
        FROM prediction_logs 
        ORDER BY timestamp DESC 
        LIMIT :limit
    """
    try:
        engine = create_engine(connection_str)
        # Using params argument for safer parameter binding, though integer injection is low risk here
        df = pd.read_sql(query, engine, params={"limit": limit})
        return df
    except SQLAlchemyError as e:
        logger.error(f"Database query failed: {e}")
        raise


def load_reference_data(path: str) -> pd.DataFrame:
    """
    Loads the reference dataset from a Parquet file.

    Args:
        path: The absolute path to the Parquet file.

    Returns:
        pd.DataFrame: The loaded reference data.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If the file is corrupted or unreadable.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Reference data not found at: {path}")

    try:
        return pd.read_parquet(path)
    except Exception as e:
        raise IOError(f"Failed to read Parquet file: {e}")


def parse_evidently_metrics(metrics_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extracts relevant drift metrics from the Evidently JSON report.

    Args:
        metrics_dict: The raw dictionary output from Evidently.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries ready for database insertion.

    Raises:
        KeyError: If the expected structure of the Evidently report is missing.
    """
    rows_to_insert: List[Dict[str, Any]] = []

    try:
        # Accessing the DataDriftPreset results
        # Note: Index 0 assumes DataDriftPreset is the first/only metric included.
        drift_data = metrics_dict.get("metrics", [{}])[0].get("result", {})

        if not drift_data:
            raise KeyError("Could not find 'result' in Evidently metrics output.")

        # 1. Dataset-level metrics
        rows_to_insert.append({
            "metric_name": "dataset_drift_share",
            "feature_name": "GLOBAL",
            "metric_value": drift_data.get("share_of_drifted_columns", 0.0)
        })
        rows_to_insert.append({
            "metric_name": "dataset_drift_detected",
            "feature_name": "GLOBAL",
            "metric_value": float(drift_data.get("dataset_drift", 0.0))
        })

        # 2. Feature-level metrics
        drift_by_columns = drift_data.get("drift_by_columns", {})
        for feature_name, details in drift_by_columns.items():
            drift_score = details.get("drift_score", 0.0)
            rows_to_insert.append({
                "metric_name": "feature_drift_score",
                "feature_name": feature_name,
                "metric_value": drift_score
            })

    except (KeyError, IndexError, TypeError) as e:
        logger.error(f"Error parsing Evidently dictionary: {e}")
        raise ValueError("Invalid Evidently report structure.") from e

    return rows_to_insert


def save_metrics_to_db(connection_str: str, metrics_rows: List[Dict[str, Any]]) -> None:
    """
    Bulk inserts drift metrics into the database.

    Args:
        connection_str: The database connection string.
        metrics_rows: A list of dictionaries containing metric data.

    Returns:
        None

    Raises:
        SQLAlchemyError: If the insert operation fails.
    """
    if not metrics_rows:
        logger.warning("No metrics to save.")
        return

    insert_query = text("""
        INSERT INTO drift_metrics (metric_name, feature_name, metric_value)
        VALUES (:metric_name, :feature_name, :metric_value)
    """)

    try:
        engine = create_engine(connection_str)
        with engine.connect() as conn:
            logger.info(f"Bulk inserting {len(metrics_rows)} drift metrics...")
            conn.execute(insert_query, metrics_rows)
            conn.commit()
        logger.success("Drift metrics successfully saved.")
    except SQLAlchemyError as e:
        logger.error(f"Failed to save metrics to DB: {e}")
        raise


def main() -> None:
    """
    Executes the data drift detection pipeline.

    Steps:
    1. Validates configuration.
    2. Loads production (current) and reference data.
    3. Aligns columns between datasets.
    4. Generates an Evidently Data Drift report.
    5. Parses and saves the results to the database.

    Returns:
        None
    """
    logger.info("Starting Drift Detection Pipeline...")

    # 1. Configuration & Data Loading
    try:
        connection_str = get_db_connection_string()
        
        current_df = fetch_production_data(connection_str, limit=5000)
        reference_df = load_reference_data(REFERENCE_DATA_PATH)

        if current_df.empty:
            logger.warning("No production data found. Skipping drift check.")
            return

    except (EnvironmentError, FileNotFoundError, SQLAlchemyError, IOError) as e:
        logger.error(f"Initialization Fatal Error: {e}")
        sys.exit(1)

    # 2. Data Alignment
    common_cols = list(set(current_df.columns) & set(reference_df.columns))
    if not common_cols:
        logger.error("Fatal Error: No common columns found between Reference and Production data.")
        sys.exit(1)
    
    logger.info(f"Aligning datasets on {len(common_cols)} common columns.")
    current_df = current_df[common_cols]
    reference_df = reference_df[common_cols]

    # 3. Report Generation
    try:
        logger.info("Running Evidently Drift Report...")
        drift_report = Report(metrics=[DataDriftPreset()])
        drift_report.run(reference_data=reference_df, current_data=current_df)
        
        metrics_dict = drift_report.as_dict()
        parsed_rows = parse_evidently_metrics(metrics_dict)
        
    except Exception as e:
        logger.error(f"Evidently Report Generation/Parsing Failed: {e}")
        sys.exit(1)

    # 4. Save Results
    try:
        save_metrics_to_db(connection_str, parsed_rows)
    except SQLAlchemyError:
        sys.exit(1)


if __name__ == "__main__":
    main()