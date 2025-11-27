import os
import sys
import time
from typing import Any, List, Optional

import pandas as pd
from loguru import logger
from sqlalchemy import create_engine, exc, text
from sqlalchemy.engine import Connection, Engine

# Ensure project root is in sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.config import load_config
from core.constant import APP_TABLE_NAME

# --- Constants ---
CSV_PATH: str = os.path.join(project_root, "data/raw/train.csv")

CREATE_PREDICTION_LOGS_TABLE: str = """
CREATE TABLE IF NOT EXISTS prediction_logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_version VARCHAR(255),
    customer_id INTEGER,
    age INTEGER,
    support_calls FLOAT,
    payment_delay FLOAT,
    total_spend FLOAT,
    last_interaction FLOAT,
    gender INTEGER,
    age_group INTEGER,
    interaction_frequency INTEGER,
    prediction INTEGER,
    probability FLOAT,
    ground_truth INTEGER DEFAULT NULL,
    is_shadow BOOLEAN DEFAULT FALSE,
    response_time_ms INTEGER
);
"""

CREATE_MODEL_PERFORMANCE_TABLE: str = """
CREATE TABLE IF NOT EXISTS model_performance (
    id SERIAL PRIMARY KEY,
    customer_id BIGINT,
    prediction INTEGER,
    actual_churn INTEGER,
    process_date DATE,
    inserted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_DRIFT_TABLE_QUERY: str = """
CREATE TABLE IF NOT EXISTS drift_metrics (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metric_name VARCHAR(255),
    feature_name VARCHAR(255),
    metric_value FLOAT
);
"""


def load_and_validate_data(file_path: str) -> pd.DataFrame:
    """
    Loads data from a CSV file and validates its content.

    Args:
        file_path: The absolute path to the CSV file.

    Returns:
        pd.DataFrame: The loaded pandas DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        pd.errors.EmptyDataError: If the CSV file is empty.
        ValueError: If the loaded DataFrame has no rows.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found at: {file_path}")

    try:
        df: pd.DataFrame = pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError(f"The CSV file at {file_path} is empty.")
    except Exception as e:
        raise RuntimeError(f"Failed to parse CSV file: {e}")

    if df.empty:
        raise ValueError("The loaded DataFrame is empty (0 rows).")

    return df


def initialize_schema(engine: Engine) -> None:
    """
    Executes SQL statements to create necessary database tables if they do not exist.

    Args:
        engine: The SQLAlchemy Engine instance connected to the target database.

    Returns:
        None

    Raises:
        exc.SQLAlchemyError: If table creation fails.
    """
    with engine.connect() as conn:
        logger.info("Initializing database schema...")
        conn.execute(text(CREATE_PREDICTION_LOGS_TABLE))
        conn.execute(text(CREATE_MODEL_PERFORMANCE_TABLE))
        conn.execute(text(CREATE_DRIFT_TABLE_QUERY))
        conn.commit()
    logger.success("Schema initialization complete.")


def get_db_url() -> str:
    """
    Constructs the database connection URL from the configuration.

    Args:
        None

    Returns:
        str: The full PostgreSQL connection string.

    Raises:
        EnvironmentError: If configuration loading fails.
    """
    try:
        config: Any = load_config()
        return (
            f"postgresql+psycopg2://{config.app_db_user}:{config.app_db_password}"
            f"@{config.db_host}:{config.db_port}/{config.app_db_name}"
        )
    except Exception as e:
        raise EnvironmentError(f"Failed to load database configuration: {e}")


def main(retries: int = 5, delay: int = 5) -> None:
    """
    Orchestrates the database population process with retry logic.

    This function performs the following steps:
    1.  Loads and validates the source CSV data.
    2.  Constructs the database connection string.
    3.  Enters a retry loop to:
        a.  Establish a database connection.
        b.  Initialize the schema (create tables).
        c.  Write the DataFrame to the database.

    Args:
        retries: Maximum number of connection attempts.
        delay: Time in seconds to wait between attempts.

    Returns:
        None

    Raises:
        SystemExit: If data loading fails or max retries are exceeded.
    """
    # --- 1. Load Data ---
    logger.info("Starting database population pipeline...")
    try:
        df = load_and_validate_data(CSV_PATH)
        logger.info(f"Loaded {len(df)} rows from {CSV_PATH}")
    except (FileNotFoundError, ValueError, Exception) as e:
        logger.error(f"Fatal Data Error: {e}")
        sys.exit(1)

    # --- 2. Configuration ---
    try:
        engine_url = get_db_url()
    except EnvironmentError as e:
        logger.error(e)
        sys.exit(1)

    # --- 3. Database Operations (Retry Loop) ---
    for i in range(retries):
        attempt_count = i + 1
        logger.info(f"Database connection attempt {attempt_count}/{retries}...")

        try:
            engine: Engine = create_engine(engine_url)

            # A. Initialize Tables
            initialize_schema(engine)

            # B. Write Data
            logger.info(f"Writing data to table '{APP_TABLE_NAME}'...")

            # utilizing chunksize helps prevent memory issues with large datasets
            df.to_sql(
                APP_TABLE_NAME, engine, if_exists="replace", index=False, chunksize=1000
            )

            logger.success(
                f"Successfully populated '{APP_TABLE_NAME}' with {len(df)} rows."
            )
            return  # Exit successfully

        except exc.OperationalError as e:
            logger.warning(f"Connection failed (OperationalError): {e}")
            if attempt_count < retries:
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logger.error("Max retries exceeded for database connection.")

        except exc.SQLAlchemyError as e:
            logger.error(f"Database integrity or query error: {e}")
            # We break here because syntax/integrity errors rarely resolve with retries
            break

        except Exception as e:
            logger.error(f"Unexpected error during database operations: {e}")
            if attempt_count < retries:
                time.sleep(delay)
            else:
                logger.error("Max retries exceeded for unexpected error.")

    logger.error("Fatal Error: Database population failed.")
    sys.exit(1)


if __name__ == "__main__":
    main()
