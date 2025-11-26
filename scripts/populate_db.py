import os
import sys
import time
from typing import Any

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
from loguru import logger
from sqlalchemy import create_engine, exc, text
from sqlalchemy.engine import Engine

from core.config import load_config
from core.constant import APP_TABLE_NAME

CSV_PATH: str = f"{project_root}/data/raw/train.csv"
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
CREATE TABLE model_performance (
    id SERIAL PRIMARY KEY,
    customer_id BIGINT,
    prediction INTEGER,
    actual_churn INTEGER,
    process_date DATE,
    inserted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

try:
    config: Any = load_config()
except EnvironmentError as e:
    logger.error(f"Configuration failed to load: {e}")
    sys.exit(1)


def main(retries: int = 5, delay: int = 5) -> None:
    """
    Connects to the PostgreSQL database, creates necessary tables, and
    populates the main application table from a CSV file.

    This function reads database configuration from the loaded 'config' object.
    It performs validation by checking for the CSV file's existence first.
    It then attempts to load the CSV data. If successful, it enters a
    retry loop to connect to the database, create tables, and write the
    data. The script will exit with an error if file validation fails or
    if all database connection retries are exhausted.

    Args:
        retries (int): The number of attempts to connect and populate the
                       database.
        delay (int): The number of seconds to wait between retry attempts.

    Returns:
        None
    """
    # --- 1. Input Validation ---
    if not os.path.exists(CSV_PATH):
        logger.error(f"Fatal Error: CSV file not found at {CSV_PATH}")
        sys.exit(1)

    # --- 2. Data Loading ---
    try:
        logger.info(f"Loading data from {CSV_PATH}...")
        df: pd.DataFrame = pd.read_csv(CSV_PATH)
    except pd.errors.ParserError as e:
        logger.error(f"Fatal Error: Failed to parse CSV file '{CSV_PATH}': {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal Error: Failed to read CSV file '{CSV_PATH}': {e}")
        sys.exit(1)

    # --- 3. Database Population with Retries ---
    db_user: str = config.app_db_user
    db_password: str = config.app_db_password
    db_name: str = config.app_db_name
    db_host: str = config.db_host
    db_port: int = config.db_port

    engine_url: str = (
        f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    )
    logger.info(f"Attempting to connect to database at {db_host}:{db_port}...")

    for i in range(retries):
        try:
            engine: Engine = create_engine(engine_url)

            # Test connection and create tables
            with engine.connect() as conn:
                logger.success("Database connection successful.")
                logger.info("Creating tables (if not exists)...")
                conn.execute(text(CREATE_PREDICTION_LOGS_TABLE))
                conn.execute(text(CREATE_MODEL_PERFORMANCE_TABLE))
                conn.commit()

            # Write data to SQL
            logger.info(f"Writing {len(df)} rows to table '{APP_TABLE_NAME}'...")
            df.to_sql(APP_TABLE_NAME, engine, if_exists="replace", index=False)

            logger.success(
                f"Successfully populated '{APP_TABLE_NAME}' with {len(df)} rows."
            )
            return  # Success

        except exc.OperationalError as e:
            logger.error(f"Database connection error: {e}")
            logger.warning(
                f"Database not ready. Retrying in {delay} seconds... "
                f"({i + 1}/{retries})"
            )

        except exc.SQLAlchemyError as e:
            logger.error(f"Database operation error (e.g., table write): {e}")
            logger.warning(f"Retrying in {delay} seconds... ({i + 1}/{retries})")

        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            logger.warning(f"Retrying in {delay} seconds... ({i + 1}/{retries})")

        time.sleep(delay)

    logger.error("Failed to populate database after several retries.")
    sys.exit(1)


if __name__ == "__main__":
    main()
