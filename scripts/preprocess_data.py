import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from datetime import datetime, timedelta
from typing import Any

import pandas as pd
from loguru import logger
from sqlalchemy import create_engine, exc
from sqlalchemy.engine import Engine

from core.config import load_config
from core.constant import APP_TABlE_NAME
from core.transformer import ChurnFeatureTransformer

OUTPUT_PATH: str = f"{project_root}/data/preprocessed/train.parquet"

try:
    config: Any = load_config()
except EnvironmentError as e:
    logger.error(f"Configuration failed to load: {e}")
    sys.exit(1)


def main() -> None:
    """
    Main function to prepare data for the churn prediction model.

    This process involves loading the raw data from the PostgreSQL database,
    applying feature transformations using the ChurnFeatureTransformer,
    generating necessary event timestamps for Feast, and saving the
    preprocessed data as a Parquet file to the offline store.
    The function handles database connection, data transformation, and
    file I/O, with specific error handling for each stage.

    Returns:
        None
    """
    logger.info("Starting data preparation from database...")

    # --- 1. Database Connection and Data Loading ---
    try:
        db_user: str = config.app_db_user
        db_password: str = config.app_db_password
        db_name: str = config.app_db_name
        db_host: str = config.db_host
        db_port: int = config.db_port

        engine_url: str = (
            f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        )
        engine: Engine = create_engine(engine_url)

        logger.info(f"Connecting to {db_host}:{db_port}...")
        df: pd.DataFrame = pd.read_sql(f"SELECT * FROM {APP_TABlE_NAME}", engine)
        logger.success(
            f"Successfully read {len(df)} rows from table '{APP_TABlE_NAME}'."
        )

    except (exc.OperationalError, exc.SQLAlchemyError) as e:
        logger.error(
            f"\n[Error] Failed to connect or read from database: {e}",
            file=sys.stderr,
        )
        logger.error("Please check:")
        logger.error(
            "  1. Are your Docker containers running? (`docker-compose up -d`)"
        )
        logger.error("  2. Are your .env variables correct?")
        logger.error("  3. Did the 'db-init' service run successfully?")
        sys.exit(1)

    # --- 2. Data Validation ---
    if df.empty:
        logger.warning(
            "No data was read from the database. " "Offline store will not be updated."
        )
        sys.exit(0)

    # --- 3. Data Transformation ---
    logger.info("Transforming data for Feast...")
    try:
        transformer = ChurnFeatureTransformer()
        df = transformer.transform(df)

        if "Id" in df.columns:
            df.rename(columns={"Id": "customer_id"}, inplace=True)

        df["event_timestamp"] = [
            datetime.now() - timedelta(days=365 - i % 365) for i in range(len(df))
        ]
        df["created_timestamp"] = datetime.now()
    except Exception as e:
        logger.error(f"Fatal Error: Failed during data transformation: {e}")
        sys.exit(1)

    # --- 4. Data Saving ---
    logger.info(f"Saving features to offline store at: {OUTPUT_PATH}")
    try:
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

        df.to_parquet(OUTPUT_PATH, index=False)
    except (IOError, OSError) as e:
        logger.error(
            f"Fatal Error: Failed to create directory or write parquet file "
            f"at '{OUTPUT_PATH}': {e}"
        )
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal Error: An unexpected error occurred during saving: {e}")
        sys.exit(1)

    logger.success("\nData preparation complete!")
    print("--- Offline Store (Parquet) Head ---")
    print(df.head())


if __name__ == "__main__":
    main()
