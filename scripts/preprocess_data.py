import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from datetime import datetime, timedelta

import pandas as pd
from loguru import logger
from sqlalchemy import create_engine, exc
from sqlalchemy.engine import Engine

from core.transformer import ChurnFeatureTransformer

OUTPUT_PATH: str = f"{project_root}/data/preprocessed/train.parquet"


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
        db_user: str = os.environ.get("APP_DB_USER")
        db_password: str = os.environ.get("APP_DB_PASSWORD")
        db_name: str = os.environ.get("APP_DB_NAME")
        db_host: str = os.environ.get("APP_DB_HOST")
        db_port: int = os.environ.get("APP_DB_PORT")

        engine_url: str = (
            f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        )
        engine: Engine = create_engine(engine_url)

        logger.info(f"Connecting to {db_host}:{db_port}...")
        df: pd.DataFrame = pd.read_sql(f"SELECT * FROM customers", engine)
        logger.success(f"Successfully read {len(df)} rows from table 'customers'.")

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
