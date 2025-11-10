import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import sys
import time

import pandas as pd
from loguru import logger
from sqlalchemy import create_engine

from core.config import load_mlops_config

TABLE_NAME = "raw_churn_data"
CSV_PATH = f"{project_root}/data/raw/train.csv"

try:
    config = load_mlops_config(project_root)
except EnvironmentError as e:
    logger.error(f"Configuration failed to load: {e}")
    sys.exit(1)


def populate_database(retries: int = 5, delay: int = 5):
    db_user = config.app_db_user
    db_password = config.app_db_password
    db_name = config.app_db_name
    db_host = config.db_host
    db_port = config.db_port

    engine_url = (
        f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    )
    logger.info(f"Connecting to database at {db_host}:{db_port}...")

    for i in range(retries):
        try:
            engine = create_engine(engine_url)

            # Test connection
            with engine.connect() as conn:
                logger.success("Database connection successful.")

            logger.info(f"Loading data from {CSV_PATH}...")
            df = pd.read_csv(CSV_PATH)

            logger.info(f"Writing data to table '{TABLE_NAME}'...")
            df.to_sql(TABLE_NAME, engine, if_exists="replace", index=False)

            logger.success(
                f"Successfully populated '{TABLE_NAME}' with {len(df)} rows."
            )
            return

        except Exception as e:
            logger.error(f"Error: {e}")
            logger.error(
                f"Database not ready. Retrying in {delay} seconds... ({i+1}/{retries})"
            )
            time.sleep(delay)

    logger.error("Failed to populate database after several retries.")
    sys.exit(1)


if __name__ == "__main__":
    populate_database()
