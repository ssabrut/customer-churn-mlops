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
from typing import Tuple

DEFAULT_APP_DB_USER = "admin"
DEFAULT_APP_DB_PASSWORD = "admin"
DEFAULT_APP_DB_NAME = "churn"
TABLE_NAME = "raw_churn_data"
CSV_PATH = "data/raw/train.csv"

def get_db_config() -> Tuple[str, str, str, str, str]:
    is_docker = os.environ.get("IS_DOCKER") == "true"

    db_user = os.environ.get("APP_DB_USER", DEFAULT_APP_DB_USER)
    db_password = os.environ.get("APP_DB_PASSWORD", DEFAULT_APP_DB_PASSWORD)
    db_name = os.environ.get("APP_DB_NAME", DEFAULT_APP_DB_NAME)

    if is_docker:
        db_host = os.environ.get("DB_HOST", "app_postgres")
        db_port = os.environ.get("DB_PORT", "5432")
        logger.info("Running inside Docker. Connecting to internal service.")
    else:
        db_host = os.environ.get("DB_HOST_LOCAL", "localhost")
        db_port = os.environ.get("DB_PORT_LOCAL", "5435") 
        logger.info(f"Running locally. Connecting to host port {db_port}.")

    return db_user, db_password, db_name, db_host, db_port

def populate_database(retries: int = 5, delay: int = 5):
    db_user, db_password, db_name, db_host, db_port = get_db_config()

    engine_url = (
        f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    )

    for i in range(retries):
        try:
            engine = create_engine(engine_url)

            with engine.connect() as conn:
                logger.success("Database connection successful.")

            logger.info(f"Loading data from {CSV_PATH}...")
            df = pd.read_csv(CSV_PATH)

            logger.info(f"Writing data to table '{TABLE_NAME}'...")
            df.to_sql(TABLE_NAME, engine, if_exists="replace", index=False)

            logger.info(f"Successfully populated '{TABLE_NAME}' with {len(df)} rows.")
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
