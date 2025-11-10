import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from datetime import datetime, timedelta

import pandas as pd
from loguru import logger
from sqlalchemy import create_engine
from typing import Tuple

from core.transformer import ChurnFeatureTransformer

DB_USER = os.environ.get("APP_DB_USER", "admin")
DB_PASSWORD = os.environ.get("APP_DB_PASSWORD", "admin")
DB_NAME = os.environ.get("APP_DB_NAME", "churn")
TABLE_NAME = "raw_churn_data"
OUTPUT_PATH = "data/preprocessed/train.parquet"

logger.info("Starting data preparation from database...")

def get_db_config() -> Tuple[str, str]:
    is_docker = os.environ.get("IS_DOCKER") == "true"
    
    if is_docker:
        db_host = os.environ.get("DB_HOST", "app_postgres")
        db_port = os.environ.get("DB_PORT", "5432")
        logger.info("Running inside Docker. Connecting to internal service.")
    else:
        db_host = os.environ.get("DB_HOST_LOCAL", "localhost")
        db_port = os.environ.get("DB_PORT_LOCAL", "5435") 
        logger.info(f"Running locally. Connecting to host port {db_port}.")

    return db_host, db_port

try:
    db_host, db_port = get_db_config()
    engine_url = (
        f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{db_host}:{db_port}/{DB_NAME}"
    )
    engine = create_engine(engine_url)

    logger.info(f"Connecting to {db_host}:{db_port}...")
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", engine)
    logger.success(f"Successfully read {len(df)} rows from table '{TABLE_NAME}'.")
except Exception as e:
    logger.error(
        f"\n[Error] Failed to connect or read from database: {e}", file=sys.stderr
    )
    logger.error("Please check:")
    logger.error("  1. Are your Docker containers running? (`docker-compose up -d`)")
    logger.error("  2. Are your .env variables correct?")
    logger.error("  3. Did the 'db-init' service run successfully?")
    sys.exit(1)

logger.info("Transforming data for Feast...")

transformer = ChurnFeatureTransformer()
df = transformer.transform(df)

if "Id" in df.columns:
    df.rename(columns={"Id": "customer_id"}, inplace=True)

df["event_timestamp"] = [
    datetime.now() - timedelta(days=365 - i % 365) for i in range(len(df))
]
df["created_timestamp"] = datetime.now()

logger.info(f"Saving features to offline store at: {OUTPUT_PATH}")

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

df.to_parquet(OUTPUT_PATH, index=False)
logger.success("\nData preparation complete!")

print("--- Offline Store (Parquet) Head ---")
print(df.head())
