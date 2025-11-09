import os
import pandas as pd
import time
import sys
from loguru import logger
from sqlalchemy import create_engine

DB_USER = os.environ.get("APP_DB_USER", "admin")
DB_PASSWORD = os.environ.get("APP_DB_PASSWORD", "admin")
DB_NAME = os.environ.get("APP_DB_NAME", "churn")
DB_HOST = "localhost"
DB_PORT = "5435"
TABLE_NAME = "raw_churn_data"

CSV_PATH = "data/raw/train.csv"

def populate_database(retries: int = 5, delay: int = 5):
    engine_url = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    
    for i in range(retries):
        try:
            engine = create_engine(engine_url)
        
            with engine.connect() as conn:
                logger.success("Database connection successful.")
            
            logger.info(f"Loading data from {CSV_PATH}...")
            df = pd.read_csv(CSV_PATH)
            
            logger.info(f"Writing data to table '{TABLE_NAME}'...")
            df.to_sql(
                TABLE_NAME, 
                engine, 
                if_exists="replace",
                index=False
            )
            
            logger.info(f"Successfully populated '{TABLE_NAME}' with {len(df)} rows.")
            return

        except Exception as e:
            logger.error(f"Error: {e}")
            logger.error(f"Database not ready. Retrying in {delay} seconds... ({i+1}/{retries})")
            time.sleep(delay)

    logger.error("Failed to populate database after several retries.")
    sys.exit(1)

if __name__ == "__main__":
    populate_database()