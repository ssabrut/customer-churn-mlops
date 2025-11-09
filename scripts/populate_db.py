import os
import pandas as pd
import time
import sys
from sqlalchemy import create_engine

DB_USER = os.environ.get("APP_DB_USER", "admin")
DB_PASSWORD = os.environ.get("APP_DB_PASSWORD", "admin")
DB_NAME = os.environ.get("APP_DB_NAME", "churn")
DB_HOST = "localhost"
DB_PORT = "5435"
TABLE_NAME = "raw_churn_data"

CSV_PATH = "data/raw/train.csv"

def populate_database(retries: int = 5, delay: int = 5):
    """
    Waits for the DB to be ready and populates it with data.
    """
    engine_url = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    
    for i in range(retries):
        try:
            # 1. Create a connection engine
            engine = create_engine(engine_url)
            
            # Test the connection (this will fail if DB is not ready)
            with engine.connect() as conn:
                print("Database connection successful.")
            
            # 2. Load the CSV data
            print(f"Loading data from {CSV_PATH}...")
            df = pd.read_csv(CSV_PATH)
            
            # 3. Write data to the SQL table
            print(f"Writing data to table '{TABLE_NAME}'...")
            df.to_sql(
                TABLE_NAME, 
                engine, 
                if_exists="replace", # Re-create the table every time
                index=False
            )
            
            print(f"Successfully populated '{TABLE_NAME}' with {len(df)} rows.")
            return

        except Exception as e:
            print(f"Error: {e}")
            print(f"Database not ready. Retrying in {delay} seconds... ({i+1}/{retries})")
            time.sleep(delay)

    print("Failed to populate database after several retries.")
    sys.exit(1)

if __name__ == "__main__":
    populate_database()