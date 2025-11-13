import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import argparse
from loguru import logger
from datetime import datetime, timedelta

from core.services.postgres import PostgresClient
from core.services.postgres.factory import make_postgres_service

def main(args):
    postgres_client: PostgresClient = make_postgres_service()
    engine = postgres_client.get_sync_engine()

    # 1. Calculate the date to process
    process_date = (datetime.fromisoformat(args.date) - 
                    timedelta(days=args.days_ago)).strftime('%Y-%m-%d')

    logger.info(f"Fetching ground truth for predictions made on: {process_date}")

    # 2. Fetch predictions from 30 days ago
    sql_preds = f"""
        SELECT customer_id, prediction
        FROM prediction_logs
        WHERE DATE(timestamp) > '{process_date}'
    """
    preds_df = pd.read_sql(sql_preds, engine)
    if preds_df.empty:
        logger.warning("No predictions found for this date. Exiting.")
        return

    # 3. Fetch actual churn data
    sql_actuals = 'SELECT "Id" AS customer_id, "Churn" AS actual_churn FROM customers'
    actuals_df = pd.read_sql(sql_actuals, engine)

    # 4. Join and save
    results_df = preds_df.merge(actuals_df, on="customer_id", how="left")
    results_df["actual_churn"] = results_df["actual_churn"].fillna(0)
    results_df["process_date"] = process_date

    logger.success(f"Found {len(results_df)} results. Saving to 'model_performance' table.")

    # 5. Write to the new performance table
    results_df.to_sql("model_performance", engine, if_exists="append", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, help="Airflow execution date (ds)")
    parser.add_argument("--days_ago", type=int, default=30, help="Churn window")
    main(parser.parse_args())