import argparse
import os
import sys
from datetime import datetime, timedelta

import pandas as pd
from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError


def main(args: argparse.Namespace) -> None:
    """
    Fetches model predictions from a specified past date, retrieves the
    corresponding actual churn data, merges the two datasets, and saves the
    combined results to a model_performance table.

    Args:
        args (argparse.Namespace): Command-line arguments containing 'date'
                                   (str, ISO format) and 'days_ago' (int).

    Raises:
        ValueError: If the provided 'date' argument is not a valid
                    ISO-formatted date string.
        ConnectionError: If the initial database connection or engine
                         creation fails.
        SQLAlchemyError: If any database query (SELECT) or write (INSERT)
                         operation fails.
    """
    try:
        db_user: str = os.environ.get("APP_DB_USER")
        db_password: str = os.environ.get("APP_DB_PASSWORD")
        db_name: str = os.environ.get("APP_DB_NAME")
        db_host: str = os.environ.get("APP_DB_HOST")
        db_port: int = os.environ.get("APP_DB_PORT")

        sync_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        engine: Engine = create_engine(sync_url)
    except Exception as e:
        logger.critical(f"Error: Failed to initialize database connection: {e}")
        # Re-raise as a more specific, expected error type
        raise ConnectionError("Database connection failed") from e

    # 1. Calculate the date to process
    try:
        process_date_dt = datetime.fromisoformat(args.date) - timedelta(
            days=args.days_ago
        )
        process_date: str = process_date_dt.strftime("%Y-%m-%d")
    except ValueError as e:
        logger.error(f"Error: Invalid date format '{args.date}'. Must be YYYY-MM-DD.")
        raise ValueError("Invalid date format provided") from e

    logger.info(f"Fetching ground truth for predictions made on: {process_date}")

    # 2. Fetch predictions from N days ago
    # Use parameterized query to prevent SQL injection
    sql_preds = text(
        """
        SELECT customer_id, prediction
        FROM prediction_logs
        WHERE DATE(timestamp) > :process_date
    """
    )
    try:
        preds_df: pd.DataFrame = pd.read_sql(
            sql_preds, engine, params={"process_date": process_date}
        )
    except SQLAlchemyError as e:
        logger.error(f"Error: Failed to fetch predictions from database: {e}")
        raise

    if preds_df.empty:
        logger.warning("No predictions found for this date. Exiting.")
        return

    # 3. Fetch actual churn data
    sql_actuals = text(
        'SELECT "Id" AS customer_id, "Churn" AS actual_churn FROM customers'
    )
    try:
        actuals_df: pd.DataFrame = pd.read_sql(sql_actuals, engine)
    except SQLAlchemyError as e:
        logger.error(f"Error: Failed to fetch actual churn data: {e}")
        raise

    # 4. Join and save
    results_df: pd.DataFrame = preds_df.merge(actuals_df, on="customer_id", how="left")
    results_df["actual_churn"] = results_df["actual_churn"].fillna(0)
    results_df["process_date"] = process_date

    logger.success(
        f"Found {len(results_df)} results. Saving to 'model_performance' table."
    )

    # 5. Write to the new performance table
    try:
        results_df.to_sql("model_performance", engine, if_exists="append", index=False)
        logger.success("Successfully saved performance results.")
    except SQLAlchemyError as e:
        logger.error(f"Error: Failed to write performance results to database: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch model predictions and compare with actuals."
    )
    parser.add_argument(
        "--date",
        required=True,
        help="Airflow execution date (ds) in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--days_ago", type=int, default=30, help="Churn window in days (default: 30)."
    )

    try:
        parsed_args = parser.parse_args()
        main(parsed_args)
    except (ValueError, ConnectionError, SQLAlchemyError) as e:
        # Catch expected, handled errors from main()
        logger.critical(f"\nScript terminated due to an error: {e}")
        sys.exit(1)
    except Exception as e:
        # Catch any other unexpected errors
        logger.critical(f"\nScript terminated due to an unexpected error: {e}")
        sys.exit(1)
