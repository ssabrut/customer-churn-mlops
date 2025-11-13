import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
import pandas as pd
from sqlalchemy import create_engine
import mlflow
import argparse
from loguru import logger
from datetime import datetime, timedelta
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

from core.constant import MODEL_NAME, APP_DB_USER, APP_DB_PASSWORD, APP_DB_NAME
from core.services.mlflow.factory import make_mlflow_service
from core.config import load_config

def main(args):
    client = make_mlflow_service()
    config = load_config()

    try:
        prod_version = client._client.get_latest_versions(MODEL_NAME)[0]
        logger.debug("Production Version:", prod_version)
        run_id = prod_version.run_id
        logger.info(f"Logging metrics to production model run: {run_id}")
    except Exception as e:
        logger.error(f"Could not find production model. {e}")
        run_id = None

    # 3. Fetch performance data for the day
    engine = create_engine(f"postgresql://{APP_DB_USER}:{APP_DB_PASSWORD}@{config.db_host}:{config.db_port}/{APP_DB_NAME}")
    process_date = (datetime.fromisoformat(args.date) - 
                    timedelta(days=args.days_ago)).strftime('%Y-%m-%d')

    sql = f"""
        SELECT prediction, actual_churn
        FROM model_performance
        WHERE process_date > '{process_date}'
    """
    df = pd.read_sql(sql, engine)

    if df.empty:
        logger.warning("No performance data found for this date. Exiting.")
        return

    # 4. Calculate metrics
    f1 = f1_score(df["actual_churn"], df["prediction"])
    acc = accuracy_score(df["actual_churn"], df["prediction"])
    auc = roc_auc_score(df["actual_churn"], df["prediction"])

    logger.success(f"Date: {process_date} | F1: {f1:.4f} | Accuracy: {acc:.4f} | AUC: {auc:.4f}")

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric(f"prod_f1_score", f1)
        mlflow.log_metric(f"prod_accuracy", acc)
        mlflow.log_metric(f"prod_roc_auc", auc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--days_ago", type=int, default=30)
    main(parser.parse_args())