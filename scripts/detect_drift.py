import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
from loguru import logger
from sqlalchemy import create_engine
from evidently import Report

from evidently.presets import DataDriftPreset
from sqlalchemy import text

try:
    APP_DB_USER: str = os.environ.get("APP_DB_USER")
    APP_DB_PASSWORD: str = os.environ.get("APP_DB_PASSWORD")
    APP_DB_NAME: str = os.environ.get("APP_DB_NAME")
    APP_DB_HOST: str = os.environ.get("APP_DB_HOST")
    APP_DB_PORT: str = os.environ.get("APP_DB_PORT")
except EnvironmentError as e:
    logger.critical(e)

REFERENCE_DATA_PATH: str = f"{project_root}/data/preprocessed/train.parquet"
DB_CONNECTION_STR: str = f"postgresql://{APP_DB_USER}:{APP_DB_PASSWORD}@{APP_DB_HOST}:{APP_DB_PORT}/{APP_DB_NAME}"
REPORT_OUTPUT_DIR = "reports/drift"

def fetch_production_data(limit: int = 5000) -> pd.DataFrame:
    engine = create_engine(DB_CONNECTION_STR)
    query = f"""
        SELECT 
            age, support_calls, payment_delay, total_spend, 
            last_interaction, gender as "Male", age_group, 
            interaction_frequency, ground_truth as "Churn"
        FROM prediction_logs 
        ORDER BY timestamp DESC 
        LIMIT {limit}
    """
    return pd.read_sql(query, engine)

def load_reference_data() -> pd.DataFrame:
    df = pd.read_parquet(REFERENCE_DATA_PATH)
    return df

def save_metrics_to_db(metrics_dict: dict):
    """Parses Evidently output and saves to Postgres."""
    engine = create_engine(DB_CONNECTION_STR)
    
    # 1. Extract Dataset-level metrics
    # Note: The exact structure depends on Evidently version, but typically:
    drift_data = metrics_dict['metrics'][0]['result']
    
    dataset_drift_share = drift_data['share_of_drifted_columns']
    dataset_drift_detected = float(drift_data['dataset_drift']) # 1.0 if drift, 0.0 if not
    
    rows_to_insert = [
        {"metric_name": "dataset_drift_share", "feature_name": "GLOBAL", "metric_value": dataset_drift_share},
        {"metric_name": "dataset_drift_detected", "feature_name": "GLOBAL", "metric_value": dataset_drift_detected},
    ]

    # 2. Extract Feature-level metrics (p-values or drift scores)
    for feature_name, details in drift_data['drift_by_columns'].items():
        drift_score = details['drift_score']
        rows_to_insert.append({
            "metric_name": "feature_drift_score",
            "feature_name": feature_name,
            "metric_value": drift_score
        })

    # 3. Insert into DB
    with engine.connect() as conn:
        logger.info(f"Inserting {len(rows_to_insert)} drift metrics...")
        stmt = text("""
            INSERT INTO drift_metrics (metric_name, feature_name, metric_value)
            VALUES (:metric_name, :feature_name, :metric_value)
        """)
        for row in rows_to_insert:
            conn.execute(stmt, row)
        conn.commit()

def main():
    logger.info("Starting Drift Detection...")
    
    # 1. Load Data
    try:
        current_df = fetch_production_data()
        reference_df = load_reference_data()
        
        if current_df.empty:
            logger.warning("No production data found. Skipping drift check.")
            return
            
        # Ensure common columns exists
        common_cols = list(set(current_df.columns) & set(reference_df.columns))
        current_df = current_df[common_cols]
        reference_df = reference_df[common_cols]
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    # Generate Report
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=reference_df, current_data=current_df)
    
    # Get the JSON output as a Python dictionary
    metrics_dict = drift_report.as_dict()
    
    # Save to DB
    save_metrics_to_db(metrics_dict)

    # Generate Drift Report (HTML)
    logger.info("Generating Data Drift Report...")
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=reference_df, current_data=current_df)
    
    os.makedirs(REPORT_OUTPUT_DIR, exist_ok=True)
    report_path = os.path.join(REPORT_OUTPUT_DIR, "drift_report.html")
    drift_report.save_html(report_path)
    logger.success(f"Drift report saved to {report_path}")

if __name__ == "__main__":
    main()