import os
import sys

import mlflow
from loguru import logger
from mlflow.tracking import MlflowClient

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configuration
try:
    MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
    MODEL_NAME = "XGBoostChurnModel"
    METRIC_TO_CHECK = "roc_auc_curve" 
except EnvironmentError as e:
    logger.error(e)
    sys.exit(1)

def get_metric(client: MlflowClient, run_id: str, metric_name: str) -> float:
    try:
        run = client.get_run(run_id)
        return run.data.metrics.get(metric_name, 0.0)
    except Exception as e:
        logger.error(f"Failed to fetch metric for run {run_id}: {e}")
        return 0.0

def main():
    logger.info("Starting Model Promotion Pipeline...")

    if not MLFLOW_TRACKING_URI:
        logger.error("MLFLOW_TRACKING_URI is missing.")
        sys.exit(1)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    # 1. Fetch Models
    logger.info(f"Fetching versions for model: {MODEL_NAME}")
    
    # Get latest Staging model
    staging_models = client.get_latest_versions(MODEL_NAME, stages=["Staging"])
    if not staging_models:
        logger.error("No 'Staging' model found. Train a model first.")
        sys.exit(1)
    staging_model = staging_models[0]
    
    # Get latest Production model
    production_models = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    production_model = production_models[0] if production_models else None

    # 2. Evaluation Logic
    staging_metric = get_metric(client, staging_model.run_id, METRIC_TO_CHECK)
    logger.info(f"Staging Model (v{staging_model.version}) {METRIC_TO_CHECK}: {staging_metric:.4f}")

    if production_model is None:
        logger.info("No Production model found. Promoting Staging immediately...")
        should_promote = True
    else:
        prod_metric = get_metric(client, production_model.run_id, METRIC_TO_CHECK)
        logger.info(f"Production Model (v{production_model.version}) {METRIC_TO_CHECK}: {prod_metric:.4f}")
        
        # Comparison: Is Staging better than Production?
        if staging_metric > prod_metric:
            logger.success(f"Staging model is better! ({staging_metric:.4f} > {prod_metric:.4f})")
            should_promote = True
        else:
            logger.warning(f"Staging model is NOT better. Keeping Production model.")
            should_promote = False

    # 3. Promotion
    if should_promote:
        logger.info(f"Promoting model version {staging_model.version} to Production...")
        
        # Move Staging -> Production
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=staging_model.version,
            stage="Production",
            archive_existing_versions=True
        )
        logger.success(f"Successfully promoted v{staging_model.version} to Production.")
    else:
        logger.info("Promotion skipped.")

if __name__ == "__main__":
    main()