import os
import sys
from loguru import logger
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

MODEL_NAME = "XGBoostChurnModel"
COMPARISON_METRIC = "f1_score"

mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5050")
if not mlflow_tracking_uri:
    logger.error("MLFLOW_TRACKING_URI environment variable not set.")
    sys.exit(1)

client = MlflowClient(tracking_uri=mlflow_tracking_uri)
logger.info(f"Connected to MLflow at {mlflow_tracking_uri}")

try:
    staging_versions = client.get_latest_versions(MODEL_NAME, stages=["Staging"])
    if not staging_versions:
        logger.info("No models found in 'Staing'. Exiting")
        sys.exit(0)

    new_version = staging_versions[0]
    new_run = client.get_run(new_version.run_id)
    new_metric = new_run.data.metrics[COMPARISON_METRIC]
    logger.info(f"New 'Staging' model found: Version {new_version.version} - {COMPARISON_METRIC}: {new_metric:.4f}")
    
except Exception as e:
    logger.error(f"Failed to get 'Staging' model: {e}")
    sys.exit(1)

try:
    prod_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    if not prod_versions:
        logger.info("No 'Production' model found. Promoting new model...")
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=new_version.version,
            stage="Production",
            archive_existing_versions=False
        )

        logger.success(f"Promoted Version {new_version.version} to 'Prodution'")
        sys.exit(0)

    prod_version = prod_versions[0]
    prod_run = client.get_run(prod_version.run_id)
    prod_metric = prod_run.data.metrics[COMPARISON_METRIC]
    logger.info(f"Current 'Production' model found: Version {prod_version.version} - {COMPARISON_METRIC}: {prod_metric:.4f}")
except Exception as e:
    logger.error(f"Failed to get 'Production' model: {e}")
    sys.exit(1)

if new_metric > prod_metric:
    logger.success(f"New model (v{new_version.version}) is better! Promoting to 'Production'")
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=new_version.version,
        stage="Production",
        archive_existing_versions=True
    )
else:
    logger.warning(f"New model (v{new_version.version}) is not better. Archiving!")
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=new_version.version,
        stage="Archived"
    )

logger.info("Promotion script finished")