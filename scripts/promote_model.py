import os
import sys
from typing import List, Optional

import mlflow
from loguru import logger
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

# Ensure project root is in sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configuration Constants
MODEL_NAME: str = "XGBoostChurnModel"
METRIC_TO_CHECK: str = "roc_auc_curve"


def validate_environment() -> str:
    """
    Validates that necessary environment variables are set.

    Args:
        None

    Returns:
        str: The validated MLFLOW_TRACKING_URI.

    Raises:
        EnvironmentError: If MLFLOW_TRACKING_URI is not set.
    """
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise EnvironmentError("MLFLOW_TRACKING_URI environment variable is missing.")
    return tracking_uri


def get_run_metric(client: MlflowClient, run_id: str, metric_name: str) -> float:
    """
    Fetches a specific metric from an MLflow run.

    Args:
        client: An initialized MlflowClient instance.
        run_id: The unique identifier for the MLflow run.
        metric_name: The name of the metric to retrieve (e.g., 'roc_auc_curve').

    Returns:
        float: The value of the requested metric.

    Raises:
        MlflowException: If the run cannot be fetched.
        KeyError: If the metric is not found in the run data.
        ValueError: If the metric value is not a valid float.
    """
    try:
        run = client.get_run(run_id)
        metrics = run.data.metrics

        if metric_name not in metrics:
            raise KeyError(f"Metric '{metric_name}' not found in run {run_id}.")

        value = metrics[metric_name]
        return float(value)

    except MlflowException as e:
        logger.error(f"MLflow API error fetching run {run_id}: {e}")
        raise
    except (ValueError, TypeError) as e:
        logger.error(f"Invalid metric value for '{metric_name}' in run {run_id}: {e}")
        raise


def get_latest_model_version(
    client: MlflowClient, model_name: str, stage: str
) -> Optional[ModelVersion]:
    """
    Retrieves the latest version of a registered model in a specific stage.

    Args:
        client: An initialized MlflowClient instance.
        model_name: The name of the registered model.
        stage: The stage to filter by (e.g., 'Staging', 'Production').

    Returns:
        Optional[ModelVersion]: The latest model version object if found, else None.

    Raises:
        MlflowException: If the query to the Model Registry fails.
    """
    try:
        versions: List[ModelVersion] = client.get_latest_versions(
            name=model_name, stages=[stage]
        )
        if not versions:
            return None
        # get_latest_versions returns a list, usually sorted by version.
        # We take the first one provided by the API as the candidate.
        return versions[0]
    except MlflowException as e:
        logger.error(
            f"Failed to fetch latest versions for {model_name} in {stage}: {e}"
        )
        raise


def main() -> None:
    """
    Executes the Model Promotion Pipeline.

    This function performs the following logic:
    1.  Validates the environment configuration.
    2.  Connects to the MLflow Model Registry.
    3.  Retrieves the latest 'Staging' model. If none exists, exits with error.
    4.  Retrieves the current 'Production' model (if any).
    5.  Compares the 'roc_auc_curve' metric of Staging vs. Production.
    6.  Promotes Staging to Production if:
        a. No Production model exists.
        b. Staging metric is strictly greater than Production metric.
    7.  Archives existing Production models upon promotion.

    Returns:
        None

    Raises:
        SystemExit: If critical steps (validation, fetching staging model) fail.
    """
    logger.info("Starting Model Promotion Pipeline...")

    # 1. Setup & Validation
    try:
        tracking_uri = validate_environment()
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()
    except EnvironmentError as e:
        logger.error(f"Configuration Error: {e}")
        sys.exit(1)

    # 2. Fetch Staging Model
    try:
        logger.info(f"Fetching 'Staging' version for model: {MODEL_NAME}")
        staging_model = get_latest_model_version(client, MODEL_NAME, "Staging")

        if staging_model is None:
            logger.error(f"No 'Staging' model found for {MODEL_NAME}. Cannot promote.")
            sys.exit(1)

        staging_metric = get_run_metric(client, staging_model.run_id, METRIC_TO_CHECK)
        logger.info(
            f"Staging Model (v{staging_model.version}) {METRIC_TO_CHECK}: {staging_metric:.4f}"
        )

    except (MlflowException, KeyError, ValueError) as e:
        logger.error(f"Fatal error evaluating Staging model: {e}")
        sys.exit(1)

    # 3. Fetch Production Model & Compare
    should_promote = False

    try:
        production_model = get_latest_model_version(client, MODEL_NAME, "Production")

        if production_model is None:
            logger.info("No 'Production' model found. Promoting Staging immediately...")
            should_promote = True
        else:
            prod_metric = get_run_metric(
                client, production_model.run_id, METRIC_TO_CHECK
            )
            logger.info(
                f"Production Model (v{production_model.version}) {METRIC_TO_CHECK}: {prod_metric:.4f}"
            )

            # Comparison Logic (Higher is Better for AUC)
            if staging_metric > prod_metric:
                diff = staging_metric - prod_metric
                logger.success(f"Staging outperforms Production by {diff:.4f}")
                should_promote = True
            else:
                logger.warning(
                    f"Staging model ({staging_metric:.4f}) is not better than "
                    f"Production ({prod_metric:.4f}). Promotion skipped."
                )
                should_promote = False

    except (MlflowException, KeyError, ValueError) as e:
        logger.error(f"Fatal error evaluating Production model: {e}")
        sys.exit(1)

    # 4. Promotion Execution
    if should_promote:
        logger.info(f"Promoting model version {staging_model.version} to Production...")
        try:
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=staging_model.version,
                stage="Production",
                archive_existing_versions=True,
            )
            logger.success(
                f"Successfully promoted v{staging_model.version} to Production."
            )
        except MlflowException as e:
            logger.error(f"Failed to transition model stage: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
