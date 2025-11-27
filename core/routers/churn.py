import time
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from feast import FeatureStore
from loguru import logger
from numpy import ndarray
from sqlalchemy import text

from core.schemas import ChurnResponse
from core.schemas.validation import InputFeatures
from core.services.mlflow import ModelManager
from core.services.postgres import factory

router = APIRouter()

# SQL for high-throughput insertion
LOG_SQL = text(
    """
    INSERT INTO prediction_logs (
        model_version, customer_id, age, support_calls, payment_delay,
        total_spend, last_interaction, gender, age_group, interaction_frequency,
        prediction, probability, ground_truth, response_time_ms, is_shadow
    )
    VALUES (
        :model_version, :customer_id, :age, :support_calls, :payment_delay,
        :total_spend, :last_interaction, :gender, :age_group, :interaction_frequency,
        :prediction, :probability, :ground_truth, :response_time_ms, :is_shadow
    )
"""
)


async def log_prediction_background(
    session_maker: Any, log_entry: Dict[str, Any]
) -> None:
    """
    Asynchronously logs prediction details to the database.

    Args:
        session_maker: The SQLAlchemy session factory.
        log_entry: Dictionary containing the data to insert.

    Returns:
        None
    """
    async with session_maker() as db:
        try:
            await db.execute(LOG_SQL, log_entry)
            await db.commit()
        except Exception as e:
            # We log locally; we do not raise, as this is a background task
            logger.error(f"Background Logging Failed: {e}")


@router.get("/predict/{customer_id}", response_model=ChurnResponse)
async def predict_customer_churn(
    customer_id: int, request: Request, background_tasks: BackgroundTasks
) -> ChurnResponse:
    """
    Orchestrates the churn prediction workflow for a specific customer.

    This endpoint:
    1. Validates the server state (Model Manager, Feature Store).
    2. Fetches real-time features from Feast.
    3. Validates data against the Pandera schema.
    4. Executes inference on the Production model.
    5. Queues a background task to log the result.
    6. Attempts a 'Shadow' prediction (Canary/Staging model) for comparison.

    Args:
        customer_id: The unique identifier of the customer.
        request: The FastAPI Request object containing app state.
        background_tasks: FastAPI utility for background execution.

    Returns:
        ChurnResponse: The prediction result and metadata.

    Raises:
        HTTPException(503): If the model or feature store is unavailable.
        HTTPException(502): If the Feature Store connection fails.
        HTTPException(400): If input data is invalid or inference fails.
    """
    start_time: float = time.perf_counter()

    # --- 1. State Validation ---
    try:
        model_manager: ModelManager = request.app.state.model_manager
        feast_store: FeatureStore = request.app.state.feast_store

        # Retrieve pipelines
        pipeline, model_version = await model_manager.get_production_model()
        shadow_pipeline, shadow_version = await model_manager.get_shadow_model()

        if pipeline is None or feast_store is None:
            raise AttributeError("Pipeline or FeatureStore is None")

    except AttributeError as e:
        logger.error(f"Service State Error: {e}")
        raise HTTPException(
            status_code=503,
            detail="Prediction service is not fully initialized.",
        )

    # --- 2. Feature Retrieval ---
    # We define feature naming conventions explicitly
    feature_names: List[str] = pipeline.feature_names_in_.tolist()
    feast_keys: List[str] = [f"customer_features:{name}" for name in feature_names]

    try:
        online_features_response = feast_store.get_online_features(
            features=feast_keys,
            entity_rows=[{"customer_id": customer_id}],
        ).to_dict()
    except Exception as e:
        logger.error(f"Feast Retrieval Error: {e}")
        raise HTTPException(
            status_code=502,
            detail="Failed to retrieve features from the Feature Store.",
        )

    # Defensive parsing: Feast returns lists. If the ID doesn't exist,
    # or the feature is null, we must handle it gracefully.
    feature_data: Dict[str, Any] = {}
    for key, values in online_features_response.items():
        if key == "customer_id":
            continue
        # Extract value safely
        feature_data[key] = values[0] if values and len(values) > 0 else None

    # --- 3. Schema Validation ---
    try:
        # Create DataFrame and Reorder columns to match model expectation
        input_df = pd.DataFrame([feature_data])

        # Ensure all columns exist, even if missing (Pandera will catch nulls)
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = None

        input_df = input_df[feature_names]

        # Use the custom validation method defined in schemas.py
        validated_df = InputFeatures.validate_instances(input_df)

    except ValueError as e:
        logger.warning(f"Validation Error for customer {customer_id}: {e}")
        raise HTTPException(status_code=400, detail=f"Data Validation Failed: {str(e)}")

    # --- 4. Production Inference ---
    try:
        yhat: ndarray = pipeline.predict(validated_df)
        yhat_proba: ndarray = pipeline.predict_proba(validated_df)
        probability: float = float(yhat_proba[0][1])
        prediction: int = int(yhat[0])
    except Exception as e:
        logger.error(f"Inference Error: {e}")
        raise HTTPException(status_code=400, detail="Model inference failed.")

    # --- 5. Logging (Background) ---
    end_time: float = time.perf_counter()
    response_time_ms: int = int((end_time - start_time) * 1000)

    log_entry: Dict[str, Any] = {
        "model_version": str(model_version),
        "customer_id": customer_id,
        "age": feature_data.get("Age"),
        "support_calls": feature_data.get("Support Calls"),
        "payment_delay": feature_data.get("Payment Delay"),
        "total_spend": feature_data.get("Total Spend"),
        "last_interaction": feature_data.get("Last Interaction"),
        "gender": feature_data.get("Male"),
        "age_group": feature_data.get("Age_Group"),
        "interaction_frequency": feature_data.get("Interaction_Frequency"),
        "prediction": prediction,
        "probability": probability,
        "ground_truth": feature_data.get("Churn"),  # May be None
        "response_time_ms": response_time_ms,
        "is_shadow": False,
    }

    postgres_svc = factory.make_postgres_service()
    background_tasks.add_task(
        log_prediction_background, postgres_svc.session_maker, log_entry
    )

    # --- 6. Shadow Model Execution ---
    # Runs silently; failures here must NOT affect the user response.
    if shadow_pipeline and shadow_version:
        try:
            shadow_probs = shadow_pipeline.predict_proba(validated_df)
            shadow_prob_val = float(shadow_probs[0][1])
            shadow_pred_val = int(shadow_prob_val > 0.5)

            shadow_entry = log_entry.copy()
            shadow_entry.update(
                {
                    "model_version": str(shadow_version),
                    "prediction": shadow_pred_val,
                    "probability": shadow_prob_val,
                    "response_time_ms": 0,  # Not relevant for shadow
                    "is_shadow": True,
                }
            )

            background_tasks.add_task(
                log_prediction_background, postgres_svc.session_maker, shadow_entry
            )
        except Exception as e:
            logger.warning(f"Shadow Pipeline Error (Ignored): {e}")

    return ChurnResponse(
        prediction=prediction,
        probability=probability,
        version=str(model_version),
        features=feature_data,
    )
