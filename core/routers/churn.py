import time
from typing import Any, Dict, List

import pandas as pd
import pandera as pa
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from feast import FeatureStore
from loguru import logger
from numpy import ndarray
from sqlalchemy import text

from core.schemas import ChurnResponse
from core.services.postgres import factory
from core.schemas.validation import InputFeatures
from core.services.mlflow import ModelManager

router = APIRouter()

LOG_SQL = text("""
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
""")


async def log_prediction_background(db_session_factory, log_entry: dict):
    async with db_session_factory() as db:
        try:
            await db.execute(LOG_SQL, log_entry)
            await db.commit()
        except Exception as e:
            # Log to file/stderr so you don't lose visibility of DB errors
            logger.error(f"Failed to log prediction: {e}")


@router.get("/predict/{customer_id}", response_model=ChurnResponse)
async def predict_customer_churn(
    customer_id: int, request: Request, background_tasks: BackgroundTasks
) -> ChurnResponse:
    """
    Performs a churn prediction for a given customer ID.

    This endpoint retrieves real-time features from the Feast feature
    store, executes a prediction using the loaded ML pipeline, and
    asynchronously logs the prediction details to the database.

    Args:
        customer_id (int): The unique identifier for the customer.
        request (Request): The incoming FastAPI request object, used to
                           access application state (model, feature store).
        db (AsyncSession): The injected asynchronous database session.

    Returns:
        ChurnResponse: An object containing the prediction (0 or 1),
                       the probability, the model version, and the
                       features used for the prediction.

    Raises:
        HTTPException:
            - 503: If the ML model or Feast feature store is not
                   initialized in the application state.
            - 502: If retrieving features from the upstream Feast store
                   fails.
            - 400: If an error occurs during the model prediction
                   (e.g., data mismatch, pipeline failure).
    """
    start_time: float = time.perf_counter()

    try:
        model_manager = request.app.state.model_manager
        feast_store: FeatureStore = request.app.state.feast_store
        pipeline, model_version = await model_manager.get_production_model()
        shadow_pipeline, shadow_version = await model_manager.get_shadow_model()
    except AttributeError as e:
        # Catch if .model, .feast_store, etc. don't exist at all
        raise HTTPException(
            status_code=503,
            detail=f"Server state is missing required attributes: {e}",
        )

    if pipeline is None or feast_store is None:
        raise HTTPException(
            status_code=503,
            detail="Model or Feast Store not initialized. Check server logs.",
        )

    FEATURE_ORDER: List[str] = pipeline.feature_names_in_.tolist()
    FEAST_REQUEST_FEATURES: List[str] = [
        f"customer_features:{name}" for name in FEATURE_ORDER
    ]

    try:
        online_features: Dict[str, Any] = feast_store.get_online_features(
            features=FEAST_REQUEST_FEATURES,
            entity_rows=[{"customer_id": customer_id}],
        ).to_dict()
    except Exception as e:
        # Handle failure to connect to the upstream feature store
        raise HTTPException(
            status_code=502,  # Bad Gateway
            detail=f"Failed to retrieve features from feature store: {e}",
        )

    feature_data: Dict[str, Any] = {
        key: val[0] for key, val in online_features.items() if key != "customer_id"
    }

    input_df: pd.DataFrame = pd.DataFrame([feature_data], columns=FEATURE_ORDER)

    # Validation block
    try:
        validated_df = InputFeatures.validate(input_df)
    except pa.errors.SchemaError as e:
        logger.error(f"Data Validation Failed: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid feature data: {e.failure_cases}"
        )
    
    try:
        yhat: ndarray = pipeline.predict(validated_df)
        yhat_proba: ndarray = pipeline.predict_proba(validated_df)
        probability: float = float(yhat_proba[0][1])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")

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
        "prediction": int(yhat[0]),
        "probability": float(probability),
        "ground_truth": feature_data.get("Churn"),
        "response_time_ms": response_time_ms,
        "is_shadow": False
    }

    postgres_factory = factory.make_postgres_service()
    background_tasks.add_task(
        log_prediction_background, postgres_factory.session_maker, log_entry
    )

    try:
        yhat_proba = shadow_pipeline.predict_proba(input_df)
        shadow_probability = float(yhat_proba[0][1])
        shadow_prediction = int(shadow_probability > 0.5)

        shadow_entry = {
            "model_version": shadow_version,
            "customer_id": customer_id,
            "age": feature_data.get("Age"),
            "support_calls": feature_data.get("Support Calls"),
            "payment_delay": feature_data.get("Payment Delay"),
            "total_spend": feature_data.get("Total Spend"),
            "last_interaction": feature_data.get("Last Interaction"),
            "gender": feature_data.get("Male"),
            "age_group": feature_data.get("Age_Group"),
            "interaction_frequency": feature_data.get("Interaction_Frequency"),
            "prediction": shadow_prediction,
            "probability": shadow_probability,
            "ground_truth": feature_data.get("Churn"),
            "response_time_ms": 0, 
            "is_shadow": True
        }
        
        background_tasks.add_task(
            log_prediction_background, postgres_factory.session_maker, shadow_entry
        )
    except Exception as e:
        logger.warning(f"Shadow prediction error: {e}")

    return ChurnResponse(
        prediction=int(yhat[0]),
        probability=probability,
        version=model_version,
        features=feature_data,
    )
