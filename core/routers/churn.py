import pandas as pd
from fastapi import APIRouter, HTTPException, Request, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from loguru import logger

from core.schemas import ChurnResponse
from core.services.postgres import factory

router = APIRouter()

FEATURE_ORDER = [
    "Age",
    "Support Calls",
    "Payment Delay",
    "Total Spend",
    "Last Interaction",
    "Male",
    "Age_Group",
    "Interaction_Frequency",
]

FEAST_REQUEST_FEATURES = [f"customer_features:{name}" for name in FEATURE_ORDER]


@router.post("/predict/{customer_id}", response_model=ChurnResponse)
async def predict_customer_churn(
    customer_id: int, request: Request, db: AsyncSession = Depends(factory.make_postgres_service().get_session)
) -> ChurnResponse:
    pipeline = request.app.state.model
    feast_store = request.app.state.feast_store
    model_version = request.app.state.model_version

    if pipeline is None or feast_store is None:
        raise HTTPException(
            status_code=503, detail="Model or Feast Store not initialized. Check server logs."
        )

    online_features = feast_store.get_online_features(
        features=FEAST_REQUEST_FEATURES,
        entity_rows=[{"customer_id": customer_id}],
    ).to_dict()

    feature_data = {
        key: val[0]
        for key, val in online_features.items()
        if key != "customer_id"
    }

    input_df = pd.DataFrame([feature_data], columns=FEATURE_ORDER)
    try:
        yhat = pipeline.predict(input_df)
        yhat_proba = pipeline.predict_proba(input_df)
        probability = yhat_proba[0][1]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")

    try:
        log_entry = {
            "model_version": model_version,
            "customer_id": customer_id,
            "age": feature_data.get("Age"),
            "total_spend": feature_data.get("Total Spend"),
            "payment_delay": feature_data.get("Payment Delay"),
            "prediction": int(yhat[0]),
            "probability": float(probability),
        }
        
        # Use sqlalchemy 'text' for safe parameter binding
        log_sql = text("""
            INSERT INTO prediction_logs (
                model_version, customer_id, age, total_spend, 
                payment_delay, prediction, probability
            )
            VALUES (
                :model_version, :customer_id, :age, :total_spend, 
                :payment_delay, :prediction, :probability
            )
        """)
        
        await db.execute(log_sql, log_entry)
        await db.commit()
        
    except Exception as e:
        logger.error(f"Failed to log prediction: {e}")

    return ChurnResponse(
        prediction=int(yhat[0]),
        probability=probability,
        version=model_version,
    )
