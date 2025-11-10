import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from core.schemas import ChurnResponse

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
    customer_id: int, request: Request
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
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")

    return ChurnResponse(
        prediction=int(yhat[0]),
        version=model_version,
    )
