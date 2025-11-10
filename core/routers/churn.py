import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from feast import FeatureStore

from core.schemas import ChurnRequest, ChurnResponse

router = APIRouter()
store = FeatureStore(repo_path="/app/feature_repo")


@router.post("/predict/{customer_id}", response_model=ChurnResponse)
async def predict_customer_churn(
    customer_id: int, request: Request, input_data: ChurnRequest
) -> ChurnResponse:
    pipeline = request.app.state.model
    model_version = request.app.state.model_version

    if pipeline is None:
        raise HTTPException(
            status_code=503, detail="Model pipeline is not loaded. Check server logs."
        )

    online_features = store.get_online_features(
        features=[
            "customer_features:Age",
            "customer_features:Support_Calls",
            "customer_features:Payment_Delay",
            "customer_features:Total_Spend",
            "customer_features:Last_Interaction",
            "customer_features:Gender",
        ],
        entity_rows=[{"customer_id": customer_id}],
    ).to_dict()

    feature_data = {
        key.split(":")[1]: val[0]
        for key, val in online_features.items()
        if key != "customer_id"
    }

    input_df = pd.DataFrame([feature_data])
    try:
        yhat = pipeline.predict(input_df)
        yhat_proba = pipeline.predict_proba(input_df)
        churn_probability = yhat_proba[0][1]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")

    return ChurnResponse(
        prediction=int(yhat[0]),
        probability=float(churn_probability),
        model_version=model_version,
    )
