import pandas as pd
from fastapi import APIRouter, Request, HTTPException

from core.schemas import ChurnResponse, ChurnRequest

router = APIRouter()

@router.post("/predict", response_model=ChurnResponse)
async def predict_customer_churn(customer_data: ChurnRequest, request: Request) -> ChurnResponse:
    pipeline = request.app.state.model
    model_version = request.app.state.model_version

    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model pipeline is not loaded. Check server logs.")

    input_df = pd.DataFrame([customer_data.model_dump(by_alias=True)])
    try:
        yhat = pipeline.predict(input_df)
        yhat_proba = pipeline.predict_proba(input_df)
        churn_probability = yhat_proba[0][1]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")

    return ChurnResponse(
        prediction=int(yhat[0]),
        probability=float(churn_probability),
        model_version=model_version
    )