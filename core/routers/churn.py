import pandas as pd
from fastapi import APIRouter, Request, HTTPException

from core.schemas import ChurnResponse, ChurnRequest
from core.transformer import ChurnFeatureTransformer

router = APIRouter()

@router.post("/predict", response_model=ChurnResponse)
async def predict_customer_churn(customer_data: ChurnRequest, request: Request) -> ChurnResponse:
    model = request.app.state.model
    scaler = request.app.state.scaler
    model_version = request.app.state.model_version
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Check server logs.")

    input_df = pd.DataFrame([customer_data.model_dump(by_alias=True)])
    transformer = ChurnFeatureTransformer()
    transformed_data = transformer.transform(input_df)
    scaled_data = pd.DataFrame(scaler.transform(transformed_data), columns=transformed_data.columns)

    try:
        yhat_proba = model.predict(scaled_data)
        yhat = (yhat_proba > 0.5).astype(int)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")

    return ChurnResponse(prediction=int(yhat[0]), version=model_version)