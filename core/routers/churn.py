from fastapi import APIRouter, Request, HTTPException

from core.schemas import ChurnResponse, ChurnRequest

router = APIRouter()

@router.post("/predict", response_model=ChurnResponse)
async def predict_customer_churn(customer_data: ChurnRequest, request: Request) -> ChurnResponse:
    model = request.app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Check server logs.")