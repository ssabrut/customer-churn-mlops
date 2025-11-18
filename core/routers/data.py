from loguru import logger
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException
from fastapi.encoders import jsonable_encoder
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from core.services.postgres.factory import make_postgres_service

router = APIRouter()

ALLOWED_TABLES = {
    "predictions": "prediction_logs",
    "performance": "model_performance",
    "customers": "customers"
}

@router.get("/data/{table}", response_model=List[Dict, str, Any])
async def get_table_data(
    table: str,
    db: AsyncSession = Depends(make_postgres_service().get_session)
):
    if table not in ALLOWED_TABLES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid tbale. Allowed tables are: {list(ALLOWED_TABLES.keys())}"
        )

    table_name = ALLOWED_TABLES[table]
    try:
        query = text(f'SELECT * FROM "{table_name}" ORDER BY 1 DESC LIMIT 100')
        result = await db.execute(query)
        data = result.mappings().all()

        if not data:
            return []

        return jsonable_encoder(data)
    except Exception as e:
        logger.error(f"Error fetching data for table {table}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Database query error: {e}"
        )