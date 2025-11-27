from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from loguru import logger
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from core.services.postgres.factory import make_postgres_service

router = APIRouter()

# Whitelist of allowed tables to prevent SQL injection via dynamic table names
ALLOWED_TABLES: Dict[str, str] = {
    "predictions": "prediction_logs",
    "performance": "model_performance",
    "customers": "customers",
}


@router.get("/data/{table}", response_model=List[Dict[str, Any]])
async def get_table_data(
    table: str,
    limit: int = Query(
        default=100, ge=1, le=1000, description="Number of rows to retrieve (max 1000)."
    ),
    db: AsyncSession = Depends(make_postgres_service().get_session),
) -> List[Dict[str, Any]]:
    """
    Retrieves the most recent rows from a specified database table.

    This endpoint acts as a generic data viewer for the allowed tables.
    It performs a descending sort on the first column (assumed to be the Primary Key)
    to show the latest data.

    Args:
        table (str): The alias of the table to query. Must be one of:
                     ['predictions', 'performance', 'customers'].
        limit (int): The maximum number of rows to return. Defaults to 100.
        db (AsyncSession): The database session dependency.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries representing the table rows.

    Raises:
        HTTPException(400): If the provided table alias is invalid.
        HTTPException(500): If there is a database connection or query error.
    """
    # 1. Validation
    if table not in ALLOWED_TABLES:
        valid_options = list(ALLOWED_TABLES.keys())
        raise HTTPException(
            status_code=400,
            detail=f"Invalid table '{table}'. Allowed options: {valid_options}",
        )

    table_name = ALLOWED_TABLES[table]

    # 2. Query Execution
    try:
        # Note: We use f-string for table_name because it is validated against
        # the ALLOWED_TABLES allowlist. We use bind parameters for 'limit'.
        # 'ORDER BY 1 DESC' assumes the first column is the Primary Key/Timestamp.
        query_str = f'SELECT * FROM "{table_name}" ORDER BY 1 DESC LIMIT :limit'
        query = text(query_str)

        result = await db.execute(query, {"limit": limit})
        data = result.mappings().all()

        if not data:
            return []

        # jsonable_encoder ensures types like datetime are converted to ISO strings
        return jsonable_encoder(data)

    except SQLAlchemyError as e:
        logger.error(f"Database error fetching table '{table}': {str(e)}")
        raise HTTPException(
            status_code=500, detail="Internal Server Error: Database operation failed."
        )
    except Exception as e:
        logger.error(f"Unexpected error in get_table_data: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error.")
