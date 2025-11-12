import os
import sys
from datetime import datetime
from typing import List

import pandas as pd
import pyarrow.parquet as pq
from feast import FeatureStore
from feast.errors import FeastConfigError
from loguru import logger
from pyarrow.lib import ArrowInvalid


def main() -> None:
    """
    Materializes features from a preprocessed Parquet file into the Feast
    online store (Redis).

    This function initializes the Feast FeatureStore, reads the data in
    batches from the specified Parquet file, performs a final check
    for required columns, and pushes each batch to the
    'customer_features_push_target'.

    Returns:
        None
    """
    try:
        store: FeatureStore = FeatureStore(repo_path="feature_repo")
    except (FeastConfigError, Exception) as e:
        logger.error(f"Fatal Error: Failed to initialize Feast FeatureStore: {e}")
        logger.error("Ensure 'feast_repo' exists and configuration is correct.")
        sys.exit(1)

    parquet_file_path: str = "/app/data/preprocessed/train.parquet"

    # --- 1. File Validation ---
    if not os.path.exists(parquet_file_path):
        logger.error(f"Fatal Error: Parquet file not found at: {parquet_file_path}")
        sys.exit(1)

    try:
        logger.info(f"Opening Parquet file: {parquet_file_path}")
        parquet_file: pq.ParquetFile = pq.ParquetFile(parquet_file_path)

        batch_size: int = 5000
        required_cols: List[str] = [
            "customer_id",
            "Age",
            "Support Calls",
            "Payment Delay",
            "Total Spend",
            "Last Interaction",
            "Churn",
            "Male",
            "Age_Group",
            "Interaction_Frequency",
            "event_timestamp",
            "created_timestamp",
        ]
        required_cols_set = set(required_cols)

        # --- 2. Batch Processing ---
        for i, batch in enumerate(parquet_file.iter_batches(batch_size=batch_size)):
            logger.info(f"Processing batch {i} with {len(batch)} rows...")

            df_chunk: pd.DataFrame = batch.to_pandas()

            # Timestamps are overwritten here for push, as per original logic
            now: datetime = datetime.utcnow()
            df_chunk["event_timestamp"] = now
            df_chunk["created_timestamp"] = now

            # --- 3. Column Validation ---
            if not required_cols_set.issubset(df_chunk.columns):
                missing = required_cols_set - set(df_chunk.columns)
                logger.error(f"Batch {i} is missing required columns: {missing}")
                logger.error("Skipping this batch.")
                continue

            df_push: pd.DataFrame = df_chunk[required_cols]

            # --- 4. Store Push ---
            logger.info(f"Pushing batch {i} to the online store...")
            store.push(
                "customer_features_push_target",
                df_push,
            )

    except ArrowInvalid as e:
        logger.error(f"Fatal Error: Failed to read Parquet file. File may be corrupt: {e}")
        sys.exit(1)
    except KeyError as e:
        logger.error(
            f"Fatal Error: A required column was missing during processing: {e}"
        )
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal Error: An unexpected error occurred during batch processing or push: {e}")
        sys.exit(1)

    logger.success("✅ Successfully ingested features into Redis!")


if __name__ == "__main__":
    main()