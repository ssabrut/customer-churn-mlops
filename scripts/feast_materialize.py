from datetime import datetime

import pyarrow.parquet as pq
from feast import FeatureStore
from loguru import logger


def main():
    store = FeatureStore(repo_path="feature_repo")
    parquet_file_path = "/app/data/preprocessed/train.parquet"

    logger.info(f"Opening Parquet file: {parquet_file_path}")
    parquet_file = pq.ParquetFile(parquet_file_path)

    batch_size = 5000
    for i, batch in enumerate(parquet_file.iter_batches(batch_size=batch_size)):
        logger.info(f"Processing batch {i} with {len(batch)} rows...")

        df_chunk = batch.to_pandas()

        now = datetime.utcnow()
        df_chunk["event_timestamp"] = now
        df_chunk["created_timestamp"] = now

        required_cols = [
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

        df_push = df_chunk[required_cols]

        logger.info(f"Pushing batch {i} to the online store...")
        store.push(
            "customer_features_push_target",
            df_push,
        )

    logger.success("✅ Successfully ingested features into Redis!")


if __name__ == "__main__":
    main()
