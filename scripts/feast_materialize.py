from datetime import datetime

import pandas as pd
from feast import FeatureStore
from loguru import logger


def main():
    store = FeatureStore(repo_path="feature_repo")

    df = pd.read_parquet("/app/data/preprocessed/train.parquet")

    now = datetime.utcnow()
    df["event_timestamp"] = now
    df["created_timestamp"] = now

    logger.info("Pushing features to the online store...")

    store.push(
        "customer_features_push_target",
        df[
            [
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
        ],
    )

    logger.success("✅ Successfully ingested features into Redis!")


if __name__ == "__main__":
    main()
