"""
Feast feature definitions for the customer churn prediction model.

This module defines the entities, data sources (batch and push), and
feature views required to manage customer features for both offline training
and online serving.

Definitions:
    customer (Entity):
        Defines the 'customer_id' as the primary entity for feature joins.

    customer_batch_source (FileSource):
        Specifies the offline batch source for features, pointing to a
        Parquet file containing preprocessed training data.

    customer_push_source (PushSource):
        Configures the target for pushing features to the online store. It
        uses the 'customer_batch_source' as its batch counterpart for
        materialization.

    customer_features (FeatureView):
        The main feature view that groups all customer-related features.
        It links the 'customer' entity to the 'customer_push_source',
        defines the schema (name and type) for each feature, and sets a
        Time-To-Live (TTL) for online features.
"""

import sys
from datetime import timedelta

try:
    from feast import (Entity, FeatureView, Field, FileSource, PushSource,
                       ValueType)
    from feast.types import Float32, Int32
except ImportError:
    print(
        "Feast library not found. Please install with 'pip install feast'",
        file=sys.stderr,
    )
    sys.exit(1)

# --- Entity Definition ---
# Defines the primary key or join key for feature views.
customer: Entity = Entity(
    name="customer_id", value_type=ValueType.INT64, description="Customer unique ID"
)

# --- Source Definitions ---
# 1. Batch source for offline training and bootstrapping the online store.
customer_batch_source: FileSource = FileSource(
    name="customer_batch_source",
    path="/app/data/preprocessed/train.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# 2. Push source for streaming ingestion into the online store.
customer_push_source: PushSource = PushSource(
    name="customer_features_push_target",
    batch_source=customer_batch_source,
)

# --- Feature View Definition ---
# Groups a set of features and links them to a data source and entity.
customer_features: FeatureView = FeatureView(
    name="customer_features",
    entities=[customer],
    ttl=timedelta(days=1),
    schema=[
        Field(name="Age", dtype=Int32),
        Field(name="Support Calls", dtype=Int32),
        Field(name="Payment Delay", dtype=Int32),
        Field(name="Total Spend", dtype=Float32),
        Field(name="Last Interaction", dtype=Int32),
        Field(name="Churn", dtype=Int32),  # Target variable
        Field(name="Male", dtype=Int32),  # Transformed feature
        Field(name="Age_Group", dtype=Int32),  # Transformed feature
        Field(name="Interaction_Frequency", dtype=Int32),  # Transformed feature
    ],
    online=True,
    source=customer_push_source,
)
