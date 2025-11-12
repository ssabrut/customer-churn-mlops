from datetime import timedelta
from feast import (
    Entity,
    FeatureView,
    Field,
    FileSource,
    PushSource,
    ValueType,
)
from feast.types import Float32, Int32

customer = Entity(
    name="customer_id", value_type=ValueType.INT64, description="Customer unique ID"
)

customer_batch_source = FileSource(
    name="customer_batch_source",
    path="/app/data/preprocessed/train.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

customer_push_source = PushSource(
    name="customer_features_push_target",
    batch_source=customer_batch_source,
)

customer_features = FeatureView(
    name="customer_features", 
    entities=[customer],
    ttl=timedelta(days=1),
    schema=[
        Field(name="Age", dtype=Int32),
        Field(name="Support Calls", dtype=Int32),
        Field(name="Payment Delay", dtype=Int32),
        Field(name="Total Spend", dtype=Float32),
        Field(name="Last Interaction", dtype=Int32),
        Field(name="Churn", dtype=Int32),
        Field(name="Male", dtype=Int32),
        Field(name="Age_Group", dtype=Int32),
        Field(name="Interaction_Frequency", dtype=Int32),
    ],
    online=True,
    source=customer_push_source,
)