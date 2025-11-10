from feast import Entity, FeatureView, FileSource, Field, ValueType
from feast.types import Float32, Int32
from datetime import timedelta

# Define the Entity
customer = Entity(
    name="customer_id", 
    value_type=ValueType.INT64, 
    description="Customer unique ID"
)

# Define the Offline Data Source
customer_source = FileSource(
    path="../data/preprocessed/train.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp"
)

# Define the Feature View
customer_features = FeatureView(
    name="customer_features",
    entities=[customer], 
    ttl=timedelta(days=365),
    source=customer_source,
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
)