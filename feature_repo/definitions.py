from feast import Entity, FeatureView, FileSource, Field, ValueType
from feast.types import Int64, Float32, String
from datetime import timedelta

# Define the Entity
customer = Entity(
    name="customer_id", 
    value_type=ValueType.INT64, 
    description="Customer unique ID"
)

# Define the Offline Data Source
customer_source = FileSource(
    path="/app/data/preprocessed/train.parquet",
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
        Field(name="Age", dtype=Int64),
        Field(name="Support_Calls", dtype=Int64),
        Field(name="Payment_Delay", dtype=Float32),
        Field(name="Total_Spend", dtype=Float32),
        Field(name="Last_Interaction", dtype=Float32),
        Field(name="Gender", dtype=String),
    ],
)