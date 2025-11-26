import pandera as pa
from pandera.typing import Series

class InputFeatures(pa.DataFrameModel):
    Age: Series[int] = pa.Field(ge=18, le=120, description="Customer age")
    Support_Calls: Series[int] = pa.Field(ge=0, alias="Support Calls", description="Number of support calls")
    Payment_Delay: Series[int] = pa.Field(ge=0, alias="Payment Delay", description="Days delayed")
    Total_Spend: Series[float] = pa.Field(ge=0.0, alias="Total Spend", description="Total money spent")
    Last_Interaction: Series[float] = pa.Field(ge=0.0, alias="Last Interaction", description="Days since last interaction")
    
    Male: Series[int] = pa.Field(isin=[0, 1], description="Gender binary flag")
    Age_Group: Series[int] = pa.Field(ge=0, description="Encoded age group")
    Interaction_Frequency: Series[float] = pa.Field(ge=0.0, alias="Interaction_Frequency")

    class Config:
        strict = True
        coerce = True