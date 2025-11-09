from pydantic import BaseModel, Field


class ChurnRequest(BaseModel):
    Age: float = Field(..., alias="Age")
    Support_Calls: float = Field(..., alias="Support Calls")
    Payment_Delay: float = Field(..., alias="Payment Delay")
    Total_Spend: float = Field(..., alias="Total Spend")
    Last_Interaction: float = Field(..., alias="Last Interaction")
    Gender: str = Field("Gender")

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "Age": 23.0,
                "Gender": "Male",
                "Support Calls": 0.0,
                "Payment Delay": 19.0,
                "Total Spend": 846.18,
                "Last Interaction": 18.0,
            }
        }


class ChurnResponse(BaseModel):
    prediction: int
    probability: float
    version: str
