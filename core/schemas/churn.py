from typing import Any, Dict

from pydantic import BaseModel, Field


class ChurnRequest(BaseModel):
    """
    Defines the structure of the input data for a churn prediction request.

    This model validates the incoming JSON payload, ensuring all required
    features are present and correctly typed. It uses aliases to map
    user-friendly JSON keys (e.g., "Support Calls") to Python-valid
    attribute names.

    Attributes:
        Age (float): The age of the customer.
        Support_Calls (float): The number of support calls made by the
                               customer.
        Payment_Delay (float): The average payment delay in days.
        Total_Spend (float): The total amount spent by the customer.
        Last_Interaction (float): The number of days since the customer's
                                  last interaction.
        Gender (str): The gender of the customer.
    """
    Age: float = Field(..., alias="Age")
    Support_Calls: float = Field(..., alias="Support Calls")
    Payment_Delay: float = Field(..., alias="Payment Delay")
    Total_Spend: float = Field(..., alias="Total Spend")
    Last_Interaction: float = Field(..., alias="Last Interaction")
    Gender: str = Field(..., alias="Gender")

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
    """
    Defines the structure of the response returned by the churn prediction
    endpoint.

    Attributes:
        prediction (int): The binary prediction, where 1 indicates churn
                          and 0 indicates no churn.
        probability (float): The model's confidence score (probability)
                             for the prediction (typically for class 1).
        version (Any): The version identifier of the model used to
                       generate the prediction.
        features (Dict[str, Any]): A dictionary of the original input
                                   features used for the prediction.
    """
    prediction: int
    probability: float
    version: Any
    features: Dict[str, Any]