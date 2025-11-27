from typing import cast

import pandas as pd
import pandera as pa
from pandera.typing import Series


class InputFeatures(pa.DataFrameModel):
    """
    Defines the schema and validation rules for customer churn model features.

    This class utilizes Pandera to enforce type safety, value ranges, and
    column existence on DataFrames passed to the model inference pipeline.
    It maps Python-friendly attribute names (with underscores) to the
    actual column names (with spaces) expected by the trained XGBoost model.

    Attributes:
        Age (Series[int]): Customer age. Must be between 18 and 120.
        Support_Calls (Series[int]): Number of support calls. Non-negative.
            Maps to column: "Support Calls".
        Payment_Delay (Series[int]): Days payment is delayed. Non-negative.
            Maps to column: "Payment Delay".
        Total_Spend (Series[float]): Total monetary spend. Non-negative.
            Maps to column: "Total Spend".
        Last_Interaction (Series[float]): Days since last interaction. Non-negative.
            Maps to column: "Last Interaction".
        Male (Series[int]): Gender binary flag (1 for Male, 0 for Female).
        Age_Group (Series[int]): Encoded age group category. Non-negative.
        Interaction_Frequency (Series[float]): Frequency score of interactions.
            Non-negative.
    """

    Age: Series[int] = pa.Field(ge=18, le=120, description="Customer age in years.")

    Support_Calls: Series[int] = pa.Field(
        ge=0, alias="Support Calls", description="Total number of support calls logged."
    )

    Payment_Delay: Series[int] = pa.Field(
        ge=0,
        alias="Payment Delay",
        description="Number of days the payment was delayed.",
    )

    Total_Spend: Series[float] = pa.Field(
        ge=0.0, alias="Total Spend", description="Total currency spent by the customer."
    )

    Last_Interaction: Series[float] = pa.Field(
        ge=0.0,
        alias="Last Interaction",
        description="Number of days since the last recorded interaction.",
    )

    Male: Series[int] = pa.Field(
        isin=[0, 1], description="Binary gender flag: 1 = Male, 0 = Female."
    )

    Age_Group: Series[int] = pa.Field(
        ge=0, description="Encoded ordinal category for age groups."
    )

    Interaction_Frequency: Series[float] = pa.Field(
        ge=0.0,
        alias="Interaction_Frequency",
        description="Calculated frequency of customer interactions.",
    )

    class Config:
        """
        Pandera model configuration.

        attributes:
            strict (bool): Ensures no extra columns are present in the DataFrame.
            coerce (bool): Attempts to convert types (e.g., int to float) if possible.
        """

        strict = True
        coerce = True

    @classmethod
    def validate_instances(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validates a DataFrame against the InputFeatures schema.

        This method employs 'lazy' validation, meaning it will scan the entire
        DataFrame and collect all errors before raising an exception, rather
        than stopping at the first error.

        Args:
            df: The pandas DataFrame to validate.

        Returns:
            pd.DataFrame: The validated (and potentially coerced) DataFrame.

        Raises:
            ValueError: If validation fails, containing a summary of all
                        schema violations (columns and failure cases).
        """
        try:
            # Cast return type for static analysis; validation returns DataFrame on success
            validated_df = cast(pd.DataFrame, cls.validate(df, lazy=True))
            return validated_df
        except pa.errors.SchemaErrors as e:
            # Aggregate errors into a readable dictionary for the API response
            failure_cases = e.failure_cases
            error_summary = failure_cases[["column", "check", "failure_case"]].to_dict(
                orient="records"
            )
            raise ValueError(f"Schema Validation Failed: {error_summary}")
