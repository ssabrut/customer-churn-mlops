from typing import Any, Dict

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class AgeInteractionMapper(BaseEstimator, TransformerMixin):
    """
    A transformer that maps categorical age and interaction groups to
    numerical (ordinal) values.

    This class stores predefined dictionaries to convert string categories
    (e.g., "Young Adult", "Highly Active") into integer representations
    (e.g., 0, 1). It is designed to be used in a pipeline after binning
    transformers like 'AgeBinner' and 'InteractionBinner'.

    Methods:
        __init__(): Initializes the transformer and its mappings.
        fit(X, y=None): Returns 'self' as it is a stateless transformer.
        transform(X): Applies the mappings to the relevant columns.
    """

    def __init__(self) -> None:
        """
        Initializes the AgeInteractionMapper and defines the static mappings.
        """
        self.AGE_MAPPING: Dict[str, int] = {
            "Young Adult": 0,
            "Adult": 1,
            "Mid-Career": 2,
            "Senior": 3,
        }
        self.INTERACTION_MAPPING: Dict[str, int] = {
            "Highly Active": 0,
            "Active": 1,
            "Dormant": 2,
        }

    def fit(self, X: pd.DataFrame, y: Any = None) -> "AgeInteractionMapper":
        """
        Fits the transformer to the data.

        Since this transformer is stateless (it only applies a
        predefined mapping), this method does nothing and returns 'self'.

        Args:
            X (pd.DataFrame): The input data.
            y (Any): The target data (ignored).

        Returns:
            AgeInteractionMapper: The fitted transformer instance.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the categorical-to-numerical mapping on 'Age_Group' and
        'Interaction_Frequency' columns.

        Args:
            X (pd.DataFrame): The input data to transform. Must contain
                              'Age_Group' and 'Interaction_Frequency'
                              columns with the expected categories.

        Returns:
            pd.DataFrame: The DataFrame with the mapped numerical columns.

        Raises:
            TypeError: If the input 'X' is not a pandas DataFrame.
            ValueError: If the input DataFrame 'X' is empty, if the
                        required 'Age_Group' or 'Interaction_Frequency'
                        columns are missing, or if an unexpected error
                        occurs during the mapping.
        """
        # --- Validation ---
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Input must be a pandas DataFrame, got {type(X)}")
        if X.empty:
            raise ValueError("Input DataFrame is empty")

        _X = X.copy()

        # --- Column Validation / Error Handling ---
        if "Age_Group" not in _X.columns:
            raise ValueError("Transformation failed: 'Age_Group' column not found.")
        if "Interaction_Frequency" not in _X.columns:
            raise ValueError(
                "Transformation failed: 'Interaction_Frequency' column not found."
            )

        # --- Transformation ---
        try:
            _X["Age_Group"] = _X["Age_Group"].map(self.AGE_MAPPING)
            _X["Interaction_Frequency"] = _X["Interaction_Frequency"].map(
                self.INTERACTION_MAPPING
            )
        except Exception as e:
            raise ValueError(
                f"An unexpected error occurred during .map() operation: {e}"
            )

        return _X
