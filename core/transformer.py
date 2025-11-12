from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from core.utils.binning import AgeBinner, InteractionBinner
from core.utils.encoding import GenderEncoder
from core.utils.mapping import AgeInteractionMapper


class ChurnFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    A custom transformer pipeline for processing churn data.

    This class chains several preprocessing steps:
    1.  Drops non-informative columns ('Tenure', 'Usage Frequency', etc.).
    2.  Optionally drops the target variable 'Churn' (if 'train=False').
    3.  Applies GenderEncoder to convert gender to a numerical format.
    4.  Applies AgeBinner to create age groups.
    5.  Applies InteractionBinner to create interaction frequency groups.
    6.  Applies AgeInteractionMapper for feature cross-binning.

    This class adheres to the scikit-learn TransformerMixin interface,
    allowing it to be used in 'sklearn.pipeline.Pipeline'.

    Methods:
        fit(X, y=None):
            Returns 'self' as this is a stateless transformer.
        transform(X, y=None, train=True):
            Applies the sequential transformations to the input DataFrame.
    """

    def __init__(self) -> None:
        """
        Initializes the ChurnFeatureTransformer.
        """
        super().__init__()

    def fit(self, X: pd.DataFrame, y: Any = None) -> "ChurnFeatureTransformer":
        """
        Fits the transformer to the data.

        Since this transformer is stateless (all sub-transformers are
        stateless), this method does nothing and returns 'self'.

        Args:
            X (pd.DataFrame): The input data.
            y (Any): The target data (ignored).

        Returns:
            ChurnFeatureTransformer: The fitted transformer instance.
        """
        return self

    def transform(
        self, X: pd.DataFrame, y: Any = None, train: bool = True
    ) -> pd.DataFrame:
        """
        Transforms the input DataFrame by applying the feature engineering
        pipeline.

        Args:
            X (pd.DataFrame): The input data to transform.
            y (Any): The target data (ignored).
            train (bool): If True, the 'Churn' column is kept. If False,
                          it is dropped.

        Returns:
            pd.DataFrame: The transformed DataFrame.

        Raises:
            TypeError: If the input 'X' is not a pandas DataFrame.
            ValueError: If the input DataFrame 'X' is empty, or if a
                        transformation step fails (e.g., due to missing
                        columns required by sub-transformers).
        """
        # --- Validation ---
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Input must be a pandas DataFrame, got {type(X)}")
        if X.empty:
            raise ValueError("Input DataFrame is empty")

        _X = X.copy()
        cols_to_drop = [
            "Tenure",
            "Usage Frequency",
            "Subscription Type",
            "Contract Length",
        ]

        if all(col in _X.columns for col in cols_to_drop):
            _X = _X.drop(cols_to_drop, axis=1)

        if not train:
            if "Churn" in _X.columns:
                _X = _X.drop(["Churn"], axis=1)

        # --- Transformation Pipeline ---
        try:
            _X = GenderEncoder().transform(_X)
            _X = AgeBinner().transform(_X)
            _X = InteractionBinner().transform(_X)
            _X = AgeInteractionMapper().transform(_X)
        except (KeyError, ValueError) as e:
            raise ValueError(
                f"Transformation failed. Check if all required columns "
                f"(e.g., 'Gender', 'Age') are present and have valid data. "
                f"Error: {e}"
            )
        except Exception as e:
            raise ValueError(f"An unexpected error occurred during transformation: {e}")

        return _X