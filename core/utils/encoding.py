from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class GenderEncoder(BaseEstimator, TransformerMixin):
    """
    A transformer that one-hot encodes the 'Gender' column.

    This transformer converts the categorical 'Gender' column into a
    binary numerical 'Male' column. It creates dummies for both 'Male'
    and 'Female' and then drops 'Female' to avoid multicollinearity
    (dummy variable trap). It also handles the edge case where only
    'Female' might be present in the data by creating a 'Male' column
    of zeros.

    Methods:
        __init__(): Initializes the transformer.
        fit(X, y=None): Returns 'self' as it is a stateless transformer.
        transform(X, y=None): Applies the gender encoding.
    """

    def __init__(self) -> None:
        """
        Initializes the GenderEncoder.
        """
        super().__init__()

    def fit(self, X: pd.DataFrame, y: Any = None) -> "GenderEncoder":
        """
        Fits the transformer to the data.

        Since this transformer is stateless, this method does nothing
        and returns 'self'.

        Args:
            X (pd.DataFrame): The input data.
            y (Any): The target data (ignored).

        Returns:
            GenderEncoder: The fitted transformer instance.
        """
        return self

    def transform(self, X: pd.DataFrame, y: Any = None) -> pd.DataFrame:
        """
        Applies one-hot encoding to the 'Gender' column.

        It creates a 'Male' column (int8) and drops the original
        'Gender' column, as well as the 'Female' dummy column.

        Args:
            X (pd.DataFrame): The input data to transform.
            y (Any): The target data (ignored).

        Returns:
            pd.DataFrame: The transformed DataFrame with 'Gender'
                          replaced by 'Male'.

        Raises:
            TypeError: If the input 'X' is not a pandas DataFrame.
            ValueError: If the input DataFrame 'X' is empty, or if the
                        'Gender' column is missing.
        """
        # --- Validation ---
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Input must be a pandas DataFrame, got {type(X)}")
        if X.empty:
            raise ValueError("Input DataFrame is empty")
        if "Gender" not in X.columns:
            raise ValueError(
                "Transformation failed: 'Gender' column not found in DataFrame."
            )

        _X = X.copy()

        # --- Transformation ---
        try:
            gender_ohe = pd.get_dummies(_X["Gender"], dtype=np.int8)
            _X = _X.drop(["Gender"], axis=1)
            _X = _X.join(gender_ohe)

            if "Female" in _X.columns and "Male" in _X.columns:
                _X = _X.drop(["Female"], axis=1)
            elif "Female" in _X.columns:
                _X["Male"] = 0
                _X = _X.drop(["Female"], axis=1)

        except (KeyError, ValueError) as e:
            raise ValueError(f"Transformation failed during 'Gender' encoding: {e}")

        return _X
