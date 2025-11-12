from typing import Any, List, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DataFrameConverter(BaseEstimator, TransformerMixin):
    """
    Converts a NumPy array back into a pandas DataFrame inside a pipeline.

    This class is typically used after a scikit-learn transformer
    (like StandardScaler) that outputs a NumPy array, allowing subsequent
    pipeline steps to receive a DataFrame with the original column names.

    Attributes:
        column_names (pd.Index | List[str]): The column names to assign to
                                             the new DataFrame.

    Methods:
        __init__(column_names): Initializes the transformer.
        fit(X, y=None): Returns 'self' as it is a stateless transformer.
        transform(X, y=None): Converts the input array to a DataFrame.
    """

    def __init__(self, column_names: Union[List[str], pd.Index]) -> None:
        """
        Initializes the DataFrameConverter.

        Args:
            column_names (Union[List[str], pd.Index]): A list or pandas Index
                of column names that should be applied to the DataFrame
                in the 'transform' step.

        Raises:
            ValueError: If 'column_names' is empty or not provided.
        """
        if len(column_names) == 0:
            raise ValueError("column_names must be a non-empty list or Index.")
        self.column_names: Union[List[str], pd.Index] = column_names

    def fit(self, X: Any, y: Any = None) -> "DataFrameConverter":
        """
        Fits the transformer to the data.

        Since this transformer is stateless (it only applies a
        predefined set of column names), this method does nothing
        and returns 'self'.

        Args:
            X (Any): The input data (ignored).
            y (Any): The target data (ignored).

        Returns:
            DataFrameConverter: The fitted transformer instance.
        """
        return self

    def transform(self, X: np.ndarray, y: Any = None) -> pd.DataFrame:
        """
        Converts the input NumPy array into a pandas DataFrame.

        Args:
            X (np.ndarray): The input NumPy array from the previous
                            pipeline step.
            y (Any): The target data (ignored).

        Returns:
            pd.DataFrame: A DataFrame created from 'X' with the stored
                          'column_names'.

        Raises:
            ValueError: If the input array 'X' is empty, or if the
                        number of columns in 'X' does not match the
                        number of 'column_names' provided during
                        initialization.
        """
        if X.size == 0:
            raise ValueError("Input array 'X' is empty.")

        try:
            df = pd.DataFrame(X, columns=self.column_names)
        except ValueError as e:
            raise ValueError(
                f"Shape mismatch: Input array has {X.shape[1]} columns, "
                f"but {len(self.column_names)} column names were provided. "
                f"Original error: {e}"
            )

        return df
