from typing import Any, List

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class AgeBinner(BaseEstimator, TransformerMixin):
    """
    A transformer that bins the 'Age' column into predefined categories.

    This transformer applies 'pd.cut' to the 'Age' column, creating a
    new 'Age_Group' column with labels: "Young Adult", "Adult",
    "Mid-Career", and "Senior".

    Methods:
        fit(X, y=None):
            Returns 'self' as this is a stateless transformer.
        transform(X):
            Applies the age binning transformation.
    """

    def fit(self, X: pd.DataFrame, y: Any = None) -> "AgeBinner":
        """
        Fits the transformer to the data.

        Since this transformer is stateless, this method does nothing
        and returns 'self'.

        Args:
            X (pd.DataFrame): The input data.
            y (Any): The target data (ignored).

        Returns:
            AgeBinner: The fitted transformer instance.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the age binning to the 'Age' column.

        Args:
            X (pd.DataFrame): The input data to transform.

        Returns:
            pd.DataFrame: The DataFrame with a new 'Age_Group' column.

        Raises:
            TypeError: If the input 'X' is not a pandas DataFrame.
            ValueError: If the input DataFrame 'X' is empty, or if the
                        'Age' column is missing or 'pd.cut' fails.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Input must be a pandas DataFrame, got {type(X)}")
        if X.empty:
            raise ValueError("Input DataFrame is empty")

        _X = X.copy()
        bins: List[float] = [17, 24, 39, 59, 100]
        labels: List[str] = ["Young Adult", "Adult", "Mid-Career", "Senior"]

        try:
            _X["Age_Group"] = pd.cut(
                _X["Age"], bins=bins, labels=labels, right=True
            )
        except KeyError:
            raise ValueError(
                "Transformation failed: 'Age' column not found in DataFrame."
            )
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Transformation failed during 'pd.cut' on 'Age' column: {e}"
            )
        return _X


class InteractionBinner(BaseEstimator, TransformerMixin):
    """
    A transformer that bins the 'Last Interaction' column into categories.

    This transformer applies 'pd.cut' to the 'Last Interaction' column,
    creating a new 'Interaction_Frequency' column with labels:
    "Highly Active", "Active", and "Dormant".

    Methods:
        fit(X, y=None):
            Returns 'self' as this is a stateless transformer.
        transform(X):
            Applies the interaction binning transformation.
    """

    def fit(self, X: pd.DataFrame, y: Any = None) -> "InteractionBinner":
        """
        Fits the transformer to the data.

        Since this transformer is stateless, this method does nothing
        and returns 'self'.

        Args:
            X (pd.DataFrame): The input data.
            y (Any): The target data (ignored).

        Returns:
            InteractionBinner: The fitted transformer instance.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies interaction binning to the 'Last Interaction' column.

        Args:
            X (pd.DataFrame): The input data to transform.

        Returns:
            pd.DataFrame: The DataFrame with a new 'Interaction_Frequency'
                          column.

        Raises:
            TypeError: If the input 'X' is not a pandas DataFrame.
            ValueError: If the input DataFrame 'X' is empty, or if the
                        'Last Interaction' column is missing or 'pd.cut'
                        fails.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Input must be a pandas DataFrame, got {type(X)}")
        if X.empty:
            raise ValueError("Input DataFrame is empty")

        _X = X.copy()
        bins: List[float] = [0, 7, 15, float("inf")]
        labels: List[str] = ["Highly Active", "Active", "Dormant"]

        try:
            _X["Interaction_Frequency"] = pd.cut(
                _X["Last Interaction"], bins=bins, labels=labels, right=True
            )
        except KeyError:
            raise ValueError(
                "Transformation failed: 'Last Interaction' column not found "
                "in DataFrame."
            )
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Transformation failed during 'pd.cut' on "
                f"'Last Interaction' column: {e}"
            )
        return _X