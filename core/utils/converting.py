import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Any

class DataFrameConverter(BaseEstimator, TransformerMixin):
    """
    Converts a NumPy array back into a pandas DataFrame inside a pipeline.
    
    It preserves the column names learned during the fit step.
    """
    def __init__(self, column_names: list[str] | pd.Index) -> None:
        self.column_names = column_names
        
    def fit(self, X: Any, y: Any = None) -> Any:
        return self

    def transform(self, X: Any, y: Any = None) -> pd.DataFrame:
        return pd.DataFrame(X, columns=self.column_names)