import pandas as pd
import numpy as np
from typing import Any
from sklearn.base import BaseEstimator, TransformerMixin

class GenderEncoder(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X: pd.DataFrame, y: Any = None) -> Any:
        return self

    def transform(self, X: pd.DataFrame, y: Any = None) -> pd.DataFrame:
        _X = X.copy()
        gender_ohe = pd.get_dummies(_X["Gender"], dtype=np.int8)
        _X = _X.drop(["Gender"], axis=1)
        _X = _X.join(gender_ohe)
        if "Female" in _X.columns and "Male" in _X.columns:
            _X = _X.drop(["Female"], axis=1)
        elif "Female" in _X.columns:
            _X["Male"] = 0
            _X = _X.drop(["Female"], axis=1)

        return _X