from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from core.utils.binning import AgeBinner, InteractionBinner
from core.utils.encoding import GenderEncoder
from core.utils.mapping import AgeInteractionMapper


class ChurnFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X: pd.DataFrame, y: Any = None) -> Any:
        return self

    def transform(
        self, X: pd.DataFrame, y: Any = None, train: bool = True
    ) -> pd.DataFrame:
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
            _X = _X.drop(["Churn"], axis=1)

        _X = GenderEncoder().transform(_X)
        _X = AgeBinner().transform(_X)
        _X = InteractionBinner().transform(_X)
        _X = AgeInteractionMapper().transform(_X)
        return _X
