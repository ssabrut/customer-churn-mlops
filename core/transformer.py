import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from core import constant

class ChurnFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        _X = X.copy()
        _X["Age Group"] = pd.cut(
            _X["Age"],
            bins=constant.AGE_BINS,
            labels=constant.AGE_LABELS,
            right=True
        )
        
        _X["Interaction Frequency"] = pd.cut(
            _X["Last Interaction"],
            bins=constant.INTERACTION_BINS,
            labels=constant.INTERACTION_LABELS,
            right=True
        )

        return _X