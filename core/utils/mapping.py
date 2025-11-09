from sklearn.base import BaseEstimator, TransformerMixin


class AgeInteractionMapper(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.AGE_MAPPING = {"Young Adult": 0, "Adult": 1, "Mid-Career": 2, "Senior": 3}
        self.INTERACTION_MAPPING = {"Highly Active": 0, "Active": 1, "Dormant": 2}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        _X = X.copy()
        _X["Age_Group"] = _X["Age_Group"].map(self.AGE_MAPPING)
        _X["Interaction_Frequency"] = _X["Interaction_Frequency"].map(
            self.INTERACTION_MAPPING
        )
        return _X
