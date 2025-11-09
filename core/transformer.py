import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

class ChurnFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()
        self.AGE_MAPPING = {
            'Young Adult': 0,
            'Adult': 1,
            'Mid-Career': 2,
            'Senior': 3
        }

        self.INTERACTION_MAPPING = {
            'Highly Active': 0,
            'Active': 1,
            'Dormant': 2
        }

        self.scaler = StandardScaler()

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None):
        _X = X.copy()
        cols_to_drop = ["Id", "Tenure", "Usage Frequency", "Subscription Type", "Contract Length", "Churn"]
        if all(col in _X.columns for col in cols_to_drop):
            _X = _X.drop(cols_to_drop, axis=1)

        # one hot encode gender
        gender_ohe = pd.get_dummies(_X["Gender"], dtype=np.int8)
        _X = _X.drop(["Gender"], axis=1)
        _X = _X.join(gender_ohe)
        if "Female" in _X.columns and "Male" in _X.columns:
            _X = _X.drop(["Female"], axis=1)
        elif "Female" in _X.columns:
            _X["Male"] = 0
            _X = _X.drop(["Female"], axis=1)
        
        # bin age
        _X["Age Group"] = _X["Age"].apply(self._classify_age_group)

        # bin interaction
        _X["Interaction Frequency"] = _X["Last Interaction"].apply(self._classify_interaction_frequency)

        # map the age group and interaction company
        _X["Age Group"] = _X["Age Group"].apply(lambda age: self.AGE_MAPPING[age])
        _X["Interaction Frequency"] = _X["Interaction Frequency"].apply(lambda age: self.INTERACTION_MAPPING[age])
        return _X

    def _classify_age_group(self, age: int):
        if 18 <= age <= 24:
            return "Young Adult"
        elif 24 < age <= 39:
            return "Adult"
        elif 39 < age <= 59:
            return "Mid-Career"
        elif age >= 60:
            return "Senior"

    def _classify_interaction_frequency(self, last_interaction: float):
        if 0 < last_interaction <= 7:
            return "Highly Active"
        elif 7 < last_interaction <= 15:
            return "Active"
        elif last_interaction > 15:
            return "Dormant"