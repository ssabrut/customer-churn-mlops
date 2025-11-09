import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class AgeBinner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        _X = X.copy()
        bins = [17, 24, 39, 59, 100]
        labels = ['Young Adult', 'Adult', 'Mid-Career', 'Senior']
        _X['Age_Group'] = pd.cut(_X['Age'], bins=bins, labels=labels, right=True)
        _X = _X.drop('Age', axis=1)
        return _X

class InteractionBinner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        _X = X.copy()
        bins = [0, 7, 15, float('inf')]
        labels = ['Highly Active', 'Active', 'Dormant']
        _X['Interaction_Frequency'] = pd.cut(_X['Last Interaction'], bins=bins, labels=labels, right=True)
        _X = _X.drop('Last Interaction', axis=1)
        return _X