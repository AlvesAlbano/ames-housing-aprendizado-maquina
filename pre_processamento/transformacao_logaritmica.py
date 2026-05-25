from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
import numpy as np

class TransformacaoLogaritmica(BaseEstimator, TransformerMixin):
    def __init__(self, colunas):
        self.colunas = colunas

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()

        for coluna in self.colunas:
            X[coluna] = np.log1p(X[coluna])

        return X