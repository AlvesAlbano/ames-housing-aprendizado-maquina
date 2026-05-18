from sklearn.base import BaseEstimator, TransformerMixin

import pandas as pd

class LotFrontageImputer(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        self.medians_ = X.groupby("Neighborhood")["LotFrontage"].median()
        return self
    
    def transform(self, X):
        X = X.copy()
        
        X["LotFrontage"] = X.apply(
            lambda row: (
                self.medians_.loc[row["Neighborhood"]]
                if pd.isna(row["LotFrontage"])
                else row["LotFrontage"]
            ),
            axis=1
        )
        
        return X