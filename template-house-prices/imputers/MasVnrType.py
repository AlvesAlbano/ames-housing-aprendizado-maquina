from sklearn.base import BaseEstimator, TransformerMixin

class MasVnrImputer(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()

        X.loc[
            (X["MasVnrType"].isna()) & (X["MasVnrArea"] == 0),
            "MasVnrType"
        ] = "None"

        X.loc[
            (X["MasVnrType"].isna()) & (X["MasVnrArea"] > 0),
            "MasVnrType"
        ] = "Unknown"

        return X