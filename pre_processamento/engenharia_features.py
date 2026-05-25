from sklearn.base import BaseEstimator, TransformerMixin

class EngenhariaFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X["IdadeCasa"] = X["YrSold"] - X["YearBuilt"]

        X["AnosAposReforma"] = (
            X["YrSold"] - X["YearRemodAdd"]
        )

        X["AreaTotal"] = (
            X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]
        )

        X["QtdBanheiros"] = (
            X["BsmtFullBath"] + (0.5 * X["BsmtHalfBath"]) + X["FullBath"] + (0.5 * X["HalfBath"])
        )

        return X