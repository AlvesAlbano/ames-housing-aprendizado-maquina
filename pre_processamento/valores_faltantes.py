from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd

class ValoresFaltantes(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = X.copy()

        self.lotfrontage_by_neighborhood_ = (
            X.groupby("Neighborhood")["LotFrontage"]
            .median()
        )

        self.lotfrontage_global_ = X["LotFrontage"].median()

        self.electrical_moda = (
            X["Electrical"].mode()[0]
        )

        return self

    def transform(self, X):
        X = X.copy()

        X["MasVnrArea"] = X["MasVnrArea"].fillna(0.0)

        X["LotFrontage"] = X["LotFrontage"].fillna(
            X["Neighborhood"].map(self.lotfrontage_by_neighborhood_)
        )

        X["LotFrontage"] = X["LotFrontage"].fillna(
            self.lotfrontage_global_
        )

        X["Electrical"] = X["Electrical"].fillna(
            self.electrical_moda
        )

        X["Alley"] = X["Alley"].fillna("NA")
        
        X.loc[
            (X["MasVnrType"].isna()) & (X["MasVnrArea"] == 0.0),
            "MasVnrType"
        ] = "None"

        X.loc[
            (X["MasVnrType"].isna()) & (X["MasVnrArea"] > 0),
            "MasVnrType"
        ] = "Unknown"

        X[["BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2"]] = X[["BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2"]].fillna("NA")
        
        X.loc[
            (X["Fireplaces"] == 0) & (X["FireplaceQu"].isna()),
            "FireplaceQu"
        ] = "NA"

        X[["GarageType","GarageFinish","GarageQual","GarageCond"]] = (
            X[["GarageType","GarageFinish","GarageQual","GarageCond"]]
            .fillna("NA")
        )

        X.loc[
              X["PoolArea"] == 0,
              "PoolQC"
        ] = "NA"

        X["Fence"] = X["Fence"].fillna("NA")

        X.loc[
              X["MiscVal"] == 0,
              "MiscFeature"
        ] = "NA"

        return X
        