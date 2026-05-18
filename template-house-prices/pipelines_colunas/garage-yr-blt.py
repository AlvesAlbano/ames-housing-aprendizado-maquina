from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

garage_yr_blt_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])