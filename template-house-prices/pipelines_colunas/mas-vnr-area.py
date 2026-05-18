from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

mas_vnr_area_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value=0.0))
])