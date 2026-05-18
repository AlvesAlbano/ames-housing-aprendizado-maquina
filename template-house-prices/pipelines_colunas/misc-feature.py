from sklearn.preprocessing import FunctionTransformer

def tratar_misc_feature(df):
    df = df.copy()

    df.loc[
        (df["MiscVal"] == 0),
        "MiscFeature"
    ] = "NA"

    df.loc[
        (df["MiscVal"] > 0) & (df["MiscFeature"].isna()),
        "MiscFeature"
    ] = "Unknown"

    return df


misc_transformer = FunctionTransformer(tratar_misc_feature)