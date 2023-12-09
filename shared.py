TARGET = "Survived"
COLS = ["Pclass", "Sex", "Age", "Fare", "Embarked"]


def fill_na(df):
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna("None")

    assert df.isna().sum().sum() == 0, df.isna().sum()

    return df


def encode_categorical(df):
    mapping = {
        "Sex": {"male": 0, "female": 1},
        "Embarked": {"None": 0, "S": 1, "C": 2, "Q": 3},
    }
    df = df.replace(mapping)

    assert df["Sex"].dtype == int, df["Sex"].dtype
    assert df["Embarked"].dtype == int, df["Embarked"].dtype

    return df
