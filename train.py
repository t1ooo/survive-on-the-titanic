import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
from shared import TARGET, COLS, fill_na, encode_categorical


def train():
    df = pd.read_csv("data/titanic.csv")
    df = df[[TARGET] + COLS]
    df = fill_na(df)
    df = encode_categorical(df)

    y = df[TARGET]
    x = df.drop(TARGET, axis=1)

    model = RandomForestClassifier()
    model.fit(x, y)
    print("score", model.score(x, y))

    if not os.path.exists("models"):
        os.mkdir("models")
    dump(model, "models/RandomForestClassifier.joblib")


if __name__ == "__main__":
    train()
