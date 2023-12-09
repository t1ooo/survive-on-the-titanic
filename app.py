import pandas as pd
import gradio as gr
from sklearn.ensemble import RandomForestClassifier
from joblib import load
from shared import TARGET, COLS, fill_na, encode_categorical

MODEL: RandomForestClassifier = load("models/RandomForestClassifier.joblib")

DF = pd.read_csv("data/titanic.csv")
DF = DF[[TARGET] + COLS]
DF = fill_na(DF)


def build_inputs():
    inputs = []
    for col in COLS:
        if DF[col].nunique() < 20:
            vals = [v for v in DF[col].unique().tolist() if v != "None"]
            vals.sort()
            if len(vals) <= 2:
                inp = gr.Radio(vals, label=col, value=vals[0])
            else:
                inp = gr.Dropdown(vals, label=col, value=vals[0])
            inputs.append(inp)
        else:
            mini, maxi = DF[col].min(), DF[col].max()
            inp = gr.Slider(mini, maxi, label=col, value=(mini + maxi) / 2)
            inputs.append(inp)
    return inputs


def predict(*args):
    df = pd.DataFrame([args], columns=COLS)
    print("input", df)
    df = encode_categorical(df)
    return MODEL.predict_proba(df)[0][1]


demo = gr.Interface(
    predict,
    inputs=build_inputs(),
    outputs=gr.Textbox(label="The probability of surviving"),
    live=True,
    allow_flagging="never",
    title="Survive On The Titanic",
)

if __name__ == "__main__":
    demo.launch()
