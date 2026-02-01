import gradio as gr
import pickle
import pandas as pd
import numpy as np

# 1) Load trained pipeline/model (includes imputers/preprocessing)
with open("h2_gradient_boosting_model.pkl", "rb") as f:
    model = pickle.load(f)

#main logic 
# def m(name):
#     return "hellik"
def clip(val, vmin, vmax):
    return max(vmin, min(val, vmax))
def predict_H2(MC, VM, FC, Ash, C, H, O, N, S, oC, ER, SB):

# --------
#  input clipping (set ranges as per data)
#  --------

    MC  = clip(MC,  0, 50)
    VM  = clip(VM,  0, 90)
    FC  = clip(FC,  0, 50)
    Ash = clip(Ash, 0, 50)

    C = clip(C,  0, 100)
    H = clip(H,  0, 15)
    O = clip(O,  0, 60)
    N = clip(N,  0, 10)
    S = clip(S,  0, 10)

    oC = clip(oC, 400, 1200)   # temperature
    ER = clip(ER, 0.0, 0.6)
    SB = clip(SB, 0.0, 2.0)


    input_df = pd.DataFrame(
        [[MC, VM, FC, Ash, C, H, O, N, S, oC, ER, SB]],
        columns=['MC', 'VM', 'FC', 'Ash', 'C', 'H',
                 'O', 'N', 'S', 'oC', 'ER', 'S/B']
    )

    prediction = model.predict(input_df)[0]
    return float(max(0, min(prediction, 100)))


#----
#input taking
#------
inputs=[
        gr.Number(label="MC"),
        gr.Number(label="VM"),
        gr.Number(label="FC"),
        gr.Number(label="Ash"),
        gr.Number(label="C"),
        gr.Number(label="H"),
        gr.Number(label="O"),
        gr.Number(label="N"),
        gr.Number(label="S"),
        gr.Number(label="Temperature (°C)"),
        gr.Number(label="ER"),
        gr.Number(label="S/B"),
    ]

outputs=gr.Number(label="Predicted H2 (%)")


#interface
app=gr.Interface(
    fn=predict_H2,
    inputs=inputs,
    outputs=outputs,
    title="H2 Prediction Model"
)
#launch
app.launch()