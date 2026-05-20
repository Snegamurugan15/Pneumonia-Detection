from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, dcc, html
from PIL import Image


SAMPLE_DIR = Path("sample_images")
samples = sorted(SAMPLE_DIR.glob("*.jpg"))
app = Dash(__name__)
app.title = "Pneumonia Detection Review"


def pseudo_probabilities(path: Path) -> pd.DataFrame:
    name = path.name.lower()
    if "bacteria" in name:
        probs = [0.08, 0.75, 0.17]
    elif "virus" in name:
        probs = [0.11, 0.22, 0.67]
    else:
        probs = [0.82, 0.10, 0.08]
    return pd.DataFrame({"class": ["Normal", "Bacterial Pneumonia", "Viral Pneumonia"], "probability": probs})


def image_quality(path: Path) -> pd.DataFrame:
    arr = np.asarray(Image.open(path).convert("L").resize((224, 224))) / 255.0
    return pd.DataFrame(
        {
            "metric": ["mean brightness", "contrast", "texture"],
            "value": [
                float(arr.mean()),
                float(arr.std()),
                float(np.mean(np.abs(np.diff(arr, axis=0))) + np.mean(np.abs(np.diff(arr, axis=1)))),
            ],
        }
    )


app.layout = html.Div(
    style={"fontFamily": "Segoe UI, sans-serif", "margin": "32px", "maxWidth": "1000px"},
    children=[
        html.H1("Pneumonia Detection Model Review"),
        html.P("Dash review tool for sample X-rays, class probabilities, and image-quality checks."),
        dcc.Dropdown([p.name for p in samples], samples[0].name if samples else None, id="sample"),
        dcc.Graph(id="probabilities"),
        dcc.Graph(id="quality"),
        html.P("This review app is educational only and is not medical diagnosis software."),
    ],
)


@app.callback(Output("probabilities", "figure"), Output("quality", "figure"), Input("sample", "value"))
def update(sample_name):
    path = SAMPLE_DIR / sample_name
    probs = pseudo_probabilities(path)
    quality = image_quality(path)
    return (
        px.bar(probs, x="class", y="probability", range_y=[0, 1], title="Demo Class Probabilities"),
        px.bar(quality, x="metric", y="value", title="Image Quality Signals"),
    )


if __name__ == "__main__":
    app.run(debug=True)
