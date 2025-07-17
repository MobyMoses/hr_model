
from typing import Literal, List, Dict, Any
from fastapi import FastAPI
import joblib, pandas as pd, shap, json
from pydantic import BaseModel
from utils.request_helper import build_payload

MODEL_VERSION = "2024-wind-v1"
app = FastAPI(title="HR Probability API")
MODEL_PATH = "models/hr_model.pkl"
model = joblib.load("models/hr_model.pkl")
explainer = shap.TreeExplainer(model)

class RawPA(BaseModel):
    launch_speed: float
    launch_angle: float
    balls: int = 0
    strikes: int = 0
    stand: Literal["R", "L"]
    p_throws: Literal["R", "L"]
    pitch_type: str
    inning: int
    month: int

@app.post("/predict")
def predict(pa: RawPA):
    features = build_payload("models/hr_model.pkl", **pa.dict())
    proba = model.predict_proba(pd.DataFrame([features]))[:, 1][0]
    return {
    "hr_probability": float(proba),
    "model_version": MODEL_VERSION
}

class RawPABatch(BaseModel):
    data: List[RawPA]

@app.post("/predict_batch")
def predict_batch(batch: RawPABatch):
    """
    Accepts: {"data": [ {...}, {...}, ... ]}
    Returns: {"hr_probability": [0.041, 0.023, ... ]}
    """
    # Expand every raw PA dict to full one-hot feature dict
    full_rows = [
        build_payload(MODEL_PATH, **pa.dict()) for pa in batch.data
    ]
    df   = pd.DataFrame(full_rows)
    probs = model.predict_proba(df)[:, 1].tolist()
    return {"hr_probability": probs}

@app.post("/explain_batch")
def explain_batch(
    batch: RawPABatch,
    top_n: int = 10       # number of features to return per PA
) -> Dict[str, Any]:
    """
    Returns:
    {
      "hr_probability": [...],
      "baseline": 0.0304,
      "contributions": [
          {"launch_speed":0.010, "launch_angle":0.004, ...},   # PA 1
          {"park_hr_index":0.005, "temp_F":0.002, ...},        # PA 2
          ...
      ]
    }
    """
    full_rows = [build_payload(MODEL_PATH, **pa.dict()) for pa in batch.data]
    df   = pd.DataFrame(full_rows)

    probs      = model.predict_proba(df)[:, 1]
    shap_vals  = explainer.shap_values(df)          # ndarray (n_rows, n_feat)

    # Build list[dict] of top_n |impact| per row
    contribs = []
    for row_vals in shap_vals:
        impact = (
            pd.Series(row_vals, index=df.columns)
              .abs()
              .sort_values(ascending=False)
              .head(top_n)
        )
        contribs.append(impact.to_dict())

    return {
        "hr_probability": probs.tolist(),
        "baseline": float(explainer.expected_value),
        "contributions": contribs
    }

@app.post("/explain")
def explain(pa: RawPA, top_n: int = 10):
    """
    Return SHAP contributions for the specified plate appearance.
    top_n: number of largest-magnitude features to return (default 10).
    """
    features = build_payload(MODEL_PATH, **pa.dict())
    X = pd.DataFrame([features])
    shap_vals = explainer.shap_values(X)[0]        # 1-row → 1-D array

    # Build {feature: value} dict sorted by |impact|
    impact = (
        pd.Series(shap_vals, index=X.columns)
          .abs()
          .sort_values(ascending=False)
          .head(top_n)
    )
    return {
        "hr_probability": float(model.predict_proba(X)[:, 1][0]),
        "baseline": float(explainer.expected_value),
        "contributions": impact.to_dict()          # e.g. {"launch_speed":0.015,…}
    }