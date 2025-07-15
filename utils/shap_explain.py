"""
Generate a SHAP force-plot (PNG) for one plate appearance.

Usage
-----
python utils/shap_explain.py pa.json --out pa_force.png
"""
import argparse, json, joblib, pandas as pd, shap, matplotlib.pyplot as plt

MODEL_PATH = "models/hr_model.pkl"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pa_json", help="JSON file with the EXACT row you fed /predict")
    ap.add_argument("--out", default="pa_force.png", help="Output PNG path")
    args = ap.parse_args()

    # ---------- load model & one-row dataframe ----------
    model = joblib.load(MODEL_PATH)
    with open(args.pa_json, encoding="utf-8-sig") as fh:
        x = pd.Series(json.load(fh)).to_frame().T  # keep column names!

    # ---------- SHAP values ----------
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(x)

    # ---------- force plot ----------
    shap.force_plot(
        explainer.expected_value,
        shap_vals,
        x,
        matplotlib=True,
        show=False                 # draw to backend
    )
    plt.tight_layout()
    plt.savefig(args.out, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"✅ wrote {args.out}")

if __name__ == "__main__":
    main()