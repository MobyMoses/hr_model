
"""Evaluate calibration and classification metrics."""
import argparse, joblib, pandas as pd
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("features_parquet")
    ap.add_argument("model_path")
    args = ap.parse_args()

    df = pd.read_parquet(args.features_parquet)
    y = df.pop("is_hr")
    X = df

    model = joblib.load(args.model_path)
    preds = model.predict_proba(X)[:,1]

    prob_true, prob_pred = calibration_curve(y, preds, n_bins=20)
    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0,1],[0,1],"--")
    plt.title("Calibration")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.savefig("calibration_plot.png", dpi=150, bbox_inches="tight")
    print("Saved calibration_plot.png")

if __name__ == "__main__":
    main()
