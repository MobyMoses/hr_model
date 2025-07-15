
"""Train an XGBoost classifier for HR probability."""
import argparse, joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss
from xgboost import XGBClassifier

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("features_parquet")
    ap.add_argument("model_out")
    args = ap.parse_args()

    df = pd.read_parquet(args.features_parquet)
    y = df.pop("is_hr")
    X = df

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    imbalance = (len(y_train) - y_train.sum()) / y_train.sum()

    model = XGBClassifier(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=imbalance
    )
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_val)[:,1]
    print("ROC-AUC:", roc_auc_score(y_val, preds))
    print("Brier:", brier_score_loss(y_val, preds))

    joblib.dump(model, args.model_out)
    print(f"Model saved to {args.model_out}")

if __name__ == "__main__":
    main()
