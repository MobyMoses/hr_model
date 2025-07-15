import json, joblib, pandas as pd, argparse

# ---------------------------------------------------------------------
# Build a 1-row dict whose keys exactly match the model's feature names
# ---------------------------------------------------------------------
def build_payload(model_path: str, **raw) -> dict:
    """
    Expand raw baseball fields into the full one-hot feature dict.

    Parameters
    ----------
    model_path : str
        Path to the trained XGBoost .pkl file.
    **raw : keyword args
        launch_speed=…, launch_angle=…, stand='R', p_throws='L',
        pitch_type='FF', inning=7, month=8, balls=0, strikes=0 …

    Returns
    -------
    dict
        Keys match every column seen during training; unmatched columns
        are filled with 0.0 so XGBoost accepts the DataFrame.
    """
    model = joblib.load(model_path)
    feat_names = model.get_booster().feature_names

    # start with a row of zeros (float to avoid dtype warnings)
    X = pd.DataFrame(0.0, index=[0], columns=feat_names)

    for field, val in raw.items():
        if field in X.columns:
            X.at[0, field] = val
        else:
            one_hot = f"{field}_{val}"
            if one_hot in X.columns:
                X.at[0, one_hot] = 1
            else:
                print(f"[warn] '{field}' -> '{val}' not matched; skipped.")

    return X.iloc[0].to_dict()

# ---------------------------------------------------------------------
# Optional CLI:  build a payload and print JSON to stdout
# ---------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build JSON payload for HR API")
    ap.add_argument("model", help="Path to hr_model.pkl")
    ap.add_argument("--launch_speed", type=float, required=True)
    ap.add_argument("--launch_angle", type=float, required=True)
    ap.add_argument("--balls", type=int, default=0)
    ap.add_argument("--strikes", type=int, default=0)
    ap.add_argument("--stand", choices=["R", "L"], required=True)
    ap.add_argument("--p_throws", choices=["R", "L"], required=True)
    ap.add_argument("--pitch_type", required=True)
    ap.add_argument("--inning", type=int, required=True)
    ap.add_argument("--month", type=int, required=True)
    args = ap.parse_args()

    payload = build_payload(
        args.model,
        launch_speed=args.launch_speed,
        launch_angle=args.launch_angle,
        balls=args.balls,
        strikes=args.strikes,
        stand=args.stand,
        p_throws=args.p_throws,
        pitch_type=args.pitch_type,
        inning=args.inning,
        month=args.month,
    )
    print(json.dumps(payload, indent=2))