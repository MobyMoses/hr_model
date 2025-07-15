
"""Convert raw Statcast to model‑ready feature set.

Usage:
    python processing/feature_engineering.py raw.parquet features.parquet
"""
import sys, argparse, pathlib, pandas as pd, numpy as np
WEATHER_PATH   = "data/weather_2024.parquet"
PARK_ORIENT_CSV = "utils/park_orientation.csv"
BASIC_KEEP = [
    "game_date", "batter", "pitcher", "stand", "p_throws",
    "pitch_type", "balls", "strikes", "inning",
    "launch_speed", "launch_angle", "home_team", "away_team",
    "events",
    "wind_out_to_cf"
]

def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df[BASIC_KEEP].copy()
    df["is_hr"] = (df["events"] == "home_run").astype("int8")
    df["month"] = pd.to_datetime(df["game_date"]).dt.month.astype("int8")
    df = df.drop(columns=["game_date"])
    cat_cols = ["stand", "p_throws", "pitch_type", "home_team", "away_team", "inning", "month"]
    df = pd.get_dummies(df, columns=cat_cols, dummy_na=False)
    return df.drop(columns=["events"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("in_path")
    ap.add_argument("out_path")
    args = ap.parse_args()

    df_raw = pd.read_parquet(args.in_path)
    weather = pd.read_parquet(WEATHER_PATH)
    orient  = pd.read_csv(PARK_ORIENT_CSV)
    weather = weather.merge(
        df_raw[["game_pk", "home_team"]].drop_duplicates(),
        on="game_pk", how="left"
    ).merge(orient, on="home_team", how="left")

    # numeric out→in metric  (positive = blowing out)
    angle_diff = (weather["wind_dir_deg"] - weather["cf_deg"] + 360) % 360
    weather["wind_out_to_cf"] = weather["wind_speed_mph"] * np.cos(np.deg2rad(angle_diff))

    df_raw = df_raw.merge(
        weather[["game_pk", "wind_out_to_cf"]],
        on="game_pk", how="left"
    )
   
    df_feat = engineer(df_raw)
    out = pathlib.Path(args.out_path)
    out.parent.mkdir(exist_ok=True, parents=True)
    df_feat.to_parquet(out, index=False)
    print(f"Features saved to {out} ({df_feat.shape[0]:,} rows, {df_feat.shape[1]} cols)")

if __name__ == "__main__":
    main()
