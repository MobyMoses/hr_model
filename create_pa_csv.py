import pandas as pd

RAW = "data/2024.parquet"         # your Statcast file
OUT = "pa_samples.csv"            # output CSV

keep = [
    "launch_speed", "launch_angle", "stand", "p_throws",
    "pitch_type", "inning", "game_date", "balls", "strikes"
]

df = pd.read_parquet(RAW, columns=keep)
df = df.dropna(subset=["launch_speed", "launch_angle"])
df["month"] = pd.to_datetime(df["game_date"]).dt.month.astype("int8")

df = df[
    ["launch_speed", "launch_angle", "stand", "p_throws",
     "pitch_type", "inning", "month", "balls", "strikes"]
]

sample = df.sample(100, random_state=42)   # adjust sample size
sample.to_csv(OUT, index=False)
print(f"Saved {len(sample)} rows → {OUT}")