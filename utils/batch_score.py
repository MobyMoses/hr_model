import argparse, requests, pandas as pd
from tqdm import tqdm
 

API   = "http://127.0.0.1:8000/predict"
MODEL = "models/hr_model.pkl"

def main():
    ap = argparse.ArgumentParser(description="Batch-score a CSV of plate appearances")
    ap.add_argument("csv_in",  help="Input CSV with raw PA columns")
    ap.add_argument("csv_out", help="Output CSV with hr_probability column")
    args = ap.parse_args()

    df = pd.read_csv(args.csv_in)
    probs = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Scoring"):
        payload = row.to_dict()
        r = requests.post(API, json=payload, timeout=3)
        r.raise_for_status()
        probs.append(r.json()["hr_probability"])

    df["hr_probability"] = probs
    df.to_csv(args.csv_out, index=False)
    print(f"✅ Saved {args.csv_out} with {len(df)} rows")

if __name__ == "__main__":
    main()