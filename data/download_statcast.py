
"""CLI to download Statcast events between two dates and store as Parquet."""
import sys, pathlib, argparse
from pybaseball import statcast
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("start", help="YYYY-MM-DD")
    ap.add_argument("end", help="YYYY-MM-DD")
    ap.add_argument("--out", default="data/statcast.parquet")
    args = ap.parse_args()

    df = statcast(start_dt=args.start, end_dt=args.end)
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df):,} rows to {out_path}")

if __name__ == "__main__":
    main()
