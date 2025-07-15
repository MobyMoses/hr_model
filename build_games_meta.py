"""
build_games_meta.py
Builds data/games_2024_meta.csv  (one row per game with game_pk, game_date, lat, lon)

Prerequisites
-------------
data/2024.parquet          – your raw Statcast file for the season
utils/park_latlon.csv      – table with home_team,lat,lon  (one row per MLB park)
"""

import pandas as pd, pathlib

STATCAST_PATH = "data/2024.parquet"
PARK_LATLON   = "utils/park_latlon.csv"
OUT_CSV       = "data/games_2024_meta.csv"

def main():
    sc = pd.read_parquet(STATCAST_PATH)
    games = (
        sc.groupby("game_pk")
          .first()
          .reset_index()[["game_pk", "game_date", "home_team"]]
    )
    parks = pd.read_csv(PARK_LATLON)          # home_team, lat, lon
    games = games.merge(parks, on="home_team", how="left")
    pathlib.Path("data").mkdir(exist_ok=True)
    games.to_csv(OUT_CSV, index=False)
    print(f"✅ wrote {OUT_CSV}  ({len(games)} games)")

if __name__ == "__main__":
    main()