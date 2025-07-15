
"""
Fetch hour-level weather for every game in games_2024_meta.csv and write
data/weather_2024.parquet.

Needs a Visual Crossing key in env var VC_API_KEY.
Free-tier key works; one season ~2,400 calls ≈ 2–3 minutes.
"""

import os, time, json, pathlib, requests
import pandas as pd, requests, tqdm
from requests.exceptions import RequestException

KEY = os.getenv("VC_API_KEY")
if not KEY:
    raise SystemExit("❗  Set VC_API_KEY before running (setx or $Env).")

BASE = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"

META_CSV = "data/games_2024_meta.csv"
OUT_PARQ = "data/weather_2024.parquet"

def one_game(lat: float, lon: float, date: str) -> dict:
    """Return weather dict for the 7 p.m. local hour, retrying on timeout."""
    url = (
        f"https://weather.visualcrossing.com/VisualCrossingWebServices/"
        f"rest/services/timeline/{lat},{lon}/{date}"
    )
    params = {
        "key": KEY,
        "unitGroup": "us",
        "include": "hours",
        "contentType": "json",
    }

    for attempt in (1, 2):                       # max 2 tries
        try:
            r = requests.get(url, params=params, timeout=(4, 60))
            r.raise_for_status()
            hr19 = r.json()["days"][0]["hours"][19]
            return {
                "temp_F": hr19.get("temp", hr19.get("tempf")),
                "wind_speed_mph": hr19.get("wspd", hr19.get("windspeed")),
                "wind_dir_deg": hr19.get("wdir", hr19.get("winddir")),
            }
        except RequestException as e:
            if attempt == 2:
                raise                     # bubble up on second failure
            print("↻ timeout, retrying in 10 s …")
            time.sleep(10)
def main() -> None:
    games = pd.read_csv(META_CSV)
    rows  = []

    for _, g in tqdm.tqdm(games.iterrows(), total=len(games)):
        wx = one_game(g.lat, g.lon, g.game_date)
        wx["game_pk"] = g.game_pk
        rows.append(wx)
        time.sleep(.1)                 # free-tier pace

    pathlib.Path("data").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_parquet(OUT_PARQ, index=False)
    print(f"✅ wrote {OUT_PARQ} ({len(rows)} rows)")
if __name__ == "__main__":
    main()
