"""
live_alerts.py
Polls Statcast live feed every 30 s, scores each PA with /predict_batch,
sends a Slack alert whenever hr_probability ≥ THRESHOLD.
"""

import os, time, asyncio, aiohttp, requests, pandas as pd, joblib, json, logging
from slack_sdk.webhook import WebhookClient
from utils.request_helper import build_payload

# ---------- CONFIG ----------------------------------------------------
MODEL_PATH = "models/hr_model.pkl"
THRESHOLD  = 0.08       # alert if prob >= 10 %
POLL_SECS  = 30         # how often to hit Statcast live endpoint
WEBHOOK_URL = os.getenv("SLACK_WEBHOOK")   # set in PowerShell
# ----------------------------------------------------------------------

model     = joblib.load(MODEL_PATH)
webhook   = WebhookClient(WEBHOOK_URL)
today     = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
seen_ids  = set()

def pa_to_payload(pb: dict) -> dict:
    """Extract raw fields needed for /predict."""
    return dict(
        launch_speed = pb["launch_speed"],
        launch_angle = pb["launch_angle"],
        stand        = pb["stand"],
        p_throws     = pb["p_throws"],
        pitch_type   = pb["pitch_type"],
        inning       = pb["inning"],
        month        = pd.to_datetime(pb["game_date"]).month,
        balls        = pb["balls"],
        strikes      = pb["strikes"],
    )

async def fetch_live(date_str: str) -> list[dict]:
    """Return list[dict] PAs. Ignores blank or malformed rows."""
    url = f"https://baseballsavant.mlb.com/gf?date={date_str}"
    async with aiohttp.ClientSession() as ses:
        async with ses.get(url, timeout=20) as r:
            r.raise_for_status()
            raw_list = await r.json()           # list[str]

    good = []
    for s in raw_list:
        if not s:              # skip empty string
            continue
        try:
            good.append(json.loads(s))
        except json.JSONDecodeError:
            logging.warning("bad JSON row skipped")
    return good

def send_slack(msg: str):
    webhook.send(text=msg)

async def main():
    if not WEBHOOK_URL:
        raise SystemExit("❗ Set SLACK_WEBHOOK before running this script.")

    while True:
        pbs = await fetch_live(today)
        new = [pb for pb in pbs if pb["play_id"] not in seen_ids and pb.get("launch_speed")]
        if new:
            seen_ids.update(pb["play_id"] for pb in new)
            rows = [pa_to_payload(pb) for pb in new]

            # score via local /predict_batch
            resp = requests.post(
                "http://127.0.0.1:8000/predict_batch",
                json={"data": rows},
                timeout=10
            ).json()

            for pb, prob in zip(new, resp["hr_probability"]):
                if prob >= THRESHOLD:
                    send_slack(
                        f":baseball: *HR Risk {prob:.1%}* | "
                        f"{pb['away_team']} @ {pb['home_team']} – "
                        f"Inning {pb['inning']} | {pb['player_name']} vs {pb['pitcher_name']}\n"
                        f"Speed {pb['launch_speed']} mph, Angle {pb['launch_angle']}°"
                    )
                    print("alert sent", pb["play_id"], prob)
        await asyncio.sleep(POLL_SECS)

if __name__ == "__main__":
    asyncio.run(main())