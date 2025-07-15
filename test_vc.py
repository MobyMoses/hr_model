import os, requests, pprint

lat, lon, date = 33.445, -112.066, "2024-04-01"   # Chase Field
url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/" \
      f"weather/timeline/{lat},{lon}/{date}"

params = dict(
    key=os.getenv("VC_API_KEY"),   # your key MUST appear here
    unitGroup="us",
    include="hours",               # bring hourly data
    contentType="json"
)

req = requests.Request("GET", url, params=params).prepare()
print("Full URL:\n", req.url)

r = requests.get(req.url, timeout=20)
print("HTTP status:", r.status_code)
print("First 200 chars:\n", r.text[:200])