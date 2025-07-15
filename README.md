
# Home‑Run Probability Model

This repo contains an end‑to‑end workflow for predicting whether a **plate appearance** results in a home run.

## Quick start

```bash
# clone / unzip repo
pip install -r requirements.txt
# 1. Pull Statcast data
python data/download_statcast.py 2024-03-28 2024-10-01 --out data/2024.parquet
# 2. Feature engineering (+ weather)
python processing/feature_engineering.py data/2024.parquet data/features_2024.parquet
# 3. Train model
python models/train_xgb.py data/features_2024.parquet models/hr_model.pkl
# 4. Evaluate
python models/evaluate.py data/features_2024.parquet models/hr_model.pkl
# 5. Run API
uvicorn serve.inference_api:app --reload
```

### Directory layout
```
data/
  download_statcast.py
  get_weather.py
processing/
  feature_engineering.py
models/
  train_xgb.py
  evaluate.py
serve/
  inference_api.py
utils/
  park_factors.csv
```

### Requirements
See `requirements.txt`.  Python 3.9+ recommended.

### TODO
* Implement `get_weather.py` to pull historical/hourly conditions.
* Improve feature set: add rolling averages, pitcher embeddings, park × weather interactions.
* Add schedule to retrain weekly via cron/GitHub Actions.
