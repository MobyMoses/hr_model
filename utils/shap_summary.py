import joblib, pandas as pd, shap, matplotlib.pyplot as plt

MODEL_PATH   = "models/hr_model.pkl"
FEATURE_PATH = "data/features_2024.parquet"

# ------------- load model & sample data -----------------
model = joblib.load(MODEL_PATH)
X      = pd.read_parquet(FEATURE_PATH).drop(columns=["is_hr"])

# (optional) sample 10 k rows to speed up SHAP
X_sample = X.sample(10_000, random_state=42)

# ------------- explain -----------------
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

# ------------- global summary bar plot --
shap.summary_plot(
    shap_values,
    X_sample,
    plot_type="bar",
    show=False,           # draw to backend
    max_display=20        # top 20 features
)
plt.tight_layout()
plt.savefig("shap_summary_bar.png", dpi=250)
plt.close()

# ------------- beeswarm (optional) ------
shap.summary_plot(
    shap_values,
    X_sample,
    show=False,
    max_display=20
)
plt.tight_layout()
plt.savefig("shap_summary_beeswarm.png", dpi=250)
plt.close()

print("✅ wrote shap_summary_bar.png and shap_summary_beeswarm.png")