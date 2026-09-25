# src/evaluate_models_cv.py
import os, json
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import xgboost as xgb
except:
    xgb = None
try:
    import lightgbm as lgb
except:
    lgb = None

DATA_PATHS = [
    "data/external/enhanced_realistic_smartgrid.csv"
]
OUT_DIR = "artifacts/visualizations"
os.makedirs(OUT_DIR, exist_ok=True)

def load_data():
    for p in DATA_PATHS:
        if os.path.exists(p):
            df = pd.read_csv(p)
            print("Loaded", p)
            return df
    raise FileNotFoundError("No data file found. Place train CSV in data/processed/ or data/raw/")

df = load_data()
# TODO: if your label column has different name, change here:
label_col = "label" if "label" in df.columns else ("attack_class" if "attack_class" in df.columns else "target")
if label_col not in df.columns:
    # Try looking for common names:
    print("Columns:", df.columns.tolist())
    raise SystemExit("Please set label_col in script to your label column name.")

# Select numeric features automatically excluding label/time cols
exclude = [label_col, "t", "scenario", "pred_label", "prob"]
features = [c for c in df.columns if c not in exclude and df[c].dtype in [int, float]]
X = df[features].values
y = df[label_col].values
le = LabelEncoder()
y_enc = le.fit_transform(y)

models = {}
models["rf"] = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
if xgb is not None:
    models["xgb"] = xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42)
if lgb is not None:
    models["lgb"] = lgb.LGBMClassifier(random_state=42)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {name: {"acc":[], "f1":[],"precision":[],"recall":[]} for name in models.keys()}

for train_idx, test_idx in skf.split(X, y_enc):
    Xtr, Xte = X[train_idx], X[test_idx]
    ytr, yte = y_enc[train_idx], y_enc[test_idx]
    for name, model in models.items():
        model.fit(Xtr, ytr)
        preds = model.predict(Xte)
        results[name]["acc"].append(accuracy_score(yte, preds))
        results[name]["f1"].append(f1_score(yte, preds, average="macro"))
        results[name]["precision"].append(precision_score(yte, preds, average="macro", zero_division=0))
        results[name]["recall"].append(recall_score(yte, preds, average="macro", zero_division=0))

# Summarize
summary = {}
for name, metrics in results.items():
    summary[name] = {m: (float(np.mean(v)), float(np.std(v))) for m,v in metrics.items()}

# Save summary
with open("artifacts/model_cv_summary.json","w") as f:
    json.dump(summary, f, indent=2)
print("Saved artifacts/model_cv_summary.json")

# Optional: make a simple bar chart for mean F1
means = {name: summary[name]["f1"][0] for name in summary}
plt.figure(figsize=(6,4))
sns.barplot(x=list(means.keys()), y=list(means.values()))
plt.ylabel("Mean Macro-F1")
plt.savefig(os.path.join(OUT_DIR,"cv_mean_macro_f1.png"))
print("Saved", os.path.join(OUT_DIR,"cv_mean_macro_f1.png"))
