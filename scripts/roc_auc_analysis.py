# scripts/roc_auc_analysis.py
"""
Multiclass ROC/AUC figure (Fig. 14, Table 2) -- REBUILT.

The previous version of this script loaded the real model and got real
predict_proba() output, but then deliberately distorted it before plotting:
  - temperature-scaled the logits (temperature=8.0) to flatten confidence
  - blended in `mix_factor=0.15` of uniform Dirichlet noise as
    "inject inter-class confusion"
That post-hoc distortion is what produced the specific AUC values currently
in Table 2 (0.979 / 0.962 / 0.943 / 0.934 / 0.912) -- they are not the
model's real ROC behavior. This version uses predict_proba() output directly,
with no post-processing.

Usage:
  python -m scripts.roc_auc_analysis
"""
from __future__ import annotations

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split

from src.train import oversample_minority_classes

# Load dataset
df = pd.read_csv("data/external/enhanced_realistic_smartgrid.csv")

with open("models/features.json", "r") as f:
    FEATURES = json.load(f)["features"]

# Split BEFORE oversampling (same leakage fix as src/train.py) so the test
# set used for the ROC curve is real, untouched, non-duplicated data.
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label"]
)

model = joblib.load("models/model_lightgbm.joblib")
scaler = joblib.load("models/scaler_lightgbm.joblib")
label_encoder = joblib.load("models/label_encoder_lightgbm.joblib")

X_test = test_df[FEATURES]
y_test = label_encoder.transform(test_df["label"].astype(str).values)
X_test_scaled = scaler.transform(X_test)

# Real predicted probabilities -- no temperature scaling, no noise blending
y_prob = model.predict_proba(X_test_scaled)

classes = np.unique(y_test)
y_test_bin = label_binarize(y_test, classes=classes)

plt.figure(figsize=(10, 7))
auc_scores = {}

for i in range(len(classes)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    class_name = label_encoder.inverse_transform([i])[0]
    auc_scores[class_name] = roc_auc
    plt.plot(fpr, tpr, linewidth=2, label=f"{class_name} (AUC = {roc_auc:.3f})")

plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Multi-Class Smart Grid IDS (real predict_proba, no post-processing)")
plt.legend()
plt.grid(True)
plt.savefig("artifacts/visualizations/roc_auc_curve.png", dpi=300, bbox_inches="tight")

with open("artifacts/roc_auc_scores.json", "w") as f:
    json.dump({k: float(v) for k, v in auc_scores.items()}, f, indent=2)

print("\n=== REAL AUC SCORES (no temperature scaling / no noise injection) ===")
for k, v in auc_scores.items():
    print(f"{k}: {v:.4f}")
print("\nSaved: artifacts/visualizations/roc_auc_curve.png")
print("Saved: artifacts/roc_auc_scores.json")
