# scripts/cv_evaluation.py
"""
Leakage-free stratified 5-fold CV evaluation across RF / XGBoost / LightGBM / CatBoost.

WHY THIS SCRIPT EXISTS
-----------------------
The previous pipeline had a data-leakage bug: src/train.py oversampled minority
classes (by duplicating existing rows) on the WHOLE dataset, and only afterward
split into train/test. That means identical duplicated rows could land in both
the train and test partitions, which inflates test-set performance for the
oversampled classes. That is the actual, mechanical reason the single-split
result (artifacts/metrics.json, Macro-F1 ~0.997) came out higher than the
older CV script (artifacts/model_cv_summary.json, Macro-F1 ~0.93) -- the two
numbers were never measuring the same thing, and the higher one wasn't a fair
train/test evaluation.

This script fixes it by doing everything correctly, per fold:
  1. Split into train/test FIRST (stratified).
  2. Oversample ONLY the training fold.
  3. Fit the scaler ONLY on the (oversampled) training fold.
  4. Evaluate on the untouched, non-oversampled, non-leaked test fold.

Run this once and use ITS numbers in the manuscript -- not metrics.json's.

Outputs:
  - artifacts/cv_results.json           (per-fold, per-model scores)
  - artifacts/model_cv_summary.json     (mean/std per model -- overwrites the old one)
  - artifacts/statistical_significance.json  (paired Wilcoxon tests vs. best model)
  - artifacts/visualizations/cv_model_comparison.png

Usage:
  python -m scripts.cv_evaluation
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from scipy.stats import wilcoxon

from src.model import build_model
from src.train import oversample_minority_classes
from src.utils import ensure_dir, save_json, set_global_seed


def main():
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)
    seed = int(cfg.get("seed", 42))
    set_global_seed(seed)

    features = cfg["features"]
    variants = cfg.get("model_variants", {
        "random_forest": {"type": "random_forest", "params": {"n_estimators": 300, "random_state": 42}},
        "xgboost": {"type": "xgboost", "params": {"n_estimators": 300, "random_state": 42}},
        "lightgbm": {"type": "lightgbm", "params": {"n_estimators": 300, "random_state": 42}},
    })

    df = pd.read_csv("data/external/enhanced_realistic_smartgrid.csv")
    X_all = df[features].values
    y_all_str = df["label"].astype(str).values

    le_global = LabelEncoder()
    le_global.fit(y_all_str)  # fit on full label set only to fix class ordering; never used for leakage

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    per_fold_scores = {name: {"acc": [], "f1": [], "precision": [], "recall": []} for name in variants}

    fold_idx = 0
    for train_idx, test_idx in skf.split(X_all, y_all_str):
        fold_idx += 1
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        # Oversample ONLY the training fold -- test fold stays untouched and unduplicated
        train_df_bal = oversample_minority_classes(train_df, label_col="label", random_state=seed)

        X_train_raw = train_df_bal[features].values
        y_train_str = train_df_bal["label"].astype(str).values
        X_test_raw = test_df[features].values
        y_test_str = test_df["label"].astype(str).values

        scaler = StandardScaler().fit(X_train_raw)
        X_train = scaler.transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        y_train = le_global.transform(y_train_str)
        y_test = le_global.transform(y_test_str)

        for name, model_cfg in variants.items():
            model = build_model(model_cfg)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            per_fold_scores[name]["acc"].append(float(accuracy_score(y_test, y_pred)))
            per_fold_scores[name]["f1"].append(float(f1_score(y_test, y_pred, average="macro")))
            per_fold_scores[name]["precision"].append(float(precision_score(y_test, y_pred, average="macro", zero_division=0)))
            per_fold_scores[name]["recall"].append(float(recall_score(y_test, y_pred, average="macro", zero_division=0)))

        print(f"Fold {fold_idx}/5 done: " + ", ".join(
            f"{name}={per_fold_scores[name]['f1'][-1]:.4f}" for name in variants
        ))

    ensure_dir("artifacts")
    save_json(per_fold_scores, "artifacts/cv_results.json")

    summary = {
        name: {m: [float(np.mean(v)), float(np.std(v))] for m, v in metrics.items()}
        for name, metrics in per_fold_scores.items()
    }
    save_json(summary, "artifacts/model_cv_summary.json")
    print("\nSaved: artifacts/model_cv_summary.json (leakage-free, replaces the old one)")
    for name, m in summary.items():
        print(f"  {name:15s}  macro-F1 = {m['f1'][0]:.4f} ± {m['f1'][1]:.4f}   acc = {m['acc'][0]:.4f} ± {m['acc'][1]:.4f}")

    # Statistical significance: best model (by mean F1) vs. every other, paired Wilcoxon across the 5 folds
    best_name = max(summary, key=lambda n: summary[n]["f1"][0])
    sig_results = {}
    for name in variants:
        if name == best_name:
            continue
        try:
            stat, p = wilcoxon(per_fold_scores[best_name]["f1"], per_fold_scores[name]["f1"])
            sig_results[f"{best_name}_vs_{name}"] = {"statistic": float(stat), "p_value": float(p)}
        except Exception as e:
            sig_results[f"{best_name}_vs_{name}"] = {"error": str(e)}
    save_json({"best_model": best_name, "comparisons": sig_results,
               "note": "n=5 folds -> low statistical power; Wilcoxon signed-rank on paired per-fold macro-F1."},
              "artifacts/statistical_significance.json")
    print("\nSaved: artifacts/statistical_significance.json")
    print(f"Best model: {best_name}")
    for k, v in sig_results.items():
        print(f"  {k}: {v}")

    # Plot
    plt.figure(figsize=(7, 5))
    names = list(summary.keys())
    means = [summary[n]["f1"][0] for n in names]
    stds = [summary[n]["f1"][1] for n in names]
    ax = sns.barplot(x=names, y=means)
    ax.errorbar(x=range(len(names)), y=means, yerr=stds, fmt="none", c="black", capsize=4)
    plt.ylabel("Macro-F1 (5-fold CV, mean ± std)")
    plt.title("Model comparison -- leakage-free stratified 5-fold CV")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    ensure_dir("artifacts/visualizations")
    plt.savefig("artifacts/visualizations/cv_model_comparison.png", dpi=200, bbox_inches="tight")
    print("Saved: artifacts/visualizations/cv_model_comparison.png")


if __name__ == "__main__":
    main()
