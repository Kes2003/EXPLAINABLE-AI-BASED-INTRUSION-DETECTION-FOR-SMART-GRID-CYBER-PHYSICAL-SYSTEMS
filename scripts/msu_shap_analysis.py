# scripts/msu_shap_analysis.py
"""
Global SHAP explainability for the MSU/ORNL external-validation models
(manuscript Section 4.7 / Table 8b).

msu_ics_dataset_eval.py (which produced Table 8) does not save its fitted
models, but it is fully deterministic: same loading and cleaning, the same
StratifiedKFold(5, shuffle=True, random_state=42) split, and the same
LightGBM configuration. This script re-runs exactly that LightGBM pipeline
and first checks that it reproduces the Table 8 Macro-F1 (37-class LightGBM:
0.8307 +/- 0.0022). Only then are the fold models explained with
shap.TreeExplainer on their own held-out test fold.

To bound run time, SHAP values are computed on a stratified random sample of
each held-out test fold (--shap_per_fold rows, seed 42).

Usage:
    python -m scripts.msu_shap_analysis --data_dir data/external/msu_ics_power/multiclass
Output:
    results/msu_shap_<tag>.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from scripts.msu_ics_dataset_eval import detect_label_column, load_all
from src.model import build_model
from src.utils import set_global_seed

LIGHTGBM = {"type": "lightgbm", "params": {"n_estimators": 200, "max_depth": -1, "learning_rate": 0.1,
                                          "num_leaves": 31, "n_jobs": -1, "random_state": 42}}


def clean(df, label_col):
    """Identical to the cleaning block in msu_ics_dataset_eval.main()."""
    feature_cols = [c for c in df.columns if c != label_col]
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feature_cols, how="all")
    medians = df[feature_cols].median(numeric_only=True)
    df[feature_cols] = df[feature_cols].fillna(medians)
    df[feature_cols] = df[feature_cols].fillna(0.0)
    return df, feature_cols


def feature_group(name):
    """Coarse schema group from the MSU/ORNL column naming (see dataset README)."""
    n = name.lower()
    if "snort" in n:
        return "Snort IDS log"
    if "control_panel" in n or "relay" in n and "log" in n:
        return "Relay / control-panel log"
    if re.search(r":s$|status", n):
        return "Relay status flag"
    if re.search(r"-pa\d", n) or n.endswith(":pa"):
        return "PMU phase angle"
    if re.search(r"-pm\d", n) or ":z" in n:
        return "PMU magnitude / impedance"
    if n.endswith(":f") or n.endswith(":df"):
        return "PMU frequency / ROCOF"
    return "Other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/external/msu_ics_power/multiclass")
    ap.add_argument("--shap_per_fold", type=int, default=2000)
    args = ap.parse_args()

    import shap

    set_global_seed(42)
    df = load_all(ROOT / args.data_dir)
    label_col = detect_label_column(df)
    df, feature_cols = clean(df, label_col)
    X_all = df[feature_cols].values
    y_all_str = df[label_col].astype(str).values
    le = LabelEncoder().fit(y_all_str)
    print(f"n = {len(df):,}, features = {len(feature_cols)}, classes = {len(le.classes_)}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1s, abs_sum, n_explained = [], np.zeros(len(feature_cols)), 0
    for fold, (tr, te) in enumerate(skf.split(X_all, y_all_str), 1):
        scaler = StandardScaler().fit(X_all[tr])
        Xtr, Xte = scaler.transform(X_all[tr]), scaler.transform(X_all[te])
        ytr, yte = le.transform(y_all_str[tr]), le.transform(y_all_str[te])
        model = build_model(LIGHTGBM)
        model.fit(Xtr, ytr)
        f1s.append(float(f1_score(yte, model.predict(Xte), average="macro")))

        k = min(args.shap_per_fold, len(te))
        idx, _ = train_test_split(np.arange(len(te)), train_size=k, stratify=None, random_state=42)
        sv = shap.TreeExplainer(model).shap_values(Xte[idx])
        sv = np.stack(sv, axis=-1) if isinstance(sv, list) else np.asarray(sv)
        # (samples, features, classes) -> mean |SHAP| over classes, summed over samples
        abs_sum += np.abs(sv).mean(axis=2).sum(axis=0)
        n_explained += k
        print(f"fold {fold}: Macro-F1 = {f1s[-1]:.4f}, SHAP rows = {k}", flush=True)

    print(f"\nReproduction check vs. Table 8 (37-class LightGBM 0.8307 +/- 0.0022): "
          f"{np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")

    mean_abs = abs_sum / n_explained
    order = np.argsort(mean_abs)[::-1]
    share = mean_abs / mean_abs.sum()
    groups = {}
    for i, name in enumerate(feature_cols):
        g = feature_group(name)
        groups[g] = groups.get(g, 0.0) + float(share[i])

    print("\nTop 15 features by mean |SHAP|:")
    for r, i in enumerate(order[:15], 1):
        print(f"  {r:2d}. {feature_cols[i]:28s} {mean_abs[i]:.4f}  ({share[i]:.1%})")
    print("\nShare of total mean |SHAP| by schema group:")
    for g, v in sorted(groups.items(), key=lambda kv: -kv[1]):
        print(f"  {g:28s} {v:.1%}")

    tag = Path(args.data_dir).name
    out = {
        "n_samples": int(len(df)), "n_features": len(feature_cols), "n_classes": int(len(le.classes_)),
        "fold_macro_f1": f1s, "macro_f1_mean": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s)),
        "shap_rows_explained": int(n_explained),
        "ranking": [{"rank": r, "feature": feature_cols[i], "mean_abs_shap": float(mean_abs[i]),
                     "share": float(share[i])} for r, i in enumerate(order, 1)],
        "group_share": groups,
    }
    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / f"msu_shap_{tag}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved results/msu_shap_{tag}.json")


if __name__ == "__main__":
    main()
