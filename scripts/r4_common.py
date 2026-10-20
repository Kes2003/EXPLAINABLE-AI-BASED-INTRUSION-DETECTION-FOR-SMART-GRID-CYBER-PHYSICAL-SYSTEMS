# scripts/r4_common.py
"""Shared loading / fold helpers for the Revision-4 analyses (scripts/r4_*.py).

Everything here reuses the existing pipeline rather than re-implementing it:
  - synthetic data + labels exactly as run_repeated_cv_and_policy_eval.load()
  - fold-local oversampling (src.train.oversample_minority_classes) and a
    StandardScaler fit on the oversampled training fold only (Algorithm 1)
  - MSU/ORNL loading and cleaning exactly as scripts/msu_ics_dataset_eval.py,
    plus the recording-file id of every row (needed for group-disjoint CV)
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.msu_ics_dataset_eval import detect_label_column, load_one_file  # noqa: E402
from src.model import build_model  # noqa: E402
from src.train import oversample_minority_classes  # noqa: E402

RESULTS = ROOT / "results" / "r4"
FIGS = RESULTS / "figures"

LIGHTGBM_CV = {"type": "lightgbm", "params": {"n_estimators": 200, "max_depth": -1, "learning_rate": 0.1,
                                             "num_leaves": 31, "n_jobs": -1, "random_state": 42}}
MSU_VARIANTS = {
    "random_forest": {"type": "random_forest", "params": {"n_estimators": 200, "max_depth": 20, "n_jobs": -1, "random_state": 42}},
    "xgboost": {"type": "xgboost", "params": {"n_estimators": 200, "max_depth": 8, "learning_rate": 0.1, "n_jobs": -1, "random_state": 42}},
    "lightgbm": LIGHTGBM_CV,
    "catboost": {"type": "catboost", "params": {"iterations": 200, "depth": 8, "learning_rate": 0.1, "random_state": 42, "verbose": False}},
}


def config():
    with open(ROOT / "config" / "config.yaml") as f:
        return yaml.safe_load(f)


def load_synthetic():
    cfg = config()
    df = pd.read_csv(ROOT / "data" / "external" / "enhanced_realistic_smartgrid.csv")
    df["label"] = df["label"].astype(str).replace({"overlo": "overload"})
    return cfg, df, cfg["features"]


def synthetic_folds(df, seed=42):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return list(skf.split(df.values, df["label"].values))


def prepare_fold(df, features, le, train_idx, test_idx, seed=42):
    """Algorithm 1, lines 3-8: split, oversample the training fold only, fit
    the scaler on the oversampled training fold only."""
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    train_bal = oversample_minority_classes(train_df, label_col="label", random_state=seed)
    scaler = StandardScaler().fit(train_bal[features].values)
    return {
        "scaler": scaler,
        "X_train": scaler.transform(train_bal[features].values),
        "y_train": le.transform(train_bal["label"].values),
        "X_test": scaler.transform(test_df[features].values),
        "y_test": le.transform(test_df["label"].values),
        "test_df": test_df,
        "train_df": train_df,
    }


def load_msu(variant="multiclass"):
    """MSU/ORNL rows, cleaned exactly as msu_ics_dataset_eval.py, with the
    recording-file id (data1..data15) of every row. Rows keep file order and,
    within a file, their recorded (temporal) order."""
    data_dir = ROOT / "data" / "external" / "msu_ics_power" / variant
    files = sorted(glob.glob(str(data_dir / "*.arff"))) + sorted(glob.glob(str(data_dir / "*.csv")))
    if not files:
        raise SystemExit(f"No MSU/ORNL files in {data_dir}")
    frames, file_ids = [], []
    for fp in files:
        part = load_one_file(Path(fp))
        frames.append(part)
        stem = Path(fp).name.split(" ")[0].split(".")[0]  # 'data7 Sampled ...arff' / 'data7.csv' -> 'data7'
        file_ids += [stem] * len(part)
    df = pd.concat(frames, ignore_index=True)
    df["_file"] = file_ids
    label_col = detect_label_column(df)
    feature_cols = [c for c in df.columns if c not in (label_col, "_file")]
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feature_cols, how="all").reset_index(drop=True)
    med = df[feature_cols].median(numeric_only=True)
    df[feature_cols] = df[feature_cols].fillna(med).fillna(0.0)
    df[label_col] = df[label_col].astype(str)
    return df, feature_cols, label_col


def save(name, obj):
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.json"
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"Saved {path.relative_to(ROOT)}")


def fresh_label_encoder(y):
    return LabelEncoder().fit(y)


__all__ = ["ROOT", "RESULTS", "FIGS", "LIGHTGBM_CV", "MSU_VARIANTS", "config", "load_synthetic",
           "synthetic_folds", "prepare_fold", "load_msu", "save", "build_model", "fresh_label_encoder"]
