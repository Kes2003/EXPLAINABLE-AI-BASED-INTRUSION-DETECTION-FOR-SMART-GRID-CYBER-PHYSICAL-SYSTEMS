from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

from .datasets.unsw_nb15 import load_unsw_nb15, prepare_unsw_for_classification
from .utils import ensure_dir, save_json, set_global_seed


def main():
    parser = argparse.ArgumentParser(description="Train model on UNSW-NB15 CSVs")
    parser.add_argument("--csvs", nargs="+", required=True, help="Paths to UNSW-NB15 CSV files")
    parser.add_argument("--out_models", default="models_unsw", help="Output models directory")
    parser.add_argument("--out_artifacts", default="artifacts_unsw", help="Output artifacts directory")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_global_seed(args.seed)

    df = load_unsw_nb15(args.csvs)
    X_df, y_raw, features = prepare_unsw_for_classification(df)

    scaler = StandardScaler()
    X = scaler.fit_transform(X_df.values)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw.values)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=1,
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=args.seed,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    out_models = Path(args.out_models)
    out_artifacts = Path(args.out_artifacts)
    ensure_dir(out_models)
    ensure_dir(out_artifacts)

    joblib.dump(scaler, out_models / "scaler.joblib")
    joblib.dump(label_encoder, out_models / "label_encoder.joblib")
    joblib.dump(model, out_models / "model.joblib")
    with open(out_models / "features.json", "w", encoding="utf-8") as f:
        json.dump({"features": features}, f, indent=2)

    save_json({"classification_report": report, "confusion_matrix": cm}, out_artifacts / "metrics.json")

    print(f"Saved UNSW model to {out_models}")
    print(f"Saved UNSW metrics to {out_artifacts / 'metrics.json'}")


if __name__ == "__main__":
    main()




