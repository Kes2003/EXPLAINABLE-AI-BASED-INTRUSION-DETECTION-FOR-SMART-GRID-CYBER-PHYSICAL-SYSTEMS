from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    from .artifacts import load_preprocessor_and_model
    from .utils import ensure_dir, project_path, log_audit_entry, init_audit_db
except Exception:
    from artifacts import load_preprocessor_and_model
    from utils import ensure_dir, project_path, log_audit_entry, init_audit_db


def predict_df(df: pd.DataFrame, models_dir: str | Path, model_type: Optional[str] = None):
    preproc, model, features, used_type = load_preprocessor_and_model(models_dir, model_type=model_type)
    # ensure df has required features (subset possible)
    X = preproc.transform(df)
    # probabilities if available
    try:
        proba = model.predict_proba(X)
        pred_idx = proba.argmax(axis=1)
        labels = preproc.label_encoder.inverse_transform(pred_idx)
        probs = proba[range(len(pred_idx)), pred_idx]
    except Exception:
        labels = model.predict(X)
        probs = [1.0] * len(labels)

    out = df.copy()
    out["prediction"] = labels
    out["prediction_probability"] = probs
    return out, used_type


def main():
    parser = argparse.ArgumentParser(description="Predict using trained intrusion detection model")
    parser.add_argument("--csv", type=str, help="Input CSV path")
    parser.add_argument("--row", type=str, help="Single row JSON object with feature values")
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--out", type=str, default="predictions.csv")
    parser.add_argument("--model_type", type=str, default=None, help="Model type to use (random_forest, xgboost, lightgbm)")
    parser.add_argument("--audit", action="store_true", help="Log each prediction to the audit DB")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)

    if args.csv:
        df = pd.read_csv(args.csv)
    elif args.row:
        obj = json.loads(args.row)
        df = pd.DataFrame([obj])
    else:
        raise SystemExit("Provide either --csv or --row")

    out_df, used_type = predict_df(df, models_dir, model_type=args.model_type)

    out_path = Path(args.out)
    ensure_dir(out_path.parent)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote predictions to {out_path} using model_type={used_type}")

    if args.audit:
        audit_db = project_path("logs", "audit.db")
        init_audit_db(audit_db)
        for _, row in out_df.iterrows():
            features_json = {k: row[k] for k in df.columns if k in df.columns}
            predicted_label = row["prediction"]
            probability = float(row["prediction_probability"])
            action = {"auto_action": None}
            log_audit_entry(audit_db, used_type, features_json, predicted_label, probability, action)
        print(f"Logged {len(out_df)} predictions to audit DB: {audit_db}")


if __name__ == "__main__":
    main()
