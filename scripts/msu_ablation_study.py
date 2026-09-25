# scripts/msu_ablation_study.py
"""
Feature-group ablation on the MSU/ORNL 37-class task (manuscript Section 4.7),
mirroring the synthetic-dataset ablation of Section 4.4.

Uses the exact Table 8 pipeline (msu_ics_dataset_eval.py loading and cleaning,
StratifiedKFold(5, shuffle=True, random_state=42), per-fold StandardScaler,
LightGBM configuration) and removes one feature group at a time. The full
feature set is re-run first as the baseline and must reproduce Table 8.

Usage:
    python -m scripts.msu_ablation_study --data_dir data/external/msu_ics_power/multiclass
Output:
    results/msu_ablation_<tag>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from scripts.msu_ics_dataset_eval import detect_label_column, load_all
from scripts.msu_shap_analysis import LIGHTGBM, clean, feature_group
from src.model import build_model
from src.utils import set_global_seed

# Ablation groups: the four PMU measurement groups, and every cyber-side column
# (control-panel/relay logs, Snort logs, relay status fields) removed together.
ABLATION_GROUPS = {
    "PMU voltage phasors": {"PMU voltage phasors"},
    "PMU current phasors": {"PMU current phasors"},
    "PMU apparent impedance": {"PMU apparent impedance"},
    "PMU frequency / frequency delta": {"PMU frequency / frequency delta"},
    "Cyber-side (logs + relay status)": {"Control-panel / relay log", "Snort IDS log", "PMU relay status flags"},
}


def cv_macro_f1(X, y_str, le):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1s = []
    for tr, te in skf.split(X, y_str):
        scaler = StandardScaler().fit(X[tr])
        model = build_model(LIGHTGBM)
        model.fit(scaler.transform(X[tr]), le.transform(y_str[tr]))
        pred = model.predict(scaler.transform(X[te]))
        f1s.append(float(f1_score(le.transform(y_str[te]), pred, average="macro")))
    return f1s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/external/msu_ics_power/multiclass")
    args = ap.parse_args()

    set_global_seed(42)
    df = load_all(ROOT / args.data_dir)
    label_col = detect_label_column(df)
    df, feature_cols = clean(df, label_col)
    y_str = df[label_col].astype(str).values
    le = LabelEncoder().fit(y_str)
    groups = {c: feature_group(c) for c in feature_cols}

    out = {"baseline": None, "ablations": {}}
    f1s = cv_macro_f1(df[feature_cols].values, y_str, le)
    base = float(np.mean(f1s))
    out["baseline"] = {"n_features": len(feature_cols), "fold_macro_f1": f1s, "mean": base, "std": float(np.std(f1s))}
    print(f"Baseline (all {len(feature_cols)} features): {base:.4f} +/- {np.std(f1s):.4f} "
          f"(Table 8: 0.8307 +/- 0.0022)", flush=True)

    for name, members in ABLATION_GROUPS.items():
        keep = [c for c in feature_cols if groups[c] not in members]
        f1s = cv_macro_f1(df[keep].values, y_str, le)
        m = float(np.mean(f1s))
        out["ablations"][name] = {"n_removed": len(feature_cols) - len(keep), "fold_macro_f1": f1s,
                                  "mean": m, "std": float(np.std(f1s)), "drop": base - m}
        print(f"without {name:34s} ({len(feature_cols) - len(keep):2d} removed): {m:.4f} +/- {np.std(f1s):.4f}"
              f"  drop = {base - m:+.4f}", flush=True)

    tag = Path(args.data_dir).name
    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / f"msu_ablation_{tag}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved results/msu_ablation_{tag}.json")


if __name__ == "__main__":
    main()
