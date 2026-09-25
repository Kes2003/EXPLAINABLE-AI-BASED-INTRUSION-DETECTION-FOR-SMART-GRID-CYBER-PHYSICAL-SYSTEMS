# scripts/ablation_study.py
"""
Feature-group ablation study (Reviewer 4, pt.6: "evaluate the contribution
of individual components of the proposed framework").

Groups the 18 features into the 4 categories the manuscript itself already
defines in Section 3.3 ("Feature Selection and Engineering"):
  - electrical:        voltage_v, current_a, active_power_kw, reactive_power_kvar,
                        power_factor, feeder_voltage_v, feeder_current_a, demand_kw,
                        line_load_percent
  - power_quality:      frequency_hz, thd_percent, phase_imbalance_percent
  - communication:      packet_rate, packet_error_rate
  - equipment_health:   breaker_closed, substation_temp_c, transformer_oil_temp_c,
                         tap_position

For each group, trains + 5-fold-CV-evaluates LightGBM with that group REMOVED
(all other features kept), and compares Macro-F1 against the full-feature
baseline. A large drop when a group is removed = that group matters a lot;
a small/no drop = the model isn't relying on it much. Uses the same
leakage-free split-then-oversample methodology as scripts/cv_evaluation.py.

Usage:
  python -m scripts.ablation_study
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, accuracy_score

from src.model import build_model
from src.train import oversample_minority_classes
from src.utils import ensure_dir, save_json, set_global_seed

FEATURE_GROUPS = {
    "electrical": ["voltage_v", "current_a", "active_power_kw", "reactive_power_kvar",
                   "power_factor", "feeder_voltage_v", "feeder_current_a", "demand_kw",
                   "line_load_percent"],
    "power_quality": ["frequency_hz", "thd_percent", "phase_imbalance_percent"],
    "communication": ["packet_rate", "packet_error_rate"],
    "equipment_health": ["breaker_closed", "substation_temp_c", "transformer_oil_temp_c",
                          "tap_position"],
}

MODEL_CFG = {"type": "lightgbm", "params": {"n_estimators": 200, "max_depth": -1,
                                             "learning_rate": 0.1, "num_leaves": 31,
                                             "n_jobs": -1, "random_state": 42}}


def cv_macro_f1(df: pd.DataFrame, feature_subset: list[str], seed: int, n_splits: int = 5) -> list[float]:
    le = LabelEncoder().fit(df["label"].astype(str).values)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for train_idx, test_idx in skf.split(df[feature_subset].values, df["label"].values):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        train_bal = oversample_minority_classes(train_df, label_col="label", random_state=seed)

        Xtr = train_bal[feature_subset].values
        ytr = le.transform(train_bal["label"].astype(str).values)
        Xte = test_df[feature_subset].values
        yte = le.transform(test_df["label"].astype(str).values)

        scaler = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

        model = build_model(MODEL_CFG)
        model.fit(Xtr_s, ytr)
        pred = model.predict(Xte_s)
        scores.append(float(f1_score(yte, pred, average="macro")))
    return scores


def main():
    set_global_seed(42)
    df = pd.read_csv("data/external/enhanced_realistic_smartgrid.csv")

    all_features = [f for group in FEATURE_GROUPS.values() for f in group]

    # sanity check: groups must cover exactly the model's feature set, no gaps/overlaps
    import yaml
    with open("config/config.yaml") as fp:
        cfg = yaml.safe_load(fp)
    configured = set(cfg["features"])
    if set(all_features) != configured:
        print("WARNING: FEATURE_GROUPS in this script does not exactly match config.yaml's "
              f"feature list. In config but not grouped: {configured - set(all_features)}. "
              f"Grouped but not in config: {set(all_features) - configured}.")

    results = {}

    print("Baseline (all features)...")
    baseline_scores = cv_macro_f1(df, all_features, seed=42)
    results["all_features"] = {
        "features_used": len(all_features),
        "macro_f1_mean": float(np.mean(baseline_scores)),
        "macro_f1_std": float(np.std(baseline_scores)),
        "per_fold": baseline_scores,
    }
    print(f"  Macro-F1 = {np.mean(baseline_scores):.4f} +/- {np.std(baseline_scores):.4f}\n")

    for group_name, group_feats in FEATURE_GROUPS.items():
        subset = [f for f in all_features if f not in group_feats]
        print(f"Removing group '{group_name}' ({len(group_feats)} features: {group_feats})...")
        scores = cv_macro_f1(df, subset, seed=42)
        drop = np.mean(baseline_scores) - np.mean(scores)
        results[f"without_{group_name}"] = {
            "removed_features": group_feats,
            "features_used": len(subset),
            "macro_f1_mean": float(np.mean(scores)),
            "macro_f1_std": float(np.std(scores)),
            "macro_f1_drop_vs_baseline": float(drop),
            "per_fold": scores,
        }
        print(f"  Macro-F1 = {np.mean(scores):.4f} +/- {np.std(scores):.4f}  "
              f"(drop = {drop:+.4f} vs. baseline)\n")

    ensure_dir("artifacts")
    save_json(results, "artifacts/ablation_study_results.json")
    print("Saved: artifacts/ablation_study_results.json")

    # Plot: Macro-F1 drop per removed group (higher bar = that group matters more)
    group_names = list(FEATURE_GROUPS.keys())
    drops = [results[f"without_{g}"]["macro_f1_drop_vs_baseline"] for g in group_names]
    order = np.argsort(drops)[::-1]

    plt.figure(figsize=(7.5, 4.5))
    plt.bar([group_names[i] for i in order], [drops[i] for i in order], color="#C44E52")
    plt.ylabel("Macro-F1 drop when group is removed\n(higher = more important)")
    plt.title("Feature-group ablation (LightGBM, 5-fold CV)")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    ensure_dir("artifacts/visualizations")
    plt.savefig("artifacts/visualizations/ablation_study.png", dpi=200, bbox_inches="tight")
    print("Saved: artifacts/visualizations/ablation_study.png")

    print("\n=== Ranked by importance (largest F1 drop first) ===")
    for i in order:
        print(f"  {group_names[i]:20s} drop = {drops[i]:+.4f}")


if __name__ == "__main__":
    main()
