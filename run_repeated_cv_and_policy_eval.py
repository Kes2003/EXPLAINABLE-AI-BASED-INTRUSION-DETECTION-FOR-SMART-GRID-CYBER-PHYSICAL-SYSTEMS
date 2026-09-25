"""
run_repeated_cv_and_policy_eval.py -- repeated cross-validation and prevention-policy evaluation.

Reuses the exact pipeline of scripts/cv_evaluation.py (the script that
produced manuscript Tables 2 and 3):
  - models built by src.model.build_model from config.yaml `model_variants`
  - minority-class oversampling via src.train.oversample_minority_classes,
    applied to the TRAINING fold only
  - StandardScaler fit on the (oversampled) training fold only
  - label encoding with a LabelEncoder fitted on the full label set

What it computes:
  1. Repeated stratified 5-fold CV over 10 fold-assignment seeds (42..51),
     i.e. n = 50 paired Macro-F1 observations per model, and paired Wilcoxon
     signed-rank tests of the best model vs. each competitor  -> Table 3c.
     Seed 42 is identical to cv_evaluation.py, so its 5 folds must reproduce
     Tables 2/3; this is printed as a check before anything else is reported.
  2. Prevention/policy-engine benchmark (Section 4.9): the out-of-fold
     LightGBM predictions from seed 42 are passed through the deployed
     src.api.compute_preventive_action and compared with the ground-truth
     class-to-action mapping (Table 10); the policy call itself is timed.

Usage (from the project root):
    python run_repeated_cv_and_policy_eval.py
Outputs:
    results/repeated_cv_and_policy_eval.json
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.api import compute_preventive_action  # deployed policy function
from src.model import build_model
from src.train import oversample_minority_classes
from src.utils import set_global_seed

SEEDS = list(range(42, 52))  # 10 fold-assignment seeds; 42 = original Tables 2/3 run
THRESHOLD = 0.7              # config.yaml thresholds.alert_probability

# Ground-truth action per TRUE class (manuscript Table 10).
GROUND_TRUTH_ACTION = {
    "normal":   (False, False),  # (trip_breaker, rate_limit_network)
    "dos":      (False, True),
    "fdi":      (False, False),  # escalate to operator only
    "tamper":   (True, False),
    "overload": (True, False),
}


def load():
    with open(ROOT / "config" / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    df = pd.read_csv(ROOT / "data" / "external" / "enhanced_realistic_smartgrid.csv")
    # The CSV copy recovered from predictions.csv spells the overload class
    # "overlo"; the deployed label encoder and src/api.py use "overload".
    # Renaming a class does not change any score, only the policy lookup.
    df["label"] = df["label"].astype(str).replace({"overlo": "overload"})
    return cfg, df


def run_seed(df, features, variants, le, seed, collect_oof=False):
    """One full 5-fold CV run, identical to scripts/cv_evaluation.py."""
    set_global_seed(seed)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    f1 = {n: [] for n in variants}
    acc = {n: [] for n in variants}
    oof = {"true": [], "pred": [], "prob": []}

    X_all = df[features].values
    y_all = df["label"].values
    for train_idx, test_idx in skf.split(X_all, y_all):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        train_bal = oversample_minority_classes(train_df, label_col="label", random_state=seed)

        scaler = StandardScaler().fit(train_bal[features].values)
        X_train = scaler.transform(train_bal[features].values)
        X_test = scaler.transform(test_df[features].values)
        y_train = le.transform(train_bal["label"].values)
        y_test = le.transform(test_df["label"].values)

        for name, model_cfg in variants.items():
            model = build_model(model_cfg)
            model.fit(X_train, y_train)
            if collect_oof and name == "lightgbm":
                proba = model.predict_proba(X_test)
                y_pred = proba.argmax(axis=1)
                oof["true"] += le.inverse_transform(y_test).tolist()
                oof["pred"] += le.inverse_transform(y_pred).tolist()
                oof["prob"] += proba[np.arange(len(y_pred)), y_pred].tolist()
            else:
                y_pred = np.asarray(model.predict(X_test)).ravel()
            f1[name].append(float(f1_score(y_test, y_pred, average="macro")))
            acc[name].append(float(accuracy_score(y_test, y_pred)))
    return f1, acc, oof


def prevention_benchmark(oof):
    rows = []
    for t, p, pr in zip(oof["true"], oof["pred"], oof["prob"]):
        a = compute_preventive_action(p, pr, THRESHOLD)
        rows.append((t, (a.should_trip_breaker, a.should_rate_limit_network)))

    n = len(rows)
    exact = sum(act == GROUND_TRUTH_ACTION[t] for t, act in rows)
    normals = [act for t, act in rows if t == "normal"]
    false_blocks = sum(any(act) for act in normals)
    needs_action = [(t, act) for t, act in rows if any(GROUND_TRUTH_ACTION[t])]
    missed = sum(not any(act) for _, act in needs_action)

    per_class = {}
    for cls in GROUND_TRUTH_ACTION:
        acts = [act for t, act in rows if t == cls]
        per_class[cls] = {
            "n": len(acts),
            "correct_action": sum(act == GROUND_TRUTH_ACTION[cls] for act in acts),
        }

    # Policy-call latency: every out-of-fold prediction, 5 passes.
    calls = list(zip(oof["pred"], oof["prob"]))
    for p, pr in calls[:1000]:  # warm-up
        compute_preventive_action(p, pr, THRESHOLD)
    per_call = []
    for _ in range(5):
        for p, pr in calls:
            t0 = time.perf_counter_ns()
            compute_preventive_action(p, pr, THRESHOLD)
            per_call.append(time.perf_counter_ns() - t0)
    lat_us = np.array(per_call) / 1000.0

    return {
        "threshold": THRESHOLD,
        "n_predictions": n,
        "response_accuracy": exact / n,
        "false_block_rate": false_blocks / len(normals),
        "false_blocks": false_blocks,
        "n_true_normal": len(normals),
        "missed_response_rate": missed / len(needs_action),
        "missed_responses": missed,
        "n_requiring_action": len(needs_action),
        "per_class": per_class,
        "latency_us": {
            "n_calls": int(lat_us.size),
            "mean": float(lat_us.mean()),
            "median": float(np.median(lat_us)),
            "p99": float(np.percentile(lat_us, 99)),
        },
    }


def main():
    cfg, df = load()
    features = cfg["features"]
    variants = cfg["model_variants"]
    le = LabelEncoder().fit(df["label"].values)
    print(f"Loaded {len(df):,} rows, classes: {df['label'].value_counts().to_dict()}")

    f1_all = {n: [] for n in variants}
    oof = None
    for i, seed in enumerate(SEEDS, 1):
        t0 = time.time()
        f1, acc, o = run_seed(df, features, variants, le, seed, collect_oof=(seed == 42))
        for n in variants:
            f1_all[n] += f1[n]
        print(f"[seed {seed}] ({i}/{len(SEEDS)}, {time.time() - t0:.0f}s) " +
              ", ".join(f"{n}={np.mean(f1[n]):.4f}" for n in variants), flush=True)
        if seed == 42:
            oof = o
            print("  Reproduction check vs. manuscript Table 2 (seed 42, 5 folds):")
            for n in variants:
                print(f"    {n:14s} Macro-F1 {np.mean(f1[n]):.4f} +/- {np.std(f1[n]):.4f}   "
                      f"acc {np.mean(acc[n]):.4f} +/- {np.std(acc[n]):.4f}")
            best5 = max(variants, key=lambda n: np.mean(f1[n]))
            for n in variants:
                if n != best5:
                    w, p = wilcoxon(f1[best5], f1[n])
                    print(f"    Table 3 check: {best5} vs {n}: W={w:.1f}, p={p:.4f}")
            prev = prevention_benchmark(oof)
            print("  Section 4.9 prevention benchmark:", json.dumps(prev, indent=2))

    summary = {n: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}
               for n, v in f1_all.items()}
    best = max(summary, key=lambda n: summary[n]["mean"])
    tests = {}
    for n in variants:
        if n == best:
            continue
        w, p = wilcoxon(f1_all[best], f1_all[n])
        diff = np.array(f1_all[best]) - np.array(f1_all[n])
        tests[n] = {"W": float(w), "p": float(p), "n": len(diff),
                    "mean_diff": float(diff.mean()), "best_wins": int((diff > 0).sum())}

    print("\nTABLE 3c -- repeated stratified 5-fold CV, 10 seeds (n = 50 per model)")
    for n, s in summary.items():
        print(f"  {n:14s} Macro-F1 {s['mean']:.4f} +/- {s['std']:.4f} (n={s['n']})")
    print(f"  best model: {best}")
    for n, t in tests.items():
        print(f"  {best} vs {n}: W={t['W']:.1f}, p={t['p']:.3g}, mean diff={t['mean_diff']:+.4f}, "
              f"{best} better in {t['best_wins']}/{t['n']} folds")

    out = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "seeds": SEEDS,
        "per_fold_macro_f1": f1_all,
        "summary": summary,
        "best_model": best,
        "wilcoxon_vs_best": tests,
        "prevention_benchmark": prev,
    }
    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / "repeated_cv_and_policy_eval.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved results/repeated_cv_and_policy_eval.json")


if __name__ == "__main__":
    main()
