"""
run_reviewer_experiments.py

WHERE TO PUT THIS: project root — the same folder as config/, data/, models/,
src/ — i.e. "AI-Enabled Intrusion Detection for Smart Grids/run_reviewer_experiments.py"

HOW TO RUN:
    python run_reviewer_experiments.py

WHAT THIS DOES (matches manuscript Sections 3.6.1 and 4.9 exactly):
  1. Loads your real dataset (data/external/enhanced_realistic_smartgrid.csv)
     and feature list (config/config.yaml), exactly as src/train.py does.
  2. Runs a repeated stratified 5x10-fold cross-validation (n=50 paired
     Macro-F1 observations per model, vs. the original n=5) and a paired
     Wilcoxon signed-rank test -> fills manuscript Table 3c.
  3. Runs a fresh leakage-safe 5-fold CV with LightGBM to get out-of-fold
     predictions (this reproduces the data behind Fig. 8's confusion
     matrix), then re-scores those predictions through your REAL,
     deployed `compute_preventive_action` policy function to get response
     accuracy, false-block rate, and policy latency -> fills manuscript
     Section 4.9.
  4. Prints ready-to-paste manuscript text for both, with numbers already
     filled in — copy those blocks straight over the [RUN & INSERT]
     markers in the manuscript.

WHAT THIS DOES NOT DO:
  SHAP-on-MSU/ORNL (manuscript Section 4.7 / Table 8b) is intentionally NOT
  included here. That must run against the specific MSU/ORNL model + test
  split you already used to produce Table 8, and that pipeline (four
  models x three label granularities) is not the one shown in
  src/train_unsw.py in this conversation (which only trains one Random
  Forest). Use the run_msu_ornl_shap() function near the bottom of this
  file — load your existing Table-8 model and test set, then call it.

RUNTIME NOTE: step 2 trains 4 models x 50 folds (200 total fits). On the
30,000-row dataset this took the earlier full single-model training run a
few seconds each, so expect this to run for several minutes — a progress
counter is printed.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import wilcoxon

import xgboost as xgb
import lightgbm as lgb

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None
    print("[WARN] catboost not installed — CatBoost will be skipped in the "
          "repeated-CV comparison. Install with: pip install catboost")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api import compute_preventive_action  # the real, deployed policy function

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# 1. Load data exactly as src/train.py does
# ---------------------------------------------------------------------------
def load_data():
    cfg_path = PROJECT_ROOT / "config" / "config.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    features = cfg["features"]
    data_path = PROJECT_ROOT / "data" / "external" / "enhanced_realistic_smartgrid.csv"
    if not data_path.exists():
        raise SystemExit(
            f"Dataset not found at {data_path}\n"
            f"Edit `data_path` in load_data() if your file lives elsewhere."
        )

    df = pd.read_csv(data_path)
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise SystemExit(f"Dataset is missing expected features: {missing}")

    X_raw = df[features].values.astype(float)
    # The dataset spells the overload class "overlo" (truncated). src/api.py's
    # policy engine matches on "overload", so normalize it here; otherwise the
    # overload class would silently fall through to "no action" on both the
    # predicted and the ground-truth side of the Section 4.9 benchmark.
    y_str = df["label"].astype(str).replace({"overlo": "overload"}).values

    le = LabelEncoder()
    y_int = le.fit_transform(y_str)

    return X_raw, y_int, y_str, le, features


# ---------------------------------------------------------------------------
# 2. Leakage-safe oversampling (mirrors src/train.py::oversample_minority_classes,
#    applied to arrays post-split, per Algorithm 1 in the manuscript)
# ---------------------------------------------------------------------------
def oversample_train_fold(X_train, y_train, random_state=RANDOM_STATE):
    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(y_train, return_counts=True)
    max_count = counts.max()
    X_parts, y_parts = [], []
    for cls, n in zip(classes, counts):
        idx = np.where(y_train == cls)[0]
        if n < max_count:
            extra = rng.integers(0, n, size=max_count - n)
            idx = np.concatenate([idx, idx[extra]])
        X_parts.append(X_train[idx])
        y_parts.append(y_train[idx])
    X_os = np.concatenate(X_parts, axis=0)
    y_os = np.concatenate(y_parts, axis=0)
    perm = rng.permutation(len(y_os))
    return X_os[perm], y_os[perm]


# ---------------------------------------------------------------------------
# 3. Model builders — hyperparameters match manuscript Section 3.9's
#    cross-validation protocol (NOT config.yaml's production single-split
#    hyperparameters — the manuscript explicitly notes these differ)
# ---------------------------------------------------------------------------
def build_models(random_state=RANDOM_STATE):
    models = {
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=200, max_depth=20, n_jobs=-1,
            class_weight="balanced", random_state=random_state),
        "xgboost": lambda: xgb.XGBClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.1,
            n_jobs=-1, random_state=random_state,
            eval_metric="mlogloss"),
        "lightgbm": lambda: lgb.LGBMClassifier(
            n_estimators=200, max_depth=-1, learning_rate=0.1,
            num_leaves=31, n_jobs=-1, random_state=random_state, verbose=-1),
    }
    if CatBoostClassifier is not None:
        models["catboost"] = lambda: CatBoostClassifier(
            iterations=200, depth=8, learning_rate=0.1,
            random_state=random_state, verbose=False)
    return models


# ---------------------------------------------------------------------------
# 4. Table 3c — repeated stratified 5x10-fold CV significance test
# ---------------------------------------------------------------------------
def repeated_cv_significance(X_raw, y_int, n_splits=5, n_repeats=10):
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                    random_state=RANDOM_STATE)
    model_builders = build_models()
    scores = {name: [] for name in model_builders}

    fold_num = 0
    total_folds = n_splits * n_repeats
    for train_idx, test_idx in rskf.split(X_raw, y_int):
        fold_num += 1
        print(f"  [repeated CV] fold {fold_num}/{total_folds}", end="\r")
        X_train, X_test = X_raw[train_idx], X_raw[test_idx]
        y_train, y_test = y_int[train_idx], y_int[test_idx]

        X_train_os, y_train_os = oversample_train_fold(X_train, y_train)
        scaler = StandardScaler().fit(X_train_os)
        X_train_s = scaler.transform(X_train_os)
        X_test_s = scaler.transform(X_test)

        for name, build_fn in model_builders.items():
            model = build_fn()
            model.fit(X_train_s, y_train_os)
            preds = model.predict(X_test_s)
            scores[name].append(f1_score(y_test, preds, average="macro"))
    print()  # newline after progress counter

    return {k: np.array(v) for k, v in scores.items()}


def print_table_3c(scores):
    print("\n" + "=" * 78)
    print("TABLE 3c — Repeated Cross-Validation (5x10 folds, n=50 paired observations)")
    print("=" * 78)
    print(f"{'Model':<15} {'Mean Macro-F1':<15} {'Std':<10} {'N':<5}")
    for name, arr in scores.items():
        print(f"{name:<15} {arr.mean():<15.4f} {arr.std():<10.4f} {len(arr):<5}")

    best = max(scores, key=lambda k: scores[k].mean())
    print(f"\nBest model: {best}\n")
    print(f"{'Comparison':<30} {'n':<5} {'W':<10} {'p-value':<10} {'Significant?'}")
    results = {}
    for name in scores:
        if name != best:
            stat, p = wilcoxon(scores[best], scores[name])
            sig = "YES" if p < 0.05 else "No"
            print(f"{best} vs. {name:<15} {len(scores[best]):<5} {stat:<10.2f} {p:<10.4f} {sig}")
            results[name] = (stat, p, sig)

    print("\n--- Ready-to-paste: Table 3c rows ---")
    for name, (stat, p, sig) in results.items():
        print(f"| {best.replace('_', ' ').title()} vs. {name.replace('_', ' ').title()} "
              f"| 50 | {stat:.2f} | {p:.4f} | {sig} |")

    any_sig = any(r[2] == "YES" for r in results.values())
    print("\n--- Ready-to-paste: interpretive sentence ---")
    if not any_sig:
        print(
            "Even with a tenfold increase in paired observations, the comparisons "
            "remain non-significant at \u03b1 = 0.05, reinforcing the interpretation "
            "that LightGBM, Random Forest, and XGBoost are practically equivalent "
            "on this task rather than distinguishable only by an underpowered test."
        )
    else:
        sig_names = [n for n, r in results.items() if r[2] == "YES"]
        print(
            f"With the increased sample size, the comparison between {best} and "
            f"{', '.join(sig_names)} newly reaches significance at \u03b1 = 0.05, "
            f"indicating a statistically distinguishable, if small, advantage for {best}."
        )
    return best, results


# ---------------------------------------------------------------------------
# 5. Out-of-fold predictions with LightGBM, for the Section 4.9 benchmark
#    (reproduces the data behind Fig. 8's confusion matrix)
# ---------------------------------------------------------------------------
def get_out_of_fold_predictions(X_raw, y_int, label_encoder, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    all_true_str, all_pred_str, all_pred_proba = [], [], []
    fold_f1 = []

    for train_idx, test_idx in skf.split(X_raw, y_int):
        X_train, X_test = X_raw[train_idx], X_raw[test_idx]
        y_train, y_test = y_int[train_idx], y_int[test_idx]

        X_train_os, y_train_os = oversample_train_fold(X_train, y_train)
        scaler = StandardScaler().fit(X_train_os)
        X_train_s = scaler.transform(X_train_os)
        X_test_s = scaler.transform(X_test)

        model = lgb.LGBMClassifier(
            n_estimators=200, max_depth=-1, learning_rate=0.1,
            num_leaves=31, n_jobs=-1, random_state=RANDOM_STATE, verbose=-1)
        model.fit(X_train_s, y_train_os)

        proba = model.predict_proba(X_test_s)
        pred_idx = proba.argmax(axis=1)
        pred_prob = proba[np.arange(len(pred_idx)), pred_idx]
        fold_f1.append(f1_score(y_test, pred_idx, average="macro"))

        all_true_str.extend(label_encoder.inverse_transform(y_test).tolist())
        all_pred_str.extend(label_encoder.inverse_transform(pred_idx).tolist())
        all_pred_proba.extend(pred_prob.tolist())

    # Sanity check against manuscript Table 2 (LightGBM 0.9285 +/- 0.0030):
    # confirms these out-of-fold predictions reproduce the published CV run.
    fold_f1 = np.array(fold_f1)
    print(f"  [check vs. Table 2] LightGBM 5-fold Macro-F1: "
          f"{fold_f1.mean():.4f} +/- {fold_f1.std():.4f} "
          f"(per fold: {', '.join(f'{v:.4f}' for v in fold_f1)})")

    return all_true_str, all_pred_str, all_pred_proba


# ---------------------------------------------------------------------------
# 6. Section 4.9 — prevention/policy-engine benchmark
# ---------------------------------------------------------------------------
GROUND_TRUTH_ACTION = {
    "dos":      {"trip": False, "rate_limit": True},
    "tamper":   {"trip": True,  "rate_limit": False},
    "overload": {"trip": True,  "rate_limit": False},
    "fdi":      {"trip": False, "rate_limit": False},   # escalation-only by design
    "normal":   {"trip": False, "rate_limit": False},
}


def benchmark_prevention(y_true_labels, y_pred_labels, y_pred_probs, threshold=0.7):
    n = len(y_true_labels)
    correct = 0
    false_blocks = 0
    n_true_normal = 0

    for true_lbl, pred_lbl, prob in zip(y_true_labels, y_pred_labels, y_pred_probs):
        action = compute_preventive_action(pred_lbl, prob, threshold)
        took_action = bool(action.should_trip_breaker or action.should_rate_limit_network)
        expected = GROUND_TRUTH_ACTION.get(true_lbl, GROUND_TRUTH_ACTION["normal"])
        expected_action = bool(expected["trip"] or expected["rate_limit"])
        if took_action == expected_action:
            correct += 1
        if true_lbl == "normal":
            n_true_normal += 1
            if took_action:
                false_blocks += 1

    return correct / n, false_blocks / max(1, n_true_normal)


def benchmark_policy_latency(sample_label, sample_prob, threshold=0.7, n_trials=10_000):
    start = time.perf_counter()
    for _ in range(n_trials):
        compute_preventive_action(sample_label, sample_prob, threshold)
    return ((time.perf_counter() - start) / n_trials) * 1e6  # microseconds


def print_section_4_9(y_true_labels, y_pred_labels, y_pred_probs):
    acc, fbr = benchmark_prevention(y_true_labels, y_pred_labels, y_pred_probs, threshold=0.7)
    latency_us = benchmark_policy_latency(y_pred_labels[0], y_pred_probs[0])

    print("\n" + "=" * 78)
    print("SECTION 4.9 — Prevention/Policy-Engine Performance Analysis")
    print("=" * 78)
    print(f"Response accuracy: {acc:.4%}")
    print(f"False-block rate:  {fbr:.4%}")
    print(f"Policy latency (mean, 10,000 calls): {latency_us:.2f} us")

    # For the record only: the same benchmark with the dataset's raw "overlo"
    # label left in place (i.e. a model trained on the raw CSV). The policy
    # engine never acts on "overlo", so every true overload is missed.
    raw_preds = ["overlo" if l == "overload" else l for l in y_pred_labels]
    acc_raw, fbr_raw = benchmark_prevention(
        y_true_labels, raw_preds, y_pred_probs, threshold=0.7)
    print(f"[as-labelled 'overlo' predictions] response accuracy: {acc_raw:.4%}, "
          f"false-block rate: {fbr_raw:.4%}")

    print("\n--- Ready-to-paste: response accuracy / false-block sentence ---")
    print(
        f"Re-scoring the leakage-safe five-fold cross-validation predictions from "
        f"Section 3.6 against this mapping, at the deployed alert-probability "
        f"threshold of 0.7, gives a response accuracy of {acc:.1%} and a "
        f"false-block rate of {fbr:.2%}, the latter driven entirely by the "
        f"normal-class false-positive rate already visible in the confusion "
        f"matrix of Fig. 8."
    )
    print("\n--- Ready-to-paste: latency sentence ---")
    print(
        f"The resulting mean per-call latency was {latency_us:.2f} \u03bcs, "
        f"negligible relative to the 1.70 ms mean ML-inference latency of Table 9."
    )
    return acc, fbr, latency_us


# ---------------------------------------------------------------------------
# 7. SHAP on MSU/ORNL — STUB ONLY (see note at top of file)
# ---------------------------------------------------------------------------
def run_msu_ornl_shap(model, X_test, feature_names, top_n=15):
    """
    Not called automatically. Load the SAME model + test split you already
    used to produce manuscript Table 8 (do not retrain a new model), then
    call: run_msu_ornl_shap(your_model, your_X_test, your_feature_names)
    """
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    if isinstance(shap_values, list):
        mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[::-1][:top_n]

    print("\n--- Ready-to-paste: Table 8b rows ---")
    for rank, i in enumerate(top_idx, start=1):
        print(f"| {rank} | {feature_names[i]} | {mean_abs_shap[i]:.4f} |")

    return [(feature_names[i], float(mean_abs_shap[i])) for i in top_idx]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading dataset and config...")
    X_raw, y_int, y_str, label_encoder, features = load_data()
    print(f"Loaded {len(y_int):,} rows, {len(features)} features, "
          f"{len(label_encoder.classes_)} classes: {list(label_encoder.classes_)}")

    print("\n[1/2] Running repeated cross-validation (5x10 folds) — trains "
          "4 models x 50 folds, this will take a few minutes...")
    scores = repeated_cv_significance(X_raw, y_int, n_splits=5, n_repeats=10)
    print_table_3c(scores)

    print("\n[2/2] Generating out-of-fold predictions (LightGBM, 5-fold) for "
          "the prevention-layer benchmark...")
    y_true_labels, y_pred_labels, y_pred_probs = get_out_of_fold_predictions(
        X_raw, y_int, label_encoder)
    print_section_4_9(y_true_labels, y_pred_labels, y_pred_probs)

    print("\n" + "=" * 78)
    print("DONE. Copy each 'Ready-to-paste' block above over the matching")
    print("[RUN & INSERT] marker in the manuscript (Table 3c, Section 3.6.1,")
    print("Section 4.9, Abstract, Conclusion, Section 4.10).")
    print("SHAP-on-MSU/ORNL (Section 4.7 / Table 8b) is NOT run by this")
    print("script — see run_msu_ornl_shap() and the note at the top of this file.")
    print("=" * 78)
