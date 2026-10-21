# scripts/r4_disjoint_validation.py
"""
Reviewer comment 5: scenario-, event- and time-disjoint validation on MSU/ORNL.

Structure of the dataset (verified by this script and printed): the release
consists of 15 recording files (data1 ... data15). Every file contains one
contiguous, time-ordered run of rows per scenario (36-37 scenarios per file),
so a (file, scenario) pair is one executed event, and each scenario is
executed once per file. Stratified random K-fold (Table 8) therefore puts
neighbouring rows of the SAME event in both training and test folds.

Protocols compared, with the Table 8 hyperparameters for LightGBM and Random
Forest (the best models in Table 8; XGBoost and CatBoost were not repeated
here to bound run time) and a StandardScaler fit on the training fold only:
  A. stratified random 5-fold (Table 8; reproduced here as the reference)
  B. event-disjoint GroupKFold(5) with group = recording file: every test
     fold holds 3 complete recordings, so every test event is an execution
     the model has never seen any part of.
  C. time-disjoint, within event: for every event the first 80% of its rows
     (in recorded order) are training data and the last 20% test data, with a
     gap of 5% of the event's rows discarded in between.

For the synthetic dataset there is no scenario, event or time identifier: its
rows are independently generated samples. The script checks this by
comparing the number of label runs in file order with the number expected
for independently ordered rows.

Usage:  python -m scripts.r4_disjoint_validation
Output: results/r4/disjoint_validation.json
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

from scripts.r4_common import MSU_VARIANTS, build_model, load_msu, load_synthetic, save
from src.utils import set_global_seed


def run_cv(X, y, splits, models):
    out = {m: {"f1": [], "acc": []} for m in models}
    for tr, te in splits:
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        for m in models:
            model = build_model(MSU_VARIANTS[m])
            model.fit(Xtr, y[tr])
            p = np.asarray(model.predict(Xte)).ravel()
            out[m]["f1"].append(float(f1_score(y[te], p, average="macro")))
            out[m]["acc"].append(float(accuracy_score(y[te], p)))
    return {m: {"macro_f1_mean": float(np.mean(v["f1"])), "macro_f1_std": float(np.std(v["f1"])),
                "accuracy_mean": float(np.mean(v["acc"])), "accuracy_std": float(np.std(v["acc"])),
                "per_fold_f1": v["f1"]} for m, v in out.items()}


def event_ids(df, label_col):
    """Consecutive rows with the same label inside one file = one event."""
    lab = df[label_col].values
    f = df["_file"].values  # multiclass release: one contiguous run per scenario per file
    new = np.r_[True, (lab[1:] != lab[:-1]) | (f[1:] != f[:-1])]
    return np.cumsum(new) - 1


def time_disjoint_split(ev, train_frac=0.8, gap_frac=0.05):
    tr, te = [], []
    for e in np.unique(ev):
        idx = np.where(ev == e)[0]  # already in recorded order
        n = len(idx)
        if n < 10:
            continue
        cut = int(np.floor(train_frac * n))
        gap = max(1, int(np.ceil(gap_frac * n)))
        tr += idx[:cut].tolist()
        te += idx[min(n, cut + gap):].tolist()
    return np.array(tr), np.array(te)


def synthetic_order_check():
    _, df, _ = load_synthetic()
    lab = df["label"].values
    runs = int(1 + np.sum(lab[1:] != lab[:-1]))
    p = df["label"].value_counts(normalize=True).values
    expected = float(1 + (len(df) - 1) * (1 - np.sum(p ** 2)))
    sd = float(np.sqrt((len(df) - 1) * (1 - np.sum(p ** 2)) * np.sum(p ** 2)))  # approx.
    return {"n_rows": len(df), "label_runs_in_file_order": runs, "expected_runs_if_independent": expected,
            "approx_sd": sd, "z": (runs - expected) / sd,
            "identifier_columns": [c for c in df.columns if c.lower() in ("time", "timestamp", "scenario", "event", "id")]}


def main():
    set_global_seed(42)
    results = {"synthetic_order_check": synthetic_order_check()}
    print("Synthetic dataset order check:", results["synthetic_order_check"])

    # The triple and binary releases contain the same 78,377 rows, in the same
    # order, as the multiclass release (verified below), only with coarser
    # labels; in them, rows are sorted by coarse label inside each file, so the
    # events are identified from the multiclass labels.
    ref, ref_feats, ref_label = load_msu("multiclass")
    ev_ref = event_ids(ref, ref_label)
    for variant in ["multiclass", "triple", "binary"]:
        t0 = time.time()
        df, feats, label_col = load_msu(variant)
        assert feats == ref_feats and len(df) == len(ref)
        assert np.isclose(df[feats].values, ref[feats].values, rtol=1e-5, atol=1e-6).all(), "row order differs"
        assert (df["_file"].values == ref["_file"].values).all()
        X = df[feats].values
        y = LabelEncoder().fit_transform(df[label_col].values)
        groups = df["_file"].values
        ev = ev_ref
        structure = {"n": len(df), "n_files": int(len(np.unique(groups))), "n_events": int(len(np.unique(ev))),
                     "rows_per_event_median": float(np.median(np.bincount(ev))),
                     "classes": int(len(np.unique(y)))}
        print(f"\n=== {variant}: {structure}", flush=True)
        models = ["lightgbm", "random_forest"]
        res = {"structure": structure}
        res["A_stratified_random_5fold"] = run_cv(
            X, y, list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, df[label_col].values)), models)
        print("  A stratified:", {m: round(v["macro_f1_mean"], 4) for m, v in res["A_stratified_random_5fold"].items()}, flush=True)
        res["B_event_disjoint_groupkfold5"] = run_cv(X, y, list(GroupKFold(5).split(X, y, groups)), models)
        print("  B group-by-file:", {m: round(v["macro_f1_mean"], 4) for m, v in res["B_event_disjoint_groupkfold5"].items()}, flush=True)
        tr, te = time_disjoint_split(ev)
        res["C_time_disjoint_within_event"] = run_cv(X, y, [(tr, te)], models)
        res["C_time_disjoint_within_event"]["n_train"], res["C_time_disjoint_within_event"]["n_test"] = len(tr), len(te)
        print("  C time-disjoint:", {m: round(v["macro_f1_mean"], 4) for m, v in res["C_time_disjoint_within_event"].items()
                                     if isinstance(v, dict)}, f"({time.time() - t0:.0f}s)", flush=True)
        results[variant] = res
        save("disjoint_validation", results)


if __name__ == "__main__":
    main()
