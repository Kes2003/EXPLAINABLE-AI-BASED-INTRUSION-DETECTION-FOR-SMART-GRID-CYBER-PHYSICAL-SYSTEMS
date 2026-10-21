# scripts/msu_improved_eval.py
"""
Improved MSU/ORNL detectors, evaluated event-disjoint only.

Protocol (identical to protocol B of scripts/r4_disjoint_validation.py):
GroupKFold(5) with group = recording file (data1..data15), so every test
fold holds 3 complete recordings and no row of a test event is ever seen in
training. Nothing is fitted on test recordings.

Events are the per-scenario recordings concatenated in each file (one
contiguous run per scenario). Features are computed causally WITHIN each
event (from its first recorded row up to the current row), so the windows
never cross into the neighbouring, differently labelled recording.

Two tasks:
  1. per-row (streaming): one prediction per PMU snapshot, LightGBM on
       raw   the 128 original columns (baseline, same booster settings)
       full  raw + physics features + causal rolling-window features of the
             raw columns + causal since-recording-start features of the raw
             and physics columns (scripts/msu_improve_features.py)
  2. per-event: one prediction per recorded event from summary statistics
     (mean, std, min, max, first, last) of its rows, ExtraTrees; also using
     only the first k rows (early decision), plus a label-shuffle control.

Usage:  python -m scripts.msu_improved_eval [rows|events|all]
Output: results/msu_improved/msu_improved_eval.json (resumable: finished
        configurations are kept and skipped on re-run)
"""
from __future__ import annotations

import json
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder

from scripts.msu_improve_features import expanding_features, phys_features, temporal_features
from scripts.r4_common import ROOT, load_msu
from scripts.r4_disjoint_validation import event_ids

OUT = ROOT / "results" / "msu_improved" / "msu_improved_eval.json"
VARIANTS = ["binary", "triple", "multiclass"]
LGBM = dict(n_estimators=200, learning_rate=0.1, num_leaves=31, max_bin=63, colsample_bytree=0.5,
            subsample=0.5, subsample_freq=1, n_jobs=4, random_state=42, verbose=-1)
FIRST_K = [5, 10, 20, 40, 80, 0]  # 0 = whole event


def load_results():
    return json.loads(OUT.read_text()) if OUT.exists() else {}


def save(res):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))


def summarise(f1s, accs, extra=None):
    d = {"macro_f1_mean": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s)),
         "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
         "per_fold_macro_f1": [float(v) for v in f1s]}
    d.update(extra or {})
    return d


def cv(make, X, y, groups):
    f1s, accs = [], []
    for tr, te in GroupKFold(5).split(X, y, groups):
        p = np.asarray(make().fit(X[tr], y[tr]).predict(X[te])).ravel()
        f1s.append(f1_score(y[te], p, average="macro"))
        accs.append(accuracy_score(y[te], p))
    return f1s, accs


def data():
    ref, feats, rl = load_msu("multiclass")
    ev = event_ids(ref, rl)
    labels = {}
    for v in VARIANTS:
        df, f2, lab = load_msu(v)
        assert f2 == feats and (df["_file"].values == ref["_file"].values).all()
        assert np.isclose(df[feats].values, ref[feats].values, rtol=1e-5, atol=1e-6).all(), "row order differs"
        labels[v] = df[lab].astype(str).values
    return ref, feats, ev, labels


def row_features(ref, feats, ev):
    """The `full` per-row feature matrix (causal, within each event)."""
    base = ref[feats].copy()
    base["_event"] = ev
    P = phys_features(ref)
    Pk = pd.concat([P, base[["_event"]]], axis=1)
    return pd.concat([ref[feats], P, temporal_features(base, feats),
                      expanding_features(base, feats), expanding_features(Pk, list(P.columns))], axis=1)


def run_rows(res, ref, feats, ev, labels):
    full = row_features(ref, feats, ev)
    sets = {"raw": ref[feats].values.astype(np.float32), "full": full.values.astype(np.float32)}
    groups = ref["_file"].values
    for v in VARIANTS:
        y = LabelEncoder().fit_transform(labels[v])
        for name, X in sets.items():
            key = f"rows/{v}/{name}"
            if key in res:
                continue
            t = time.time()
            f1s, accs = cv(lambda: lgb.LGBMClassifier(**LGBM), X, y, groups)
            res[key] = summarise(f1s, accs, {"n_features": int(X.shape[1]), "n_rows": int(len(y)),
                                             "seconds": round(time.time() - t)})
            print(key, round(res[key]["macro_f1_mean"], 4), f"({res[key]['seconds']}s)", flush=True)
            save(res)


def event_matrix(ref, feats, ev, first_k):
    F = pd.concat([ref[feats], phys_features(ref)], axis=1)
    F["_event"] = ev
    if first_k:
        F = F.groupby("_event", sort=False).head(first_k)
    agg = F.groupby("_event").agg(["mean", "std", "min", "max", "first", "last"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return agg.fillna(0.0)


def run_events(res, ref, feats, ev, labels):
    files = pd.Series(ref["_file"].values).groupby(ev).first()
    make = lambda: ExtraTreesClassifier(500, n_jobs=4, random_state=42)  # noqa: E731
    for k in FIRST_K:
        agg = event_matrix(ref, feats, ev, k)
        X, g = agg.values.astype(np.float32), files.loc[agg.index].values
        for v in VARIANTS:
            key = f"events/{v}/first_{k or 'all'}"
            if key in res:
                continue
            ylab = pd.Series(labels[v]).groupby(ev).agg(lambda s: s.mode().iloc[0]).loc[agg.index].values
            y = LabelEncoder().fit_transform(ylab)
            f1s, accs = cv(make, X, y, g)
            res[key] = summarise(f1s, accs, {"n_events": int(len(y)), "n_classes": int(len(np.unique(y)))})
            if k == 0:
                rng = np.random.default_rng(42)
                sh = [cv(make, X, rng.permutation(y), g) for _ in range(3)]
                res[key]["label_shuffle_control_macro_f1"] = float(np.mean([np.mean(s[0]) for s in sh]))
            print(key, round(res[key]["macro_f1_mean"], 4), flush=True)
            save(res)


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    res = load_results()
    ref, feats, ev, labels = data()
    res["structure"] = {"n_rows": int(len(ref)), "n_files": int(ref["_file"].nunique()),
                        "n_events": int(len(np.unique(ev))), "rows_per_event_median": float(np.median(np.bincount(ev)))}
    if what in ("events", "all"):
        run_events(res, ref, feats, ev, labels)
    if what in ("rows", "all"):
        run_rows(res, ref, feats, ev, labels)
    print(json.dumps({k: round(v["macro_f1_mean"], 4) for k, v in res.items() if "macro_f1_mean" in v}, indent=1))


if __name__ == "__main__":
    main()
