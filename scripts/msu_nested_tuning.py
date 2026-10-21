# scripts/msu_nested_tuning.py
"""
Nested, event-disjoint hyperparameter tuning for the MSU/ORNL detectors of
scripts/msu_improved_eval.py.

Outer loop: GroupKFold(5) by recording file, exactly as msu_improved_eval.py.
Inner loop (training files of the outer fold only): candidates are scored by
Macro-F1 on held-out training recordings; the best candidate is refit on the
whole outer training fold and scored once on the outer test fold. The outer
test recordings never take part in selection.

  events  ExtraTrees and LightGBM grids on the per-event summary features,
          inner GroupKFold(4) by file (model family is selected too)
  rows    LightGBM grid on the `full` per-row features; to bound run time the
          inner split is one hold-out of 3 of the 12 training recordings

Usage:  python -m scripts.msu_nested_tuning [events|rows] [variant ...]
Output: results/msu_improved/msu_nested_tuning.json (resumable)
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
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

from scripts.msu_improved_eval import LGBM, VARIANTS, data, event_matrix, row_features, summarise
from scripts.r4_common import ROOT

OUT = ROOT / "results" / "msu_improved" / "msu_nested_tuning.json"

EVENT_GRID = (
    [("et", dict(n_estimators=500, max_features=mf, min_samples_leaf=leaf, n_jobs=4, random_state=42))
     for mf in ("sqrt", 0.1, 0.3) for leaf in (1, 2)]
    + [("lgbm", dict(n_estimators=300, learning_rate=0.05, num_leaves=nl, min_child_samples=3,
                     colsample_bytree=cs, n_jobs=4, verbose=-1, random_state=42))
       for nl in (7, 15) for cs in (0.1, 0.3)]
)
ROW_GRID = [dict(LGBM, num_leaves=nl, min_child_samples=mcs) for nl in (15, 63) for mcs in (20, 100)]


def make(kind, params):
    return ExtraTreesClassifier(**params) if kind == "et" else lgb.LGBMClassifier(**params)


def f1(m, X, y, tr, te):
    p = np.asarray(m.fit(X[tr], y[tr]).predict(X[te])).ravel()
    return f1_score(y[te], p, average="macro"), p


def nested(X, y, groups, grid, inner_splits):
    f1s, accs, chosen = [], [], []
    for tr, te in GroupKFold(5).split(X, y, groups):
        scores = []
        for kind, params in grid:
            s = [f1(make(kind, params), X[tr], y[tr], a, b)[0] for a, b in inner_splits(X[tr], y[tr], groups[tr])]
            scores.append(float(np.mean(s)))
        best = int(np.argmax(scores))
        kind, params = grid[best]
        score, p = f1(make(kind, params), X, y, tr, te)
        f1s.append(score); accs.append(accuracy_score(y[te], p))
        chosen.append({"model": kind, "params": {k: v for k, v in params.items() if k not in ("n_jobs", "verbose", "random_state")},
                       "inner_macro_f1": scores[best]})
        print(f"   outer fold {len(f1s)}: {kind} inner {scores[best]:.4f} outer {score:.4f}", flush=True)
    return summarise(f1s, accs, {"selected_per_fold": chosen})


def main():
    what = sys.argv[1]
    variants = sys.argv[2:] or VARIANTS
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    ref, feats, ev, labels = data()
    files = pd.Series(ref["_file"].values)
    if what == "events":
        agg = event_matrix(ref, feats, ev, 0)
        X, g = agg.values.astype(np.float32), files.groupby(ev).first().loc[agg.index].values
        inner = lambda X_, y_, g_: list(GroupKFold(4).split(X_, y_, g_))  # noqa: E731
        for v in variants:
            key = f"events/{v}/first_all"
            if key in res:
                continue
            y = LabelEncoder().fit_transform(pd.Series(labels[v]).groupby(ev).agg(lambda s: s.mode().iloc[0]).loc[agg.index].values)
            res[key] = nested(X, y, g, EVENT_GRID, inner)
            print(key, round(res[key]["macro_f1_mean"], 4), flush=True)
            OUT.write_text(json.dumps(res, indent=2))
    else:
        X = row_features(ref, feats, ev).values.astype(np.float32)
        g = files.values
        inner = lambda X_, y_, g_: list(GroupShuffleSplit(1, test_size=3, random_state=42).split(X_, y_, g_))  # noqa: E731
        for v in variants:
            key = f"rows/{v}/full"
            if key in res:
                continue
            t = time.time()
            y = LabelEncoder().fit_transform(labels[v])
            res[key] = nested(X, y, g, [("lgbm", p) for p in ROW_GRID], inner)
            res[key]["seconds"] = round(time.time() - t)
            print(key, round(res[key]["macro_f1_mean"], 4), flush=True)
            OUT.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
