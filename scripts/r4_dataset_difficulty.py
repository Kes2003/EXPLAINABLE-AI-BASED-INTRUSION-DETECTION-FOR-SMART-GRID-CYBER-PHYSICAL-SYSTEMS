# scripts/r4_dataset_difficulty.py
"""
Reviewer comment 8: is the higher synthetic-data performance due to an easier,
more controlled data distribution?

Measured on both datasets with identical code (standardized features):
  1. Data-complexity measures (Ho & Basu 2002; Lorena et al. 2019):
       F1   maximum Fisher discriminant ratio over features (one-vs-rest,
            averaged over classes) -> higher = easier
       N1   fraction of points on the class boundary (minimum spanning tree
            edges joining different classes), 4,000-row stratified subsample
            -> lower = easier
       kNN  leave-one-out 1-NN error on the same subsample -> lower = easier
  2. Model-capacity curve: Macro-F1 (5-fold stratified CV) of models of
     increasing capacity -- majority class, logistic regression, decision
     stumps / trees of depth 1, 2, 4, 8, and the Table 2/8 LightGBM. On an
     'easy' distribution low-capacity models already approach the ceiling.
  3. Controlled comparisons that remove known confounders:
       (a) same label granularity: binary attack-vs-non-attack on both
       (b) same training-set size: MSU/ORNL subsampled to 30,000 rows
       (c) same number of features: MSU/ORNL restricted to its 18 most
           important features (global mean |SHAP| ranking of
           results/msu_shap_multiclass.json)
       (d) event-disjoint MSU/ORNL evaluation (results/r4/disjoint_validation.json)
           is reported alongside, from scripts/r4_disjoint_validation.py.
       (e) why random row-level splitting is optimistic on MSU/ORNL: for 4,000
           random query rows, the fraction whose nearest neighbour among all
           other rows (standardized features) comes from the same recorded
           event, and the 1-NN accuracy when the neighbour may / may not come
           from the same event.
  4. Generator artefacts in the synthetic data: the generator clips every
     feature to a fixed range, so a share of rows sits EXACTLY on a range
     bound, and this share differs strongly between classes. We report, per
     feature and class, the fraction of rows on a bound; the Macro-F1 of a
     LightGBM that sees ONLY 36 binary 'on lower bound' / 'on upper bound'
     indicators (no measured values at all); and the Macro-F1 of the normal
     pipeline when every row that lies on any bound of a continuous feature
     is removed (from training and test folds alike).

Synthetic folds use the leakage-safe protocol (fold-local oversampling); MSU
folds follow Table 8 (no oversampling).

Usage:  python -m scripts.r4_dataset_difficulty
Output: results/r4/dataset_difficulty.json
"""
from __future__ import annotations

import json

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from scripts.r4_common import LIGHTGBM_CV, ROOT, build_model, load_msu, load_synthetic, save
from src.train import oversample_minority_classes
from src.utils import set_global_seed

N_SUB = 4000


def fisher_f1(X, y):
    vals = []
    for c in np.unique(y):
        a, b = X[y == c], X[y != c]
        num = (a.mean(0) - b.mean(0)) ** 2
        den = a.var(0) + b.var(0) + 1e-12
        vals.append(np.max(num / den))
    return float(np.mean(vals))


def n1_and_knn(X, y):
    D = cdist(X, X)
    mst = minimum_spanning_tree(D).tocoo()
    boundary = np.zeros(len(X), bool)
    diff = y[mst.row] != y[mst.col]
    boundary[mst.row[diff]] = True
    boundary[mst.col[diff]] = True
    np.fill_diagonal(D, np.inf)
    nn = D.argmin(1)
    return float(boundary.mean()), float(np.mean(y[nn] != y))


def capacity_models():
    return {
        "logistic_regression": lambda: LogisticRegression(max_iter=2000),
        "tree_depth1": lambda: DecisionTreeClassifier(max_depth=1, random_state=42),
        "tree_depth2": lambda: DecisionTreeClassifier(max_depth=2, random_state=42),
        "tree_depth4": lambda: DecisionTreeClassifier(max_depth=4, random_state=42),
        "tree_depth8": lambda: DecisionTreeClassifier(max_depth=8, random_state=42),
        "lightgbm": lambda: build_model(LIGHTGBM_CV),
    }


def cv_scores(X, y, oversample, models, seed=42):
    out = {m: [] for m in models}
    out["majority_class"] = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        Xtr, ytr = X[tr], y[tr]
        if oversample:
            import pandas as pd
            d = pd.DataFrame(Xtr); d["label"] = ytr
            d = oversample_minority_classes(d, label_col="label", random_state=seed)
            ytr = d.pop("label").values; Xtr = d.values
        sc = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(X[te])
        maj = np.bincount(y[tr]).argmax()
        out["majority_class"].append(f1_score(y[te], np.full(len(te), maj), average="macro"))
        for name, mk in models.items():
            m = mk(); m.fit(Xtr_s, ytr)
            out[name].append(f1_score(y[te], np.asarray(m.predict(Xte_s)).ravel(), average="macro"))
    return {k: {"macro_f1_mean": float(np.mean(v)), "macro_f1_std": float(np.std(v))} for k, v in out.items()}


def complexity(X, y, seed=42):
    Xs = StandardScaler().fit_transform(X)
    idx, _ = train_test_split(np.arange(len(y)), train_size=N_SUB, stratify=y, random_state=seed)
    n1, knn = n1_and_knn(Xs[idx], y[idx])
    return {"F1_max_fisher_ratio": fisher_f1(Xs, y), "N1_boundary_fraction": n1, "knn1_loo_error": knn,
            "n_features": int(X.shape[1]), "n_classes": int(len(np.unique(y))), "n_rows": int(len(y))}


def saturation(df, features):
    cont = [c for c in features if df[c].nunique() > 2]
    lo, hi = df[cont].min(), df[cont].max()
    on_lo = df[cont].eq(lo).astype(int)
    on_hi = df[cont].eq(hi).astype(int)
    per = {}
    for c in cont:
        per[c] = {cls: float((on_lo[c] | on_hi[c])[df["label"] == cls].mean()) for cls in sorted(df["label"].unique())}
    flags = np.hstack([on_lo.values, on_hi.values])
    y = LabelEncoder().fit_transform(df["label"].values)
    flags_only = cv_scores(flags.astype(float), y, True, {"lightgbm": lambda: build_model(LIGHTGBM_CV)})
    any_bound = (on_lo.values | on_hi.values).any(axis=1)
    keep = ~any_bound
    sub = df[keep]
    ys = LabelEncoder().fit_transform(sub["label"].values)
    unsat = cv_scores(sub[features].values, ys, True, {"lightgbm": lambda: build_model(LIGHTGBM_CV)})
    return {"per_feature_class_fraction_on_bound": per,
            "rows_on_any_bound_fraction": float(any_bound.mean()),
            "rows_on_any_bound_by_class": {cls: float(any_bound[df["label"].values == cls].mean())
                                           for cls in sorted(df["label"].unique())},
            "bound_indicators_only_lightgbm": flags_only["lightgbm"],
            "bound_indicators_only_majority": flags_only["majority_class"],
            "without_rows_on_bounds": {"n_rows": int(keep.sum()),
                                       "class_counts": sub["label"].value_counts().to_dict(),
                                       "lightgbm": unsat["lightgbm"]}}


def nn_same_event(X, y, ev, n_query=4000, seed=42):
    Xs = StandardScaler().fit_transform(X).astype(np.float32)
    q = np.random.default_rng(seed).choice(len(Xs), n_query, replace=False)
    sq = (Xs ** 2).sum(1)
    nn_any = np.empty(n_query, int)
    nn_other = np.empty(n_query, int)
    for s0 in range(0, n_query, 500):
        idx = q[s0:s0 + 500]
        D = sq[idx, None] - 2 * Xs[idx] @ Xs.T + sq[None, :]
        D[np.arange(len(idx)), idx] = np.inf
        nn_any[s0:s0 + 500] = D.argmin(1)
        D[ev[idx][:, None] == ev[None, :]] = np.inf
        nn_other[s0:s0 + 500] = D.argmin(1)
    return {"n_query": n_query,
            "nn_from_same_event_fraction": float(np.mean(ev[nn_any] == ev[q])),
            "knn1_accuracy_any_neighbour": float(np.mean(y[nn_any] == y[q])),
            "knn1_accuracy_neighbour_from_other_event": float(np.mean(y[nn_other] == y[q]))}


def main():
    set_global_seed(42)
    models = capacity_models()
    res = {}
    _, syn0, feats0 = load_synthetic()
    res["synthetic_saturation"] = saturation(syn0, feats0)
    print("synthetic saturation:", {k: v for k, v in res["synthetic_saturation"].items()
                                    if k != "per_feature_class_fraction_on_bound"}, flush=True)

    _, syn, features = load_synthetic()
    Xs = syn[features].values
    ys = LabelEncoder().fit_transform(syn["label"].values)
    ysb = (syn["label"].values != "normal").astype(int)
    res["synthetic_5class"] = {"complexity": complexity(Xs, ys), "capacity": cv_scores(Xs, ys, True, models)}
    print("synthetic 5-class:", json.dumps(res["synthetic_5class"], indent=1), flush=True)
    res["synthetic_binary"] = {"complexity": complexity(Xs, ysb), "capacity": cv_scores(Xs, ysb, True, models)}
    print("synthetic binary:", {k: round(v["macro_f1_mean"], 4) for k, v in res["synthetic_binary"]["capacity"].items()}, flush=True)

    msu, feats, label_col = load_msu("multiclass")
    Xm = msu[feats].values
    ym = LabelEncoder().fit_transform(msu[label_col].values)
    mb, _, lb = load_msu("binary")
    ymb = (mb[lb].values == "Attack").astype(int)
    # 37-class LightGBM under these folds is exactly Table 8 (not refitted here)
    no_lgb = {k: v for k, v in models.items() if k != "lightgbm"}
    res["msu_37class"] = {"complexity": complexity(Xm, ym), "capacity": cv_scores(Xm, ym, False, no_lgb)}
    t8 = json.loads((ROOT / "artifacts" / "msu_external_validation_results_multiclass.json").read_text())
    res["msu_37class"]["capacity"]["lightgbm"] = {"macro_f1_mean": t8["summary"]["lightgbm"]["macro_f1_mean"],
                                                  "macro_f1_std": t8["summary"]["lightgbm"]["macro_f1_std"],
                                                  "source": "Table 8 (msu_ics_dataset_eval.py)"}
    print("MSU 37-class:", json.dumps(res["msu_37class"], indent=1), flush=True)
    res["msu_binary"] = {"complexity": complexity(Xm, ymb), "capacity": cv_scores(Xm, ymb, False, models)}
    print("MSU binary:", {k: round(v["macro_f1_mean"], 4) for k, v in res["msu_binary"]["capacity"].items()}, flush=True)

    # (e) nearest neighbours and recorded events
    from scripts.r4_disjoint_validation import event_ids
    ev = event_ids(msu, label_col)
    res["msu_nn_events_37class"] = nn_same_event(Xm, ym, ev)
    res["msu_nn_events_binary"] = nn_same_event(Xm, ymb, ev)
    print("MSU nearest-neighbour / event check:", res["msu_nn_events_37class"], res["msu_nn_events_binary"], flush=True)

    # (b) same training-set size
    sub, _ = train_test_split(np.arange(len(ym)), train_size=30000, stratify=ym, random_state=42)
    res["msu_37class_n30000"] = cv_scores(Xm[sub], ym[sub], False, {"lightgbm": models["lightgbm"]})
    res["msu_binary_n30000"] = cv_scores(Xm[sub], ymb[sub], False, {"lightgbm": models["lightgbm"]})
    # (c) same number of features: top-18 by global mean |SHAP| (Table 8b analysis)
    shap_json = json.loads((ROOT / "results" / "msu_shap_multiclass.json").read_text())
    ranking = shap_json.get("ranking") or shap_json.get("features") or shap_json.get("top_features")
    names = [r["feature"] if isinstance(r, dict) else r for r in ranking][:18]
    cols = [feats.index(n) for n in names]
    res["msu_37class_top18"] = cv_scores(Xm[:, cols], ym, False, {"lightgbm": models["lightgbm"]})
    res["msu_binary_top18"] = cv_scores(Xm[:, cols], ymb, False, {"lightgbm": models["lightgbm"]})
    res["msu_top18_features"] = names
    for k in ["msu_37class_n30000", "msu_binary_n30000", "msu_37class_top18", "msu_binary_top18"]:
        print(k, {m: round(v["macro_f1_mean"], 4) for m, v in res[k].items()}, flush=True)
    save("dataset_difficulty", res)


if __name__ == "__main__":
    main()
