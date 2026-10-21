# scripts/r4_shap_faithfulness.py
"""
Reviewer comment 7: are the SHAP explanations FAITHFUL to the model, not
merely stable?

Deletion / insertion tests (Petsiuk et al. 2018; Samek et al. 2017 style),
computed per explained sample x with predicted class c:
  deletion : features are replaced, one at a time and in a given order, by a
             baseline value; f_c is recorded after every step. A faithful
             ranking makes f_c fall fastest -> LOW area under the curve.
  insertion: start from the all-baseline input and restore features in the
             given order; a faithful ranking makes f_c rise fastest -> HIGH
             area under the curve.
Orderings compared: instance-level |SHAP| for class c (descending), the
reverse of it (least important first), the global mean-|SHAP| ranking
(same order for every sample), LightGBM gain importance, and random orders
(10 per sample). Two baselines: the training-fold mean (0 after scaling),
and 'marginal' -- values taken from randomly drawn training rows (averaged
over 5 draws), which keeps each replaced feature within its observed range.
f_c is the predicted probability of class c. AUCs are the mean of the curve
over k = 0..d deleted/inserted features (d = number of features).
We also report the fraction of samples whose predicted class changes after
deleting the top-1 and top-3 SHAP features vs. top-1/top-3 random features.

Global check (remove-and-retrain, ROAR; Hooker et al. 2019): the k globally
most important SHAP features are removed and the model RETRAINED, vs. k
random features (5 draws), k in {1, 2, 3, 5}; Macro-F1 under the same folds.

Datasets: synthetic (5-fold seed-42 CV, LightGBM, 400 test rows per fold)
and MSU/ORNL 37-class (Table 8 folds, LightGBM, 200 test rows per fold; with
128 features the curves use steps of 4 features; ROAR with k = 8 and two
random draws, to bound the cost of retraining 37-class models).

Usage:  python -m scripts.r4_shap_faithfulness
Output: results/r4/shap_faithfulness.json, results/r4/figures/faithfulness_*.png
"""
from __future__ import annotations

import numpy as np
import shap
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.r4_common import (FIGS, LIGHTGBM_CV, build_model, fresh_label_encoder, load_msu, load_synthetic,
                               prepare_fold, save, synthetic_folds)
from src.utils import set_global_seed


def class_shap(explainer, X, cls):
    sv = explainer.shap_values(X)
    sv = np.stack(sv, axis=-1) if isinstance(sv, list) else np.asarray(sv)  # (n, d, C)
    return sv[np.arange(len(X)), :, cls], sv


def curves(model, X, cls, orders, base_fn, steps):
    """orders: (n, d) feature indices per sample. Returns deletion and
    insertion curves (n, len(steps)) of f_c."""
    n, d = X.shape
    dele = np.zeros((n, len(steps)))
    ins = np.zeros((n, len(steps)))
    for si, k in enumerate(steps):
        Xd = X.copy()
        B = base_fn()
        Xi = B.copy()
        rows = np.repeat(np.arange(n), k)
        cols = orders[:, :k].ravel()
        if k:
            Xd[rows, cols] = B[rows, cols]
            Xi[rows, cols] = X[rows, cols]
        dele[:, si] = model.predict_proba(Xd)[np.arange(n), cls]
        ins[:, si] = model.predict_proba(Xi)[np.arange(n), cls]
    return dele, ins


def flip_rate(model, X, cls, orders, base, k):
    Xd = X.copy()
    rows = np.repeat(np.arange(len(X)), k)
    cols = orders[:, :k].ravel()
    Xd[rows, cols] = base[rows, cols]
    return float(np.mean(model.predict(Xd) != cls))


def faithfulness_fold(model, Xtr, Xte, rng, steps, n_random=10, n_marginal=5):
    d = Xte.shape[1]
    cls = model.predict(Xte).astype(int)
    expl = shap.TreeExplainer(model)
    phi, full = class_shap(expl, Xte, cls)
    global_rank = np.argsort(-np.abs(full).mean(axis=(0, 2)))
    gain_rank = np.argsort(-model.booster_.feature_importance(importance_type="gain"))
    orders = {
        "shap_instance": np.argsort(-np.abs(phi), axis=1),
        "shap_reverse": np.argsort(np.abs(phi), axis=1),
        "shap_global": np.tile(global_rank, (len(Xte), 1)),
        "gain_global": np.tile(gain_rank, (len(Xte), 1)),
    }
    mean_base = lambda: np.tile(Xtr.mean(axis=0), (len(Xte), 1))
    out = {}
    for bname in ["mean", "marginal"]:
        res = {}
        for oname, ordr in list(orders.items()) + [("random", None)]:
            reps = n_random if oname == "random" else 1
            dl, il = [], []
            for r in range(reps):
                o = ordr if ordr is not None else np.array([rng.permutation(d) for _ in range(len(Xte))])
                if bname == "mean":
                    dd, ii = curves(model, Xte, cls, o, mean_base, steps)
                else:
                    acc_d, acc_i = 0, 0
                    for _ in range(n_marginal):
                        draw = Xtr[rng.integers(0, len(Xtr), len(Xte))]
                        dd_, ii_ = curves(model, Xte, cls, o, lambda: draw, steps)
                        acc_d, acc_i = acc_d + dd_, acc_i + ii_
                    dd, ii = acc_d / n_marginal, acc_i / n_marginal
                dl.append(dd); il.append(ii)
            dd, ii = np.mean(dl, axis=0), np.mean(il, axis=0)
            res[oname] = {"deletion_curve": dd.mean(axis=0).tolist(), "insertion_curve": ii.mean(axis=0).tolist(),
                          "deletion_auc_per_sample": dd.mean(axis=1), "insertion_auc_per_sample": ii.mean(axis=1)}
        out[bname] = res
    base = mean_base()
    rand_orders = np.array([rng.permutation(d) for _ in range(len(Xte))])
    flips = {f"shap_top{k}": flip_rate(model, Xte, cls, orders["shap_instance"], base, k) for k in (1, 3)}
    flips.update({f"random_top{k}": flip_rate(model, Xte, cls, rand_orders, base, k) for k in (1, 3)})
    return out, flips, global_rank


def summarise(per_fold, steps):
    summary = {}
    for bname in per_fold[0][0]:
        summary[bname] = {}
        for oname in per_fold[0][0][bname]:
            da = np.concatenate([f[0][bname][oname]["deletion_auc_per_sample"] for f in per_fold])
            ia = np.concatenate([f[0][bname][oname]["insertion_auc_per_sample"] for f in per_fold])
            summary[bname][oname] = {
                "deletion_auc_mean": float(da.mean()), "deletion_auc_sem": float(da.std() / np.sqrt(len(da))),
                "insertion_auc_mean": float(ia.mean()), "insertion_auc_sem": float(ia.std() / np.sqrt(len(ia))),
                "deletion_curve": np.mean([f[0][bname][oname]["deletion_curve"] for f in per_fold], axis=0).tolist(),
                "insertion_curve": np.mean([f[0][bname][oname]["insertion_curve"] for f in per_fold], axis=0).tolist(),
                "n_samples": int(len(da))}
        # paired comparison SHAP-instance vs random (sign test proportion + Wilcoxon)
        from scipy.stats import wilcoxon
        a = np.concatenate([f[0][bname]["shap_instance"]["deletion_auc_per_sample"] for f in per_fold])
        b = np.concatenate([f[0][bname]["random"]["deletion_auc_per_sample"] for f in per_fold])
        c = np.concatenate([f[0][bname]["shap_instance"]["insertion_auc_per_sample"] for f in per_fold])
        e = np.concatenate([f[0][bname]["random"]["insertion_auc_per_sample"] for f in per_fold])
        summary[bname]["shap_vs_random"] = {
            "deletion_shap_lower_fraction": float(np.mean(a < b)), "deletion_wilcoxon_p": float(wilcoxon(a, b).pvalue),
            "insertion_shap_higher_fraction": float(np.mean(c > e)), "insertion_wilcoxon_p": float(wilcoxon(c, e).pvalue)}
    flips = {k: float(np.mean([f[1][k] for f in per_fold])) for k in per_fold[0][1]}
    return {"steps": list(steps), "auc": summary, "prediction_flip_rate": flips}


def roar(d, folds_fn, rank, ks, rng, n_random=5, baseline=None):
    """Remove-and-retrain over d features with the given folds; rank = global SHAP order.
    folds_fn(cols) yields (X_train, X_test, y_train, y_test) restricted to cols."""
    def cv_f1(cols):
        f1s = []
        for Xtr, Xte, ytr, yte in folds_fn(cols):
            m = build_model(LIGHTGBM_CV)
            m.fit(Xtr, ytr)
            f1s.append(f1_score(yte, m.predict(Xte), average="macro"))
        return float(np.mean(f1s))
    out = {"all_features": baseline if baseline is not None else cv_f1(np.arange(d))}
    for k in ks:
        keep_shap = np.setdiff1d(np.arange(d), rank[:k])
        rnd = [cv_f1(np.setdiff1d(np.arange(d), rng.choice(d, k, replace=False))) for _ in range(n_random)]
        out[f"k{k}"] = {"remove_top_shap": cv_f1(keep_shap), "remove_random_mean": float(np.mean(rnd)),
                        "remove_random_min": float(np.min(rnd)), "remove_random_max": float(np.max(rnd))}
        print(f"   ROAR k={k}: top-SHAP removed {out[f'k{k}']['remove_top_shap']:.4f} | "
              f"random removed {out[f'k{k}']['remove_random_mean']:.4f}", flush=True)
    return out


def main():
    set_global_seed(42)
    rng = np.random.default_rng(42)
    results = {}

    # ---------------- synthetic ----------------
    _, df, features = load_synthetic()
    le = fresh_label_encoder(df["label"].values)
    folds = synthetic_folds(df, 42)
    steps = list(range(0, len(features) + 1))
    per_fold, ranks, prepared = [], [], []
    for k, (tr, te) in enumerate(folds, 1):
        f = prepare_fold(df, features, le, tr, te, 42)
        prepared.append(f)
        m = build_model(LIGHTGBM_CV); m.fit(f["X_train"], f["y_train"])
        sel = rng.choice(len(f["X_test"]), 400, replace=False)
        out, flips, rank = faithfulness_fold(m, f["X_train"], f["X_test"][sel], rng, steps)
        per_fold.append((out, flips)); ranks.append(rank)
        print(f"synthetic fold {k}: SHAP del-AUC {out['mean']['shap_instance']['deletion_auc_per_sample'].mean():.4f} "
              f"vs random {out['mean']['random']['deletion_auc_per_sample'].mean():.4f}", flush=True)
    syn = summarise(per_fold, steps)
    global_rank = np.argsort(np.mean([np.argsort(r) for r in ranks], axis=0))  # mean rank position
    syn["global_shap_rank"] = [features[i] for i in global_rank]

    def syn_folds(cols):
        for (tr, te), f in zip(folds, prepared):
            yield f["X_train"][:, cols], f["X_test"][:, cols], f["y_train"], f["y_test"]
    syn["roar"] = roar(len(features), syn_folds, global_rank, [1, 2, 3, 5], rng)
    results["synthetic"] = syn
    save("shap_faithfulness", results)

    # ---------------- MSU/ORNL 37-class ----------------
    msu, feats, label_col = load_msu("multiclass")
    X = msu[feats].values
    y_str = msu[label_col].values
    y = LabelEncoder().fit_transform(y_str)
    mfolds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y_str))
    msteps = list(range(0, len(feats) + 1, 4))
    per_fold, ranks, mprep, mf1 = [], [], [], []
    for k, (tr, te) in enumerate(mfolds, 1):
        sc = StandardScaler().fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        mprep.append((Xtr, Xte))
        m = build_model(LIGHTGBM_CV); m.fit(Xtr, y[tr])
        mf1.append(f1_score(y[te], m.predict(Xte), average="macro"))
        sel = rng.choice(len(Xte), 200, replace=False)
        out, flips, rank = faithfulness_fold(m, Xtr, Xte[sel], rng, msteps, n_random=5, n_marginal=3)
        per_fold.append((out, flips)); ranks.append(rank)
        print(f"MSU fold {k}: SHAP del-AUC {out['mean']['shap_instance']['deletion_auc_per_sample'].mean():.4f} "
              f"vs random {out['mean']['random']['deletion_auc_per_sample'].mean():.4f}", flush=True)
    ms = summarise(per_fold, msteps)
    mrank = np.argsort(np.mean([np.argsort(r) for r in ranks], axis=0))
    ms["global_shap_rank_top10"] = [feats[i] for i in mrank[:10]]

    def msu_folds(cols):
        for (tr, te), (Xtr, Xte) in zip(mfolds, mprep):
            yield Xtr[:, cols], Xte[:, cols], y[tr], y[te]
    ms["roar"] = roar(len(feats), msu_folds, mrank, [8], rng, n_random=2, baseline=float(np.mean(mf1)))
    results["msu_37class"] = ms
    save("shap_faithfulness", results)

    # ---------------- figure ----------------
    FIGS.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for row, (name, r) in enumerate([("Synthetic", syn), ("MSU/ORNL 37-class", ms)]):
        for col, kind in enumerate(["deletion", "insertion"]):
            ax = axes[row, col]
            for oname, st in [("shap_instance", "o-"), ("shap_global", "s--"), ("gain_global", "d-."),
                              ("random", "x:"), ("shap_reverse", "v:")]:
                ax.plot(r["steps"], r["auc"]["mean"][oname][f"{kind}_curve"], st, ms=3, label=oname)
            ax.set_title(f"{name}: {kind} (mean baseline)", fontsize=10)
            ax.set_xlabel(f"Features {'removed' if kind == 'deletion' else 'restored'}")
            ax.set_ylabel("Predicted probability of original class"); ax.grid(alpha=.3)
    axes[0, 0].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(FIGS / "shap_faithfulness_curves.png", dpi=200); plt.close()


if __name__ == "__main__":
    main()
