# scripts/shap_stability.py
"""
SHAP explanation stability (manuscript Section 4.3, Fig. 9c).

  1. Cross-fold: LightGBM is trained on each of the five seed-42 folds
     (leakage-safe protocol) and the global mean(|SHAP|) feature ranking is
     computed on that fold's held-out test data; pairwise Spearman rank
     correlations over the 10 fold pairs, and the top-5 feature sets.
  2. Perturbation: with the fold-1 model, SHAP values of 200 held-out rows
     are recomputed after adding Gaussian noise of 1% and 5% of each
     feature's scale (standardized units); per-sample Spearman correlation of
     the |SHAP| vector (predicted class) with the clean-input one, averaged
     over the 200 rows; 10 trials.

Usage:  python -m scripts.shap_stability
Output: results/r4/shap_stability.json, results/r4/figures/shap_stability_heatmap.png
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import shap
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from scripts.r4_common import (FIGS, LIGHTGBM_CV, build_model, fresh_label_encoder, load_synthetic, prepare_fold,
                               save, synthetic_folds)
from src.utils import set_global_seed


def shap3(expl, X):
    sv = expl.shap_values(X)
    return np.stack(sv, axis=-1) if isinstance(sv, list) else np.asarray(sv)


def main():
    set_global_seed(42)
    rng = np.random.default_rng(42)
    _, df, features = load_synthetic()
    le = fresh_label_encoder(df["label"].values)
    imps, fold1 = [], None
    for k, (tr, te) in enumerate(synthetic_folds(df, 42), 1):
        f = prepare_fold(df, features, le, tr, te, 42)
        m = build_model(LIGHTGBM_CV); m.fit(f["X_train"], f["y_train"])
        expl = shap.TreeExplainer(m)
        sv = shap3(expl, f["X_test"])
        imps.append(np.abs(sv).mean(axis=(0, 2)))
        if k == 1:
            fold1 = (m, expl, f["X_test"])
        print(f"fold {k}: top-5 = {[features[i] for i in np.argsort(-imps[-1])[:5]]}", flush=True)
    pairs = {f"{a + 1}-{b + 1}": float(spearmanr(imps[a], imps[b]).statistic) for a, b in combinations(range(5), 2)}
    mat = np.array([[spearmanr(imps[a], imps[b]).statistic for b in range(5)] for a in range(5)])
    top5 = [[features[i] for i in np.argsort(-v)[:5]] for v in imps]
    modal = max(set(map(frozenset, top5)), key=lambda s: sum(frozenset(t) == s for t in top5))

    m, expl, Xte = fold1
    idx = rng.choice(len(Xte), 200, replace=False)
    X = Xte[idx]
    cls = m.predict(X).astype(int)
    base = np.abs(shap3(expl, X)[np.arange(len(X)), :, cls])
    pert = {}
    for lvl in (0.01, 0.05):
        trials = []
        for _ in range(10):
            Xn = X + rng.normal(0, lvl, X.shape)
            sn = np.abs(shap3(expl, Xn)[np.arange(len(X)), :, cls])
            trials.append(np.mean([spearmanr(base[i], sn[i]).statistic for i in range(len(X))]))
        pert[str(lvl)] = {"mean": float(np.mean(trials)), "std": float(np.std(trials)), "trials": trials}

    out = {"pairwise_spearman": pairs, "mean_pairwise_spearman": float(np.mean(list(pairs.values()))),
           "min_pairwise_spearman": float(np.min(list(pairs.values()))), "top5_per_fold": top5,
           "modal_top5": sorted(modal), "folds_with_modal_top5": int(sum(frozenset(t) == modal for t in top5)),
           "perturbation": pert}
    print(f"mean pairwise Spearman {out['mean_pairwise_spearman']:.3f} (min {out['min_pairwise_spearman']:.3f}); "
          f"modal top-5 in {out['folds_with_modal_top5']}/5 folds; perturbation "
          + ", ".join(f"{k}: {v['mean']:.3f} +/- {v['std']:.3f}" for k, v in pert.items()))
    save("shap_stability", out)
    FIGS.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4.5, 3.8))
    sns.heatmap(mat, annot=True, fmt=".2f", vmin=0.8, vmax=1.0, cmap="viridis",
                xticklabels=[f"F{i}" for i in range(1, 6)], yticklabels=[f"F{i}" for i in range(1, 6)])
    plt.title("Cross-fold SHAP ranking (Spearman)", fontsize=10)
    plt.tight_layout(); plt.savefig(FIGS / "shap_stability_heatmap.png", dpi=200); plt.close()


if __name__ == "__main__":
    main()
