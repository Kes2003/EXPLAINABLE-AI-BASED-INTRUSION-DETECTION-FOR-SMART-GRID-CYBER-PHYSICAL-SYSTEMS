# scripts/r4_drift_detection.py
"""
Reviewer comment 2: does KS/PSI drift monitoring actually detect the shifts
of Table 7, and does shadow retraining actually recover performance?

Setting (same model, split and shift operator as Table 7 / Section 4.5):
  - deployed LightGBM (models/model_lightgbm.joblib), trained by src.train on
    the 80% stratified split (seed 42); the 20% test split (6,000 rows) is
    the incoming telemetry
  - shift operator of scripts/noise_robustness_test.apply_shift: every
    feature x -> x + s * mean(x), s in {0, 0.01, 0.02, 0.05, 0.10, 0.20}

Detection (label-free):
  reference window = 2,000 rows drawn from the TRAINING split (before
  oversampling); monitoring windows of W = 250 rows drawn from the
  'recent' + 'evaluation' slices of the test split (see below); the
  confidence reference is the held-back, unshifted 'validation' slice. A window raises an alarm if
    KS : any of the 18 two-sample KS tests has p < 0.01 / 18 (Bonferroni)
    PSI: any feature has PSI > 0.2 (10 reference-quantile bins)
    CONF: mean top-class probability falls more than 3 standard errors below
          its value on the reference window
  Detection rate = fraction of 500 random windows that alarm at a given s;
  at s = 0 this is the false-alarm rate. Detection delay after an abrupt
  change = number of post-change windows until the first alarm (geometric
  with the per-window detection rate; the empirical mean is reported).

Mitigation, after drift is detected (test split divided into three disjoint
slices: 'recent' 2,000 rows, 'validation' 1,000 rows, 'evaluation' 3,000 rows;
all three shifted by the same s):
  M0  no action (deployed model)
  M1  label-free re-centering: subtract the per-feature mean difference
      between the recent window and the reference window, then apply the
      deployed model
  M2  shadow retraining: a candidate LightGBM (deployed configuration) is
      trained on the original training split plus n_lab labelled rows of the
      recent shifted slice (n_lab in {250, 500, 1000, 2000}), with fold-local
      oversampling and a scaler refit, and is PROMOTED only if its Macro-F1 on
      the validation slice exceeds the deployed model's; the reported
      Macro-F1 is that of whichever model is active, on the evaluation slice.

Usage:  python -m scripts.r4_drift_detection
Output: results/r4/drift_detection.json, results/r4/figures/drift_*.png
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.noise_robustness_test import apply_shift
from scripts.r4_common import FIGS, ROOT, build_model, config, load_synthetic, save
from src.train import oversample_minority_classes
from src.utils import set_global_seed

SHIFTS = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]
W = 250
N_WINDOWS = 500
N_REF = 2000
ALPHA = 0.01
PSI_LIMIT = 0.2
N_LAB = [250, 500, 1000, 2000]


def psi(ref, cur, bins=10):
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # (near-)constant or binary feature: use its distinct values
        vals = np.unique(ref)
        r = np.array([(ref == v).mean() for v in vals])
        c = np.array([(cur == v).mean() for v in vals])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
        r = np.histogram(ref, edges)[0] / len(ref)
        c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def window_alarms(ref, cur, ref_conf_mean, ref_conf_se, cur_conf):
    ks_p = np.array([ks_2samp(ref[:, j], cur[:, j]).pvalue for j in range(ref.shape[1])])
    ps = np.array([psi(ref[:, j], cur[:, j]) for j in range(ref.shape[1])])
    return {"ks": bool((ks_p < ALPHA / ref.shape[1]).any()),
            "psi": bool((ps > PSI_LIMIT).any()),
            "conf": bool(cur_conf.mean() < ref_conf_mean - 3 * ref_conf_se),
            "min_ks_p": float(ks_p.min()), "max_psi": float(ps.max())}


def macro_f1(model, scaler, le, X_df, y):
    return float(f1_score(le.transform(y), model.predict(scaler.transform(X_df.values)), average="macro"))


def main():
    cfg = config()
    seed = int(cfg.get("seed", 42))
    set_global_seed(seed)
    rng = np.random.default_rng(seed)
    _, df, features = load_synthetic()
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], shuffle=True, random_state=seed)
    train_df, test_df = train_df.reset_index(drop=True), test_df.reset_index(drop=True)

    model = joblib.load(ROOT / "models" / "model_lightgbm.joblib")
    scaler = joblib.load(ROOT / "models" / "scaler_lightgbm.joblib")
    le = joblib.load(ROOT / "models" / "label_encoder_lightgbm.joblib")
    print("Deployed model, unshifted test split Macro-F1:",
          round(macro_f1(model, scaler, le, test_df[features], test_df["label"].values), 4))

    # test split -> 'recent' 2,000 / 'validation' 1,000 / 'evaluation' 3,000 rows
    recent_df, rest = train_test_split(test_df, train_size=2000, stratify=test_df["label"], random_state=seed)
    val_df, eval_df = train_test_split(rest, train_size=1000, stratify=rest["label"], random_state=seed)
    recent_df, val_df, eval_df = (d.reset_index(drop=True) for d in (recent_df, val_df, eval_df))
    stream_df = pd.concat([recent_df, eval_df], ignore_index=True)  # monitored telemetry

    ref_df = train_df.sample(N_REF, random_state=seed)
    ref = ref_df[features].values
    # confidence reference: held-back, unshifted validation slice (training rows
    # would give an optimistic, too-high reference confidence)
    ref_conf = model.predict_proba(scaler.transform(val_df[features].values)).max(axis=1)
    ref_conf_mean = ref_conf.mean()
    ref_conf_se = ref_conf.std() / np.sqrt(W)

    detection = []
    for s in SHIFTS:
        shifted = apply_shift(stream_df[features], s).values
        conf_all = model.predict_proba(scaler.transform(shifted)).max(axis=1)
        al = {"ks": [], "psi": [], "conf": []}
        min_p, max_psi = [], []
        for _ in range(N_WINDOWS):
            idx = rng.choice(len(shifted), W, replace=False)
            a = window_alarms(ref, shifted[idx], ref_conf_mean, ref_conf_se, conf_all[idx])
            for k in al:
                al[k].append(a[k])
            min_p.append(a["min_ks_p"]); max_psi.append(a["max_psi"])
        row = {"shift": s, "window": W, "n_windows": N_WINDOWS,
               "macro_f1_deployed": macro_f1(model, scaler, le, pd.DataFrame(shifted, columns=features),
                                             stream_df["label"].values),
               "mean_top_class_probability": float(conf_all.mean())}
        for k, v in al.items():
            rate = float(np.mean(v))
            row[f"{k}_detection_rate"] = rate
            # empirical delay: windows until first alarm, over random post-change sequences
            v = np.array(v)
            delays = []
            for _ in range(200):
                order = rng.permutation(len(v))
                hit = np.where(v[order])[0]
                delays.append(int(hit[0]) + 1 if len(hit) else None)
            ok = [d for d in delays if d is not None]
            row[f"{k}_mean_delay_windows"] = float(np.mean(ok)) if ok else None
        row["median_min_ks_p"] = float(np.median(min_p))
        row["median_max_psi"] = float(np.median(max_psi))
        detection.append(row)
        print(f"shift {s:.2f}: F1 {row['macro_f1_deployed']:.4f} | KS {row['ks_detection_rate']:.3f} "
              f"PSI {row['psi_detection_rate']:.3f} CONF {row['conf_detection_rate']:.3f} | "
              f"mean conf {row['mean_top_class_probability']:.3f}", flush=True)

    # --- mitigation ----------------------------------------------------------
    mitigation = []
    for s in SHIFTS[1:]:
        rec = apply_shift(recent_df[features], s); rec["label"] = recent_df["label"].values
        val = apply_shift(val_df[features], s)
        ev = apply_shift(eval_df[features], s)
        m0_val = macro_f1(model, scaler, le, val, val_df["label"].values)
        m0 = macro_f1(model, scaler, le, ev, eval_df["label"].values)
        # M1: label-free re-centering using the recent window (no labels used)
        delta = rec[features].mean().values - ref_df[features].mean().values
        m1 = macro_f1(model, scaler, le, ev - delta, eval_df["label"].values)
        row = {"shift": s, "M0_no_action": m0, "M1_recentering": m1, "M2_shadow": []}
        for n in N_LAB:
            lab = rec.sample(n, random_state=seed) if n < len(rec) else rec
            cand_train = pd.concat([train_df[features + ["label"]], lab[features + ["label"]]], ignore_index=True)
            bal = oversample_minority_classes(cand_train, label_col="label", random_state=seed)
            sc = StandardScaler().fit(bal[features].values)
            cand = build_model(cfg["model"])
            cand.fit(sc.transform(bal[features].values), le.transform(bal["label"].values))
            c_val = macro_f1(cand, sc, le, val, val_df["label"].values)
            promoted = c_val > m0_val
            c_eval = macro_f1(cand, sc, le, ev, eval_df["label"].values)
            row["M2_shadow"].append({"n_labelled": n, "candidate_val_f1": c_val, "deployed_val_f1": m0_val,
                                     "promoted": bool(promoted), "candidate_eval_f1": c_eval,
                                     "active_model_eval_f1": c_eval if promoted else m0})
        mitigation.append(row)
        print(f"shift {s:.2f}: M0 {m0:.4f} | M1 re-centre {m1:.4f} | M2 " +
              ", ".join(f"n={r['n_labelled']}: {r['active_model_eval_f1']:.4f}{'*' if r['promoted'] else ''}"
                        for r in row["M2_shadow"]), flush=True)

    # promotion guard on unshifted data (should not replace a healthy model with a worse one)
    lab = recent_df.sample(500, random_state=seed)
    cand_train = pd.concat([train_df[features + ["label"]], lab[features + ["label"]]], ignore_index=True)
    bal = oversample_minority_classes(cand_train, label_col="label", random_state=seed)
    sc = StandardScaler().fit(bal[features].values)
    cand = build_model(cfg["model"]); cand.fit(sc.transform(bal[features].values), le.transform(bal["label"].values))
    guard = {"candidate_val_f1": macro_f1(cand, sc, le, val_df[features], val_df["label"].values),
             "deployed_val_f1": macro_f1(model, scaler, le, val_df[features], val_df["label"].values),
             "candidate_eval_f1": macro_f1(cand, sc, le, eval_df[features], eval_df["label"].values),
             "deployed_eval_f1": macro_f1(model, scaler, le, eval_df[features], eval_df["label"].values)}
    print("No-drift guard:", guard)

    save("drift_detection", {"settings": {"W": W, "n_windows": N_WINDOWS, "n_ref": N_REF, "alpha": ALPHA,
                                          "bonferroni_features": len(features), "psi_limit": PSI_LIMIT},
                             "detection": detection, "mitigation": mitigation, "no_drift_guard": guard})

    FIGS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    xs = [100 * r["shift"] for r in detection]
    for k, mk in [("ks", "o-"), ("psi", "s--"), ("conf", "^-.")]:
        ax[0].plot(xs, [r[f"{k}_detection_rate"] for r in detection], mk, label=k.upper())
    ax[0].set_xlabel("Shift (% of feature mean)"); ax[0].set_ylabel(f"Alarm rate per {W}-row window")
    ax[0].legend(); ax[0].grid(alpha=.3)
    xm = [100 * r["shift"] for r in mitigation]
    ax[1].plot(xm, [r["M0_no_action"] for r in mitigation], "o-", label="No action")
    ax[1].plot(xm, [r["M1_recentering"] for r in mitigation], "s--", label="Label-free re-centring")
    for j, n in enumerate(N_LAB):
        ax[1].plot(xm, [r["M2_shadow"][j]["active_model_eval_f1"] for r in mitigation], ":", marker="^",
                   label=f"Shadow retraining, {n} labels")
    ax[1].set_xlabel("Shift (% of feature mean)"); ax[1].set_ylabel("Macro-F1 (evaluation slice)")
    ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(FIGS / "drift_detection_mitigation.png", dpi=200); plt.close()


if __name__ == "__main__":
    main()
