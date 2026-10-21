# scripts/r4_drift_single_sensor.py
"""
Reviewer comment 2 (sensitivity limit of the drift monitor).

The Table 7 shift operator moves every feature by a fraction of its MEAN at
once; for features with a small coefficient of variation (e.g. frequency,
50 Hz +/- 0.05 Hz) even a 1% shift is many standard deviations, so it is an
easy case for a two-sample test. This script measures the harder case of a
single drifting sensor: one feature at a time is offset by k standard
deviations (k in {0.05, 0.1, 0.25, 0.5, 1.0}), all others untouched, and the
same KS (Bonferroni over 18 features) and PSI (> 0.2) window monitors of
scripts/r4_drift_detection.py are applied (reference 2,000 training rows,
windows of 250 rows, 300 windows per setting). The deployed model's
Macro-F1 on the shifted stream is reported next to the detection rate, so
undetected drifts can be checked for harm.

Usage:  python -m scripts.r4_drift_single_sensor
Output: results/r4/drift_single_sensor.json
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from scripts.r4_common import ROOT, config, load_synthetic, save
from scripts.r4_drift_detection import ALPHA, N_REF, PSI_LIMIT, W, psi
from scipy.stats import ks_2samp
from sklearn.metrics import f1_score
from src.utils import set_global_seed

KS_SIGMA = [0.05, 0.1, 0.25, 0.5, 1.0]
N_WIN = 300


def main():
    cfg = config()
    seed = int(cfg.get("seed", 42))
    set_global_seed(seed)
    rng = np.random.default_rng(seed)
    _, df, features = load_synthetic()
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], shuffle=True, random_state=seed)
    model = joblib.load(ROOT / "models" / "model_lightgbm.joblib")
    scaler = joblib.load(ROOT / "models" / "scaler_lightgbm.joblib")
    le = joblib.load(ROOT / "models" / "label_encoder_lightgbm.joblib")
    ref = train_df.sample(N_REF, random_state=seed)[features].values
    sd = train_df[features].std().values
    X0 = test_df[features].values
    y = le.transform(test_df["label"].values)
    base_f1 = f1_score(y, model.predict(scaler.transform(X0)), average="macro")

    rows = []
    for j, feat in enumerate(features):
        for k in KS_SIGMA:
            X = X0.copy()
            X[:, j] = X[:, j] + k * sd[j]
            f1 = f1_score(y, model.predict(scaler.transform(X)), average="macro")
            ks_hit = psi_hit = 0
            for _ in range(N_WIN):
                idx = rng.choice(len(X), W, replace=False)
                cur = X[idx]
                p = min(ks_2samp(ref[:, i], cur[:, i]).pvalue for i in range(len(features)))
                ks_hit += p < ALPHA / len(features)
                psi_hit += max(psi(ref[:, i], cur[:, i]) for i in range(len(features))) > PSI_LIMIT
            rows.append({"feature": feat, "k_sigma": k, "ks_detection_rate": ks_hit / N_WIN,
                         "psi_detection_rate": psi_hit / N_WIN, "macro_f1": float(f1),
                         "macro_f1_drop": float(base_f1 - f1)})
        print(feat, [(r["k_sigma"], round(r["ks_detection_rate"], 2), round(r["macro_f1_drop"], 4))
                     for r in rows[-len(KS_SIGMA):]], flush=True)

    tab = pd.DataFrame(rows)
    summary = {}
    for k in KS_SIGMA:
        t = tab[tab.k_sigma == k]
        summary[str(k)] = {"ks_detection_rate_mean": float(t.ks_detection_rate.mean()),
                           "ks_detection_rate_min": float(t.ks_detection_rate.min()),
                           "psi_detection_rate_mean": float(t.psi_detection_rate.mean()),
                           "macro_f1_drop_mean": float(t.macro_f1_drop.mean()),
                           "macro_f1_drop_max": float(t.macro_f1_drop.max()),
                           "max_drop_among_ks_undetected_lt50pct": float(t[t.ks_detection_rate < 0.5].macro_f1_drop.max())
                           if (t.ks_detection_rate < 0.5).any() else None}
        print(k, summary[str(k)])
    save("drift_single_sensor", {"base_macro_f1": float(base_f1), "window": W, "n_windows": N_WIN,
                                 "rows": rows, "summary": summary})


if __name__ == "__main__":
    main()
