# scripts/r4_cross_domain_transfer.py
"""
Reviewer comment 1: is a model trained on the synthetic dataset usable on the
MSU/ORNL data WITHOUT retraining?

The two datasets share no feature schema (18 engineered quantities vs. 128
PMU / log channels) and no label taxonomy (normal / dos / fdi / overload /
tamper vs. attack / natural / no-event). A direct test therefore needs (a) a
feature mapping and (b) a common label space. We use the only defensible
common label space, binary attack vs. non-attack, and map the synthetic
features that have a physical counterpart in the PMU data:

  voltage_v            <- R1-PM1:V   (relay R1, phase-A voltage magnitude)
  current_a            <- R1-PM4:I   (relay R1, phase-A current magnitude)
  frequency_hz         <- R1:F
  power_factor         <- |cos(R1-PA1:VH - R1-PA4:IH)|      (angles in degrees)
  active_power_kw      <- V * I * power_factor
  reactive_power_kvar  <- V * I * |sin(angle difference)|
  demand_kw            <- active power (no separate demand channel)
  phase_imbalance_pct  <- 100 * max_k |V_k - mean(V)| / mean(V), V_k = R1-PM1..3:V
  feeder_voltage_v     <- R2-PM1:V   (neighbouring relay R2)
  feeder_current_a     <- R2-PM4:I

Magnitudes (V, I, f, P, Q, demand, feeder V/I) are converted to the synthetic
operating scale per unit: x_syn = x_msu / median_MSU(x) * median_syn_normal(x),
where median_MSU is computed over all MSU rows without labels (label-free).
Dimensionless quantities (power factor, phase imbalance) are kept as they are
("mapping P"); a sensitivity variant also rescales them per unit ("mapping PU").
The eight synthetic features with no PMU counterpart (thd_percent,
breaker_closed, packet_rate, packet_error_rate, substation_temp_c,
transformer_oil_temp_c, line_load_percent, tap_position) are set to their
synthetic normal-class median, i.e. treated as missing and imputed.

Models, none of which sees any MSU/ORNL row during training:
  T1  the deployed synthetic LightGBM (models/model_lightgbm.joblib, all 18
      features, 5 classes); attack = any predicted class other than 'normal'
  T2  LightGBM trained on the synthetic data restricted to the 10 mapped
      features, with binary normal-vs-attack labels
Reference (with retraining on MSU/ORNL):
  R   LightGBM trained and tested on MSU/ORNL with the same 10 mapped
      features, (a) stratified random 5-fold CV as in Table 8 and (b)
      event-disjoint 5-fold CV with whole recordings held out (GroupKFold
      by recording file, as in Table 8e), i.e. the in-domain ceiling for
      this feature set.

Usage:  python -m scripts.r4_cross_domain_transfer
Output: results/r4/cross_domain_transfer.json
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from scripts.r4_common import LIGHTGBM_CV, ROOT, build_model, load_msu, load_synthetic, save
from src.train import oversample_minority_classes
from src.utils import set_global_seed

MAPPED = ["voltage_v", "current_a", "frequency_hz", "power_factor", "active_power_kw", "reactive_power_kvar",
          "demand_kw", "phase_imbalance_percent", "feeder_voltage_v", "feeder_current_a"]
DIMENSIONLESS = {"power_factor", "phase_imbalance_percent"}


def msu_physical(msu):
    v = msu["R1-PM1:V"].values
    i = msu["R1-PM4:I"].values
    d = np.deg2rad(msu["R1-PA1:VH"].values - msu["R1-PA4:IH"].values)
    pf = np.abs(np.cos(d))
    vp = msu[["R1-PM1:V", "R1-PM2:V", "R1-PM3:V"]].values
    vm = vp.mean(axis=1)
    imb = 100 * np.abs(vp - vm[:, None]).max(axis=1) / np.where(vm == 0, np.nan, vm)
    out = pd.DataFrame({
        "voltage_v": v, "current_a": i, "frequency_hz": msu["R1:F"].values, "power_factor": pf,
        "active_power_kw": v * i * pf, "reactive_power_kvar": v * i * np.abs(np.sin(d)),
        "demand_kw": v * i * pf, "phase_imbalance_percent": imb,
        "feeder_voltage_v": msu["R2-PM1:V"].values, "feeder_current_a": msu["R2-PM4:I"].values,
    })
    return out.replace([np.inf, -np.inf], np.nan).fillna(out.median())


def to_synthetic_scale(phys, syn_normal_median, per_unit_all):
    out = phys.copy()
    for c in MAPPED:
        if c in DIMENSIONLESS and not per_unit_all:
            continue
        m = np.median(phys[c].values)
        out[c] = phys[c].values / (m if m != 0 else 1.0) * syn_normal_median[c]
    return out


def scores(y_true, y_pred, p_attack=None):
    r = {"macro_f1": float(f1_score(y_true, y_pred, average="macro")),
         "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
         "attack_recall": float(y_pred[y_true == 1].mean()),
         "false_alarm_rate_non_attack": float(y_pred[y_true == 0].mean()),
         "fraction_predicted_attack": float(y_pred.mean())}
    if p_attack is not None:
        r["roc_auc"] = float(roc_auc_score(y_true, p_attack))
    return r


def main():
    set_global_seed(42)
    cfg, syn, features = load_synthetic()
    msu, _, label_col = load_msu("binary")
    y_msu = (msu[label_col].values == "Attack").astype(int)
    syn_normal_median = syn.loc[syn["label"] == "normal", features].median()
    phys = msu_physical(msu)

    # chance references
    chance = {"all_attack": scores(y_msu, np.ones_like(y_msu)),
              "attack_prevalence": float(y_msu.mean())}
    print("MSU/ORNL binary: n =", len(y_msu), "attack prevalence =", round(y_msu.mean(), 4))
    print("All-attack predictor:", chance["all_attack"])

    # T1: deployed 18-feature synthetic model
    model = joblib.load(ROOT / "models" / "model_lightgbm.joblib")
    scaler = joblib.load(ROOT / "models" / "scaler_lightgbm.joblib")
    le = joblib.load(ROOT / "models" / "label_encoder_lightgbm.joblib")
    normal_idx = list(le.classes_).index("normal")
    res = {"chance": chance, "n_msu": int(len(y_msu))}
    for name, pu in [("P", False), ("PU", True)]:
        mapped = to_synthetic_scale(phys, syn_normal_median, pu)
        X = pd.DataFrame({f: (mapped[f] if f in MAPPED else syn_normal_median[f]) for f in features})[features]
        proba = model.predict_proba(scaler.transform(X.values))
        pred_cls = le.classes_[proba.argmax(axis=1)]
        p_attack = 1 - proba[:, normal_idx]
        r = scores(y_msu, (pred_cls != "normal").astype(int), p_attack)
        r["predicted_class_distribution"] = pd.Series(pred_cls).value_counts(normalize=True).round(4).to_dict()
        res[f"T1_deployed_18feat_mapping_{name}"] = r
        print(f"T1 (deployed model, mapping {name}):", r)

        # T2: synthetic model on the mapped features only, binary labels
        syn_b = syn.copy()
        syn_b["label"] = np.where(syn_b["label"] == "normal", "normal", "attack")
        bal = oversample_minority_classes(syn_b, label_col="label", random_state=42)
        sc2 = StandardScaler().fit(bal[MAPPED].values)
        m2 = build_model(LIGHTGBM_CV)
        m2.fit(sc2.transform(bal[MAPPED].values), (bal["label"].values == "attack").astype(int))
        p2 = m2.predict_proba(sc2.transform(mapped[MAPPED].values))[:, 1]
        r2 = scores(y_msu, (p2 >= 0.5).astype(int), p2)
        res[f"T2_synthetic_10feat_binary_mapping_{name}"] = r2
        print(f"T2 (10 mapped features, binary, mapping {name}):", r2)

    # R: in-domain reference with the same 10 features (retrained on MSU/ORNL), under the random
    # stratified folds of Table 8 and under event-disjoint folds (whole recordings held out, Table 8e)
    Xr = phys[MAPPED].values
    splits = {"R_in_domain_10feat_retrained_cv": StratifiedKFold(5, shuffle=True, random_state=42).split(Xr, y_msu),
              "R_in_domain_10feat_event_disjoint": GroupKFold(5).split(Xr, y_msu, msu["_file"].values)}
    for key, folds in splits.items():
        f1s, bas, aucs = [], [], []
        for tr, te in folds:
            sc = StandardScaler().fit(Xr[tr])
            m = build_model(LIGHTGBM_CV)
            m.fit(sc.transform(Xr[tr]), y_msu[tr])
            p = m.predict_proba(sc.transform(Xr[te]))[:, 1]
            f1s.append(f1_score(y_msu[te], (p >= .5).astype(int), average="macro"))
            bas.append(balanced_accuracy_score(y_msu[te], (p >= .5).astype(int)))
            aucs.append(roc_auc_score(y_msu[te], p))
        res[key] = {"macro_f1_mean": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s)),
                    "balanced_accuracy_mean": float(np.mean(bas)), "roc_auc_mean": float(np.mean(aucs))}
        print(f"{key}:", res[key])
    save("cross_domain_transfer", res)


if __name__ == "__main__":
    main()
