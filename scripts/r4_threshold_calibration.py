# scripts/r4_threshold_calibration.py
"""
Reviewer comment 4: experimental justification for the confidence threshold
tau = 0.7 of the policy engine.

Uses the same seed-42 five-fold out-of-fold LightGBM predictions as Section
4.9 (all 30,000 samples, each predicted once by a model that never saw it)
and the deployed policy function src.api.compute_preventive_action.

  1. Threshold sweep, tau = 0.20 ... 0.95: response accuracy, false-block
     rate, missed-response rate, wrong-action rate, and detection performance
     of the gated detector (a prediction below tau raises no alert, i.e. it
     is treated as 'normal').
  2. Calibration of the top-class probability: expected calibration error
     (15 equal-width bins), multiclass Brier score, reliability curve.
  3. Nested threshold selection: inside each outer training fold, an inner
     stratified 5-fold CV produces out-of-fold probabilities for the training
     rows only; tau is chosen on those (maximum response accuracy, ties ->
     larger tau) and then applied to the untouched outer test fold. This
     tests whether 0.7 is what a data-driven procedure would pick, without
     using the test data to pick it.

Usage:  python -m scripts.r4_threshold_calibration
Output: results/r4/threshold_calibration.json, results/r4/figures/threshold_*.png
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.r4_common import (FIGS, LIGHTGBM_CV, build_model, fresh_label_encoder, load_synthetic,
                               prepare_fold, save, synthetic_folds)
from src.api import compute_preventive_action
from src.utils import set_global_seed

GROUND_TRUTH_ACTION = {"normal": (False, False), "dos": (False, True), "fdi": (False, False),
                       "tamper": (True, False), "overload": (True, False)}
TAUS = [round(t, 2) for t in np.arange(0.20, 0.951, 0.05)]


def policy_metrics(true, pred, prob, tau):
    acts = [compute_preventive_action(p, pr, tau) for p, pr in zip(pred, prob)]
    acts = [(a.should_trip_breaker, a.should_rate_limit_network) for a in acts]
    gt = [GROUND_TRUTH_ACTION[t] for t in true]
    n = len(true)
    exact = sum(a == g for a, g in zip(acts, gt))
    normal = [a for a, t in zip(acts, true) if t == "normal"]
    need = [(a, g) for a, g in zip(acts, gt) if any(g)]
    missed = sum(not any(a) for a, _ in need)
    wrong = sum(any(a) and a != g for a, g in need)  # acted, but with the wrong action
    # gated detector: below tau -> no alert -> 'normal'
    gated = [p if pr >= tau else "normal" for p, pr in zip(pred, prob)]
    attack_true = np.array([t != "normal" for t in true])
    alert = np.array([g != "normal" for g in gated])
    return {
        "tau": tau,
        "response_accuracy": exact / n,
        "false_block_rate": sum(any(a) for a in normal) / len(normal),
        "false_blocks": int(sum(any(a) for a in normal)),
        "missed_response_rate": missed / len(need),
        "missed_responses": int(missed),
        "wrong_action_rate": wrong / len(need),
        "gated_macro_f1": float(f1_score(true, gated, average="macro")),
        "alert_recall": float((alert & attack_true).sum() / attack_true.sum()),
        "alert_precision": float((alert & attack_true).sum() / max(alert.sum(), 1)),
        "fraction_below_tau": float(np.mean(np.asarray(prob) < tau)),
        # FDI is escalated to the operator only when predicted as fdi with prob >= tau
        "fdi_escalation_rate": float(np.mean([p == "fdi" and pr >= tau
                                              for t, p, pr in zip(true, pred, prob) if t == "fdi"])),
    }


def calibration(true_idx, proba, n_bins=15):
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == true_idx).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    bins, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        acc, c = correct[m].mean(), conf[m].mean()
        ece += m.mean() * abs(acc - c)
        bins.append({"lo": lo, "hi": hi, "n": int(m.sum()), "accuracy": float(acc), "confidence": float(c)})
    onehot = np.eye(proba.shape[1])[true_idx]
    brier = float(((proba - onehot) ** 2).sum(axis=1).mean())
    # accuracy of predictions at or above 0.7 vs. below
    hi = conf >= 0.7
    return {"ece": float(ece), "brier": brier, "bins": bins,
            "accuracy_conf_ge_0.7": float(correct[hi].mean()), "n_conf_ge_0.7": int(hi.sum()),
            "accuracy_conf_lt_0.7": float(correct[~hi].mean()), "n_conf_lt_0.7": int((~hi).sum())}


def oof_lightgbm(df, features, le, folds, seed=42):
    true, pred, prob = [None] * len(df), [None] * len(df), [None] * len(df)
    proba_all = np.zeros((len(df), len(le.classes_)))
    for tr, te in folds:
        f = prepare_fold(df, features, le, tr, te, seed)
        model = build_model(LIGHTGBM_CV)
        model.fit(f["X_train"], f["y_train"])
        pr = model.predict_proba(f["X_test"])
        proba_all[te] = pr
        for j, i in enumerate(te):
            true[i] = df["label"].iloc[i]
            pred[i] = le.classes_[pr[j].argmax()]
            prob[i] = float(pr[j].max())
    return true, pred, prob, proba_all


def main():
    set_global_seed(42)
    _, df, features = load_synthetic()
    le = fresh_label_encoder(df["label"].values)
    folds = synthetic_folds(df, 42)

    true, pred, prob, proba = oof_lightgbm(df, features, le, folds)
    print(f"OOF Macro-F1 (ungated) = {f1_score(true, pred, average='macro'):.4f}")

    sweep = [policy_metrics(true, pred, prob, t) for t in TAUS]
    print("\n tau   RespAcc  FalseBlock  Missed  WrongAct  GatedMacroF1  AlertRec  AlertPrec  <tau")
    for s in sweep:
        print(f" {s['tau']:.2f}  {s['response_accuracy']:.4f}   {s['false_block_rate']:.4f}    "
              f"{s['missed_response_rate']:.4f}  {s['wrong_action_rate']:.4f}    {s['gated_macro_f1']:.4f}      "
              f"{s['alert_recall']:.4f}    {s['alert_precision']:.4f}   {s['fraction_below_tau']:.4f}")

    cal = calibration(le.transform(true), proba)
    print(f"\nCalibration: ECE = {cal['ece']:.4f}, Brier = {cal['brier']:.4f}; "
          f"accuracy when conf>=0.7: {cal['accuracy_conf_ge_0.7']:.4f} (n={cal['n_conf_ge_0.7']}), "
          f"conf<0.7: {cal['accuracy_conf_lt_0.7']:.4f} (n={cal['n_conf_lt_0.7']})")

    # nested selection of tau
    nested = []
    for k, (tr, te) in enumerate(folds, 1):
        inner_df = df.iloc[tr].reset_index(drop=True)
        inner_folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(inner_df.values, inner_df["label"].values))
        it, ip, ipr, _ = oof_lightgbm(inner_df, features, le, inner_folds)
        inner = [policy_metrics(it, ip, ipr, t) for t in TAUS]
        best = max(inner, key=lambda s: (round(s["response_accuracy"], 6), s["tau"]))
        te_true = [true[i] for i in te]
        te_pred = [pred[i] for i in te]
        te_prob = [prob[i] for i in te]
        outer_sel = policy_metrics(te_true, te_pred, te_prob, best["tau"])
        outer_07 = policy_metrics(te_true, te_pred, te_prob, 0.7)
        nested.append({"fold": k, "selected_tau": best["tau"], "inner_best_response_accuracy": best["response_accuracy"],
                       "outer_at_selected": outer_sel, "outer_at_0.7": outer_07})
        print(f"fold {k}: inner-selected tau = {best['tau']:.2f}; outer resp.acc {outer_sel['response_accuracy']:.4f} "
              f"(tau=0.7: {outer_07['response_accuracy']:.4f})", flush=True)

    save("threshold_calibration", {"oof_macro_f1": float(f1_score(true, pred, average="macro")),
                                   "sweep": sweep, "calibration": cal, "nested_selection": nested})

    FIGS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    t = [s["tau"] for s in sweep]
    ax[0].plot(t, [s["response_accuracy"] for s in sweep], "o-", label="Response accuracy")
    ax[0].plot(t, [s["gated_macro_f1"] for s in sweep], "s--", label="Gated detection Macro-F1")
    ax[0].plot(t, [1 - s["missed_response_rate"] for s in sweep], "^-.", label="1 - missed-response rate")
    ax[0].axvline(0.7, color="grey", lw=0.8, ls=":")
    ax[0].set_xlabel("Confidence threshold τ"); ax[0].set_ylabel("Score"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax0b = ax[0].twinx()
    ax0b.plot(t, [100 * s["false_block_rate"] for s in sweep], "x:", color="tab:red", label="False-block rate (%)")
    ax0b.set_ylabel("False-block rate (%)", color="tab:red")
    b = cal["bins"]
    ax[1].plot([0, 1], [0, 1], "k:", lw=0.8)
    ax[1].plot([x["confidence"] for x in b], [x["accuracy"] for x in b], "o-")
    ax[1].set_xlabel("Mean predicted top-class probability"); ax[1].set_ylabel("Observed accuracy")
    ax[1].set_title(f"Reliability (ECE = {cal['ece']:.3f})", fontsize=10); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(FIGS / "threshold_sensitivity_calibration.png", dpi=200); plt.close()


if __name__ == "__main__":
    main()
