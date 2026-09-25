# scripts/corrected_resampled_ttest.py
"""
Nadeau & Bengio (2003) corrected resampled t-test for the repeated-CV
comparison of Table 3c.

Per-fold scores from repeated k-fold CV are not independent (training sets
overlap), so the naive variance of the paired differences is too small and
tests at n = 50 are anti-conservative. The corrected statistic inflates the
variance by the test/train size ratio:

    t = mean(d) / sqrt( (1/J + n_test/n_train) * var(d) ),   df = J - 1

with J = 50 paired fold differences and n_test/n_train = 1/4 for 5-fold CV.

Reads results/repeated_cv_and_policy_eval.json; writes results/corrected_ttest.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent


def main():
    res = json.loads((ROOT / "results" / "repeated_cv_and_policy_eval.json").read_text())
    scores = {k: np.array(v) for k, v in res["per_fold_macro_f1"].items()}
    best = res["best_model"]
    ratio = 1 / 4  # 5-fold CV: test fold is 1/4 the size of the training folds
    out = {"best_model": best, "test_train_ratio": ratio, "comparisons": {}}
    for name, s in scores.items():
        if name == best:
            continue
        d = scores[best] - s
        J = len(d)
        t = d.mean() / np.sqrt((1 / J + ratio) * d.var(ddof=1))
        p = 2 * stats.t.sf(abs(t), df=J - 1)
        out["comparisons"][name] = {"J": J, "mean_diff": float(d.mean()), "t": float(t), "p": float(p)}
        print(f"{best} vs {name}: mean diff {d.mean():+.4f}, corrected t({J - 1}) = {t:.3f}, p = {p:.3g}")
    (ROOT / "results" / "corrected_ttest.json").write_text(json.dumps(out, indent=2))
    print("Saved results/corrected_ttest.json")


if __name__ == "__main__":
    main()
