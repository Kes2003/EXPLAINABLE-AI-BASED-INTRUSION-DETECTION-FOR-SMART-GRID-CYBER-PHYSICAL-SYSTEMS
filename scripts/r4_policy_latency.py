# scripts/r4_policy_latency.py
"""Policy-layer latency (manuscript Section 4.9), re-measured on an otherwise idle machine.

run_repeated_cv_and_policy_eval.py measures the policy-call latency while it is
running the repeated cross-validation, i.e. while other jobs may share the CPU.
This script recomputes only the seed-42 out-of-fold LightGBM predictions (same
code path, run_seed(..., collect_oof=True)), checks that the action-selection
results equal those of the repeated-CV script, and times the deployed policy
function exactly as that script does (1,000 warm-up calls, then 5 passes over
all 30,000 predictions). Run it together with the latency benchmark, alone.

Usage:  python -m scripts.r4_policy_latency
Output: results/r4/policy_latency.json
"""
from __future__ import annotations

import json

from sklearn.preprocessing import LabelEncoder

from run_repeated_cv_and_policy_eval import load, prevention_benchmark, run_seed
from scripts.r4_common import ROOT, save


def main():
    cfg, df = load()
    le = LabelEncoder().fit(df["label"].values)
    variants = {"lightgbm": cfg["model_variants"]["lightgbm"]}
    _, _, oof = run_seed(df, cfg["features"], variants, le, 42, collect_oof=True)
    res = prevention_benchmark(oof)
    ref = json.loads((ROOT / "results" / "repeated_cv_and_policy_eval.json").read_text())["prevention_benchmark"]
    for k in ("response_accuracy", "false_blocks", "missed_responses"):
        assert res[k] == ref[k], (k, res[k], ref[k])
    print(res["latency_us"])
    save("policy_latency", res)


if __name__ == "__main__":
    main()
