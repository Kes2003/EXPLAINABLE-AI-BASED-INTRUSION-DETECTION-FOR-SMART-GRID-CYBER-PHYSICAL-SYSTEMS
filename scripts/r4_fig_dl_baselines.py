# scripts/r4_fig_dl_baselines.py
"""Fig. 9b: Macro-F1 of LightGBM vs. the deep-learning baselines (results/r4/dl_baselines.json).

Usage:  python -m scripts.r4_fig_dl_baselines
Output: results/r4/figures/dl_baselines.png
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.r4_common import FIGS, ROOT


def main():
    s = json.loads((ROOT / "results" / "r4" / "dl_baselines.json").read_text())["summary"]
    names = [("lightgbm_same_90pct_split", "LightGBM"), ("cnn_lstm", "CNN-LSTM"), ("transformer", "Transformer")]
    m = [s[k]["macro_f1_mean"] for k, _ in names]
    e = [s[k]["macro_f1_std"] for k, _ in names]
    FIGS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([l for _, l in names], m, yerr=e, capsize=6, color=["#2a6f97", "#9a9a9a", "#bdbdbd"])
    for i, v in enumerate(m):
        ax.text(i, v + max(e) + 0.005, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylabel("Macro-F1 (5-fold CV, mean ± std)")
    ax.set_ylim(min(m) - 0.1, 1.0)
    ax.grid(axis="y", alpha=.3)
    plt.tight_layout()
    plt.savefig(FIGS / "dl_baselines.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
