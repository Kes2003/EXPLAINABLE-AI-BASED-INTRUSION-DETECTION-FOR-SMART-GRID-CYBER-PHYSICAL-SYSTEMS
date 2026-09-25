# scripts/dos_attack.py
"""
DoS attack-simulation figure (Fig. 4 / Fig. 8 in the manuscript) -- REBUILT.

The previous version set the "IDS Attack Probability" curve with
np.random.uniform(...), never touching the trained model. This version
interpolates between the REAL empirical median 'normal' operating point and
the REAL empirical median 'dos' operating point (both computed directly from
the training dataset), runs every timestep through the actual trained model,
and plots the model's genuine predicted probability of the 'dos' class.
Nothing in the probability curve is randomly generated.

Usage:
  python -m scripts.dos_attack
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts._sim_utils import (
    load_pipeline, run_sequence, attack_probability_series,
    empirical_class_medians, build_ramp_rows,
)

T = 100
ATTACK_START, ATTACK_END = 40, 70


def main():
    model, preproc, features = load_pipeline(model_type="lightgbm")
    medians = empirical_class_medians(features)
    rows = build_ramp_rows(features, medians["normal"], medians["dos"], T, ATTACK_START, ATTACK_END)
    _, _, proba, class_names = run_sequence(model, preproc, features, rows)
    ids_prob = attack_probability_series(proba, class_names, "dos")
    packet_rate_series = np.array([r["packet_rate"] for r in rows])

    t = np.arange(T)
    plt.figure(figsize=(10, 5))

    plt.subplot(2, 1, 1)
    plt.plot(t, packet_rate_series, color="red")
    plt.axvspan(ATTACK_START, ATTACK_END, color="red", alpha=0.15)
    plt.ylabel("Packet Rate")
    plt.title("DoS Attack Simulation in Smart Grid (model-driven, LightGBM)")

    plt.subplot(2, 1, 2)
    plt.plot(t, ids_prob, color="blue")
    plt.axvspan(ATTACK_START, ATTACK_END, color="red", alpha=0.15)
    plt.ylabel("IDS P(dos) -- real model output")
    plt.ylim(-0.02, 1.02)
    plt.xlabel("Time step")

    plt.tight_layout()
    out = "artifacts/visualizations/dos_attack.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    print(f"Mean P(dos) during attack window: {ids_prob[ATTACK_START:ATTACK_END].mean():.4f}")
    print(f"Mean P(dos) outside attack window: {np.concatenate([ids_prob[:ATTACK_START], ids_prob[ATTACK_END:]]).mean():.4f}")


if __name__ == "__main__":
    main()
