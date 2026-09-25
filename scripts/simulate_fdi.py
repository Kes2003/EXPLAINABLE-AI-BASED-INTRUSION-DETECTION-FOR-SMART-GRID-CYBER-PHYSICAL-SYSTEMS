# scripts/simulate_fdi.py
"""
False Data Injection attack-simulation figure (Fig. 5 / Fig. 9) -- REBUILT.
Same fix and rationale as scripts/dos_attack.py: interpolate between real
empirical 'normal' and 'fdi' median operating points, plot real model output.

Usage:
  python -m scripts.simulate_fdi
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
    rows = build_ramp_rows(features, medians["normal"], medians["fdi"], T, ATTACK_START, ATTACK_END)
    _, _, proba, class_names = run_sequence(model, preproc, features, rows)
    ids_prob = attack_probability_series(proba, class_names, "fdi")
    voltage_series = np.array([r["voltage_v"] for r in rows])
    reactive_series = np.array([r["reactive_power_kvar"] for r in rows])

    t = np.arange(T)
    plt.figure(figsize=(10, 6))

    plt.subplot(3, 1, 1)
    plt.plot(t, voltage_series)
    plt.axvspan(ATTACK_START, ATTACK_END, color="red", alpha=0.15)
    plt.ylabel("Voltage (V)")
    plt.title("FDI Attack Simulation (model-driven, LightGBM)")

    plt.subplot(3, 1, 2)
    plt.plot(t, reactive_series)
    plt.axvspan(ATTACK_START, ATTACK_END, color="red", alpha=0.15)
    plt.ylabel("Reactive Power (kVAR)")

    plt.subplot(3, 1, 3)
    plt.plot(t, ids_prob, color="blue")
    plt.axvspan(ATTACK_START, ATTACK_END, color="red", alpha=0.15)
    plt.ylabel("IDS P(fdi) -- real model output")
    plt.ylim(-0.02, 1.02)
    plt.xlabel("Time step")

    plt.tight_layout()
    out = "artifacts/visualizations/fdi.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")
    print(f"Mean P(fdi) during attack window: {ids_prob[ATTACK_START:ATTACK_END].mean():.4f}")
    print(f"Mean P(fdi) outside attack window: {np.concatenate([ids_prob[:ATTACK_START], ids_prob[ATTACK_END:]]).mean():.4f}")


if __name__ == "__main__":
    main()
