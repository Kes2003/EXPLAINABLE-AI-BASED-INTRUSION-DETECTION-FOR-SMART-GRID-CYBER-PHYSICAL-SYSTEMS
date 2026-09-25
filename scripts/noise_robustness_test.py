# scripts/noise_robustness_test.py
"""
Robustness evaluation for the trained IDPS model (Reviewer 2 / Reviewer 4 pt.7).

Evaluates degradation under three realistic corruption types, at multiple
severities, using the ACTUAL trained model and the ACTUAL preprocessing
pipeline (no synthetic/random stand-ins):

  1. Gaussian sensor noise on continuous features (simulates measurement noise)
  2. Missing values / dropped telemetry (simulates comms loss -> imputed via
     the same fillna(median) strategy the Preprocessor already uses)
  3. Feature-wise distribution shift (simulates sensor drift / miscalibration)

Outputs:
  - artifacts/robustness_results.json   (machine-readable, for the paper's tables)
  - artifacts/visualizations/robustness_degradation.png (Fig. for the paper)

Usage:
  python -m scripts.noise_robustness_test --model lightgbm
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, accuracy_score

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.preprocess import Preprocessor
from src.utils import set_global_seed, ensure_dir, save_json


NOISE_LEVELS = [0.0, 0.02, 0.05, 0.10, 0.20]      # fraction of feature std added as Gaussian noise
MISSING_LEVELS = [0.0, 0.05, 0.10, 0.20, 0.30]    # fraction of cells randomly dropped then imputed
SHIFT_LEVELS = [0.0, 0.05, 0.10, 0.20]            # fraction constant shift of each feature's mean


def load_test_split(df: pd.DataFrame, features: list[str], seed: int):
    from sklearn.model_selection import train_test_split
    X_df = df[features].copy()
    y = df["label"].astype(str).values
    _, X_test_df, _, y_test = train_test_split(
        X_df, y, test_size=0.2, stratify=y, random_state=seed
    )
    return X_test_df.reset_index(drop=True), y_test


def apply_gaussian_noise(X_df: pd.DataFrame, level: float, rng: np.random.Generator) -> pd.DataFrame:
    if level == 0.0:
        return X_df.copy()
    out = X_df.copy()
    stds = out.std(numeric_only=True)
    for col in out.columns:
        noise = rng.normal(0, level * (stds[col] if stds[col] > 0 else 1.0), size=len(out))
        out[col] = out[col] + noise
    return out


def apply_missingness(X_df: pd.DataFrame, level: float, rng: np.random.Generator) -> pd.DataFrame:
    if level == 0.0:
        return X_df.copy()
    out = X_df.copy()
    mask = rng.random(out.shape) < level
    arr = out.to_numpy(dtype=float)
    arr[mask] = np.nan
    out = pd.DataFrame(arr, columns=out.columns)
    # Same imputation strategy the real Preprocessor uses in production (median fill)
    out = out.fillna(out.median())
    return out


def apply_shift(X_df: pd.DataFrame, level: float) -> pd.DataFrame:
    if level == 0.0:
        return X_df.copy()
    out = X_df.copy()
    means = out.mean(numeric_only=True)
    for col in out.columns:
        out[col] = out[col] + level * means[col]
    return out


def evaluate(model, preproc: Preprocessor, X_df: pd.DataFrame, y_true_str: np.ndarray) -> dict:
    X_scaled = preproc.transform(X_df)
    y_pred = model.predict(X_scaled)
    y_true_enc = preproc.label_encoder.transform(y_true_str)
    return {
        "accuracy": float(accuracy_score(y_true_enc, y_pred)),
        "macro_f1": float(f1_score(y_true_enc, y_pred, average="macro")),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="lightgbm",
                         choices=["random_forest", "xgboost", "lightgbm", "catboost"])
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--data", type=str, default="data/external/enhanced_realistic_smartgrid.csv")
    parser.add_argument("--artifacts_dir", type=str, default="artifacts")
    parser.add_argument("--models_dir", type=str, default="models")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    seed = int(cfg.get("seed", 42))
    set_global_seed(seed)
    rng = np.random.default_rng(seed)

    features = cfg["features"]
    artifacts_dir = Path(args.artifacts_dir)
    models_dir = Path(args.models_dir)
    ensure_dir(artifacts_dir / "visualizations")

    df = pd.read_csv(args.data)

    model_path = models_dir / f"model_{args.model}.joblib"
    if not model_path.exists():
        model_path = models_dir / "model.joblib"
    model = joblib.load(model_path)

    preproc = Preprocessor(
        features=features,
        scaler_path=models_dir / f"scaler_{args.model}.joblib",
        label_encoder_path=models_dir / f"label_encoder_{args.model}.joblib",
    )
    if not preproc.load_from_paths():
        # fall back to the untyped artifacts saved by train.py
        preproc = Preprocessor(
            features=features,
            scaler_path=models_dir / "scaler.joblib",
            label_encoder_path=models_dir / "label_encoder.joblib",
        )
        preproc.load_from_paths()

    X_test_df, y_test = load_test_split(df, features, seed)

    results = {"model": args.model, "noise": [], "missing": [], "shift": []}

    print("Running Gaussian-noise robustness sweep...")
    for level in NOISE_LEVELS:
        X_corrupt = apply_gaussian_noise(X_test_df, level, rng)
        m = evaluate(model, preproc, X_corrupt, y_test)
        m["level"] = level
        results["noise"].append(m)
        print(f"  noise={level:>4.2f}  acc={m['accuracy']:.4f}  macro_f1={m['macro_f1']:.4f}")

    print("Running missing-data robustness sweep...")
    for level in MISSING_LEVELS:
        X_corrupt = apply_missingness(X_test_df, level, rng)
        m = evaluate(model, preproc, X_corrupt, y_test)
        m["level"] = level
        results["missing"].append(m)
        print(f"  missing={level:>4.2f}  acc={m['accuracy']:.4f}  macro_f1={m['macro_f1']:.4f}")

    print("Running distribution-shift robustness sweep...")
    for level in SHIFT_LEVELS:
        X_corrupt = apply_shift(X_test_df, level)
        m = evaluate(model, preproc, X_corrupt, y_test)
        m["level"] = level
        results["shift"].append(m)
        print(f"  shift={level:>4.2f}  acc={m['accuracy']:.4f}  macro_f1={m['macro_f1']:.4f}")

    save_json(results, artifacts_dir / "robustness_results.json")
    print(f"Saved: {artifacts_dir / 'robustness_results.json'}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, key, xlabel in zip(
        axes,
        ["noise", "missing", "shift"],
        ["Gaussian noise (fraction of feature std)", "Missing-data fraction (median-imputed)", "Constant feature shift (fraction of mean)"],
    ):
        levels = [r["level"] for r in results[key]]
        f1s = [r["macro_f1"] for r in results[key]]
        accs = [r["accuracy"] for r in results[key]]
        ax.plot(levels, f1s, marker="o", label="Macro-F1")
        ax.plot(levels, accs, marker="s", label="Accuracy", linestyle="--")
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Score")
    axes[0].legend()
    fig.suptitle(f"Robustness of {args.model} under telemetry corruption", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_path = artifacts_dir / "visualizations" / "robustness_degradation.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
