# scripts/latency_benchmark.py
"""
Real inference-latency / throughput / memory benchmark (Reviewer 4, pt.8:
"inference time, computational complexity, memory consumption, hardware
configuration"). The file previously at this path was a leftover, mislabeled
copy of the old noise-robustness script -- it never measured latency at all.

Measures, using the real trained model on real held-out test data:
  - Single-sample inference latency (ms), mean/median/p95/p99
  - Batch throughput (samples/sec) at several batch sizes
  - Full pipeline latency (scaler.transform + model.predict), since that's
    what a deployed REST call actually pays for
  - Model artifact size on disk (proxy for memory footprint)
  - Reports the CPU it ran on, since latency numbers are meaningless without that

Usage:
  python -m scripts.latency_benchmark --model lightgbm --n_runs 200
"""
from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil
import yaml

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.utils import save_json
from sklearn.model_selection import train_test_split


def timeit_ms(fn, n_runs: int) -> np.ndarray:
    times = np.empty(n_runs)
    for i in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times[i] = (time.perf_counter() - t0) * 1000.0  # ms
    return times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="lightgbm",
                         choices=["random_forest", "xgboost", "lightgbm", "catboost"])
    parser.add_argument("--n_runs", type=int, default=200, help="repeats for single-sample timing")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 8, 32, 128, 512])
    args = parser.parse_args()

    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)
    features = cfg["features"]

    models_dir = Path("models")
    model_path = models_dir / f"model_{args.model}.joblib"
    if not model_path.exists():
        model_path = models_dir / "model.joblib"
    scaler_path = models_dir / f"scaler_{args.model}.joblib"
    if not scaler_path.exists():
        scaler_path = models_dir / "scaler.joblib"

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    df = pd.read_csv("data/external/enhanced_realistic_smartgrid.csv")
    # Use a proper held-out split -- never benchmark against training rows
    _, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    X_test_raw = test_df[features].values

    rng = np.random.default_rng(42)

    print(f"Model artifact: {model_path}  ({model_path.stat().st_size / 1024:.1f} KB on disk)")
    print(f"CPU: {platform.processor() or platform.machine()}  |  "
          f"Cores: {psutil.cpu_count(logical=False)} physical / {psutil.cpu_count()} logical  |  "
          f"RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"Python: {platform.python_version()}  |  OS: {platform.system()} {platform.release()}\n")

    # --- Single-sample, full-pipeline latency (scaler + model, one row at a time) ---
    def single_pipeline_call():
        idx = rng.integers(0, len(X_test_raw))
        row = X_test_raw[idx:idx + 1]
        row_scaled = scaler.transform(row)
        model.predict(row_scaled)

    single_times = timeit_ms(single_pipeline_call, args.n_runs)

    print("=== Single-sample end-to-end latency (scaler + predict), ms ===")
    print(f"  mean   = {single_times.mean():.3f}")
    print(f"  median = {np.median(single_times):.3f}")
    print(f"  p95    = {np.percentile(single_times, 95):.3f}")
    print(f"  p99    = {np.percentile(single_times, 99):.3f}")
    print(f"  min/max = {single_times.min():.3f} / {single_times.max():.3f}\n")

    # --- Batch throughput at several batch sizes ---
    print("=== Batch throughput ===")
    throughput_results = {}
    for bs in args.batch_sizes:
        bs = min(bs, len(X_test_raw))
        batch_raw = X_test_raw[:bs]

        def batch_call(batch_raw=batch_raw):
            batch_scaled = scaler.transform(batch_raw)
            model.predict(batch_scaled)

        t = timeit_ms(batch_call, max(5, args.n_runs // 10))
        mean_ms = t.mean()
        throughput = bs / (mean_ms / 1000.0)
        throughput_results[bs] = {"mean_ms": float(mean_ms), "throughput_samples_per_sec": float(throughput)}
        print(f"  batch={bs:>4d}  mean={mean_ms:8.3f} ms  throughput={throughput:10.1f} samples/sec")

    results = {
        "model": args.model,
        "model_artifact_kb": model_path.stat().st_size / 1024,
        "hardware": {
            "cpu": platform.processor() or platform.machine(),
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(),
            "ram_gb": round(psutil.virtual_memory().total / (1024**3), 1),
            "os": f"{platform.system()} {platform.release()}",
            "python": platform.python_version(),
        },
        "single_sample_latency_ms": {
            "mean": float(single_times.mean()),
            "median": float(np.median(single_times)),
            "p95": float(np.percentile(single_times, 95)),
            "p99": float(np.percentile(single_times, 99)),
            "min": float(single_times.min()),
            "max": float(single_times.max()),
        },
        "batch_throughput": throughput_results,
    }
    save_json(results, "artifacts/latency_benchmark_results.json")
    print("\nSaved: artifacts/latency_benchmark_results.json")
    print("\nNote: these numbers are specific to the machine they ran on -- report the")
    print("hardware line above alongside the latency numbers in the paper (Reviewer 4, pt.8).")


if __name__ == "__main__":
    main()
