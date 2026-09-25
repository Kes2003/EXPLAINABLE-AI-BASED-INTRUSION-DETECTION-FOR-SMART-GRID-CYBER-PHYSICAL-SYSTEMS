# scripts/msu_ics_dataset_eval.py
"""
External real-world validation on the Mississippi State University / Oak
Ridge National Laboratory "Power System Attack Datasets" (Adhikari, Pan,
Morris et al.) -- addresses Reviewer 1 & Reviewer 4's request for validation
on a public/benchmark dataset, as a SEPARATE generalization check alongside
the synthetic dataset (not a forced relabeling onto dos/fdi/overload/tamper --
see the note in IDPS_Q1_Revision_Plan.md on why that would be dishonest).

SETUP (I can't download this myself -- ece.uah.edu isn't reachable from my
sandbox's network allowlist):
  1. Download one of these (README explains the difference):
       Multiclass: http://www.ece.uah.edu/~thm0009/icsdatasets/multiclass.7z   (extracts to .arff)
       3-class:    http://www.ece.uah.edu/~thm0009/icsdatasets/triple.7z       (extracts to .csv)
       Binary:     http://www.ece.uah.edu/~thm0009/icsdatasets/binaryAllNaturalPlusNormalVsAttacks.7z  (extracts to .csv)
       README:     http://www.ece.uah.edu/~thm0009/icsdatasets/PowerSystem_Dataset_README.pdf
  2. Extract each .7z (7-Zip / `py7zr` / `unar`). Note the three variants use
     DIFFERENT feature/column layouts -- don't mix them in one folder.
  3. Put each variant in its OWN subfolder, e.g.:
       data/external/msu_ics_power/multiclass/   (the .arff files)
       data/external/msu_ics_power/triple/       (the .csv files)
       data/external/msu_ics_power/binary/       (the .csv files)
  4. Run once per variant, pointing --data_dir at that subfolder:
       python -m scripts.msu_ics_dataset_eval --data_dir data/external/msu_ics_power/multiclass
       python -m scripts.msu_ics_dataset_eval --data_dir data/external/msu_ics_power/triple
       python -m scripts.msu_ics_dataset_eval --data_dir data/external/msu_ics_power/binary
     Output files are auto-tagged by the subfolder name so runs don't overwrite each other.

This script accepts both .arff and .csv (the multiclass set extracts to ARFF;
the triple/binary sets extract to CSV -- both are handled).

Outputs:
  - artifacts/msu_external_validation_results.json
  - artifacts/visualizations/msu_external_validation.png
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, classification_report

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from src.model import build_model
from src.utils import ensure_dir, save_json, set_global_seed

CANDIDATE_LABEL_NAMES = ["marker", "class", "label", "target", "result"]


def load_arff_as_df(path: Path) -> pd.DataFrame:
    """Load a single .arff file into a DataFrame. Tries scipy first, falls
    back to liac-arff (handles a wider range of malformed/real-world ARFF)."""
    try:
        from scipy.io import arff as scipy_arff
        data, meta = scipy_arff.loadarff(str(path))
        df = pd.DataFrame(data)
        # scipy returns byte-strings for nominal/string columns -- decode them
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].apply(lambda v: v.decode("utf-8") if isinstance(v, bytes) else v)
        return df
    except Exception as e:
        print(f"  scipy.io.arff failed ({e}); trying liac-arff...")
        import arff as liac_arff
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            obj = liac_arff.load(f)
        cols = [a[0] for a in obj["attributes"]]
        return pd.DataFrame(obj["data"], columns=cols)


def load_one_file(path: Path) -> pd.DataFrame:
    """Load a single .arff or .csv file into a DataFrame."""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return load_arff_as_df(path)


def load_all(data_dir: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(data_dir / "*.arff"))) + sorted(glob.glob(str(data_dir / "*.csv")))
    if not files:
        raise SystemExit(
            f"No .arff or .csv files found in {data_dir}.\n"
            f"Download and extract one of the .7z files listed at the top of this script first."
        )
    print(f"Found {len(files)} file(s) in {data_dir}: "
          f"{sum(1 for f in files if f.endswith('.arff'))} .arff, "
          f"{sum(1 for f in files if f.endswith('.csv'))} .csv")
    frames = []
    cols_ref = None
    for fp in files:
        print(f"  Loading {Path(fp).name} ...")
        df_part = load_one_file(Path(fp))
        if cols_ref is None:
            cols_ref = list(df_part.columns)
        elif list(df_part.columns) != cols_ref:
            print(f"    Warning: column mismatch vs. first file "
                  f"({len(df_part.columns)} vs {len(cols_ref)} columns) -- "
                  f"this file may be from a different dataset variant (e.g. triple vs binary vs multiclass). "
                  f"Keep each variant in its own subfolder and run this script once per subfolder.")
        frames.append(df_part)
    df = pd.concat(frames, ignore_index=True)
    print(f"Combined shape: {df.shape}")
    return df


def detect_label_column(df: pd.DataFrame) -> str:
    for c in CANDIDATE_LABEL_NAMES:
        if c in df.columns:
            return c
    # fall back: last column is the ARFF convention for the class attribute
    print(f"  Warning: no column named {CANDIDATE_LABEL_NAMES} found; "
          f"falling back to the last column ('{df.columns[-1]}') per ARFF convention. "
          f"VERIFY this is correct before trusting results.")
    return df.columns[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/external/msu_ics_power")
    parser.add_argument("--n_splits", type=int, default=5)
    args = parser.parse_args()

    set_global_seed(42)
    data_dir = ROOT / args.data_dir
    df = load_all(data_dir)

    label_col = detect_label_column(df)
    print(f"\nDetected label column: '{label_col}'")
    print("Detected classes and counts:")
    print(df[label_col].value_counts())

    feature_cols = [c for c in df.columns if c != label_col]
    # coerce all feature columns to numeric; drop any that can't be (e.g. stray IDs)
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # This dataset's PMU impedance features (R1-R4, computed as V/I) go to +/-inf
    # when current is near zero during certain fault/attack scenarios -- a known
    # quirk of this exact dataset. Treat inf the same as missing, THEN impute.
    n_inf = int(np.isinf(df[feature_cols].to_numpy(dtype=float)).sum())
    if n_inf > 0:
        print(f"  Found {n_inf} +/-inf values across feature columns (expected -- "
              f"PMU impedance ratios blow up when current is near zero). Treating as missing.")
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    df = df.dropna(subset=feature_cols, how="all")
    medians = df[feature_cols].median(numeric_only=True)
    df[feature_cols] = df[feature_cols].fillna(medians)
    # any column that was ALL inf/NaN (median itself is NaN) -- fall back to 0
    df[feature_cols] = df[feature_cols].fillna(0.0)

    remaining_bad = int(np.isinf(df[feature_cols].to_numpy(dtype=float)).sum()
                         + df[feature_cols].isna().to_numpy().sum())
    if remaining_bad > 0:
        raise RuntimeError(f"{remaining_bad} non-finite values remain after cleaning -- inspect the data.")
    print(f"  Cleaned feature matrix: {df[feature_cols].shape}, all finite.")

    X_all = df[feature_cols].values
    y_all_str = df[label_col].astype(str).values
    le = LabelEncoder().fit(y_all_str)

    variants = {
        "random_forest": {"type": "random_forest", "params": {"n_estimators": 200, "max_depth": 20, "n_jobs": -1, "random_state": 42}},
        "xgboost": {"type": "xgboost", "params": {"n_estimators": 200, "max_depth": 8, "learning_rate": 0.1, "n_jobs": -1, "random_state": 42}},
        "lightgbm": {"type": "lightgbm", "params": {"n_estimators": 200, "max_depth": -1, "learning_rate": 0.1, "num_leaves": 31, "n_jobs": -1, "random_state": 42}},
        "catboost": {"type": "catboost", "params": {"iterations": 200, "depth": 8, "learning_rate": 0.1, "random_state": 42, "verbose": False}},
    }

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    scores = {name: {"acc": [], "f1": []} for name in variants}

    fold = 0
    for train_idx, test_idx in skf.split(X_all, y_all_str):
        fold += 1
        Xtr_raw, Xte_raw = X_all[train_idx], X_all[test_idx]
        ytr_str, yte_str = y_all_str[train_idx], y_all_str[test_idx]
        scaler = StandardScaler().fit(Xtr_raw)
        Xtr, Xte = scaler.transform(Xtr_raw), scaler.transform(Xte_raw)
        ytr, yte = le.transform(ytr_str), le.transform(yte_str)

        for name, cfg in variants.items():
            model = build_model(cfg)
            model.fit(Xtr, ytr)
            pred = model.predict(Xte)
            scores[name]["acc"].append(float(accuracy_score(yte, pred)))
            scores[name]["f1"].append(float(f1_score(yte, pred, average="macro")))
        print(f"Fold {fold}/{args.n_splits}: " + ", ".join(f"{n}={scores[n]['f1'][-1]:.4f}" for n in variants))

    summary = {n: {"macro_f1_mean": float(np.mean(v["f1"])), "macro_f1_std": float(np.std(v["f1"])),
                    "accuracy_mean": float(np.mean(v["acc"])), "accuracy_std": float(np.std(v["acc"]))}
               for n, v in scores.items()}

    ensure_dir("artifacts")
    tag = Path(args.data_dir).name  # e.g. "multiclass" or "binary" -> keeps runs from overwriting each other
    save_json({"label_column": label_col, "classes": list(le.classes_), "n_samples": len(df), "summary": summary,
               "per_fold": scores}, f"artifacts/msu_external_validation_results_{tag}.json")
    print(f"\nSaved: artifacts/msu_external_validation_results_{tag}.json")
    for n, m in summary.items():
        print(f"  {n:15s} macro-F1 = {m['macro_f1_mean']:.4f} +/- {m['macro_f1_std']:.4f}")

    plt.figure(figsize=(7, 5))
    names = list(summary.keys())
    means = [summary[n]["macro_f1_mean"] for n in names]
    stds = [summary[n]["macro_f1_std"] for n in names]
    ax = sns.barplot(x=names, y=means)
    ax.errorbar(x=range(len(names)), y=means, yerr=stds, fmt="none", c="black", capsize=4)
    plt.ylabel("Macro-F1 (CV, mean +/- std)")
    plt.title(f"External validation: MSU/ORNL Power System dataset (n={len(df)})")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    ensure_dir("artifacts/visualizations")
    plt.savefig(f"artifacts/visualizations/msu_external_validation_{tag}.png", dpi=200, bbox_inches="tight")
    print(f"Saved: artifacts/visualizations/msu_external_validation_{tag}.png")


if __name__ == "__main__":
    main()
