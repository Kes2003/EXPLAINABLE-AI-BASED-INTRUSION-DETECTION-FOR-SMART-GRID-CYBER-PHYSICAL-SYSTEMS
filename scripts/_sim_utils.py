# scripts/_sim_utils.py
"""
Shared helper for the attack-simulation figures (Figs. 4-11 in the manuscript).

Every "IDS probability" value plotted by scripts/{dos_attack,simulate_fdi,
simulate_overload,simulate_tamper}.py comes from this function actually
calling the trained model's predict_proba on a constructed row -- never from
np.random. The only thing that is "simulated" is the INPUT (a physically
plausible ramp from normal operating values into attack-range values, which
is a standard and legitimate way to build an illustrative time-series test
case); the OUTPUT (the probability curve) is always genuine model inference.
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import joblib
import numpy as np
import pandas as pd
import yaml

from src.preprocess import Preprocessor

BASELINE_NORMAL_ROW = {
    "voltage_v": 230.0,
    "current_a": 10.5,
    "frequency_hz": 50.01,
    "power_factor": 0.97,
    "active_power_kw": 2.3,
    "reactive_power_kvar": 0.65,
    "thd_percent": 2.1,
    "breaker_closed": 1,
    "packet_rate": 120,
    "packet_error_rate": 0.002,
    "substation_temp_c": 28,
    "transformer_oil_temp_c": 40,
    "line_load_percent": 55,
    "tap_position": 10,
    "phase_imbalance_percent": 1.2,
    "feeder_voltage_v": 229,
    "feeder_current_a": 10.4,
    "demand_kw": 2.4,
}


def load_pipeline(model_type: str = "lightgbm", config_path: str = "config/config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    features = cfg["features"]
    models_dir = ROOT / "models"

    model_path = models_dir / f"model_{model_type}.joblib"
    if not model_path.exists():
        model_path = models_dir / "model.joblib"
    model = joblib.load(model_path)

    preproc = Preprocessor(
        features=features,
        scaler_path=models_dir / f"scaler_{model_type}.joblib",
        label_encoder_path=models_dir / f"label_encoder_{model_type}.joblib",
    )
    if not preproc.load_from_paths():
        preproc = Preprocessor(
            features=features,
            scaler_path=models_dir / "scaler.joblib",
            label_encoder_path=models_dir / "label_encoder.joblib",
        )
        preproc.load_from_paths()

    return model, preproc, features


def run_sequence(model, preproc: Preprocessor, features: list[str], rows: list[dict]):
    """
    rows: list of feature dicts (one per timestep).
    Returns: (pred_labels: list[str], prob_of_pred: np.ndarray, full_proba: np.ndarray, class_names: list[str])
    All values are genuine model.predict_proba output -- no randomness added.
    """
    df = pd.DataFrame(rows)[features]
    X = preproc.transform(df)
    proba = model.predict_proba(X)
    pred_idx = proba.argmax(axis=1)
    class_names = list(preproc.label_encoder.classes_)
    pred_labels = [class_names[i] for i in pred_idx]
    prob_of_pred = proba.max(axis=1)
    return pred_labels, prob_of_pred, proba, class_names


def attack_probability_series(proba: np.ndarray, class_names: list[str], attack_label: str) -> np.ndarray:
    """The model's predicted probability of the specific attack class at each timestep
    (rather than just max-probability), which is what Figs. 8-11 actually plot."""
    if attack_label not in class_names:
        raise ValueError(f"'{attack_label}' not in trained class list {class_names}")
    idx = class_names.index(attack_label)
    return proba[:, idx]


def empirical_class_medians(features: list[str], data_path: str = "data/external/enhanced_realistic_smartgrid.csv") -> dict:
    """
    Median feature vector per label, computed directly from the real dataset.
    Used to build attack-simulation ramps that interpolate toward a REAL,
    in-distribution operating point for that attack class, rather than a
    hand-guessed perturbation that may not match what the model actually learned.
    """
    df = pd.read_csv(ROOT / data_path)
    return df.groupby("label")[features].median().to_dict(orient="index")


def build_ramp_rows(features: list[str], normal_point: dict, attack_point: dict,
                     T: int = 100, attack_start: int = 40, attack_end: int = 70) -> list[dict]:
    """
    Build a T-step time series that sits at `normal_point` outside the attack
    window and ramps smoothly to `attack_point` (both real, empirically
    grounded feature vectors) during [attack_start, attack_end).
    """
    rows = []
    for t in range(T):
        if attack_start <= t < attack_end:
            progress = (t - attack_start) / max(1, (attack_end - attack_start - 1))
            row = {f: normal_point[f] + progress * (attack_point[f] - normal_point[f]) for f in features}
        else:
            row = dict(normal_point)
        rows.append(row)
    return rows
