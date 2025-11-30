from __future__ import annotations

from pathlib import Path
from typing import Tuple, Any, List
import json
import joblib

try:
    from .preprocess import Preprocessor
except Exception:
    from preprocess import Preprocessor


def load_preprocessor_and_model(
    models_dir: str | Path,
    model_type: str | None = None,
) -> Tuple[Preprocessor, Any, List[str], str]:
    """
    Robust loader for preprocessor + model artifacts.

    - Supports typed artifact names: model_{type}.joblib, scaler_{type}.joblib, label_encoder_{type}.joblib
    - Falls back to untyped artifacts model.joblib, scaler.joblib, label_encoder.joblib
    - Returns (preproc, model, features_list, used_model_type)
    """
    p = Path(models_dir)
    if not p.exists():
        raise FileNotFoundError(f"Models directory not found: {p}")

    features_path = p / "features.json"
    if not features_path.exists():
        raise FileNotFoundError(f"features.json not found in models directory: {features_path}")

    with features_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    features = data.get("features", [])
    if not isinstance(features, list) or not features:
        raise ValueError("Invalid or empty 'features' entry in models/features.json")

    # helper to resolve candidate path
    def resolve(name: str):
        """
        Try typed then untyped. Return tuple (Path_or_None, used_type_or_None).
        """
        if model_type:
            typed = p / f"{name}_{model_type}.joblib"
            if typed.exists():
                return typed, model_type
        untyped = p / f"{name}.joblib"
        if untyped.exists():
            return untyped, "default"
        return None, None

    scaler_path, scaler_type = resolve("scaler")
    le_path, le_type = resolve("label_encoder")
    model_path, model_type_used = resolve("model")

    if model_path is None:
        raise FileNotFoundError(f"No model file found in {p} for model_type={model_type}")

    # load available artifacts
    scaler = joblib.load(scaler_path) if scaler_path is not None else None
    label_encoder = joblib.load(le_path) if le_path is not None else None
    model = joblib.load(model_path)

    # create Preprocessor and inject loaded scaler/encoder
    preproc = Preprocessor(features, scaler_path=scaler_path if scaler_path is not None else None, label_encoder_path=le_path if le_path is not None else None)
    preproc.scaler = scaler
    preproc.label_encoder = label_encoder

    # determine used type string
    used_type = model_type if (model_type and model_path.name.endswith(f"_{model_type}.joblib")) else (model_type_used or "model")
    return preproc, model, features, used_type
