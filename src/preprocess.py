from __future__ import annotations

from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

try:
    from .utils import ensure_dir
except Exception:
    from utils import ensure_dir


class Preprocessor:
    """
    Preprocessor for the Smart Grid IDS project.

    Usage:
        preproc = Preprocessor(features, scaler_path="models/scaler.joblib", label_encoder_path="models/label_encoder.joblib")
        X, y = preproc.fit_transform(df)   # for training
        X = preproc.transform(df_new)      # for inference (uses loaded/fitted scaler)
    """

    def __init__(self, features: List[str], scaler_path: Optional[Path | str] = None, label_encoder_path: Optional[Path | str] = None):
        self.features = list(features)
        self.scaler_path = Path(scaler_path) if scaler_path is not None else None
        self.label_encoder_path = Path(label_encoder_path) if label_encoder_path is not None else None
        self.scaler: Optional[StandardScaler] = None
        self.label_encoder: Optional[LabelEncoder] = None

    # ------------------------------
    # Utilities
    # ------------------------------
    def _validate_features_present(self, df: pd.DataFrame):
        missing = [f for f in self.features if f not in df.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing}")

    def _coerce_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Coerce specific features to numeric types; ensure no object columns for features.
        Keep a safe approach: attempt conversion, if fails raise informative error.
        """
        out = df.copy()
        for f in self.features:
            if f not in out.columns:
                continue
            # attempt numeric coercion
            try:
                out[f] = pd.to_numeric(out[f], errors="raise")
            except Exception as e:
                raise ValueError(f"Feature '{f}' could not be converted to numeric type: {e}")
        return out

    # ------------------------------
    # Fit / Transform for training
    # ------------------------------
    def fit_transform(self, df: pd.DataFrame, label_col: str = "label") -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit scaler and label encoder on the provided dataframe and return X (scaled) and y (encoded).
        Saves scaler and label_encoder to the configured paths if provided.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame")

        # validate features exist
        self._validate_features_present(df)

        # coerce types for features and fill NaNs with median
        X_df = self._coerce_types(df[self.features])
        X_df = X_df.replace([np.inf, -np.inf], np.nan)
        X_df = X_df.fillna(X_df.median())

        # labels
        if label_col not in df.columns:
            raise ValueError(f"Label column '{label_col}' not found in training DataFrame.")
        y_ser = df[label_col].astype(str).copy()

        # scaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_df.values)

        # label encoder
        self.label_encoder = LabelEncoder()
        y_enc = self.label_encoder.fit_transform(y_ser.values)

        # save artifacts if paths provided
        if self.scaler_path:
            ensure_dir(self.scaler_path.parent)
            joblib.dump(self.scaler, self.scaler_path)
        if self.label_encoder_path:
            ensure_dir(self.label_encoder_path.parent)
            joblib.dump(self.label_encoder, self.label_encoder_path)

        return X_scaled, y_enc

    # ------------------------------
    # Transform for inference
    # ------------------------------
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform a dataframe into the scaled numpy array expected by models.
        Expects that `self.scaler` and (optionally) `self.label_encoder` are already set.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame")

        self._validate_features_present(df)

        X_df = self._coerce_types(df[self.features])
        X_df = X_df.replace([np.inf, -np.inf], np.nan)
        X_df = X_df.fillna(X_df.median())

        if self.scaler is None:
            # Try loading from disk if path provided
            if self.scaler_path and Path(self.scaler_path).exists():
                self.scaler = joblib.load(self.scaler_path)
            else:
                raise RuntimeError("Scaler is not fitted or available. Call fit_transform first or provide a scaler_path.")

        X_scaled = self.scaler.transform(X_df.values)
        return X_scaled

    # ------------------------------
    # Helpers for saving / loading named/versioned artifacts
    # ------------------------------
    def save_versioned(self, models_dir: Path | str, model_type: str):
        """
        Save copies of scaler and label encoder with model_type suffix,
        e.g. scaler_random_forest.joblib and label_encoder_random_forest.joblib
        """
        md = Path(models_dir)
        ensure_dir(md)
        if self.scaler is not None:
            target = md / f"scaler_{model_type}.joblib"
            joblib.dump(self.scaler, target)
            # also save untyped fallback for convenience
            joblib.dump(self.scaler, md / "scaler.joblib")
        if self.label_encoder is not None:
            target = md / f"label_encoder_{model_type}.joblib"
            joblib.dump(self.label_encoder, target)
            joblib.dump(self.label_encoder, md / "label_encoder.joblib")

    def load_from_paths(self):
        """
        Load scaler and label encoder from configured paths (if set).
        Returns True if at least scaler loaded, otherwise False.
        """
        loaded = False
        if self.scaler_path and Path(self.scaler_path).exists():
            self.scaler = joblib.load(self.scaler_path)
            loaded = True
        if self.label_encoder_path and Path(self.label_encoder_path).exists():
            self.label_encoder = joblib.load(self.label_encoder_path)
            # label encoder alone doesn't count as loaded in the same way
        return loaded
