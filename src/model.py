from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
import joblib

from .utils import ensure_dir

# Optional models
try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    import lightgbm as lgb
except Exception:
    lgb = None

try:
    import catboost as cb
except Exception:
    cb = None


# ---------------------------------------------------------
# MODEL FACTORY
# ---------------------------------------------------------
def build_model(model_cfg: Dict[str, Any]):
    model_type = model_cfg.get("type", "random_forest").lower()
    params = model_cfg.get("params", {}) or {}

    if model_type == "random_forest":
        return RandomForestClassifier(**params)

    if model_type == "xgboost":
        if xgb is None:
            raise RuntimeError("xgboost not installed. Install via: pip install xgboost")
        
        # Remove eval_metric and use_label_encoder if they're in params
        # These will be set explicitly
        xgb_params = params.copy()
        xgb_params.pop('eval_metric', None)
        xgb_params.pop('use_label_encoder', None)
        
        return xgb.XGBClassifier(
            use_label_encoder=False,
            **xgb_params
        )

    if model_type == "lightgbm":
        if lgb is None:
            raise RuntimeError("lightgbm not installed. Install via: pip install lightgbm")
        
        # Remove verbose if present (LightGBM uses 'verbose' differently)
        lgb_params = params.copy()
        lgb_params.pop('verbose', None)
        
        return lgb.LGBMClassifier(**lgb_params)

    if model_type == "catboost":
        if cb is None:
            raise RuntimeError("catboost not installed. Install via: pip install catboost")

        cb_params = params.copy()
        cb_params.setdefault("verbose", False)
        cb_params.setdefault("random_seed", cb_params.pop("random_state", 42))
        return cb.CatBoostClassifier(**cb_params)

    raise ValueError(f"Unsupported model type: {model_type}")


# ---------------------------------------------------------
# TRAIN + EVAL
# ---------------------------------------------------------
def train_eval(
    X: np.ndarray,
    y: np.ndarray,
    model_cfg: Dict[str, Any],
    test_size: float = 0.2,
    stratify: bool = True,
    shuffle: bool = True,
    artifacts_dir: Path | str = "artifacts",
) -> Tuple[Any, Dict[str, Any], np.ndarray, np.ndarray, np.ndarray | None]:

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y if stratify else None,
        shuffle=shuffle,
        random_state=42,
    )

    model = build_model(model_cfg)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Try to get probabilities
    try:
        y_proba = model.predict_proba(X_test)
    except Exception:
        y_proba = None

    # Metrics
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics: Dict[str, Any] = {
        "classification_report": report,
        "confusion_matrix": cm,
    }

    # Only binary supports ROC-AUC properly
    unique = np.unique(y)
    if y_proba is not None and len(unique) == 2:
        try:
            auc_val = float(roc_auc_score(y_test, y_proba[:, 1]))
            fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
            precision, recall, _ = precision_recall_curve(y_test, y_proba[:, 1])
            metrics["roc_auc"] = auc_val
            metrics["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
            metrics["precision_recall_curve"] = {
                "precision": precision.tolist(),
                "recall": recall.tolist(),
            }
        except Exception:
            pass

    ensure_dir(artifacts_dir)
    joblib.dump(model, Path(artifacts_dir) / "last_model.joblib")

    return model, metrics, y_test, y_pred, y_proba


# ---------------------------------------------------------
# TRAIN + EVAL (pre-split variant -- avoids leakage from external oversampling)
# ---------------------------------------------------------
def train_eval_presplit(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_cfg: Dict[str, Any],
    artifacts_dir: Path | str = "artifacts",
) -> Tuple[Any, Dict[str, Any], np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Same as train_eval, but the caller supplies an already-correct train/test
    split (e.g. oversampled train + untouched test). Use this whenever any
    oversampling/augmentation happens outside this function, so the split
    itself stays leakage-free.
    """
    model = build_model(model_cfg)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    try:
        y_proba = model.predict_proba(X_test)
    except Exception:
        y_proba = None

    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics: Dict[str, Any] = {
        "classification_report": report,
        "confusion_matrix": cm,
    }

    unique = np.unique(np.concatenate([y_train, y_test]))
    if y_proba is not None and len(unique) == 2:
        try:
            auc_val = float(roc_auc_score(y_test, y_proba[:, 1]))
            fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
            precision, recall, _ = precision_recall_curve(y_test, y_proba[:, 1])
            metrics["roc_auc"] = auc_val
            metrics["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
            metrics["precision_recall_curve"] = {"precision": precision.tolist(), "recall": recall.tolist()}
        except Exception:
            pass

    ensure_dir(artifacts_dir)
    joblib.dump(model, Path(artifacts_dir) / "last_model.joblib")

    return model, metrics, y_test, y_pred, y_proba


# ---------------------------------------------------------
# SAVE / LOAD
# ---------------------------------------------------------
def save_model(model, path: str | Path) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    joblib.dump(model, p)
    return p


def load_model(path: str | Path):
    return joblib.load(path)