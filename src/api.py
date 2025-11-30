from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional

import json
import logging

import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException, Query, Header
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

try:
    from .artifacts import load_preprocessor_and_model
    from .utils import ensure_dir, project_path, init_audit_db, log_audit_entry
except ImportError:
    from artifacts import load_preprocessor_and_model
    from utils import ensure_dir, project_path, init_audit_db, log_audit_entry

APP_VERSION = "1.3.0"

app = FastAPI(
    title="Smart Grid IDS API",
    version=APP_VERSION,
    description="AI-Enabled Intrusion Detection & Prevention for Smart Grids"
)

# Directories
MODELS_DIR = Path("models")
ARTIFACTS_DIR = Path("artifacts")

# Logging
logger = logging.getLogger("smartgrid_ids_api")
logger.setLevel(logging.INFO)
LOGS_DIR = project_path("logs")
ensure_dir(LOGS_DIR)
file_handler = logging.FileHandler(LOGS_DIR / "api.log")
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(file_handler)

# Audit DB
AUDIT_DB = LOGS_DIR / "audit.db"
init_audit_db(AUDIT_DB)

# Defaults (overwritten by config on startup)
ALERT_THRESHOLD: float = 0.7
DEFAULT_MODEL_TYPE: str = "random_forest"

# In-memory cache for preloaded models: model_type -> (preproc, model)
_LOADED_MODELS: Dict[str, tuple] = {}


# ------------------------------
# Config loader
# ------------------------------
def _load_config():
    global ALERT_THRESHOLD, DEFAULT_MODEL_TYPE
    cfg_path = project_path("config", "config.yaml")
    if cfg_path.exists():
        try:
            with cfg_path.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            ALERT_THRESHOLD = float(cfg.get("thresholds", {}).get("alert_probability", ALERT_THRESHOLD))
            DEFAULT_MODEL_TYPE = str(cfg.get("default_model", cfg.get("model", {}).get("type", DEFAULT_MODEL_TYPE)))
            logger.info("Loaded config: alert_threshold=%.3f, default_model=%s", ALERT_THRESHOLD, DEFAULT_MODEL_TYPE)
        except Exception as e:
            logger.exception("Failed to load config: %s", e)
    else:
        logger.info("No config file found at config/config.yaml - using defaults.")


# ------------------------------
# Pydantic schemas
# ------------------------------
class TelemetryRow(BaseModel):
    voltage_v: float
    current_a: float
    frequency_hz: float
    power_factor: float
    active_power_kw: float
    reactive_power_kvar: float
    thd_percent: float
    breaker_closed: int
    packet_rate: float
    packet_error_rate: float
    substation_temp_c: float
    transformer_oil_temp_c: float
    line_load_percent: float
    tap_position: int
    phase_imbalance_percent: float
    feeder_voltage_v: float
    feeder_current_a: float
    demand_kw: float


class PredictRequest(BaseModel):
    rows: List[TelemetryRow] = Field(default_factory=list, description="Telemetry rows to classify")


class PredictionResult(BaseModel):
    label: str
    probability: Optional[float]
    proba_available: bool
    model_type: str


class PredictResponse(BaseModel):
    predictions: List[PredictionResult]


class PreventAction(BaseModel):
    should_trip_breaker: bool = False
    should_rate_limit_network: bool = False
    reason: str = ""
    predicted_label: Optional[str] = None
    probability: Optional[float] = None
    model_type: Optional[str] = None


class PreventResponse(BaseModel):
    actions: List[PreventAction]


# ------------------------------
# Model cache helpers
# ------------------------------
def preload_default_model():
    """
    Load default model on startup into cache (if available).
    """
    try:
        preproc, model, _, used_type = load_preprocessor_and_model(MODELS_DIR, model_type=DEFAULT_MODEL_TYPE)
        _LOADED_MODELS[used_type] = (preproc, model)
        logger.info("Preloaded default model: %s", used_type)
    except Exception as e:
        logger.warning("Could not preload default model '%s': %s", DEFAULT_MODEL_TYPE, e)
        logger.warning("API will start but predictions will fail until a model is trained and available.")


def _get_model_from_cache_or_load(model_type: Optional[str]):
    mt = model_type or DEFAULT_MODEL_TYPE
    if mt in _LOADED_MODELS:
        return _LOADED_MODELS[mt], mt
    # attempt to load
    try:
        preproc, model, _, used_type = load_preprocessor_and_model(MODELS_DIR, model_type=mt)
        _LOADED_MODELS[used_type] = (preproc, model)
        return _LOADED_MODELS[used_type], used_type
    except Exception as e:
        logger.error("Failed to load model type '%s': %s", mt, e)
        raise HTTPException(
            status_code=503,
            detail=f"Model '{mt}' not available. Please train a model first using: python -m src.train"
        )


def _available_models() -> List[str]:
    p = MODELS_DIR
    ensure_dir(p)
    models = []
    for child in sorted(p.glob("*.joblib")):
        stem = child.stem  # model_random_forest or model
        if stem.startswith("model_"):
            models.append(stem.split("model_")[-1])
        elif stem == "model":
            models.append("model")
    # unique preserve order
    out = []
    for m in models:
        if m not in out:
            out.append(m)
    return out


# ------------------------------
# Prediction utilities
# ------------------------------
def _prepare_df_and_validate(preproc, rows: List[TelemetryRow]) -> pd.DataFrame:
    df = pd.DataFrame([r.model_dump() for r in rows])
    # validate presence of features expected by preproc
    missing = [f for f in preproc.features if f not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required features: {missing}")
    return df


def _predict_with_model(preproc, model, df: pd.DataFrame):
    X = preproc.transform(df)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        pred_idx = np.argmax(proba, axis=1)
        labels = preproc.label_encoder.inverse_transform(pred_idx)
        probs = proba[np.arange(len(pred_idx)), pred_idx]
        proba_available = True
    else:
        labels = model.predict(X)
        probs = [None] * len(labels)
        proba_available = False
    return labels, probs, proba_available


# ------------------------------
# Prevention policy (uses ALERT_THRESHOLD and per-class optional thresholds)
# ------------------------------
def compute_preventive_action(prediction: str, probability: Optional[float], threshold: float) -> PreventAction:
    prob_val = float(probability) if probability is not None else 0.0

    if probability is None or prob_val < threshold:
        return PreventAction(
            should_trip_breaker=False,
            should_rate_limit_network=False,
            reason=f"Prediction '{prediction}' below confidence threshold ({prob_val:.2f} < {threshold:.2f}); no automated action",
            predicted_label=prediction,
            probability=prob_val,
        )

    if prediction == "dos":
        return PreventAction(
            should_trip_breaker=False,
            should_rate_limit_network=True,
            reason="High-confidence DoS detected: recommend rate limiting",
            predicted_label=prediction,
            probability=prob_val,
        )

    if prediction == "tamper":
        return PreventAction(
            should_trip_breaker=True,
            should_rate_limit_network=False,
            reason="High-confidence tampering detected: recommend isolating feeder / trip breaker (operator review needed)",
            predicted_label=prediction,
            probability=prob_val,
        )

    if prediction == "overload":
        return PreventAction(
            should_trip_breaker=True,
            should_rate_limit_network=False,
            reason="High-confidence overload detected: recommend load shedding / trip relay (operator review needed)",
            predicted_label=prediction,
            probability=prob_val,
        )

    if prediction == "fdi":
        return PreventAction(
            should_trip_breaker=False,
            should_rate_limit_network=False,
            reason="False Data Injection detected: escalate to operator for manual investigation",
            predicted_label=prediction,
            probability=prob_val,
        )

    return PreventAction(
        should_trip_breaker=False,
        should_rate_limit_network=False,
        reason="No automated action for normal state",
        predicted_label=prediction,
        probability=prob_val,
    )


# ------------------------------
# Routes
# ------------------------------
@app.on_event("startup")
async def startup_event():
    ensure_dir(MODELS_DIR)
    ensure_dir(ARTIFACTS_DIR)
    _load_config()
    preload_default_model()
    logger.info("Smart Grid IDS API v%s started (default_model=%s, alert_threshold=%.2f)", APP_VERSION, DEFAULT_MODEL_TYPE, ALERT_THRESHOLD)


@app.get("/", response_class=HTMLResponse)
async def root():
    available = _available_models()
    html = f"""
    <html>
      <head><title>Smart Grid IDS API</title></head>
      <body style="font-family:Arial,Helvetica,sans-serif;margin:24px">
        <h1>Smart Grid IDS API (v{APP_VERSION})</h1>
        <p><strong>Default model:</strong> {DEFAULT_MODEL_TYPE}</p>
        <p><strong>Available models:</strong> {', '.join(available) if available else 'none found'}</p>
        <p>Interactive docs: <a href="/docs">/docs</a></p>
        <h3>Example: Predict (normal)</h3>
        <pre><code>curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type:application/json" -d '{{"rows":[{{"voltage_v":230.0,"current_a":10.5,"frequency_hz":50.01,"power_factor":0.97,"active_power_kw":2.3,"reactive_power_kvar":0.65,"thd_percent":2.1,"breaker_closed":1,"packet_rate":120,"packet_error_rate":0.002,"substation_temp_c":28,"transformer_oil_temp_c":40,"line_load_percent":55,"tap_position":10,"phase_imbalance_percent":1.2,"feeder_voltage_v":229,"feeder_current_a":10.4,"demand_kw":2.4}}]}}'</code></pre>
        <h3>Example: Prevent (threat)</h3>
        <pre><code>curl -X POST "http://127.0.0.1:8000/prevent" -H "Content-Type:application/json" -d '{{"rows":[{{"voltage_v":227,"current_a":9.7,"frequency_hz":50.0,"power_factor":0.95,"active_power_kw":2.0,"reactive_power_kvar":0.5,"thd_percent":3.0,"breaker_closed":1,"packet_rate":800.0,"packet_error_rate":0.15,"substation_temp_c":29,"transformer_oil_temp_c":41,"line_load_percent":58,"tap_position":10,"phase_imbalance_percent":1.5,"feeder_voltage_v":227,"feeder_current_a":10.1,"demand_kw":2.35}}]}}'</code></pre>
        <p>To explicitly choose model: add <code>?model_type=xgboost</code> or header <code>X-Model-Type: lightgbm</code>.</p>
      </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.get("/health")
async def health():
    model_status = "ready" if (MODELS_DIR / "model.joblib").exists() else "not_loaded"
    return {
        "status": "ok",
        "api_version": APP_VERSION,
        "model_status": model_status,
        "default_model": DEFAULT_MODEL_TYPE,
        "alert_threshold": ALERT_THRESHOLD,
        "available_models": _available_models(),
    }


@app.get("/models")
async def list_models():
    return {"models": _available_models()}


@app.post("/predict", response_model=PredictResponse)
async def predict(
    req: PredictRequest,
    model_type: Optional[str] = Query(None, description="Optional model override (query param)"),
    x_model_type: Optional[str] = Header(None, alias="X-Model-Type", description="Optional model override (header)")
):
    if not req.rows:
        raise HTTPException(status_code=400, detail="No rows provided")

    chosen = model_type or x_model_type or DEFAULT_MODEL_TYPE
    logger.info("Predict called with %d rows, model_type=%s", len(req.rows), chosen)

    (preproc, model), used_type = _get_model_from_cache_or_load(chosen)
    df = _prepare_df_and_validate(preproc, req.rows)

    labels, probs, proba_available = _predict_with_model(preproc, model, df)
    results: List[PredictionResult] = []
    for lbl, p in zip(labels, probs):
        res = PredictionResult(
            label=str(lbl),
            probability=(float(p) if p is not None else None),
            proba_available=bool(proba_available),
            model_type=str(used_type),
        )
        results.append(res)

    # audit log
    for r_row, lbl, p in zip(req.rows, labels, probs):
        try:
            log_audit_entry(AUDIT_DB, used_type, r_row.model_dump(), str(lbl), (float(p) if p is not None else None), {"endpoint": "/predict"})
        except Exception:
            logger.exception("Failed to write audit entry")

    return PredictResponse(predictions=results)


@app.post("/prevent", response_model=PreventResponse)
async def prevent(
    req: PredictRequest,
    model_type: Optional[str] = Query(None, description="Optional model override (query param)"),
    x_model_type: Optional[str] = Header(None, alias="X-Model-Type", description="Optional model override (header)")
):
    if not req.rows:
        raise HTTPException(status_code=400, detail="No rows provided")

    chosen = model_type or x_model_type or DEFAULT_MODEL_TYPE
    logger.info("Prevent called with %d rows, model_type=%s", len(req.rows), chosen)

    (preproc, model), used_type = _get_model_from_cache_or_load(chosen)
    df = _prepare_df_and_validate(preproc, req.rows)

    labels, probs, proba_available = _predict_with_model(preproc, model, df)

    actions: List[PreventAction] = []
    for r_row, lbl, p in zip(req.rows, labels, probs):
        action = compute_preventive_action(str(lbl), (float(p) if p is not None else None), ALERT_THRESHOLD)
        action.model_type = used_type  # add model info
        actions.append(action)
        try:
            log_audit_entry(AUDIT_DB, used_type, r_row.model_dump(), str(lbl), (float(p) if p is not None else None), action.model_dump())
        except Exception:
            logger.exception("Failed to write audit entry for prevent")

    return PreventResponse(actions=actions)