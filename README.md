# Explainable AI-Based Intrusion Detection for Smart Grid Cyber-Physical Systems

An end-to-end, explainable machine-learning system for **detecting and responding to cyber-physical attacks on smart grids**. It fuses electrical, power-quality, communication-network, and equipment-health telemetry. It then classifies each sample as normal operation or one of four attack types, explains every decision with SHAP, and serves predictions and prevention recommendations through a real-time FastAPI service.

**Highlights**

- **Multidomain detection**: 18 telemetry features across four domains; 5 classes (normal, DoS, FDI, tamper, overload).
- **Four ensemble models**: Random Forest, XGBoost, LightGBM (default), and CatBoost, selectable per request.
- **Leakage-safe evaluation**: stratified cross-validation that oversamples the training fold only, with paired Wilcoxon and corrected resampled t-tests.
- **Explainability**: SHAP (TreeExplainer) global and per-class explanations, validated with feature-group ablation.
- **Robustness testing** under sensor noise, missing telemetry, and distribution shift.
- **External validation** on the public MSU/ORNL Power System Attack Dataset (78,377 real PMU/relay samples).
- **Real-time API** with a confidence-gated prevention policy, per-request model selection, and an SQLite audit trail.

---

## Contents

1. [How it works](#1-how-it-works)
2. [Repository structure](#2-repository-structure)
3. [Installation](#3-installation)
4. [Data](#4-data)
5. [Configuration](#5-configuration)
6. [Quick start](#6-quick-start)
7. [Training and prediction](#7-training-and-prediction)
8. [REST API](#8-rest-api)
9. [Prevention policy](#9-prevention-policy)
10. [Explainability](#10-explainability)
11. [Evaluation and experiments](#11-evaluation-and-experiments)
12. [Results](#12-results)
13. [Reproducibility](#13-reproducibility)
14. [Testing and demo clients](#14-testing-and-demo-clients)
15. [Limitations](#15-limitations)
16. [Troubleshooting](#16-troubleshooting)

---

## 1. How it works

```text
 telemetry row (18 features)
          │
          ▼
 ┌──────────────────┐   ┌──────────────────────┐   ┌──────────────────┐   ┌───────────────────────┐
 │  Preprocessing   │──▶│  Ensemble classifier │──▶│  SHAP explainer  │──▶│  Prevention policy    │
 │ impute · scale   │   │  LightGBM (default)  │   │  per-feature     │   │  action if p ≥ 0.7,   │
 │ (fit on train)   │   │  RF · XGB · CatBoost │   │  attribution     │   │  else escalate        │
 └──────────────────┘   └──────────────────────┘   └──────────────────┘   └───────────┬───────────┘
                                                                                      │
                                         FastAPI service  ◀──────────────────────────┘
                                         /predict · /prevent · audit log (SQLite)
```

**Input features** (feature order is fixed in `config/config.yaml` and `models/features.json`):

| Domain | Features |
|---|---|
| Electrical (9) | `voltage_v`, `current_a`, `active_power_kw`, `reactive_power_kvar`, `power_factor`, `feeder_voltage_v`, `feeder_current_a`, `demand_kw`, `line_load_percent` |
| Power quality (3) | `frequency_hz`, `thd_percent`, `phase_imbalance_percent` |
| Communication (2) | `packet_rate`, `packet_error_rate` |
| Equipment health (4) | `breaker_closed`, `substation_temp_c`, `transformer_oil_temp_c`, `tap_position` |

**Classes**

| Class | Description | Typical signature |
|---|---|---|
| `normal` | Nominal operation | — |
| `dos` | Denial-of-service / network flooding | High packet rate and packet error rate |
| `fdi` | False data injection into measurements | Correlated voltage / THD / power-factor deviations |
| `tamper` | Physical tampering / unauthorized breaker operation | Breaker state change, phase-imbalance spike |
| `overload` | Thermal overload / high loading | Rising line load and transformer/substation temperature |

---

## 2. Repository structure

```text
├── src/                               # Core package
│   ├── api.py                         # FastAPI service (/health, /models, /predict, /prevent)
│   ├── cli.py                         # `python -m src.cli serve`
│   ├── train.py                       # Training pipeline (split BEFORE oversampling)
│   ├── model.py                       # Model factory (RF, XGBoost, LightGBM, CatBoost) + evaluation
│   ├── preprocess.py                  # Missing-value handling, scaling, label encoding
│   ├── predict.py                     # Batch / single-row prediction CLI with optional audit logging
│   ├── explain.py                     # SHAP explanations for a trained model
│   ├── artifacts.py                   # Loads model + preprocessor by model type
│   ├── utils.py                       # Paths, seeding, JSON helpers, audit DB
│   ├── generate_data.py               # Synthetic smart-grid data generator
│   ├── evaluate_models_cv.py          # Legacy multi-model CV comparison
│   ├── train_unsw.py                  # UNSW-NB15 network-intrusion training pipeline
│   └── datasets/unsw_nb15.py          # UNSW-NB15 adapter
│
├── scripts/
│   ├── cv_evaluation.py               # Leakage-safe 5-fold CV of all four models + Wilcoxon tests
│   ├── corrected_resampled_ttest.py   # Nadeau–Bengio corrected t-test for repeated CV
│   ├── ablation_study.py              # Feature-group ablation (primary dataset)
│   ├── roc_auc_analysis.py            # Multiclass ROC / AUC
│   ├── noise_robustness_test.py       # Noise, missingness, distribution-shift robustness
│   ├── latency_benchmark.py           # Inference latency / throughput / memory
│   ├── dos_attack.py, simulate_fdi.py, simulate_overload.py, simulate_tamper.py
│   │                                  # Attack-scenario simulations driven by the trained model
│   ├── msu_ics_dataset_eval.py        # External validation on MSU/ORNL (4 models)
│   ├── msu_shap_analysis.py           # SHAP explainability on MSU/ORNL
│   ├── msu_ablation_study.py          # Feature-group ablation on MSU/ORNL
│   ├── live_stream.py, simulate_attack_stream.py   # Streaming demos against the API
│   ├── start_api.py                   # API launcher
│   └── setup.py                       # Environment / folder checks
│
├── run_repeated_cv_and_policy_eval.py # Repeated CV (10 seeds × 5 folds) + prevention-policy evaluation
├── results/                           # JSON results and console logs of the evaluation runs
├── config/config.yaml                 # Features, model hyperparameters, model_variants, alert threshold
├── models/features.json               # Feature order used by trained models (model files are gitignored)
├── data/                              # raw/, processed/, external/ (dataset files are gitignored)
├── artifacts/visualizations/          # Generated figures
├── tests/test_api.py                  # API endpoint tests
├── test_normal.py, test_dos_attack.py, compare_models.py, view_audit.py, testing.py
│                                      # Example API clients and utilities
├── Dockerfile
├── requirements.txt
└── docs/                              # Project-structure and quick-reference notes
```

---

## 3. Installation

Requires **Python 3.10 or newer**.

```bash
git clone https://github.com/Kes2003/EXPLAINABLE-AI-BASED-INTRUSION-DETECTION-FOR-SMART-GRID-CYBER-PHYSICAL-SYSTEMS.git
cd EXPLAINABLE-AI-BASED-INTRUSION-DETECTION-FOR-SMART-GRID-CYBER-PHYSICAL-SYSTEMS
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/setup.py                               # optional: creates the expected folders
```

The main dependencies are scikit-learn, XGBoost, LightGBM, CatBoost, SHAP, SciPy, pandas, FastAPI, and Uvicorn (see `requirements.txt`).

---

## 4. Data

Dataset files are **not** stored in the repository (see `.gitignore`).

| Dataset | Location | Used for |
|---|---|---|
| Primary smart-grid dataset: 30,000 labeled samples, 18 features + `label` | `data/external/enhanced_realistic_smartgrid.csv` | Training and all primary evaluations |
| Synthetic dataset (generated) | `data/raw/smartgrid_synthetic.csv` | Quick experiments and demos |
| MSU/ORNL Power System Attack Dataset: 78,377 samples, 128 PMU/relay/log features, 37 classes | `data/external/msu_ics_power/multiclass/` | External validation |
| UNSW-NB15 (optional) | anywhere, passed via `--csvs` | Network-only intrusion baseline |

**Generate a synthetic dataset:**

```bash
python -m src.generate_data --rows 50000 --out data/raw/smartgrid_synthetic.csv --seed 42
```

**Download the MSU/ORNL dataset:** get `multiclass.7z` from http://www.ece.uah.edu/~thm0009/icsdatasets/ (the `triple.7z` and `binaryAllNaturalPlusNormalVsAttacks.7z` variants are optional). Extract each variant into its own folder, for example:

```text
data/external/msu_ics_power/multiclass/   # 15 .arff files (data1 … data15)
data/external/msu_ics_power/triple/
data/external/msu_ics_power/binary/
```

---

## 5. Configuration

All settings live in `config/config.yaml`:

| Key | Meaning |
|---|---|
| `seed` | Global random seed (42) |
| `default_model` | Model used by training and the API when none is requested (`lightgbm`) |
| `features` | Ordered list of the 18 input features |
| `model` | Hyperparameters of the model trained by `src.train` |
| `model_variants` | Hyperparameters of all four models for the cross-validation scripts |
| `train` | `test_size`, `stratify`, `shuffle` for the train/test split |
| `thresholds.alert_probability` | Minimum confidence for an automated prevention action (0.7) |

Alternative single-model configurations are provided in `config/randomforestconfig.txt`, `config/xgboostconfig.txt`, and `config/LightGBMconfig.txt`.

---

## 6. Quick start

```bash
# 1. Train the default model
python -m src.train --data data/external/enhanced_realistic_smartgrid.csv

# 2. Start the API
python -m src.cli serve --port 8000

# 3. In another terminal: send a DoS-like sample
python test_dos_attack.py
```

Interactive API documentation is then available at http://127.0.0.1:8000/docs.

---

## 7. Training and prediction

**Train**

```bash
python -m src.train --data data/external/enhanced_realistic_smartgrid.csv \
                    --config config/config.yaml --models_dir models --artifacts_dir artifacts
```

- The data is split into train and test sets **first**. Minority-class oversampling is then applied to the training split only, so no duplicated row can appear in both sets.
- The model, `StandardScaler`, and `LabelEncoder` are saved to `models/` (both untyped and typed, e.g. `model_lightgbm.joblib`). Metrics are saved to `artifacts/metrics.json`, and figures (attack distribution, feature correlations, model performance, feature analysis) to `artifacts/visualizations/`.
- By default, training is only allowed on datasets under `data/external/` or with "real"/"realistic" in the path. Pass `--allow-synthetic` to train on a generated dataset.
- To train a different model, set `model.type` / `default_model` in the config (e.g. `random_forest`, `xgboost`, `lightgbm`, `catboost`).

**Predict**

```bash
# Batch prediction from a CSV file (adds prediction and prediction_probability columns)
python -m src.predict --csv data/raw/smartgrid_synthetic.csv --out predictions.csv

# Single row, choosing the model and writing to the audit log
python -m src.predict --row '{"voltage_v": 228.5, "current_a": 9.8, ...}' --model_type lightgbm --audit
```

**UNSW-NB15 (optional)**

```bash
python -m src.train_unsw --csvs UNSW_NB15_training-set.csv UNSW_NB15_testing-set.csv
```

---

## 8. REST API

**Start the server**

```bash
python -m src.cli serve --host 0.0.0.0 --port 8000     # or: python scripts/start_api.py
```

**Docker**

```bash
docker build -t smartgrid-ids .
docker run -p 8000:8000 smartgrid-ids
```

The Docker image copies the `models/` folder, so train a model before building.

**Endpoints**

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | HTML landing page with usage notes |
| `/health` | GET | Service status, API version, default model, alert threshold, available models |
| `/models` | GET | List of trained models found in `models/` |
| `/predict` | POST | Predicted class and probability for each telemetry row |
| `/prevent` | POST | Prediction plus the recommended prevention action for each row |

Both POST endpoints accept `{"rows": [ { <18 telemetry fields> }, ... ]}`. To choose a model for one request, add `?model_type=xgboost` or the header `X-Model-Type: xgboost`. Otherwise `default_model` is used.

**Example: `/predict`**

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "rows": [{
    "voltage_v": 228.5, "current_a": 9.8, "frequency_hz": 50.01, "power_factor": 0.96,
    "active_power_kw": 2.15, "reactive_power_kvar": 0.6, "thd_percent": 2.5, "breaker_closed": 1,
    "packet_rate": 110, "packet_error_rate": 0.004, "substation_temp_c": 27,
    "transformer_oil_temp_c": 38, "line_load_percent": 60, "tap_position": 16,
    "phase_imbalance_percent": 1.2, "feeder_voltage_v": 229, "feeder_current_a": 9.9, "demand_kw": 2.1
  }]
}'
```

```json
{
  "predictions": [
    { "label": "normal", "probability": 0.99999, "proba_available": true, "model_type": "lightgbm" }
  ]
}
```

**Example: `/prevent`** (same row with `"packet_rate": 800, "packet_error_rate": 0.15`)

```json
{
  "actions": [
    {
      "should_trip_breaker": false,
      "should_rate_limit_network": true,
      "reason": "High-confidence DoS detected: recommend rate limiting",
      "predicted_label": "dos",
      "probability": 0.99999,
      "model_type": "lightgbm"
    }
  ]
}
```

**Audit log**: every `/predict` and `/prevent` call is recorded in `logs/audit.db` (timestamp, model, input features, prediction, probability, action). Inspect it with:

```bash
python view_audit.py
```

---

## 9. Prevention policy

The policy (`src/api.py::compute_preventive_action`) is a deterministic mapping from the predicted class and its probability to an action. It acts only when the probability reaches `thresholds.alert_probability` (default **0.7**). Below that threshold, no automated action is taken.

| Predicted class | Action (probability ≥ 0.7) |
|---|---|
| `normal` | None |
| `dos` | Rate-limit network traffic |
| `tamper` | Trip / isolate breaker (operator review) |
| `overload` | Trip relay / load shedding (operator review) |
| `fdi` | Escalate to operator; no automated action |

The policy returns recommendations. It does not actuate breakers or network equipment itself, so connecting it to field devices is left to the integrating SCADA/EMS system.

---

## 10. Explainability

- **SHAP explanations** for a trained model (plots and JSON in `artifacts/explanations/`):
  ```bash
  python -m src.explain --input <rows.csv>
  ```
- **Feature-group ablation** measures how much each domain contributes by removing it and re-running cross-validation:
  ```bash
  python -m scripts.ablation_study
  ```
- **External dataset**: `scripts/msu_shap_analysis.py` and `scripts/msu_ablation_study.py` run the same analyses on MSU/ORNL. They report a global mean |SHAP| ranking and a group-level breakdown (voltage phasors, current phasors, impedance, frequency, relay status, logs).

---

## 11. Evaluation and experiments

All scripts are run from the project root and use seed 42.

| Purpose | Command | Output |
|---|---|---|
| 5-fold CV of all four models + Wilcoxon tests | `python -m scripts.cv_evaluation` | `artifacts/cv_results.json`, `model_cv_summary.json`, `statistical_significance.json` |
| Repeated CV (10 seeds × 5 folds) + prevention-policy evaluation | `python run_repeated_cv_and_policy_eval.py` | `results/repeated_cv_and_policy_eval.json` |
| Corrected resampled t-test on the repeated CV | `python scripts/corrected_resampled_ttest.py` | `results/corrected_ttest.json` |
| Feature-group ablation | `python -m scripts.ablation_study` | `artifacts/ablation_study_results.json` |
| ROC / AUC | `python -m scripts.roc_auc_analysis` | `artifacts/roc_auc_scores.json` |
| Robustness (noise, missing data, distribution shift) | `python -m scripts.noise_robustness_test` | `artifacts/robustness_results.json` |
| Latency and throughput | `python -m scripts.latency_benchmark --model lightgbm --n_runs 200` | `artifacts/latency_benchmark_results.json` |
| Attack simulations | `python -m scripts.dos_attack` (also `simulate_fdi`, `simulate_overload`, `simulate_tamper`) | `artifacts/visualizations/` |
| External validation, MSU/ORNL | `python -m scripts.msu_ics_dataset_eval --data_dir data/external/msu_ics_power/multiclass` | `artifacts/msu_external_validation_results_<variant>.json` |
| SHAP on MSU/ORNL | `python -m scripts.msu_shap_analysis --data_dir data/external/msu_ics_power/multiclass` | `results/msu_shap_multiclass.json` |
| Ablation on MSU/ORNL | `python -m scripts.msu_ablation_study --data_dir data/external/msu_ics_power/multiclass` | `results/msu_ablation_multiclass.json` |

`scripts/msu_ics_dataset_eval.py` can be run once per MSU/ORNL variant (`multiclass`, `triple`, `binary`); output files are tagged with the folder name.

---

## 12. Results

All numbers below were produced by the scripts in Section 11; the raw outputs are in `results/`.

**Detection: primary dataset (30,000 samples, leakage-safe stratified CV)**

| Model | Macro-F1, 5-fold | Accuracy, 5-fold | Macro-F1, 10 × 5-fold |
|---|---|---|---|
| LightGBM | 0.9285 ± 0.0030 | 96.73% ± 0.10% | 0.9300 ± 0.0045 |
| XGBoost | 0.9266 ± 0.0042 | 96.62% ± 0.15% | 0.9275 ± 0.0046 |
| Random Forest | 0.9264 ± 0.0025 | 96.59% ± 0.09% | 0.9275 ± 0.0042 |
| CatBoost | 0.8933 ± 0.0053 | 94.62% ± 0.31% | 0.8947 ± 0.0047 |

LightGBM has the highest mean, but its lead over Random Forest and XGBoost is small (0.0024 Macro-F1). It is not significant under the corrected resampled t-test (p = 0.105 and p = 0.093), so the three are practically equivalent. CatBoost is significantly worse (p < 0.001).

**Prevention policy** (30,000 out-of-fold LightGBM predictions passed through the deployed policy, threshold 0.7)

| Metric | Value |
|---|---|
| Response accuracy (exact match with the intended action) | 98.1% |
| False-block rate on normal traffic | 0.44% (102 of 23,427) |
| Missed-response rate (DoS / tamper / overload with no action) | 8.7% (417 of 4,811) |
| Policy latency per decision | 1.21 μs mean, 2.13 μs p99 |

**External validation: MSU/ORNL (37-class, LightGBM, 5-fold CV)**

- Macro-F1: **0.8307 ± 0.0022**.
- About 85% of the SHAP attribution goes to PMU voltage and current phasors, and about 1.3% to the relay, control-panel, and Snort log signals.
- In the ablation, removing the current or voltage phasors lowers Macro-F1 by 0.061 and 0.047. Removing all 16 cyber-side columns lowers it by only 0.004.

---

## 13. Reproducibility

- All randomness is seeded (`seed: 42`; the repeated CV uses seeds 42–51).
- The cross-validation scripts fit the scaler and apply oversampling **inside each training fold**, never on the full dataset.
- The repeated-CV, SHAP, and ablation scripts first re-run the corresponding baseline and print a reproduction check. For example, seed 42 of the repeated CV reproduces the 5-fold results above exactly.
- Every JSON file in `results/` is accompanied by the console log of the run that produced it.

---

## 14. Testing and demo clients

With the API running:

```bash
python tests/test_api.py        # health, predict and prevent endpoint checks
python test_normal.py           # normal-operation sample
python test_dos_attack.py       # DoS sample
python compare_models.py        # same sample through every available model
python scripts/live_stream.py   # live streaming demo
python scripts/simulate_attack_stream.py
```

---

## 15. Limitations

- The primary dataset is synthetic, but physically grounded. Performance on the real MSU/ORNL data is lower and is the more realistic estimate for real telemetry.
- The model is robust to sensor noise and missing values, but sensitive to systematic sensor drift (distribution shift). Deployments should include drift monitoring and periodic retraining.
- The prevention policy has been evaluated offline on cross-validated predictions only. It has not been validated in closed-loop operation (actual breaker actuation, rate-limit enforcement, or operator response).
- Models trained on one grid's data should be retrained and re-validated before use on another grid.

---

## 16. Troubleshooting

| Problem | Fix |
|---|---|
| `No --data provided and no default real dataset found` | Place the dataset at `data/external/enhanced_realistic_smartgrid.csv` or pass `--data`. |
| Training refuses a generated dataset | Add `--allow-synthetic`. |
| API reports `model_status: not_loaded` | Train a model first (`python -m src.train ...`) so that `models/model.joblib` exists. |
| `catboost not installed` | `pip install catboost` (it is listed in `requirements.txt`). |
| MSU/ORNL script finds no files | Check that the extracted `.arff`/`.csv` files are directly inside the folder passed to `--data_dir`. |
| Connection refused from test clients | Start the API first (`python -m src.cli serve`). |
