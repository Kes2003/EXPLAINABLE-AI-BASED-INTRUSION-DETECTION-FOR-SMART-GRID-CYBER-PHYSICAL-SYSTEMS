# AI-Enabled Intrusion Detection for Smart Grids

This repository contains a complete, configurable machine learning pipeline for **AI-enabled intrusion detection and prevention in smart grids**. The system supports:

- Synthetic cyber–physical smart grid data (with labeled attack scenarios)
- Realistic external datasets
- A structured **training pipeline**
- A production-style **REST API** (FastAPI + Uvicorn)
- **Automated prevention recommendations** based on model predictions and confidence
- **Research-grade metrics and visualizations** for analysis and reporting

The project is suitable for:

- Academic coursework / internships
- Research prototypes for cyber-physical security
- Demonstrations of end-to-end ML systems (data → training → API → prevention)

---

## 1. System Overview

The system models a **smart grid substation / feeder** with telemetry such as:

- Electrical variables: `voltage_v`, `current_a`, `frequency_hz`, `power_factor`, `active_power_kw`, `reactive_power_kvar`
- Power quality: `thd_percent`, `phase_imbalance_percent`
- Network behavior: `packet_rate`, `packet_error_rate`
- Asset health: `substation_temp_c`, `transformer_oil_temp_c`, `line_load_percent`, `tap_position`
- Downstream measurements: `feeder_voltage_v`, `feeder_current_a`, `demand_kw`

Attack classes:

- `normal` – nominal operation
- `dos` – Denial-of-Service (network flooding)
- `fdi` – False Data Injection (manipulated sensor values)
- `tamper` – physical tampering / breaker toggling
- `overload` – thermal overload / high loading

A **RandomForest** classifier is trained to distinguish between these classes. At inference time, the API returns both **predicted label** and **probability**, and a simple rule-based prevention policy uses the probability to decide whether to automatically act or escalate to a human.

---

## 2. Project Structure

(Short version; see `docs/PROJECT_STRUCTURE.md` for full detail.)

```text
AI-Enabled Intrusion Detection for Smart Grids/
│
├── src/
│   ├── api.py                # FastAPI REST API (v1.1.0, with probabilities and prevention)
│   ├── cli.py                # CLI interface (start API)
│   ├── generate_data.py      # Synthetic smart grid data generator
│   ├── train.py              # Training pipeline (metrics + visualizations)
│   ├── train_unsw.py         # UNSW-NB15 training pipeline
│   ├── predict.py            # Batch prediction CLI (CSV / single row)
│   ├── preprocess.py         # Preprocessing (scaling, label encoding)
│   ├── model.py              # Model utilities + evaluation
│   ├── artifacts.py          # Centralized artifact (model + preprocessor) loader
│   ├── utils.py              # Helper utilities (paths, seeding)
│   └── datasets/
│       └── unsw_nb15.py      # UNSW-NB15 dataset adapter
│
├── data/
│   ├── raw/                  # Raw datasets
│   │   └── smartgrid_synthetic.csv
│   ├── processed/            # (optional) processed datasets
│   └── external/
│       └── enhanced_realistic_smartgrid.csv
│
├── models/
│   ├── model.joblib          # Trained model
│   ├── scaler.joblib         # StandardScaler
│   ├── label_encoder.joblib  # LabelEncoder for attack classes
│   └── features.json         # Feature list used during training
│
├── artifacts/
│   ├── metrics.json          # Classification report + confusion matrix
│   ├── last_model.joblib     # Last trained model checkpoint
│   └── visualizations/
│       ├── 01_attack_distribution.png
│       ├── 02_feature_correlations.png
│       ├── 03_model_performance.png
│       ├── 04_feature_analysis.png
│       ├── 05_smart_grid_architecture.png
│       └── 06_project_summary_fixed.png
│
├── config/
│   └── config.yaml           # Features, model hyperparameters, thresholds
│
├── scripts/
│   ├── start_api.py          # API launcher
│   └── setup.py              # Environment / structure checks
│
├── tests/
│   └── test_api.py           # Simple API tests
│
├── docs/
│   ├── PROJECT_STRUCTURE.md
│   ├── QUICK_REFERENCE.md
│   └── REORGANIZATION_SUMMARY.md
│
├── logs/                     # Runtime logs (e.g., logs/api.log)
├── requirements.txt
├── Dockerfile
├── README.md                 # This file
└── PROJECT_COMPLETE.md
