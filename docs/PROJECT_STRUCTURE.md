# Project Structure

## Complete Directory Organization

```
AI-Enabled Intrusion Detection for Smart Grids/
│
├── 📁 src/                          # Source code
│   ├── __init__.py
│   ├── api.py                      # FastAPI REST API server
│   ├── cli.py                      # Command-line interface
│   ├── generate_data.py            # Synthetic data generation
│   ├── train.py                    # Model training pipeline
│   ├── train_unsw.py               # UNSW-NB15 dataset training
│   ├── predict.py                  # Prediction CLI tool
│   ├── preprocess.py               # Data preprocessing
│   ├── model.py                    # Model utilities
│   ├── utils.py                    # Utility functions
│   └── 📁 datasets/                # Dataset adapters
│       ├── __init__.py
│       └── unsw_nb15.py            # UNSW-NB15 dataset loader
│
├── 📁 data/                         # Data directory
│   ├── 📁 raw/                     # Raw datasets
│   │   └── smartgrid_synthetic.csv
│   ├── 📁 processed/              # Processed datasets
│   └── 📁 external/                # External/real datasets
│       └── enhanced_realistic_smartgrid.csv
│
├── 📁 models/                       # Trained models
│   ├── model.joblib                # Trained RandomForest model
│   ├── scaler.joblib               # Feature scaler
│   ├── label_encoder.joblib        # Label encoder
│   └── features.json               # Feature configuration
│
├── 📁 artifacts/                    # Training artifacts
│   ├── metrics.json                 # Performance metrics
│   ├── last_model.joblib           # Model checkpoint
│   └── 📁 visualizations/          # Generated charts
│       ├── 01_attack_distribution.png
│       ├── 02_feature_correlations.png
│       ├── 03_model_performance.png
│       ├── 04_feature_analysis.png
│       ├── 05_smart_grid_architecture.png
│       └── 06_project_summary_fixed.png
│
├── 📁 config/                       # Configuration files
│   └── config.yaml                 # Model and training config
│
├── 📁 scripts/                      # Utility scripts
│   └── start_api.py                # API server launcher
│
├── 📁 tests/                        # Test files
│   └── test_api.py                 # API endpoint tests
│
├── 📁 docs/                         # Documentation
│   └── PROJECT_STRUCTURE.md        # This file
│
├── 📁 logs/                         # Application logs
│
├── 📄 requirements.txt             # Python dependencies
├── 📄 README.md                    # Main documentation
├── 📄 Dockerfile                   # Docker deployment
└── 📄 .gitignore                   # Git ignore rules
```

## Directory Purposes

### `src/`

Core application source code. All Python modules for data generation, training, prediction, and API.

### `data/`

- **`raw/`**: Original, unprocessed datasets
- **`processed/`**: Cleaned and preprocessed datasets ready for training
- **`external/`**: Real-world datasets from external sources

### `models/`

Persisted trained models, preprocessors, and model configuration files.

### `artifacts/`

Training outputs including metrics, model checkpoints, and generated visualizations.

### `config/`

Configuration files for models, training parameters, and system settings.

### `scripts/`

Utility scripts for deployment, automation, and system management.

### `tests/`

Test files for validating functionality and API endpoints.

### `docs/`

Project documentation, guides, and reference materials.

### `logs/`

Application logs and runtime information.

## File Naming Conventions

- **Python files**: `snake_case.py`
- **Data files**: `descriptive_name.csv`
- **Config files**: `config.yaml`, `settings.json`
- **Model files**: `model.joblib`, `scaler.joblib`
- **Documentation**: `UPPERCASE.md` for main docs, `snake_case.md` for specific topics

## Path References

When referencing files in code:

- Use relative paths from project root
- For data: `data/raw/`, `data/processed/`, `data/external/`
- For config: `config/config.yaml`
- For models: `models/`
- For artifacts: `artifacts/`
