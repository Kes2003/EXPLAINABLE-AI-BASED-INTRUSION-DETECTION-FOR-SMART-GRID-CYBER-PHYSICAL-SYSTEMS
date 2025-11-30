# Quick Reference Guide

## Common Commands

### Setup

```powershell
# Install dependencies
pip install -r requirements.txt

# Verify setup
python scripts/setup.py
```

### Data Generation

```powershell
# Generate synthetic dataset
python -m src.generate_data --rows 20000 --out data/raw/smartgrid_synthetic.csv
```

### Training

```powershell
# Train on synthetic data
python -m src.train --data data/raw/smartgrid_synthetic.csv --config config/config.yaml

# Train on real-world data
python -m src.train --data data/external/enhanced_realistic_smartgrid.csv --config config/config.yaml
```

### Prediction

```powershell
# Batch prediction
python -m src.predict --csv data/raw/smartgrid_synthetic.csv --out predictions.csv
```

### API Server

```powershell
# Start server
python scripts/start_api.py

# Test API
python tests/test_api.py
```

### Docker

```bash
docker build -t smartgrid-ids .
docker run -p 8000:8000 smartgrid-ids
```

## File Paths Reference

| Purpose        | Path                        |
| -------------- | --------------------------- |
| Raw Data       | `data/raw/`                 |
| Processed Data | `data/processed/`           |
| External Data  | `data/external/`            |
| Models         | `models/`                   |
| Artifacts      | `artifacts/`                |
| Visualizations | `artifacts/visualizations/` |
| Config         | `config/config.yaml`        |
| Logs           | `logs/`                     |

## API Endpoints

| Endpoint   | Method | Purpose                |
| ---------- | ------ | ---------------------- |
| `/`        | GET    | API information page   |
| `/health`  | GET    | Health check           |
| `/docs`    | GET    | Interactive API docs   |
| `/predict` | POST   | Predict intrusions     |
| `/prevent` | POST   | Get prevention actions |

## Attack Types

- `normal` - Normal operations
- `dos` - Denial-of-Service
- `fdi` - False Data Injection
- `tamper` - Physical tampering
- `overload` - Thermal overload
