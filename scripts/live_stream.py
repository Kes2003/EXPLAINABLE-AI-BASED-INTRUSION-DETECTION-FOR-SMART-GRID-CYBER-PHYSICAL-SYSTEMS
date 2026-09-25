"""
Live Smart Grid Intrusion Detection Demo
-----------------------------------------
✔ Real model
✔ Real scaler
✔ Real features
✔ Continuous stream
✔ Terminal table
✔ CSV export
"""

import json
import time
import random
import joblib
import pandas as pd
from pathlib import Path


# ======================================================
# PATHS
# ======================================================
ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "model_lightgbm.joblib"
SCALER_PATH = ROOT / "models" / "scaler_lightgbm.joblib"
ENCODER_PATH = ROOT / "models" / "label_encoder_lightgbm.joblib"
FEATURE_PATH = ROOT / "models" / "features.json"


# ======================================================
# LOAD ARTIFACTS
# ======================================================
print("Loading model artifacts...")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
encoder = joblib.load(ENCODER_PATH)

with open(FEATURE_PATH) as f:
    data = json.load(f)

# handle dict or list
if isinstance(data, dict):
    FEATURES = data.get("features", list(data.values())[0])
else:
    FEATURES = data

print("✅ System Ready")


# ======================================================
# SIMPLE SCENARIO GENERATOR
# ======================================================
def generate_row(scenario):
    """Generate one telemetry sample depending on scenario"""

    row = {}

    for f in FEATURES:
        row[f] = random.uniform(0, 1)

    # introduce obvious pattern changes
    if scenario == "dos":
        if "packet_rate" in row:
            row["packet_rate"] = random.uniform(500, 900)

    if scenario == "fdi":
        if "voltage_v" in row:
            row["voltage_v"] = random.uniform(150, 180)

    if scenario == "overload":
        if "line_load_percent" in row:
            row["line_load_percent"] = random.uniform(85, 120)

    if scenario == "tamper":
        if "breaker_closed" in row:
            row["breaker_closed"] = 0

    return row


# ======================================================
# LIVE STREAM
# ======================================================
print("\nStarting live smart grid stream...\n")

results = []

TOTAL_STEPS = 80

for t in range(TOTAL_STEPS):

    # scenario scheduling
    if t < 20:
        actual = "normal"
    elif t < 35:
        actual = "dos"
    elif t < 50:
        actual = "fdi"
    elif t < 65:
        actual = "overload"
    else:
        actual = "tamper"

    row = generate_row(actual)

    df = pd.DataFrame([row])[FEATURES]
    X = scaler.transform(df)

    pred_id = model.predict(X)[0]
    prob = model.predict_proba(X).max()

    predicted = encoder.inverse_transform([pred_id])[0]

    # live print
    print(
        f"t={t:02d} | actual={actual:<8} | predicted={predicted:<8} | confidence={prob:.3f}"
    )

    # store for table
    results.append({
        "time": t,
        "actual": actual,
        "predicted": predicted,
        "confidence": round(float(prob), 3)
    })

    time.sleep(0.2)  # makes demo readable


# ======================================================
# TABLE OUTPUT
# ======================================================
df_results = pd.DataFrame(results)

print("\n📋 STREAM SUMMARY (first rows)\n")
print(df_results.head(20))


# ======================================================
# SAVE CSV
# ======================================================
out_path = ROOT / "artifacts" / "live_stream_results.csv"
df_results.to_csv(out_path, index=False)

print(f"\n✅ Results saved to: {out_path}")
print("\n🎉 Demo Complete")

