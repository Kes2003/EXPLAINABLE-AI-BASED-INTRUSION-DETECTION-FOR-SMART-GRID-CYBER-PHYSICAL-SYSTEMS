# scripts/simulate_attack_stream.py
import time, requests, csv, os
from copy import deepcopy

API_URL = "http://127.0.0.1:8000/predict"  # change if needed
OUT_CSV = "artifacts/demo_stream.csv"
SLEEP = 0.25  # 250ms between posts to simulate streaming

# Baseline normal row (match keys used in your test_normal.py)
BASE_ROW = {
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
    "demand_kw": 2.4
}

def inject_dos(row, intensity=6):
    r = deepcopy(row)
    r["packet_rate"] = int(r["packet_rate"] * intensity)
    r["packet_error_rate"] = min(1.0, r["packet_error_rate"] + 0.01 * intensity)
    r["frequency_hz"] += 0.05  # small jitter
    return r

def inject_fdi(row, pct_change=0.2):
    r = deepcopy(row)
    # change voltage and reactive power artificially
    r["voltage_v"] = r["voltage_v"] * (1 - pct_change)
    r["reactive_power_kvar"] = r["reactive_power_kvar"] * (1 + pct_change)
    return r

def inject_tamper(row):
    r = deepcopy(row)
    r["breaker_closed"] = 0 if r["breaker_closed"] == 1 else 1
    r["tap_position"] = max(0, r["tap_position"] - 5)
    return r

def post_row(row):
    payload = {"rows": [row]}
    try:
        r = requests.post(API_URL, json=payload, timeout=5)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def main():
    os.makedirs("artifacts", exist_ok=True)
    with open(OUT_CSV, mode="w", newline="") as f:
        writer = csv.writer(f)
        header = ["t", "scenario", "voltage_v", "packet_rate", "pred_label", "prob"]
        writer.writerow(header)

        total_steps = 220
        for t in range(total_steps):
            # normal baseline
            row = deepcopy(BASE_ROW)

            # Inject attack windows
            if 60 <= t < 100:
                row = inject_dos(row)
                scenario = "dos"
            elif 120 <= t < 150:
                row = inject_fdi(row)
                scenario = "fdi"
            elif 170 <= t < 190:
                row = inject_tamper(row)
                scenario = "tamper"
            else:
                scenario = "normal"

            resp = post_row(row)
            # parse response: expected something like {'label': 'dos', 'prob': 0.95}
            ppred_label = (
    resp.get("prediction")
    or resp.get("label")
    or resp.get("predicted_label")
    or resp.get("class")
    or "error"
)
            prob = (
    resp.get("confidence")
    or resp.get("probability")
    or resp.get("prob")
    or resp.get("score")
    or 0.0
)

if __name__ == "__main__":
    main()
