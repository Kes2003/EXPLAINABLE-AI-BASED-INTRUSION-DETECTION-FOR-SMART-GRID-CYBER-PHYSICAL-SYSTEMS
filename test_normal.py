import requests
import json

normal_data = {
    "rows": [{
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
    }]
}

print("=" * 80)
print("Testing NORMAL Operation")
print("=" * 80)

response = requests.post("http://127.0.0.1:8000/predict", json=normal_data)
print("\n/predict Response:")
print(json.dumps(response.json(), indent=2))

response = requests.post("http://127.0.0.1:8000/prevent", json=normal_data)
print("\n/prevent Response:")
print(json.dumps(response.json(), indent=2))