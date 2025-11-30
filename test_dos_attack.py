import requests
import json

dos_attack_data = {
    "rows": [{
        "voltage_v": 227.0,
        "current_a": 9.7,
        "frequency_hz": 50.0,
        "power_factor": 0.95,
        "active_power_kw": 2.0,
        "reactive_power_kvar": 0.5,
        "thd_percent": 3.0,
        "breaker_closed": 1,
        "packet_rate": 800.0,
        "packet_error_rate": 0.15,
        "substation_temp_c": 29,
        "transformer_oil_temp_c": 41,
        "line_load_percent": 58,
        "tap_position": 10,
        "phase_imbalance_percent": 1.5,
        "feeder_voltage_v": 227,
        "feeder_current_a": 10.1,
        "demand_kw": 2.35
    }]
}

print("=" * 80)
print("Testing DoS ATTACK Scenario")
print("=" * 80)

response = requests.post("http://127.0.0.1:8000/predict", json=dos_attack_data)
print("\n/predict Response:")
print(json.dumps(response.json(), indent=2))

response = requests.post("http://127.0.0.1:8000/prevent", json=dos_attack_data)
print("\n/prevent Response:")
print(json.dumps(response.json(), indent=2))