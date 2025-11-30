#!/usr/bin/env python3
"""Simple script to test the API endpoints"""

import requests
import json
import sys

BASE_URL = "http://127.0.0.1:8000"

def test_health():
    """Test health endpoint"""
    print("Testing /health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        print("[ERROR] Connection refused. Is the API server running?")
        return False
    except Exception as e:
        print(f"[ERROR] {e}")
        return False

def test_predict():
    """Test predict endpoint"""
    print("\nTesting /predict endpoint...")
    test_data = {
        "rows": [{
            "voltage_v": 228.5,
            "current_a": 9.8,
            "frequency_hz": 50.01,
            "power_factor": 0.96,
            "active_power_kw": 2.15,
            "reactive_power_kvar": 0.6,
            "thd_percent": 2.5,
            "breaker_closed": 1,
            "packet_rate": 110,
            "packet_error_rate": 0.004,
            "substation_temp_c": 27,
            "transformer_oil_temp_c": 38,
            "line_load_percent": 62,
            "tap_position": 10,
            "phase_imbalance_percent": 1.8,
            "feeder_voltage_v": 229,
            "feeder_current_a": 10.2,
            "demand_kw": 2.3
        }]
    }
    try:
        response = requests.post(f"{BASE_URL}/predict", json=test_data, timeout=10)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {json.dumps(response.json(), indent=2)}")
            return True
        else:
            print(f"[ERROR] {response.text}")
            return False
    except Exception as e:
        print(f"[ERROR] {e}")
        return False

def test_prevent():
    """Test prevent endpoint"""
    print("\nTesting /prevent endpoint...")
    test_data = {
        "rows": [{
            "voltage_v": 228.5,
            "current_a": 9.8,
            "frequency_hz": 50.01,
            "power_factor": 0.96,
            "active_power_kw": 2.15,
            "reactive_power_kvar": 0.6,
            "thd_percent": 2.5,
            "breaker_closed": 1,
            "packet_rate": 250,  # High packet rate (potential DoS)
            "packet_error_rate": 0.015,  # High error rate
            "substation_temp_c": 27,
            "transformer_oil_temp_c": 38,
            "line_load_percent": 62,
            "tap_position": 10,
            "phase_imbalance_percent": 1.8,
            "feeder_voltage_v": 229,
            "feeder_current_a": 10.2,
            "demand_kw": 2.3
        }]
    }
    try:
        response = requests.post(f"{BASE_URL}/prevent", json=test_data, timeout=10)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {json.dumps(response.json(), indent=2)}")
            return True
        else:
            print(f"[ERROR] {response.text}")
            return False
    except Exception as e:
        print(f"[ERROR] {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Smart Grid IDS API Test")
    print("=" * 60)
    print(f"\nTesting API at: {BASE_URL}")
    print("Make sure the API server is running.")
    print("If not, start it with: python scripts/start_api.py\n")
    
    health_ok = test_health()
    if health_ok:
        predict_ok = test_predict()
        prevent_ok = test_prevent()
        
        print("\n" + "=" * 60)
        print("Test Summary:")
        print("=" * 60)
        print(f"Health Check: {'[PASS]' if health_ok else '[FAIL]'}")
        print(f"Predict Endpoint: {'[PASS]' if predict_ok else '[FAIL]'}")
        print(f"Prevent Endpoint: {'[PASS]' if prevent_ok else '[FAIL]'}")
        
        if health_ok and predict_ok and prevent_ok:
            print("\n[SUCCESS] All tests passed!")
            sys.exit(0)
        else:
            print("\n[WARNING] Some tests failed!")
            sys.exit(1)
    else:
        print("\n[ERROR] API server is not running or not accessible!")
        print("Please start the server first with: python scripts/start_api.py")
        sys.exit(1)