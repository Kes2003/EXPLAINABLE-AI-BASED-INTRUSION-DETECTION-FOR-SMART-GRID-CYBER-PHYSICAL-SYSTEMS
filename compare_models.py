#!/usr/bin/env python3
"""Compare predictions from all available models"""

import requests
import json

# Test data - DoS attack scenario
test_data = {
    "rows": [{
        "voltage_v": 227.0,
        "current_a": 9.7,
        "frequency_hz": 50.0,
        "power_factor": 0.95,
        "active_power_kw": 2.0,
        "reactive_power_kvar": 0.5,
        "thd_percent": 3.0,
        "breaker_closed": 1,
        "packet_rate": 800.0,          # High (DoS indicator)
        "packet_error_rate": 0.15,     # High (DoS indicator)
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

BASE_URL = "http://127.0.0.1:8000"

def compare_models():
    """Compare predictions from all available models"""
    
    # Get available models
    try:
        response = requests.get(f"{BASE_URL}/models", timeout=5)
        available_models = response.json()["models"]
        print("=" * 80)
        print("Model Comparison - DoS Attack Scenario")
        print("=" * 80)
        print(f"Available models: {', '.join(available_models)}\n")
    except Exception as e:
        print(f"Error connecting to API: {e}")
        print("Make sure the API is running: python scripts/start_api.py")
        return
    
    results = []
    
    for model in available_models:
        print(f"Testing {model.upper()}...")
        try:
            response = requests.post(
                f"{BASE_URL}/predict",
                json=test_data,
                params={"model_type": model},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()["predictions"][0]
                results.append({
                    "model": model,
                    "label": result["label"],
                    "probability": result["probability"]
                })
                print(f"  ✅ Predicted: {result['label']} "
                      f"(confidence: {result['probability']:.4f})")
            else:
                print(f"  ❌ Error: {response.status_code}")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # Summary
    if results:
        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)
        print(f"{'Model':<20} {'Prediction':<15} {'Confidence':<15}")
        print("-" * 80)
        for r in results:
            print(f"{r['model']:<20} {r['label']:<15} {r['probability']:.4f}")
        
        # Find best prediction
        best = max(results, key=lambda x: x['probability'] or 0)
        print("-" * 80)
        print(f"Highest confidence: {best['model']} → {best['label']} "
              f"({best['probability']:.4f})")

if __name__ == "__main__":
    compare_models()