from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from .utils import ensure_dir
except ImportError:  # when executed as a script
    from utils import ensure_dir


def simulate_smartgrid(rows: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Normal operating conditions
    voltage = rng.normal(230, 5, rows)
    current = rng.normal(10, 3, rows)
    frequency = rng.normal(50.0, 0.05, rows)
    power_factor = rng.beta(20, 2, rows)  # near 1.0

    active_power = voltage * current * power_factor / 1000.0
    reactive_power = voltage * current * np.sqrt(1 - np.clip(power_factor, 0, 1) ** 2) / 1000.0

    thd = rng.normal(3.0, 1.0, rows)
    breaker_closed = rng.integers(0, 2, rows)

    packet_rate = rng.normal(100, 15, rows)
    packet_error_rate = np.abs(rng.normal(0.002, 0.002, rows))

    substation_temp = rng.normal(25, 5, rows)
    transformer_oil_temp = substation_temp + rng.normal(10, 2, rows)

    line_load = rng.normal(60, 15, rows)
    tap_position = rng.integers(0, 33, rows)

    phase_imbalance = np.abs(rng.normal(1.5, 0.8, rows))

    feeder_voltage = voltage + rng.normal(0, 3, rows)
    feeder_current = current + rng.normal(0, 1, rows)

    demand_kw = active_power + rng.normal(0, 2, rows)

    df = pd.DataFrame(
        {
            "voltage_v": voltage,
            "current_a": current,
            "frequency_hz": frequency,
            "power_factor": power_factor,
            "active_power_kw": active_power,
            "reactive_power_kvar": reactive_power,
            "thd_percent": thd,
            "breaker_closed": breaker_closed,
            "packet_rate": packet_rate,
            "packet_error_rate": packet_error_rate,
            "substation_temp_c": substation_temp,
            "transformer_oil_temp_c": transformer_oil_temp,
            "line_load_percent": line_load,
            "tap_position": tap_position,
            "phase_imbalance_percent": phase_imbalance,
            "feeder_voltage_v": feeder_voltage,
            "feeder_current_a": feeder_current,
            "demand_kw": demand_kw,
        }
    )

    # Inject attacks/anomalies
    labels = np.array(["normal"] * rows, dtype=object)

    # 1) DoS/network anomalies: high packet_rate & error_rate
    idx = rng.choice(rows, size=int(0.08 * rows), replace=False)
    df.loc[idx, "packet_rate"] += rng.normal(120, 30, len(idx))
    df.loc[idx, "packet_error_rate"] += np.abs(rng.normal(0.02, 0.01, len(idx)))
    labels[idx] = "dos"

    # 2) False data injection: tweak electrical variables inconsistently
    idx = rng.choice(rows, size=int(0.07 * rows), replace=False)
    df.loc[idx, "power_factor"] = np.clip(df.loc[idx, "power_factor"] - rng.normal(0.2, 0.05, len(idx)), 0.1, 1.0)
    df.loc[idx, "voltage_v"] += rng.normal(15, 5, len(idx))
    df.loc[idx, "current_a"] -= rng.normal(4, 1.5, len(idx))
    df.loc[idx, "thd_percent"] += rng.normal(3, 1, len(idx))
    labels[idx] = "fdi"

    # 3) Physical tampering: breaker toggling, phase imbalance spikes
    idx = rng.choice(rows, size=int(0.06 * rows), replace=False)
    df.loc[idx, "breaker_closed"] = 1 - df.loc[idx, "breaker_closed"].values
    df.loc[idx, "phase_imbalance_percent"] += rng.normal(5, 2, len(idx))
    labels[idx] = "tamper"

    # 4) Thermal overload: temps and load rise
    idx = rng.choice(rows, size=int(0.05 * rows), replace=False)
    df.loc[idx, "substation_temp_c"] += rng.normal(12, 3, len(idx))
    df.loc[idx, "transformer_oil_temp_c"] += rng.normal(15, 3, len(idx))
    df.loc[idx, "line_load_percent"] += rng.normal(25, 8, len(idx))
    labels[idx] = "overload"

    # Some noise/clipping
    df = df.clip(lower=df.quantile(0.001), upper=df.quantile(0.999), axis=1)

    df["label"] = labels
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic smart grid dataset")
    parser.add_argument("--rows", type=int, default=50000, help="Number of rows to generate")
    parser.add_argument("--out", type=str, default="data/raw/smartgrid_synthetic.csv", help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    print(f"Generating {args.rows:,} rows of synthetic smart grid data...")
    df = simulate_smartgrid(args.rows, args.seed)
    
    out_path = Path(args.out)
    ensure_dir(out_path.parent)
    df.to_csv(out_path, index=False)
    
    print(f"Successfully wrote {len(df):,} rows to {out_path}")
    print(f"\nLabel distribution:")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()