"""
data_generator.py
-----------------
Generates synthetic multi-sensor time-series data for three marine vessels.
Sensors modelled: engine RPM, coolant temperature, exhaust gas temperature,
fuel consumption rate, vibration (RMS), lube-oil pressure, and vessel speed.

Fault types injected at random windows:
  0 - Normal operation
  1 - Cooling-system failure  (coolant temp spike, slow rise)
  2 - Fuel-system degradation (fuel consumption drifts upward)
  3 - Bearing wear            (vibration rises, RPM becomes noisy)
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)

# ── config ────────────────────────────────────────────────────────────────────
VESSELS        = ["VS_Haida_Gwaii", "VS_Spirit_Sound", "VS_Pacific_Wren"]
DAYS           = 90          # simulation window
SAMPLE_MIN     = 10          # minutes between readings
FAULT_PROB     = 0.06        # probability that any given day starts a fault
FAULT_LEN_H    = (6, 36)     # fault duration range (hours)

# Normal operating ranges (Gaussian: mean, std)
NORMAL = {
    "rpm"          : (650, 30),
    "coolant_temp" : (82,  3),
    "egt"          : (380, 20),
    "fuel_lph"     : (85,  8),
    "vibration"    : (1.2, 0.2),
    "oil_pressure" : (4.5, 0.3),
    "speed_kn"     : (12,  1.5),
}


def _timestamps(days: int, freq_min: int) -> pd.DatetimeIndex:
    return pd.date_range("2025-01-01", periods=days * 24 * 60 // freq_min,
                         freq=f"{freq_min}min")


def _base_signals(n: int) -> dict:
    """Generate correlated normal-operation signals for n timesteps."""
    rpm        = RNG.normal(*NORMAL["rpm"],          n).clip(350, 950)
    rpm_norm   = (rpm - NORMAL["rpm"][0]) / NORMAL["rpm"][0]

    # fuel and speed loosely correlated with RPM
    fuel = NORMAL["fuel_lph"][0] + rpm_norm * 40 + RNG.normal(0, NORMAL["fuel_lph"][1], n)
    speed = NORMAL["speed_kn"][0] + rpm_norm * 3  + RNG.normal(0, NORMAL["speed_kn"][1], n)

    return {
        "rpm"          : rpm,
        "coolant_temp" : RNG.normal(*NORMAL["coolant_temp"], n).clip(60, 105),
        "egt"          : RNG.normal(*NORMAL["egt"],          n).clip(250, 550),
        "fuel_lph"     : fuel.clip(40, 200),
        "vibration"    : RNG.normal(*NORMAL["vibration"],    n).clip(0.3, 6.0),
        "oil_pressure" : RNG.normal(*NORMAL["oil_pressure"], n).clip(2.0, 8.0),
        "speed_kn"     : speed.clip(0, 22),
    }


def _inject_faults(df: pd.DataFrame) -> pd.DataFrame:
    """Mark fault windows and distort the relevant signals."""
    df = df.copy()
    df["fault_label"] = 0

    n = len(df)
    steps_per_day = 24 * 60 // SAMPLE_MIN
    i = 0
    while i < n:
        if RNG.random() < FAULT_PROB:
            fault_type = RNG.integers(1, 4)   # 1, 2, or 3
            dur_steps  = int(RNG.integers(*[h * 60 // SAMPLE_MIN for h in FAULT_LEN_H]))
            end = min(i + dur_steps, n)
            window = slice(i, end)
            t = np.linspace(0, 1, end - i)

            if fault_type == 1:          # cooling failure – temp rises then plateaus
                df.loc[df.index[window], "coolant_temp"] += (t ** 0.5) * RNG.uniform(15, 35)
                df.loc[df.index[window], "egt"]          += t * RNG.uniform(30, 80)

            elif fault_type == 2:        # fuel degradation – consumption creeps up
                df.loc[df.index[window], "fuel_lph"] += t * RNG.uniform(20, 60)
                df.loc[df.index[window], "speed_kn"] -= t * RNG.uniform(1, 4)

            elif fault_type == 3:        # bearing wear – vibration spikes, RPM noise
                df.loc[df.index[window], "vibration"] += t * RNG.uniform(1.5, 4.0)
                noise = RNG.normal(0, 40, end - i) * t
                df.loc[df.index[window], "rpm"] += noise

            df.loc[df.index[window], "fault_label"] = fault_type
            i = end + steps_per_day      # gap before next fault
        else:
            i += steps_per_day

    # clip physical bounds after injection
    df["coolant_temp"] = df["coolant_temp"].clip(60, 120)
    df["egt"]          = df["egt"].clip(250, 650)
    df["fuel_lph"]     = df["fuel_lph"].clip(40, 250)
    df["vibration"]    = df["vibration"].clip(0.3, 8.0)
    df["rpm"]          = df["rpm"].clip(200, 1100)
    return df


def generate(output_dir: str | Path = "data") -> pd.DataFrame:
    """
    Generate sensor data for all vessels and save to CSV.

    Returns
    -------
    pd.DataFrame  – combined dataframe with a 'vessel_id' column
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    timestamps = _timestamps(DAYS, SAMPLE_MIN)
    n = len(timestamps)

    for vessel in VESSELS:
        signals = _base_signals(n)
        df = pd.DataFrame(signals, index=timestamps)
        df.index.name = "timestamp"
        df = _inject_faults(df)
        df["vessel_id"] = vessel

        # introduce ~1 % missing values (sensor dropout)
        for col in ["coolant_temp", "vibration", "oil_pressure"]:
            mask = RNG.random(n) < 0.01
            df.loc[df.index[mask], col] = np.nan

        frames.append(df)

    combined = pd.concat(frames).reset_index()
    combined.to_csv(output_dir / "sensor_data_raw.csv", index=False)
    print(f"[data_generator] Saved {len(combined):,} rows → {output_dir}/sensor_data_raw.csv")
    return combined


if __name__ == "__main__":
    generate()
