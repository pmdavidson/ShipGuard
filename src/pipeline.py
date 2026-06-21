"""
Data ingestion, cleaning, and preprocessing pipeline.

Steps:
  1. Load raw CSV
  2. Parse timestamps, sort by vessel + time
  3. Interpolate short sensor gaps (≤ 3 consecutive NaNs)
  4. Clip residual outliers (IQR fence)
  5. Return a clean DataFrame ready for feature engineering
"""

import pandas as pd
import numpy as np
from pathlib import Path

SENSOR_COLS = [
    "rpm", "coolant_temp", "egt", "fuel_lph",
    "vibration", "oil_pressure", "speed_kn",
]

# Load
def load_raw(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values(["vessel_id", "timestamp"]).reset_index(drop=True)
    
    print(f"[pipeline] Loaded {len(df):,} rows from {path}")
    return df


# Clean
def _interpolate_gaps(group: pd.DataFrame, limit: int = 3) -> pd.DataFrame:
    """Linear interpolation for short gaps; forward-fill for residuals."""
    group = group.copy()
    group[SENSOR_COLS] = (
        group[SENSOR_COLS]
        .interpolate(method="linear", limit=limit)
        .ffill()
        .bfill()
    )
    return group

def _clip_outliers(df: pd.DataFrame, k: float = 4.0) -> pd.DataFrame:
    """Clip values beyond k × IQR from the median (per-vessel per-column)."""
    df = df.copy()

    for col in SENSOR_COLS:
        q1  = df[col].quantile(0.25)
        q3  = df[col].quantile(0.75)
        iqr = q3 - q1
        lo  = q1 - k * iqr
        hi  = q3 + k * iqr
        df[col] = df[col].clip(lo, hi)
    return df

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pass: interpolation → outlier clipping."""
    parts = [_interpolate_gaps(grp) for _, grp in df.groupby("vessel_id")]
    df    = pd.concat(parts)
    parts = [_clip_outliers(grp) for _, grp in df.groupby("vessel_id")]
    df    = pd.concat(parts)
    missing_after = df[SENSOR_COLS].isna().sum().sum()

    print(f"[pipeline] Cleaning done. Remaining NaNs: {missing_after}")
    return df.reset_index(drop=True)


# Normalize
def compute_stats(df: pd.DataFrame) -> dict:
    """Compute mean/std from training data only (call before train/test split)."""
    return {
        col: {"mean": df[col].mean(), "std": df[col].std()}
        for col in SENSOR_COLS
    }

def normalise(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    df = df.copy()

    for col in SENSOR_COLS:
        df[col] = (df[col] - stats[col]["mean"]) / (stats[col]["std"] + 1e-8)
    return df


# Train / test split
def split_by_time(df: pd.DataFrame,
                  test_frac: float = 0.20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronological split per vessel (no data-leakage).
    Returns (train_df, test_df).
    """
    trains, tests = [], []

    for _, grp in df.groupby("vessel_id"):
        n       = len(grp)
        cutoff  = int(n * (1 - test_frac))
        trains.append(grp.iloc[:cutoff])
        tests.append(grp.iloc[cutoff:])
    train = pd.concat(trains).reset_index(drop=True)
    test  = pd.concat(tests).reset_index(drop=True)

    print(f"[pipeline] Train: {len(train):,} rows | Test: {len(test):,} rows")
    return train, test


# Convenience entry point
def run(raw_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Full pipeline: load → clean → split → normalise.

    Returns (train_df, test_df, stats)
    where train_df / test_df contain both raw sensor columns AND normalised
    ones (prefixed with 'z_') so visualisations can still use raw values.
    """
    df     = load_raw(raw_path)
    df     = clean(df)
    train, test = split_by_time(df)
    stats  = compute_stats(train)

    # add z-scored columns alongside raw
    for col in SENSOR_COLS:
        train[f"z_{col}"] = (train[col] - stats[col]["mean"]) / (stats[col]["std"] + 1e-8)
        test[f"z_{col}"]  = (test[col]  - stats[col]["mean"]) / (stats[col]["std"] + 1e-8)

    return train, test, stats


if __name__ == "__main__":
    run("data/sensor_data_raw.csv")
