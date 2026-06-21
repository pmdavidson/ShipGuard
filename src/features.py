"""
Feature engineering for ship sensor time-series.

Derived features (computed per-vessel to avoid boundary leakage):
  - Rolling statistics: mean & std over 1 h, 6 h, 24 h windows
  - Rate of change: first difference (delta per timestep)
  - Lag features: t-1, t-6, t-12 timesteps (i.e. 10, 60, 120 min)
  - Cross-sensor ratios: diagnostically meaningful combinations
  - Remaining-Useful-Life (RUL) proxy: steps until next fault onset
"""

import numpy as np
import pandas as pd

SENSOR_COLS = [
    "rpm", "coolant_temp", "egt", "fuel_lph",
    "vibration", "oil_pressure", "speed_kn",
]

# window sizes in samples (10min intervals)
WINDOWS = {"1h": 6, "6h": 36, "24h": 144}
LAGS    = [1, 6, 12]          # in timesteps


def rolling_features(group: pd.DataFrame) -> pd.DataFrame:
    group = group.copy()
    for col in SENSOR_COLS:
        for label, w in WINDOWS.items():
            group[f"{col}_roll_mean_{label}"] = (
                group[col].rolling(w, min_periods=1).mean()
            )
            group[f"{col}_roll_std_{label}"] = (
                group[col].rolling(w, min_periods=1).std().fillna(0)
            )
    return group


def delta_features(group: pd.DataFrame) -> pd.DataFrame:
    group = group.copy()
    for col in SENSOR_COLS:
        group[f"{col}_delta"] = group[col].diff().fillna(0)
    return group


def lag_features(group: pd.DataFrame) -> pd.DataFrame:
    group = group.copy()
    for col in SENSOR_COLS:
        for lag in LAGS:
            group[f"{col}_lag{lag}"] = group[col].shift(lag).bfill()
    return group


def ratio_features(group: pd.DataFrame) -> pd.DataFrame:
    group = group.copy()
    eps = 1e-6
    # thermal efficiency proxy
    group["fuel_per_speed"] = group["fuel_lph"] / (group["speed_kn"] + eps)
    # engine thermal load
    group["egt_per_rpm"]    = group["egt"] / (group["rpm"] + eps)
    # vibration anomaly index
    group["vib_roll_ratio"] = (group["vibration"] / (group["vibration"].rolling(144, min_periods=1).mean() + eps))
    
    return group

"""
Assign a RUL (remaining useful life) label in timesteps.
RUL = 0 during a fault; before a fault onset it decreases linearly
from a cap of 144 steps (24 h) down to 1.
RUL = 144 when no upcoming fault is within the look-ahead window.
"""
def compute_rul(group: pd.DataFrame) -> pd.DataFrame:
    RUL_CAP = 144
    group = group.copy()
    labels = group["fault_label"].values
    n      = len(labels)
    rul    = np.full(n, RUL_CAP, dtype=float)

    i = 0
    while i < n:
        if labels[i] != 0:
            # find end of current fault
            j = i
            while j < n and labels[j] != 0:
                j += 1
            # mark fault window with RUL = 0
            rul[i:j] = 0
            # back-fill pre-fault ramp
            start = max(0, i - RUL_CAP)
            length = i - start
            if length > 0:
                rul[start:i] = np.minimum(rul[start:i],
                                           np.arange(length, 0, -1, dtype=float))
            i = j
        
        else:
            i += 1

    group["rul"] = rul
    return group


"""
Apply all feature engineering transforms per vessel,
then drop any residual NaNs.
"""
def build(df: pd.DataFrame) -> pd.DataFrame:
    transforms = [rolling_features, delta_features, lag_features,
                  ratio_features, compute_rul]

    frames = []
    for _, grp in df.groupby("vessel_id"):
        for fn in transforms:
            grp = fn(grp)
        frames.append(grp)

    out = pd.concat(frames).reset_index(drop=True)
    before = len(out)
    out = out.dropna()
    print(f"[features] Engineered {len(out.columns)} columns. "
          f"Dropped {before - len(out)} rows with NaNs.")
    return out


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return ML-ready feature column names (exclude meta & target columns)."""
    exclude = {"timestamp", "vessel_id", "fault_label", "rul"}
    return [c for c in df.columns if c not in exclude]


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data_generator import generate
    from src.pipeline import run as pipeline_run

    generate()
    train, test, _ = pipeline_run("data/sensor_data_raw.csv")
    train_fe = build(train)
    print(train_fe.head())
    print(f"Feature count: {len(get_feature_cols(train_fe))}")
