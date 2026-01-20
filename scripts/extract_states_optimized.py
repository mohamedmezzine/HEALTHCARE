"""Optimized state feature extraction (pre-filters data for speed)."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.features.config import DEFAULT_CONFIG
from src.features.builder import build_state_features

print("=" * 80)
print("OPTIMIZED STATE FEATURE EXTRACTION")
print("=" * 80)

# Load data
data_dir = Path("artifacts/cleaned")
print(f"\nLoading data from {data_dir}...")

icustays = pd.read_parquet(data_dir / "ICUSTAYS.parquet")
chartevents = pd.read_parquet(data_dir / "CHARTEVENTS.parquet")
labevents = pd.read_parquet(data_dir / "LABEVENTS.parquet")

print(f"  ICUSTAYS: {len(icustays):,} rows")
print(f"  CHARTEVENTS: {len(chartevents):,} rows")
print(f"  LABEVENTS: {len(labevents):,} rows")

# Convert timestamps
icustays["intime"] = pd.to_datetime(icustays["intime"])
icustays["outtime"] = pd.to_datetime(icustays["outtime"])
chartevents["charttime"] = pd.to_datetime(chartevents["charttime"])
labevents["charttime"] = pd.to_datetime(labevents["charttime"])

# PRE-FILTER: Only keep relevant item IDs and ICU stays
print("\nPre-filtering data for speed...")

config = DEFAULT_CONFIG
vital_itemids = [
    itemid
    for itemids in config.get_vital_itemids().values()
    for itemid in itemids
]
lab_itemids = [
    itemid
    for itemids in config.get_lab_itemids().values()
    for itemid in itemids
]

icustay_ids = set(icustays["icustay_id"].unique())

# Filter chartevents
chartevents_filtered = chartevents[
    (chartevents["icustay_id"].isin(icustay_ids))
    & (chartevents["itemid"].isin(vital_itemids))
].copy()

print(f"  Filtered CHARTEVENTS: {len(chartevents):,} -> {len(chartevents_filtered):,} rows")

# Filter labevents
hadm_ids = set(icustays["hadm_id"].unique())
labevents_filtered = labevents[
    (labevents["hadm_id"].isin(hadm_ids))
    & (labevents["itemid"].isin(lab_itemids))
].copy()

print(f"  Filtered LABEVENTS: {len(labevents):,} -> {len(labevents_filtered):,} rows")

# Extract features
print("\nExtracting features...")
state_df, normalizer = build_state_features(
    icustays,
    chartevents_filtered,
    labevents_filtered,
    config=config,
    normalize=True,
    verbose=True,
)

# Save
output_path = Path("artifacts/states.parquet")
output_path.parent.mkdir(parents=True, exist_ok=True)
state_df.to_parquet(output_path, index=False)

print(f"\n✓ Saved to {output_path}")
print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

# Quick stats
print("\n" + "=" * 80)
print("QUICK STATS")
print("=" * 80)
print(f"Total hourly records: {len(state_df):,}")
print(f"Unique episodes: {state_df['icustay_id'].nunique()}")
print(f"Features per time step: {config.num_features()}")
print("\nSample data:")
print(state_df.head())
print("\nFeature statistics:")
feature_cols = config.vital_features + config.lab_features
print(state_df[feature_cols].describe())
