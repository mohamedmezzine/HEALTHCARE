# State Features System - User Guide

## Overview

The state features system extracts vital signs and lab values from MIMIC-III and constructs state vectors for each hourly time step of ICU episodes.

## Architecture

```
Input: CHARTEVENTS, LABEVENTS, ICUSTAYS
         ↓
    Extraction
         ↓
    [Vitals]  [Labs]
         ↓
    Aggregation (hourly)
         ↓
    Forward-fill (missing data)
         ↓
    Normalization (z-score)
         ↓
Output: State vectors (T x D per episode)
```

## Features

### Default Vital Signs (8 features)
- **heart_rate**: Heart rate (bpm)
- **sbp**: Systolic blood pressure (mmHg)
- **dbp**: Diastolic blood pressure (mmHg)
- **mbp**: Mean blood pressure (mmHg)
- **resp_rate**: Respiratory rate (breaths/min)
- **spo2**: Oxygen saturation (%)
- **temperature**: Body temperature (°C)
- **gcs**: Glasgow Coma Scale (points)

### Default Lab Values (4 features)
- **lactate**: Lactate (mmol/L) - sepsis marker
- **creatinine**: Creatinine (mg/dL) - kidney function
- **wbc**: White blood cell count (K/uL) - infection
- **potassium**: Potassium (mEq/L) - cardiac function

**Total: 12 features per time step**

## Quick Start

### 1. Command Line Usage

```bash
# Build all state features
python -m src.features.builder \
    --data-dir artifacts/cleaned \
    --output artifacts/states.parquet

# Or extract vitals/labs separately
python -m src.features.vitals \
    --data-dir artifacts/cleaned \
    --output artifacts/vitals.parquet \
    --aggregation last

python -m src.features.labs \
    --data-dir artifacts/cleaned \
    --output artifacts/labs.parquet \
    --aggregation last
```

### 2. Python API Usage

```python
from src.features import build_all_state_features, StateFeatureConfig

# Use default configuration
state_df, normalizer = build_all_state_features(
    data_dir='artifacts/cleaned',
    normalize=True,
    verbose=True
)

# Result: DataFrame with columns
# - icustay_id
# - hour
# - hour_start, hour_end
# - heart_rate, sbp, dbp, mbp, resp_rate, spo2, temperature, gcs
# - lactate, creatinine, wbc, potassium
```

### 3. Custom Configuration

```python
from src.features import StateFeatureConfig, build_all_state_features

# Customize features
config = StateFeatureConfig(
    vital_features=['heart_rate', 'sbp', 'spo2'],  # Subset of vitals
    lab_features=['lactate', 'creatinine'],  # Subset of labs
    vital_aggregation='mean',  # mean, median, or last
    lab_aggregation='last',
    forward_fill_limit=6,  # Max hours to forward-fill
    normalize=True,
    normalization_method='z-score',  # z-score, min-max, or robust
    clip_outliers=True,
    outlier_std=5.0
)

state_df, normalizer = build_all_state_features(
    data_dir='artifacts/cleaned',
    config=config,
    normalize=True
)
```

## Missing Data Handling

### Strategy
1. **Hourly aggregation**: If multiple measurements in one hour, take last/mean/median
2. **Forward-fill**: Propagate last known value up to 6 hours (configurable)
3. **Population mean**: Fill remaining NaN with training set mean

### Example
```
Hour  Raw HR  Forward-fill  Mean-fill
  0     85         85           85
  1     NaN        85           85     ← Forward-fill
  2     NaN        85           85     ← Forward-fill
  3     90         90           90
  4     NaN        90           90     ← Forward-fill
  ...
  10    NaN        NaN          82     ← Past fill limit, use mean
```

## Normalization

### Z-score (default)
```python
x_norm = (x - mean) / std
```

- Mean ≈ 0, Std ≈ 1
- Preserves outliers (with clipping)
- Best for neural networks

### Min-Max
```python
x_norm = (x - min) / (max - min)
```

- Range: [0, 1]
- Sensitive to outliers

### Robust
```python
x_norm = (x - median) / IQR
```

- Median-centered
- Resistant to outliers

## Accessing State Vectors

### For RL Training

```python
from src.features import get_state_vector, get_state_matrix

# Single time step
state_t = get_state_vector(
    state_df,
    icustay_id=250055,
    hour=5,
    config=config
)
# Returns: numpy array shape (12,)

# Entire episode
episode_states = get_state_matrix(
    state_df,
    icustay_id=250055,
    config=config
)
# Returns: numpy array shape (T, 12) where T = episode length
```

### For Analysis

```python
# Get states for specific episode
episode_data = state_df[state_df['icustay_id'] == 250055]

# Feature columns
feature_cols = config.vital_features + config.lab_features

# Extract as matrix
X = episode_data[feature_cols].values  # shape (T, 12)
```

## Validation Checks

### 1. Check Data Availability

```python
# Missingness before forward-fill
vitals_raw_missing = vitals_df[[f'{v}_raw' for v in config.vital_features]].isna().mean()
print(vitals_raw_missing)

# Missingness after forward-fill
vitals_missing = vitals_df[config.vital_features].isna().mean()
print(vitals_missing)
```

### 2. Check Normalization

```python
# After normalization, features should have mean≈0, std≈1
state_df[config.vital_features + config.lab_features].describe()
```

### 3. Check for Outliers

```python
# Identify extreme values (beyond ±5 std)
for col in config.vital_features + config.lab_features:
    extreme = state_df[col].abs() > 5
    if extreme.any():
        print(f'{col}: {extreme.sum()} extreme values')
```

## Integration with Actions

```python
from src.actions import load_and_extract_actions
from src.features import build_all_state_features

# Extract actions
actions_df = load_and_extract_actions('artifacts/cleaned')

# Extract states
state_df, normalizer = build_all_state_features('artifacts/cleaned')

# Merge
rl_data = state_df.merge(
    actions_df[['icustay_id', 'hour', 'action']],
    on=['icustay_id', 'hour'],
    how='inner'
)

# Now you have: state features + action for each hour
# Ready for RL dataset construction!
```

## File Outputs

Running the full pipeline creates:

```
artifacts/
  ├── vitals.parquet       # Hourly vital signs (optional intermediate)
  ├── labs.parquet         # Hourly lab values (optional intermediate)
  └── states.parquet       # Combined normalized state features
```

## Performance

- **Extraction time**: ~30-60 seconds for MIMIC-III demo (136 episodes)
- **Memory**: ~50-100 MB for feature DataFrame
- **State vector size**: 12 features × 4 bytes = 48 bytes per time step

For 136 episodes × ~50 hours avg = ~6,800 time steps → ~325 KB total

## Next Steps

After extracting state features:

1. **Combine with actions**: Merge state_df with actions_df
2. **Add rewards**: Compute rewards based on outcomes
3. **Create episodes**: Build (s, a, r, s') tuples
4. **Train RL**: Use offline RL algorithms (DQN, CQL)

See `notebooks/07_state_features_analysis.ipynb` for detailed exploration.
