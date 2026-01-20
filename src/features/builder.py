"""Combine vitals and labs into complete state feature vectors."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

from .config import StateFeatureConfig, DEFAULT_CONFIG
from .vitals import extract_vitals_for_all_episodes
from .labs import extract_labs_for_all_episodes
from .normalization import FeatureNormalizer, handle_missing_values


def build_state_features(
    icustays,
    chartevents,
    labevents,
    config: StateFeatureConfig = None,
    normalize=True,
    verbose=True,
):
    """
    Build complete state feature matrix (vitals + labs).

    Args:
        icustays: DataFrame of ICUSTAYS
        chartevents: DataFrame of CHARTEVENTS
        labevents: DataFrame of LABEVENTS
        config: Feature configuration
        normalize: Whether to normalize features
        verbose: Print progress

    Returns:
        DataFrame with all state features per hour
    """
    if config is None:
        config = DEFAULT_CONFIG

    if verbose:
        print("=" * 80)
        print("BUILDING STATE FEATURES")
        print("=" * 80)

    # Extract vitals
    if verbose:
        print("\n1. Extracting vital signs...")
    vitals_df = extract_vitals_for_all_episodes(
        icustays, chartevents, config=config, verbose=verbose
    )

    # Extract labs
    if verbose:
        print("\n2. Extracting lab values...")
    labs_df = extract_labs_for_all_episodes(icustays, labevents, config=config, verbose=verbose)

    # Merge vitals and labs
    if verbose:
        print("\n3. Merging vitals and labs...")

    # Merge on icustay_id and hour
    state_df = vitals_df.merge(
        labs_df.drop(columns=["hour_start", "hour_end"], errors="ignore"),
        on=["icustay_id", "hour"],
        how="outer",
        suffixes=("_vital", "_lab"),
    )

    # Fill remaining NaN with population mean
    if verbose:
        print("\n4. Handling missing values...")

    state_df, fill_values = handle_missing_values(state_df, strategy="mean")

    # Normalize features
    if normalize and config.normalize:
        if verbose:
            print(f"\n5. Normalizing features ({config.normalization_method})...")

        normalizer = FeatureNormalizer(config)
        state_df = normalizer.fit_transform(state_df, verbose=verbose)
    else:
        normalizer = None

    # Summary
    if verbose:
        print("\n" + "=" * 80)
        print("STATE FEATURES SUMMARY")
        print("=" * 80)
        print(f"Total hours: {len(state_df):,}")
        print(f"Unique episodes: {state_df['icustay_id'].nunique()}")
        print(f"Feature count: {config.num_features()}")
        print(f"  Vitals: {len(config.vital_features)}")
        print(f"  Labs: {len(config.lab_features)}")

        # Check for remaining NaN
        feature_cols = [col for col in state_df.columns if col in config.vital_features + config.lab_features]
        if feature_cols:
            nan_counts = state_df[feature_cols].isna().sum()
            if nan_counts.sum() > 0:
                print(f"\nWarning: {nan_counts.sum()} NaN values remaining:")
                for col in nan_counts[nan_counts > 0].index:
                    print(f"  {col}: {nan_counts[col]}")

        print("=" * 80)

    return state_df, normalizer


def build_all_state_features(
    data_dir,
    config: StateFeatureConfig = None,
    normalize=True,
    save_path: Optional[str] = None,
    verbose=True,
):
    """
    Load data and build complete state features.

    Args:
        data_dir: Path to cleaned data directory
        config: Feature configuration
        normalize: Whether to normalize
        save_path: Optional path to save features
        verbose: Print progress

    Returns:
        (state_df, normalizer)
    """
    data_dir = Path(data_dir)

    if verbose:
        print("Loading data from", data_dir)

    # Load tables
    icustays = pd.read_parquet(data_dir / "ICUSTAYS.parquet")
    chartevents = pd.read_parquet(data_dir / "CHARTEVENTS.parquet")
    labevents = pd.read_parquet(data_dir / "LABEVENTS.parquet")

    # Ensure timestamps
    icustays["intime"] = pd.to_datetime(icustays["intime"])
    icustays["outtime"] = pd.to_datetime(icustays["outtime"])
    chartevents["charttime"] = pd.to_datetime(chartevents["charttime"])
    labevents["charttime"] = pd.to_datetime(labevents["charttime"])

    # Build features
    state_df, normalizer = build_state_features(
        icustays,
        chartevents,
        labevents,
        config=config,
        normalize=normalize,
        verbose=verbose,
    )

    # Save if requested
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        state_df.to_parquet(save_path, index=False)
        if verbose:
            print(f"\nSaved state features to {save_path}")

    return state_df, normalizer


def get_state_vector(
    state_df, icustay_id, hour, config: StateFeatureConfig = None
) -> np.ndarray:
    """
    Extract state vector for a specific episode-hour.

    Args:
        state_df: DataFrame with state features
        icustay_id: ICU stay identifier
        hour: Hour index
        config: Feature configuration (to determine feature order)

    Returns:
        1D numpy array of feature values
    """
    if config is None:
        config = DEFAULT_CONFIG

    # Get row
    row = state_df[(state_df["icustay_id"] == icustay_id) & (state_df["hour"] == hour)]

    if len(row) == 0:
        raise ValueError(f"No state found for icustay_id={icustay_id}, hour={hour}")

    row = row.iloc[0]

    # Extract features in order
    feature_names = config.vital_features + config.lab_features
    state_vector = np.array([row[name] for name in feature_names])

    return state_vector


def get_state_matrix(
    state_df, icustay_id, config: StateFeatureConfig = None
) -> np.ndarray:
    """
    Extract state matrix for an entire episode (T x D).

    Args:
        state_df: DataFrame with state features
        icustay_id: ICU stay identifier
        config: Feature configuration

    Returns:
        2D numpy array (num_hours x num_features)
    """
    if config is None:
        config = DEFAULT_CONFIG

    # Get episode data
    episode_data = state_df[state_df["icustay_id"] == icustay_id].sort_values("hour")

    if len(episode_data) == 0:
        raise ValueError(f"No state found for icustay_id={icustay_id}")

    # Extract features
    feature_names = config.vital_features + config.lab_features
    state_matrix = episode_data[feature_names].values

    return state_matrix


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Build state features from MIMIC-III")
    parser.add_argument(
        "--data-dir", type=str, default="artifacts/cleaned", help="Cleaned data directory"
    )
    parser.add_argument(
        "--output", type=str, default="artifacts/states.parquet", help="Output file"
    )
    parser.add_argument("--no-normalize", action="store_true", help="Skip normalization")

    args = parser.parse_args()

    # Build features
    state_df, normalizer = build_all_state_features(
        data_dir=args.data_dir,
        normalize=not args.no_normalize,
        save_path=args.output,
        verbose=True,
    )

    print("\nDone!")
