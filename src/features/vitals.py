"""Extract vital sign features from CHARTEVENTS for state construction."""

import numpy as np
import pandas as pd
from pathlib import Path

from .config import StateFeatureConfig, VITAL_FEATURES, DEFAULT_CONFIG


def extract_vitals_for_episode(
    icustay_id,
    intime,
    outtime,
    chartevents,
    config: StateFeatureConfig = None,
):
    """
    Extract hourly vital sign features for a single ICU episode.

    Args:
        icustay_id: ICU stay identifier
        intime: ICU admission timestamp
        outtime: ICU discharge timestamp
        chartevents: DataFrame of CHARTEVENTS
        config: Feature extraction configuration

    Returns:
        DataFrame with columns: icustay_id, hour, hour_start, hour_end,
                                 {vital_name}_raw, {vital_name}
    """
    if config is None:
        config = DEFAULT_CONFIG

    # Create hourly bins
    intime = pd.to_datetime(intime)
    outtime = pd.to_datetime(outtime)
    duration_hours = int(np.ceil((outtime - intime).total_seconds() / 3600))
    hourly_bins = pd.date_range(start=intime, periods=duration_hours + 1, freq="1H")

    # Filter chart events for this ICU stay
    stay_events = chartevents[chartevents["icustay_id"] == icustay_id].copy()

    # Get vital item IDs
    vital_itemids = config.get_vital_itemids()

    # Initialize result
    vital_data = []

    for hour_idx in range(duration_hours):
        hour_start = hourly_bins[hour_idx]
        hour_end = hourly_bins[hour_idx + 1]

        hour_record = {
            "icustay_id": icustay_id,
            "hour": hour_idx,
            "hour_start": hour_start,
            "hour_end": hour_end,
        }

        # Extract each vital sign
        for vital_name in config.vital_features:
            if vital_name not in vital_itemids:
                hour_record[f"{vital_name}_raw"] = np.nan
                continue

            itemids = vital_itemids[vital_name]

            # Get events for this vital in this hour
            hour_vital_events = stay_events[
                (stay_events["charttime"] >= hour_start)
                & (stay_events["charttime"] < hour_end)
                & (stay_events["itemid"].isin(itemids))
                & (stay_events["valuenum"].notna())
            ]

            # Aggregate based on configuration
            if len(hour_vital_events) == 0:
                value = np.nan
            elif config.vital_aggregation == "last":
                # Most recent value
                value = hour_vital_events.sort_values("charttime").iloc[-1]["valuenum"]
            elif config.vital_aggregation == "mean":
                value = hour_vital_events["valuenum"].mean()
            elif config.vital_aggregation == "median":
                value = hour_vital_events["valuenum"].median()
            else:
                raise ValueError(f"Unknown aggregation: {config.vital_aggregation}")

            hour_record[f"{vital_name}_raw"] = value

        vital_data.append(hour_record)

    # Convert to DataFrame
    vitals_df = pd.DataFrame(vital_data)

    # Apply forward fill
    for vital_name in config.vital_features:
        col_name = f"{vital_name}_raw"
        if col_name in vitals_df.columns:
            vitals_df[vital_name] = vitals_df[col_name].fillna(
                method="ffill", limit=config.forward_fill_limit
            )

            # Optional backward fill for first hours
            if config.backward_fill:
                vitals_df[vital_name] = vitals_df[vital_name].fillna(method="bfill", limit=1)

    return vitals_df


def extract_vitals_for_all_episodes(
    icustays, chartevents, config: StateFeatureConfig = None, verbose=True
):
    """
    Extract vital signs for all ICU episodes.

    Args:
        icustays: DataFrame of ICUSTAYS
        chartevents: DataFrame of CHARTEVENTS
        config: Feature extraction configuration
        verbose: Print progress

    Returns:
        DataFrame with all episode-hour-vital records
    """
    if config is None:
        config = DEFAULT_CONFIG

    if verbose:
        print(f"Extracting vitals for {len(icustays)} episodes...")
        print(f"  Vital features: {config.vital_features}")
        print(f"  Aggregation: {config.vital_aggregation}")

    all_vitals = []

    for idx, row in icustays.iterrows():
        if verbose and (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(icustays)} episodes")

        episode_vitals = extract_vitals_for_episode(
            icustay_id=row["icustay_id"],
            intime=row["intime"],
            outtime=row["outtime"],
            chartevents=chartevents,
            config=config,
        )

        all_vitals.append(episode_vitals)

    # Combine
    vitals_df = pd.concat(all_vitals, ignore_index=True)

    if verbose:
        print(f"\nExtracted {len(vitals_df):,} hourly vital records")

        # Report missingness
        print("\nMissing data after forward fill:")
        for vital_name in config.vital_features:
            if vital_name in vitals_df.columns:
                missing_pct = vitals_df[vital_name].isna().mean() * 100
                print(f"  {vital_name}: {missing_pct:.1f}%")

    return vitals_df


def load_and_extract_vitals(
    data_dir, config: StateFeatureConfig = None, verbose=True
):
    """
    Convenience function to load data and extract vitals.

    Args:
        data_dir: Path to cleaned data directory
        config: Feature extraction configuration
        verbose: Print progress

    Returns:
        DataFrame with vital features
    """
    data_dir = Path(data_dir)

    if verbose:
        print("Loading data...")

    icustays = pd.read_parquet(data_dir / "ICUSTAYS.parquet")
    chartevents = pd.read_parquet(data_dir / "CHARTEVENTS.parquet")

    # Ensure timestamps
    icustays["intime"] = pd.to_datetime(icustays["intime"])
    icustays["outtime"] = pd.to_datetime(icustays["outtime"])
    chartevents["charttime"] = pd.to_datetime(chartevents["charttime"])

    return extract_vitals_for_all_episodes(icustays, chartevents, config, verbose)


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Extract vital sign features from MIMIC-III")
    parser.add_argument(
        "--data-dir", type=str, default="artifacts/cleaned", help="Cleaned data directory"
    )
    parser.add_argument(
        "--output", type=str, default="artifacts/vitals.parquet", help="Output file"
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        choices=["last", "mean", "median"],
        default="last",
        help="Aggregation method",
    )

    args = parser.parse_args()

    # Create config
    config = DEFAULT_CONFIG
    config.vital_aggregation = args.aggregation

    # Extract
    vitals_df = load_and_extract_vitals(data_dir=args.data_dir, config=config)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vitals_df.to_parquet(output_path, index=False)

    print(f"\nSaved to {output_path}")
