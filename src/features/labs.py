"""Extract lab value features from LABEVENTS for state construction."""

import numpy as np
import pandas as pd
from pathlib import Path

from .config import StateFeatureConfig, LAB_FEATURES, DEFAULT_CONFIG


def extract_labs_for_episode(
    icustay_id,
    hadm_id,
    intime,
    outtime,
    labevents,
    config: StateFeatureConfig = None,
):
    """
    Extract hourly lab value features for a single ICU episode.

    Note: Labs are ordered by HADM_ID, not ICUSTAY_ID, so we need both.

    Args:
        icustay_id: ICU stay identifier
        hadm_id: Hospital admission identifier
        intime: ICU admission timestamp
        outtime: ICU discharge timestamp
        labevents: DataFrame of LABEVENTS
        config: Feature extraction configuration

    Returns:
        DataFrame with columns: icustay_id, hour, hour_start, hour_end,
                                 {lab_name}_raw, {lab_name}
    """
    if config is None:
        config = DEFAULT_CONFIG

    # Create hourly bins
    intime = pd.to_datetime(intime)
    outtime = pd.to_datetime(outtime)
    duration_hours = int(np.ceil((outtime - intime).total_seconds() / 3600))
    hourly_bins = pd.date_range(start=intime, periods=duration_hours + 1, freq="1H")

    # Filter lab events for this admission within ICU timeframe
    admission_labs = labevents[labevents["hadm_id"] == hadm_id].copy()
    admission_labs["charttime"] = pd.to_datetime(admission_labs["charttime"])

    # Only keep labs during ICU stay (prevent temporal leakage)
    stay_labs = admission_labs[
        (admission_labs["charttime"] >= intime) & (admission_labs["charttime"] <= outtime)
    ]

    # Get lab item IDs
    lab_itemids = config.get_lab_itemids()

    # Initialize result
    lab_data = []

    for hour_idx in range(duration_hours):
        hour_start = hourly_bins[hour_idx]
        hour_end = hourly_bins[hour_idx + 1]

        hour_record = {
            "icustay_id": icustay_id,
            "hour": hour_idx,
            "hour_start": hour_start,
            "hour_end": hour_end,
        }

        # Extract each lab value
        for lab_name in config.lab_features:
            if lab_name not in lab_itemids:
                hour_record[f"{lab_name}_raw"] = np.nan
                continue

            itemids = lab_itemids[lab_name]

            # Get lab results for this hour
            hour_lab_events = stay_labs[
                (stay_labs["charttime"] >= hour_start)
                & (stay_labs["charttime"] < hour_end)
                & (stay_labs["itemid"].isin(itemids))
                & (stay_labs["valuenum"].notna())
            ]

            # Aggregate
            if len(hour_lab_events) == 0:
                value = np.nan
            elif config.lab_aggregation == "last":
                value = hour_lab_events.sort_values("charttime").iloc[-1]["valuenum"]
            elif config.lab_aggregation == "mean":
                value = hour_lab_events["valuenum"].mean()
            elif config.lab_aggregation == "median":
                value = hour_lab_events["valuenum"].median()
            else:
                raise ValueError(f"Unknown aggregation: {config.lab_aggregation}")

            hour_record[f"{lab_name}_raw"] = value

        lab_data.append(hour_record)

    # Convert to DataFrame
    labs_df = pd.DataFrame(lab_data)

    # Apply forward fill (labs are less frequent than vitals)
    for lab_name in config.lab_features:
        col_name = f"{lab_name}_raw"
        if col_name in labs_df.columns:
            labs_df[lab_name] = labs_df[col_name].fillna(
                method="ffill", limit=config.forward_fill_limit
            )

            # Backward fill for first hours (first lab may come hours after admission)
            if config.backward_fill:
                labs_df[lab_name] = labs_df[lab_name].fillna(method="bfill", limit=3)

    return labs_df


def extract_labs_for_all_episodes(
    icustays, labevents, config: StateFeatureConfig = None, verbose=True
):
    """
    Extract lab values for all ICU episodes.

    Args:
        icustays: DataFrame of ICUSTAYS (must have hadm_id)
        labevents: DataFrame of LABEVENTS
        config: Feature extraction configuration
        verbose: Print progress

    Returns:
        DataFrame with all episode-hour-lab records
    """
    if config is None:
        config = DEFAULT_CONFIG

    if verbose:
        print(f"Extracting labs for {len(icustays)} episodes...")
        print(f"  Lab features: {config.lab_features}")
        print(f"  Aggregation: {config.lab_aggregation}")

    all_labs = []

    for idx, row in icustays.iterrows():
        if verbose and (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(icustays)} episodes")

        episode_labs = extract_labs_for_episode(
            icustay_id=row["icustay_id"],
            hadm_id=row["hadm_id"],
            intime=row["intime"],
            outtime=row["outtime"],
            labevents=labevents,
            config=config,
        )

        all_labs.append(episode_labs)

    # Combine
    labs_df = pd.concat(all_labs, ignore_index=True)

    if verbose:
        print(f"\nExtracted {len(labs_df):,} hourly lab records")

        # Report missingness
        print("\nMissing data after forward fill:")
        for lab_name in config.lab_features:
            if lab_name in labs_df.columns:
                missing_pct = labs_df[lab_name].isna().mean() * 100
                print(f"  {lab_name}: {missing_pct:.1f}%")

    return labs_df


def load_and_extract_labs(data_dir, config: StateFeatureConfig = None, verbose=True):
    """
    Convenience function to load data and extract labs.

    Args:
        data_dir: Path to cleaned data directory
        config: Feature extraction configuration
        verbose: Print progress

    Returns:
        DataFrame with lab features
    """
    data_dir = Path(data_dir)

    if verbose:
        print("Loading data...")

    icustays = pd.read_parquet(data_dir / "ICUSTAYS.parquet")
    labevents = pd.read_parquet(data_dir / "LABEVENTS.parquet")

    # Ensure timestamps
    icustays["intime"] = pd.to_datetime(icustays["intime"])
    icustays["outtime"] = pd.to_datetime(icustays["outtime"])
    labevents["charttime"] = pd.to_datetime(labevents["charttime"])

    return extract_labs_for_all_episodes(icustays, labevents, config, verbose)


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Extract lab value features from MIMIC-III")
    parser.add_argument(
        "--data-dir", type=str, default="artifacts/cleaned", help="Cleaned data directory"
    )
    parser.add_argument(
        "--output", type=str, default="artifacts/labs.parquet", help="Output file"
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
    config.lab_aggregation = args.aggregation

    # Extract
    labs_df = load_and_extract_labs(data_dir=args.data_dir, config=config)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labs_df.to_parquet(output_path, index=False)

    print(f"\nSaved to {output_path}")
