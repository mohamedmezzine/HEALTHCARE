"""Extract ground-truth actions from INPUTEVENTS for offline RL.

This module identifies vasopressor administration in the data and extracts
discrete actions for each hourly time step of an ICU episode.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .discretizer import discretize_vasopressor_rate


# Vasopressor item IDs in MIMIC-III
# Source: https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iii/concepts/durations/vasoactive-durations.sql
VASOPRESSOR_ITEMIDS = {
    # Norepinephrine
    30047,  # CareVue
    30120,  # CareVue
    221906,  # MetaVision
    # Epinephrine
    30044,  # CareVue
    30119,  # CareVue
    221289,  # MetaVision
    # Phenylephrine
    30127,  # CareVue
    30128,  # CareVue
    221749,  # MetaVision
    # Vasopressin
    30051,  # CareVue
    222315,  # MetaVision
    # Dopamine
    30043,  # CareVue
    30307,  # CareVue
    221662,  # MetaVision
}


def identify_vasopressor_events(inputevents_cv, inputevents_mv):
    """
    Identify all vasopressor events from INPUTEVENTS tables.

    Args:
        inputevents_cv: DataFrame of INPUTEVENTS_CV
        inputevents_mv: DataFrame of INPUTEVENTS_MV

    Returns:
        DataFrame with columns: icustay_id, charttime, rate, itemid
    """
    # Filter CareVue events
    vaso_cv = inputevents_cv[inputevents_cv["itemid"].isin(VASOPRESSOR_ITEMIDS)].copy()

    # Filter MetaVision events
    vaso_mv = inputevents_mv[inputevents_mv["itemid"].isin(VASOPRESSOR_ITEMIDS)].copy()

    # Standardize columns
    vaso_cv["charttime"] = pd.to_datetime(vaso_cv["charttime"])
    vaso_cv = vaso_cv[["icustay_id", "charttime", "rate", "itemid"]]

    # MetaVision uses starttime instead of charttime
    vaso_mv["charttime"] = pd.to_datetime(vaso_mv["starttime"])
    vaso_mv = vaso_mv[["icustay_id", "charttime", "rate", "itemid"]]

    # Combine
    vaso_events = pd.concat([vaso_cv, vaso_mv], ignore_index=True)

    # Remove events with missing rate
    vaso_events = vaso_events[vaso_events["rate"].notna()]

    # Sort by ICU stay and time
    vaso_events = vaso_events.sort_values(["icustay_id", "charttime"])

    return vaso_events


def extract_actions_for_episode(
    icustay_id, intime, outtime, vaso_events, aggregation="mean"
):
    """
    Extract hourly actions for a single ICU episode.

    Args:
        icustay_id: ICU stay identifier
        intime: ICU admission timestamp
        outtime: ICU discharge timestamp
        vaso_events: DataFrame of vasopressor events (from identify_vasopressor_events)
        aggregation: How to aggregate multiple events in one hour ('mean', 'max', 'last')

    Returns:
        DataFrame with columns: icustay_id, hour, hour_start, hour_end, rate, action
    """
    # Create hourly bins
    intime = pd.to_datetime(intime)
    outtime = pd.to_datetime(outtime)

    duration_hours = int(np.ceil((outtime - intime).total_seconds() / 3600))

    # Generate hourly time bins
    hourly_bins = pd.date_range(start=intime, periods=duration_hours + 1, freq="1H")

    # Filter events for this ICU stay
    stay_events = vaso_events[vaso_events["icustay_id"] == icustay_id].copy()

    # Initialize result
    actions = []

    for hour_idx in range(duration_hours):
        hour_start = hourly_bins[hour_idx]
        hour_end = hourly_bins[hour_idx + 1]

        # Get events in this hour
        hour_events = stay_events[
            (stay_events["charttime"] >= hour_start) & (stay_events["charttime"] < hour_end)
        ]

        # Aggregate rate
        if len(hour_events) == 0:
            # No vasopressor in this hour
            rate = 0.0
        elif aggregation == "mean":
            rate = hour_events["rate"].mean()
        elif aggregation == "max":
            rate = hour_events["rate"].max()
        elif aggregation == "last":
            rate = hour_events.iloc[-1]["rate"]
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        # Discretize to action
        action = discretize_vasopressor_rate(rate)

        actions.append(
            {
                "icustay_id": icustay_id,
                "hour": hour_idx,
                "hour_start": hour_start,
                "hour_end": hour_end,
                "rate": rate,
                "action": action,
            }
        )

    return pd.DataFrame(actions)


def extract_actions_for_all_episodes(
    icustays, inputevents_cv, inputevents_mv, aggregation="mean", verbose=True
):
    """
    Extract actions for all ICU episodes.

    Args:
        icustays: DataFrame of ICUSTAYS
        inputevents_cv: DataFrame of INPUTEVENTS_CV
        inputevents_mv: DataFrame of INPUTEVENTS_MV
        aggregation: How to aggregate within hour ('mean', 'max', 'last')
        verbose: Print progress

    Returns:
        DataFrame with all episode-hour-action records
    """
    if verbose:
        print("Identifying vasopressor events...")

    # Get all vasopressor events
    vaso_events = identify_vasopressor_events(inputevents_cv, inputevents_mv)

    if verbose:
        print(f"  Found {len(vaso_events):,} vasopressor events")
        print(f"  Across {vaso_events['icustay_id'].nunique()} ICU stays")

    # Extract actions for each episode
    all_actions = []

    if verbose:
        print(f"\nExtracting actions for {len(icustays)} episodes...")

    for idx, row in icustays.iterrows():
        if verbose and (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(icustays)} episodes")

        episode_actions = extract_actions_for_episode(
            icustay_id=row["icustay_id"],
            intime=row["intime"],
            outtime=row["outtime"],
            vaso_events=vaso_events,
            aggregation=aggregation,
        )

        all_actions.append(episode_actions)

    # Combine all episodes
    actions_df = pd.concat(all_actions, ignore_index=True)

    if verbose:
        print(f"\nExtracted {len(actions_df):,} hourly actions")
        print(f"Action distribution:")
        print(actions_df["action"].value_counts().sort_index())

    return actions_df


def load_and_extract_actions(data_dir, aggregation="mean", verbose=True):
    """
    Convenience function to load data and extract actions.

    Args:
        data_dir: Path to cleaned data directory (e.g., 'artifacts/cleaned')
        aggregation: How to aggregate within hour
        verbose: Print progress

    Returns:
        DataFrame with all episode-hour-action records
    """
    data_dir = Path(data_dir)

    if verbose:
        print("Loading data...")

    # Load required tables
    icustays = pd.read_parquet(data_dir / "ICUSTAYS.parquet")
    inputevents_cv = pd.read_parquet(data_dir / "INPUTEVENTS_CV.parquet")
    inputevents_mv = pd.read_parquet(data_dir / "INPUTEVENTS_MV.parquet")

    # Ensure timestamps are datetime
    icustays["intime"] = pd.to_datetime(icustays["intime"])
    icustays["outtime"] = pd.to_datetime(icustays["outtime"])

    # Extract actions
    actions_df = extract_actions_for_all_episodes(
        icustays, inputevents_cv, inputevents_mv, aggregation=aggregation, verbose=verbose
    )

    return actions_df


def save_actions(actions_df, output_path, format="parquet"):
    """
    Save extracted actions to disk.

    Args:
        actions_df: DataFrame of actions
        output_path: Where to save
        format: 'parquet' or 'csv'
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "parquet":
        actions_df.to_parquet(output_path, index=False)
    elif format == "csv":
        actions_df.to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unknown format: {format}")

    print(f"Saved actions to {output_path}")


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Extract vasopressor actions from MIMIC-III")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="artifacts/cleaned",
        help="Directory with cleaned parquet files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/actions.parquet",
        help="Output file path",
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        choices=["mean", "max", "last"],
        default="mean",
        help="How to aggregate rates within an hour",
    )
    parser.add_argument(
        "--format", type=str, choices=["parquet", "csv"], default="parquet", help="Output format"
    )

    args = parser.parse_args()

    # Extract actions
    actions_df = load_and_extract_actions(data_dir=args.data_dir, aggregation=args.aggregation)

    # Save
    save_actions(actions_df, output_path=args.output, format=args.format)

    print("\nDone!")
