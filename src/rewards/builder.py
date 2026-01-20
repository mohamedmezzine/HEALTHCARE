"""Build rewards for all episodes from data."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Union

from .config import RewardConfig, RewardType, SPARSE_CONFIG, SHAPED_CONFIG
from .sparse import SparseReward
from .shaped import ShapedReward


def compute_rewards_for_episode(
    icustay_id: int,
    survived: bool,
    episode_length: int,
    config: RewardConfig = None,
    vitals_df: Optional[pd.DataFrame] = None,
) -> np.ndarray:
    """
    Compute rewards for a single episode.

    Args:
        icustay_id: ICU stay identifier
        survived: Whether patient survived
        episode_length: Number of hourly time steps
        config: Reward configuration
        vitals_df: Optional DataFrame with vital signs (for shaped rewards)

    Returns:
        Array of rewards, shape (episode_length,)
    """
    if config is None:
        config = SPARSE_CONFIG

    # Select reward function based on config
    if config.reward_type == RewardType.SPARSE:
        reward_fn = SparseReward(config)
        rewards = reward_fn.compute_episode_rewards(episode_length, survived)

    elif config.reward_type == RewardType.SHAPED:
        reward_fn = ShapedReward(config)

        # Extract vitals for this episode if available
        if vitals_df is not None:
            episode_vitals = vitals_df[vitals_df["icustay_id"] == icustay_id]
        else:
            episode_vitals = None

        rewards = reward_fn.compute_episode_rewards(episode_length, survived, episode_vitals)

    elif config.reward_type == RewardType.SURVIVAL_LOS:
        # Survival with length-of-stay penalty
        reward_fn = SparseReward(config)
        rewards = reward_fn.compute_episode_rewards(episode_length, survived)

        # Add LOS penalty
        los_penalty = -config.los_penalty_weight * episode_length
        rewards[-1] += los_penalty  # Add to terminal reward

    else:
        raise ValueError(f"Unknown reward type: {config.reward_type}")

    return rewards


def compute_all_rewards(
    icustays: pd.DataFrame,
    admissions: pd.DataFrame,
    config: RewardConfig = None,
    vitals_df: Optional[pd.DataFrame] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Compute rewards for all episodes.

    Args:
        icustays: DataFrame of ICUSTAYS (with intime, outtime)
        admissions: DataFrame of ADMISSIONS (with hospital_expire_flag)
        config: Reward configuration
        vitals_df: Optional DataFrame with vitals (for shaped rewards)
        verbose: Print progress

    Returns:
        DataFrame with columns: icustay_id, hour, reward, is_terminal, survived
    """
    if config is None:
        config = SPARSE_CONFIG

    if verbose:
        print("=" * 80)
        print("COMPUTING REWARDS")
        print("=" * 80)
        print(f"Reward type: {config.reward_type.value}")
        print(f"Episodes: {len(icustays)}")

    # Merge with outcomes
    episode_data = icustays.merge(
        admissions[["hadm_id", "hospital_expire_flag"]], on="hadm_id", how="left"
    )

    # Convert timestamps
    episode_data["intime"] = pd.to_datetime(episode_data["intime"])
    episode_data["outtime"] = pd.to_datetime(episode_data["outtime"])

    # Compute episode lengths
    episode_data["duration_hours"] = (
        (episode_data["outtime"] - episode_data["intime"]).dt.total_seconds() / 3600
    )
    episode_data["episode_length"] = episode_data["duration_hours"].apply(lambda x: int(np.ceil(x)))

    all_rewards = []

    for idx, row in episode_data.iterrows():
        if verbose and (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(episode_data)} episodes")

        icustay_id = row["icustay_id"]
        survived = row["hospital_expire_flag"] == 0  # 0 = survived, 1 = died
        episode_length = row["episode_length"]

        # Compute rewards
        rewards = compute_rewards_for_episode(
            icustay_id=icustay_id,
            survived=survived,
            episode_length=episode_length,
            config=config,
            vitals_df=vitals_df,
        )

        # Create records
        for hour in range(episode_length):
            all_rewards.append(
                {
                    "icustay_id": icustay_id,
                    "hour": hour,
                    "reward": rewards[hour],
                    "is_terminal": hour == episode_length - 1,
                    "survived": survived,
                }
            )

    rewards_df = pd.DataFrame(all_rewards)

    if verbose:
        print(f"\nComputed {len(rewards_df):,} reward values")
        print("\nReward statistics:")
        print(rewards_df["reward"].describe())
        print(f"\nNon-zero rewards: {(rewards_df['reward'] != 0).sum():,}")
        print(f"Terminal rewards: {rewards_df['is_terminal'].sum():,}")

    return rewards_df


def load_and_compute_rewards(
    data_dir: Union[str, Path],
    config: RewardConfig = None,
    vitals_path: Optional[Union[str, Path]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load data and compute rewards.

    Args:
        data_dir: Directory with cleaned data
        config: Reward configuration
        vitals_path: Optional path to vitals DataFrame (for shaped rewards)
        verbose: Print progress

    Returns:
        DataFrame with rewards
    """
    data_dir = Path(data_dir)

    if verbose:
        print("Loading data...")

    icustays = pd.read_parquet(data_dir / "ICUSTAYS.parquet")
    admissions = pd.read_parquet(data_dir / "ADMISSIONS.parquet")

    # Load vitals if path provided
    vitals_df = None
    if vitals_path is not None:
        vitals_path = Path(vitals_path)
        if vitals_path.exists():
            vitals_df = pd.read_parquet(vitals_path)
            if verbose:
                print(f"Loaded vitals from {vitals_path}")

    return compute_all_rewards(icustays, admissions, config, vitals_df, verbose)


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Compute rewards for MIMIC-III episodes")
    parser.add_argument(
        "--data-dir", type=str, default="artifacts/cleaned", help="Cleaned data directory"
    )
    parser.add_argument(
        "--vitals",
        type=str,
        default=None,
        help="Optional vitals parquet file (for shaped rewards)",
    )
    parser.add_argument(
        "--reward-type",
        type=str,
        choices=["sparse", "shaped", "survival_los"],
        default="sparse",
        help="Reward function type",
    )
    parser.add_argument(
        "--output", type=str, default="artifacts/rewards.parquet", help="Output file"
    )

    args = parser.parse_args()

    # Create config
    if args.reward_type == "sparse":
        config = SPARSE_CONFIG
    elif args.reward_type == "shaped":
        config = SHAPED_CONFIG
    elif args.reward_type == "survival_los":
        from .config import SURVIVAL_LOS_CONFIG

        config = SURVIVAL_LOS_CONFIG

    # Compute rewards
    rewards_df = load_and_compute_rewards(
        data_dir=args.data_dir, config=config, vitals_path=args.vitals, verbose=True
    )

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rewards_df.to_parquet(output_path, index=False)

    print(f"\nSaved to {output_path}")
