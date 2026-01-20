"""Build RL transitions and dataset splits."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd


def _feature_columns(
    states_df: pd.DataFrame,
    include_raw: bool = False,
    include_time: bool = False,
) -> list[str]:
    exclude = {"icustay_id", "hour", "hadm_id"}
    if not include_time:
        exclude |= {"hour_start", "hour_end"}

    cols = []
    for col in states_df.columns:
        if col in exclude:
            continue
        if not include_raw and col.endswith("_raw"):
            continue
        cols.append(col)
    return cols


def build_transitions(
    states_df: pd.DataFrame,
    actions_df: pd.DataFrame,
    rewards_df: pd.DataFrame,
    include_raw: bool = False,
    include_time: bool = False,
) -> pd.DataFrame:
    """
    Build (s, a, r, s_next, done) transitions from state/action/reward tables.

    Returns:
        DataFrame with keys, action, reward, done, survived, and state columns.
    """
    feature_cols = _feature_columns(
        states_df, include_raw=include_raw, include_time=include_time
    )

    # Merge on icustay_id + hour
    merged = states_df.merge(
        actions_df[["icustay_id", "hour", "action"]],
        on=["icustay_id", "hour"],
        how="inner",
    ).merge(
        rewards_df[["icustay_id", "hour", "reward", "is_terminal", "survived"]],
        on=["icustay_id", "hour"],
        how="inner",
    )

    merged = merged.sort_values(["icustay_id", "hour"]).reset_index(drop=True)

    # Compute next-state features by shifting within each episode
    next_features = (
        merged.groupby("icustay_id", group_keys=False)[feature_cols]
        .shift(-1)
        .add_suffix("_next")
    )

    transitions = pd.concat([merged, next_features], axis=1)

    transitions = transitions.rename(columns={"is_terminal": "done"})

    # For terminal steps, next state is undefined
    transitions.loc[transitions["done"], [f"{c}_next" for c in feature_cols]] = np.nan

    # Keep only relevant columns in a consistent order
    ordered_cols = (
        ["icustay_id", "hour"]
        + feature_cols
        + ["action", "reward", "done", "survived"]
        + [f"{c}_next" for c in feature_cols]
    )
    return transitions[ordered_cols]


def split_by_icustay(
    transitions: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 7,
    stratify_by: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split transitions by icustay_id to avoid leakage.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    rng = np.random.default_rng(seed)
    icustays = transitions[["icustay_id"]].drop_duplicates()

    if stratify_by is not None:
        strat = transitions.groupby("icustay_id")[stratify_by].first().reset_index()
        icustays = icustays.merge(strat, on="icustay_id", how="left")

        train_ids = []
        val_ids = []
        test_ids = []
        for _, group in icustays.groupby(stratify_by, dropna=False):
            ids = group["icustay_id"].to_numpy()
            rng.shuffle(ids)
            n = len(ids)
            n_train = int(round(n * train_ratio))
            n_val = int(round(n * val_ratio))
            train_ids.extend(ids[:n_train])
            val_ids.extend(ids[n_train : n_train + n_val])
            test_ids.extend(ids[n_train + n_val :])
    else:
        ids = icustays["icustay_id"].to_numpy()
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        train_ids = ids[:n_train]
        val_ids = ids[n_train : n_train + n_val]
        test_ids = ids[n_train + n_val :]

    train_df = transitions[transitions["icustay_id"].isin(train_ids)].copy()
    val_df = transitions[transitions["icustay_id"].isin(val_ids)].copy()
    test_df = transitions[transitions["icustay_id"].isin(test_ids)].copy()

    return train_df, val_df, test_df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", default="artifacts/states.parquet")
    parser.add_argument("--actions", default="artifacts/actions.parquet")
    parser.add_argument("--rewards", default="artifacts/rewards_sparse.parquet")
    parser.add_argument("--output", default="artifacts/rl/transitions.parquet")
    parser.add_argument("--include-raw", action="store_true")
    parser.add_argument("--include-time", action="store_true")
    parser.add_argument("--split", action="store_true", help="Write train/val/test splits")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--stratify-by", type=str, default=None)
    return parser.parse_args()


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / path


def main() -> None:
    args = _parse_args()
    states_path = _resolve_path(args.states)
    actions_path = _resolve_path(args.actions)
    rewards_path = _resolve_path(args.rewards)

    if not states_path.exists():
        raise FileNotFoundError(f"States file not found: {states_path}")
    if not actions_path.exists():
        raise FileNotFoundError(f"Actions file not found: {actions_path}")
    if not rewards_path.exists():
        raise FileNotFoundError(f"Rewards file not found: {rewards_path}")

    states_df = pd.read_parquet(states_path)
    actions_df = pd.read_parquet(actions_path)
    rewards_df = pd.read_parquet(rewards_path)

    transitions = build_transitions(
        states_df,
        actions_df,
        rewards_df,
        include_raw=args.include_raw,
        include_time=args.include_time,
    )

    output_path = _resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    transitions.to_parquet(output_path, index=False)
    print(f"Wrote transitions to {output_path}")

    if args.split:
        train_df, val_df, test_df = split_by_icustay(
            transitions,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            stratify_by=args.stratify_by,
        )
        split_dir = output_path.parent
        train_df.to_parquet(split_dir / "transitions_train.parquet", index=False)
        val_df.to_parquet(split_dir / "transitions_val.parquet", index=False)
        test_df.to_parquet(split_dir / "transitions_test.parquet", index=False)
        print(f"Wrote splits to {split_dir}")


if __name__ == "__main__":
    main()
