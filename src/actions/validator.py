"""Validation and quality checks for extracted actions.

This module provides sanity checks to ensure extracted actions make clinical sense.
"""

import numpy as np
import pandas as pd


def validate_action_distribution(actions_df, warn_threshold=0.05):
    """
    Check if action distribution is reasonable.

    Args:
        actions_df: DataFrame with 'action' column
        warn_threshold: Warn if any action has < this fraction

    Returns:
        dict: Validation results
    """
    action_counts = actions_df["action"].value_counts(normalize=True).sort_index()

    results = {
        "total_actions": len(actions_df),
        "action_distribution": action_counts.to_dict(),
        "warnings": [],
    }

    # Check for missing actions
    for action_id in range(4):
        if action_id not in action_counts.index:
            results["warnings"].append(f"Action {action_id} never occurs (0%)")
        elif action_counts[action_id] < warn_threshold:
            results["warnings"].append(
                f"Action {action_id} is rare ({action_counts[action_id]:.1%})"
            )

    # Check for extreme imbalance
    if action_counts.max() > 0.8:
        results["warnings"].append(
            f"Highly imbalanced: {action_counts.idxmax()} = {action_counts.max():.1%}"
        )

    return results


def validate_action_transitions(actions_df):
    """
    Check if action transitions are clinically reasonable.

    Smooth transitions (±1 step) are expected for vasopressor titration.

    Args:
        actions_df: DataFrame with 'icustay_id', 'hour', 'action'

    Returns:
        dict: Transition statistics
    """
    # Sort by episode and hour
    df = actions_df.sort_values(["icustay_id", "hour"]).copy()

    # Compute action changes within episodes
    df["action_diff"] = df.groupby("icustay_id")["action"].diff()

    # Remove first hour of each episode (no previous action)
    transitions = df["action_diff"].dropna()

    results = {
        "total_transitions": len(transitions),
        "smooth_transitions": (transitions.abs() <= 1).sum(),
        "smooth_pct": (transitions.abs() <= 1).mean() * 100,
        "mean_abs_change": transitions.abs().mean(),
        "max_jump": transitions.abs().max(),
    }

    # Transition matrix
    transition_matrix = pd.crosstab(
        df.groupby("icustay_id")["action"].shift(1), df["action"], normalize="index"
    )

    results["transition_matrix"] = transition_matrix

    return results


def validate_correlation_with_outcome(actions_df, icustays, admissions):
    """
    Check if action usage correlates with outcomes.

    Expect: Higher max action → higher mortality (sicker patients need more support)

    Args:
        actions_df: DataFrame with actions
        icustays: DataFrame of ICUSTAYS
        admissions: DataFrame of ADMISSIONS

    Returns:
        dict: Correlation statistics
    """
    # Merge outcome information
    episode_outcomes = icustays.merge(
        admissions[["hadm_id", "hospital_expire_flag"]], on="hadm_id", how="left"
    )

    # Get max action per episode
    max_actions = actions_df.groupby("icustay_id")["action"].max().reset_index()
    max_actions.columns = ["icustay_id", "max_action"]

    # Get mean action per episode
    mean_actions = actions_df.groupby("icustay_id")["action"].mean().reset_index()
    mean_actions.columns = ["icustay_id", "mean_action"]

    # Merge
    analysis = episode_outcomes.merge(max_actions, on="icustay_id", how="left")
    analysis = analysis.merge(mean_actions, on="icustay_id", how="left")

    # Fill NaN (episodes with no vasopressor) with 0
    analysis["max_action"] = analysis["max_action"].fillna(0)
    analysis["mean_action"] = analysis["mean_action"].fillna(0)

    # Mortality by max action
    mortality_by_max = (
        analysis.groupby("max_action")["hospital_expire_flag"].agg(["sum", "count", "mean"])
    )
    mortality_by_max["mortality_pct"] = mortality_by_max["mean"] * 100

    results = {
        "mortality_by_max_action": mortality_by_max.to_dict(),
        "correlation_max_action_mortality": analysis[["max_action", "hospital_expire_flag"]]
        .corr()
        .iloc[0, 1],
        "correlation_mean_action_mortality": analysis[["mean_action", "hospital_expire_flag"]]
        .corr()
        .iloc[0, 1],
    }

    return results


def validate_temporal_patterns(actions_df):
    """
    Check temporal patterns (e.g., escalation over time, de-escalation near discharge).

    Args:
        actions_df: DataFrame with actions

    Returns:
        dict: Temporal statistics
    """
    # Divide episodes into early/mid/late phases
    episode_lengths = actions_df.groupby("icustay_id")["hour"].max()

    # For each episode, categorize hours
    phase_data = []
    for icustay_id, length in episode_lengths.items():
        episode_actions = actions_df[actions_df["icustay_id"] == icustay_id].copy()

        for _, row in episode_actions.iterrows():
            hour = row["hour"]
            if hour < length * 0.33:
                phase = "early"
            elif hour < length * 0.67:
                phase = "mid"
            else:
                phase = "late"

            phase_data.append({"phase": phase, "action": row["action"]})

    phase_df = pd.DataFrame(phase_data)

    # Mean action by phase
    mean_by_phase = phase_df.groupby("phase")["action"].mean()

    results = {
        "mean_action_by_phase": mean_by_phase.to_dict(),
        "early_vs_late_diff": mean_by_phase.get("late", 0) - mean_by_phase.get("early", 0),
    }

    return results


def validate_rate_discretization(actions_df):
    """
    Check if rate discretization is working correctly.

    Args:
        actions_df: DataFrame with 'rate' and 'action'

    Returns:
        dict: Discretization statistics
    """
    # Check rate ranges for each action
    rate_stats = actions_df.groupby("action")["rate"].agg(["min", "max", "mean", "std", "count"])

    # Expected ranges
    expected_ranges = {
        0: (0.0, 0.0),
        1: (0.0, 0.1),
        2: (0.1, 0.3),
        3: (0.3, float("inf")),
    }

    warnings = []
    for action_id in range(4):
        if action_id not in rate_stats.index:
            continue

        actual_min = rate_stats.loc[action_id, "min"]
        actual_max = rate_stats.loc[action_id, "max"]
        expected_min, expected_max = expected_ranges[action_id]

        # Check boundaries
        if action_id > 0 and actual_min < expected_min:
            warnings.append(
                f"Action {action_id}: min rate {actual_min:.3f} < expected {expected_min:.3f}"
            )

        if action_id < 3 and actual_max >= expected_max:
            warnings.append(
                f"Action {action_id}: max rate {actual_max:.3f} >= expected {expected_max:.3f}"
            )

    results = {
        "rate_stats_by_action": rate_stats.to_dict(),
        "warnings": warnings,
    }

    return results


def run_all_validations(actions_df, icustays=None, admissions=None, verbose=True):
    """
    Run all validation checks.

    Args:
        actions_df: DataFrame with extracted actions
        icustays: Optional ICUSTAYS DataFrame for outcome correlation
        admissions: Optional ADMISSIONS DataFrame for outcome correlation
        verbose: Print results

    Returns:
        dict: All validation results
    """
    results = {}

    if verbose:
        print("=" * 80)
        print("ACTION VALIDATION REPORT")
        print("=" * 80)

    # 1. Action distribution
    if verbose:
        print("\n1. Action Distribution")
    dist_results = validate_action_distribution(actions_df)
    results["distribution"] = dist_results

    if verbose:
        print(f"  Total actions: {dist_results['total_actions']:,}")
        for action_id, pct in dist_results["action_distribution"].items():
            print(f"    Action {action_id}: {pct:.1%}")
        if dist_results["warnings"]:
            print("  Warnings:")
            for w in dist_results["warnings"]:
                print(f"    - {w}")

    # 2. Action transitions
    if verbose:
        print("\n2. Action Transitions")
    trans_results = validate_action_transitions(actions_df)
    results["transitions"] = trans_results

    if verbose:
        print(f"  Total transitions: {trans_results['total_transitions']:,}")
        print(f"  Smooth transitions (±1): {trans_results['smooth_pct']:.1f}%")
        print(f"  Mean absolute change: {trans_results['mean_abs_change']:.2f}")
        print(f"  Max jump: {trans_results['max_jump']:.0f}")

    # 3. Rate discretization
    if verbose:
        print("\n3. Rate Discretization")
    disc_results = validate_rate_discretization(actions_df)
    results["discretization"] = disc_results

    if verbose:
        for action_id, stats in disc_results["rate_stats_by_action"].items():
            if "mean" in stats:
                print(
                    f"  Action {action_id}: rate range [{stats['min']:.3f}, {stats['max']:.3f}], "
                    f"mean={stats['mean']:.3f}, n={stats['count']}"
                )
        if disc_results["warnings"]:
            print("  Warnings:")
            for w in disc_results["warnings"]:
                print(f"    - {w}")

    # 4. Temporal patterns
    if verbose:
        print("\n4. Temporal Patterns")
    temp_results = validate_temporal_patterns(actions_df)
    results["temporal"] = temp_results

    if verbose:
        for phase, mean_action in temp_results["mean_action_by_phase"].items():
            print(f"  {phase.capitalize()} phase: mean action = {mean_action:.2f}")
        print(f"  Early → Late change: {temp_results['early_vs_late_diff']:+.2f}")

    # 5. Outcome correlation (if data provided)
    if icustays is not None and admissions is not None:
        if verbose:
            print("\n5. Correlation with Outcomes")
        outcome_results = validate_correlation_with_outcome(actions_df, icustays, admissions)
        results["outcomes"] = outcome_results

        if verbose:
            print(f"  Correlation (max action ↔ mortality): {outcome_results['correlation_max_action_mortality']:.3f}")
            print(f"  Correlation (mean action ↔ mortality): {outcome_results['correlation_mean_action_mortality']:.3f}")
            print("  Mortality by max action:")
            for action_id, stats in outcome_results["mortality_by_max_action"]["mortality_pct"].items():
                print(f"    Max action {action_id}: {stats:.1f}%")

    if verbose:
        print("\n" + "=" * 80)

    return results
