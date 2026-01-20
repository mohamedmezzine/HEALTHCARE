"""Discretize continuous vasopressor rates into discrete actions.

Action Space (Phase 1):
    0: No vasopressor
    1: Low dose (< 0.1 mcg/kg/min)
    2: Medium dose (0.1 - 0.3 mcg/kg/min)
    3: High dose (> 0.3 mcg/kg/min)
"""

import numpy as np
import pandas as pd


class VasopressorActionSpace:
    """Define the vasopressor action space and discretization thresholds."""

    # Action definitions
    NO_PRESSOR = 0
    LOW_DOSE = 1
    MEDIUM_DOSE = 2
    HIGH_DOSE = 3

    # Dose thresholds (mcg/kg/min for norepinephrine)
    LOW_THRESHOLD = 0.1
    MEDIUM_THRESHOLD = 0.3

    # Action names for interpretability
    ACTION_NAMES = {
        NO_PRESSOR: "No vasopressor",
        LOW_DOSE: "Low dose",
        MEDIUM_DOSE: "Medium dose",
        HIGH_DOSE: "High dose",
    }

    @classmethod
    def num_actions(cls):
        """Return the number of discrete actions."""
        return 4

    @classmethod
    def get_action_bounds(cls):
        """Return the dose bounds for each action."""
        return {
            cls.NO_PRESSOR: (0.0, 0.0),
            cls.LOW_DOSE: (0.0, cls.LOW_THRESHOLD),
            cls.MEDIUM_DOSE: (cls.LOW_THRESHOLD, cls.MEDIUM_THRESHOLD),
            cls.HIGH_DOSE: (cls.MEDIUM_THRESHOLD, float("inf")),
        }


def discretize_vasopressor_rate(rate_mcg_kg_min):
    """
    Discretize a continuous vasopressor rate into a discrete action.

    Args:
        rate_mcg_kg_min: Vasopressor rate in mcg/kg/min
                        Can be scalar, np.array, or pd.Series
                        NaN/None treated as no vasopressor

    Returns:
        Discrete action in {0, 1, 2, 3}
        Same type as input (scalar → int, array → np.array, Series → pd.Series)
    """
    # Handle different input types
    is_scalar = np.isscalar(rate_mcg_kg_min)
    is_series = isinstance(rate_mcg_kg_min, pd.Series)

    # Convert to numpy array for vectorized operations
    if is_scalar:
        rates = np.array([rate_mcg_kg_min])
    elif is_series:
        rates = rate_mcg_kg_min.values
    else:
        rates = np.asarray(rate_mcg_kg_min)

    # Initialize with NO_PRESSOR (handles NaN)
    actions = np.full(rates.shape, VasopressorActionSpace.NO_PRESSOR, dtype=int)

    # Discretize based on thresholds
    valid_mask = ~np.isnan(rates) & (rates > 0)

    actions[valid_mask & (rates < VasopressorActionSpace.LOW_THRESHOLD)] = (
        VasopressorActionSpace.LOW_DOSE
    )
    actions[
        valid_mask
        & (rates >= VasopressorActionSpace.LOW_THRESHOLD)
        & (rates < VasopressorActionSpace.MEDIUM_THRESHOLD)
    ] = VasopressorActionSpace.MEDIUM_DOSE
    actions[valid_mask & (rates >= VasopressorActionSpace.MEDIUM_THRESHOLD)] = (
        VasopressorActionSpace.HIGH_DOSE
    )

    # Return same type as input
    if is_scalar:
        return int(actions[0])
    elif is_series:
        return pd.Series(actions, index=rate_mcg_kg_min.index)
    else:
        return actions


def get_action_name(action_id):
    """
    Get human-readable name for an action.

    Args:
        action_id: Integer action in {0, 1, 2, 3} or array-like

    Returns:
        String name or list of names
    """
    if np.isscalar(action_id):
        return VasopressorActionSpace.ACTION_NAMES.get(
            action_id, f"Unknown action {action_id}"
        )
    else:
        return [get_action_name(int(a)) for a in action_id]


def validate_action(action_id):
    """
    Validate that action_id is valid.

    Args:
        action_id: Integer action

    Returns:
        bool: True if valid

    Raises:
        ValueError: If action is invalid
    """
    if not isinstance(action_id, (int, np.integer)):
        raise ValueError(f"Action must be integer, got {type(action_id)}")

    if not 0 <= action_id < VasopressorActionSpace.num_actions():
        raise ValueError(
            f"Action {action_id} out of range [0, {VasopressorActionSpace.num_actions()})"
        )

    return True


def get_representative_rate(action_id):
    """
    Get a representative rate for an action (for visualization/analysis).

    Args:
        action_id: Integer action in {0, 1, 2, 3}

    Returns:
        float: Representative rate in mcg/kg/min
    """
    bounds = VasopressorActionSpace.get_action_bounds()
    lower, upper = bounds[action_id]

    if action_id == VasopressorActionSpace.NO_PRESSOR:
        return 0.0
    elif action_id == VasopressorActionSpace.HIGH_DOSE:
        # Use 2x medium threshold as representative for high dose
        return VasopressorActionSpace.MEDIUM_THRESHOLD * 2
    else:
        # Use midpoint of range
        return (lower + upper) / 2
