"""Configuration for reward functions."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional


class RewardType(Enum):
    """Types of reward functions."""

    SPARSE = "sparse"  # Only terminal reward (survival)
    SHAPED = "shaped"  # Intermediate rewards based on vitals
    SURVIVAL_LOS = "survival_los"  # Survival + length of stay penalty
    CLINICAL = "clinical"  # Based on clinical scores (e.g., SOFA)


@dataclass
class RewardConfig:
    """Configuration for reward computation."""

    # Reward type
    reward_type: RewardType = RewardType.SPARSE

    # Terminal rewards (for survival)
    survival_reward: float = 10.0  # Reward for surviving
    death_penalty: float = -10.0  # Penalty for death

    # Step-based rewards
    timestep_penalty: float = -0.05  # Small penalty per hour (encourage efficiency)

    # Shaped reward parameters (for RewardType.SHAPED)
    vital_stability_bonus: float = 0.1  # Bonus for normal vital signs
    vital_abnormal_penalty: float = -0.2  # Penalty for abnormal vitals
    improvement_bonus: float = 0.2  # Bonus for improving vitals

    # Length of stay weighting (for RewardType.SURVIVAL_LOS)
    los_penalty_weight: float = 0.1  # How much to penalize long stays

    # Vital sign ranges for shaped rewards (dict of {vital_name: (low, high)})
    vital_normal_ranges: Optional[Dict[str, tuple]] = None

    # Discounting
    use_discounting: bool = True
    gamma: float = 0.99  # Discount factor

    def __post_init__(self):
        """Set default vital ranges if not provided."""
        if self.vital_normal_ranges is None:
            # Default normal ranges
            self.vital_normal_ranges = {
                "heart_rate": (60, 100),
                "sbp": (90, 140),
                "dbp": (60, 90),
                "mbp": (70, 105),
                "resp_rate": (12, 20),
                "spo2": (95, 100),
                "temperature": (36.5, 37.5),
                "gcs": (13, 15),
            }


# Predefined configurations
SPARSE_CONFIG = RewardConfig(
    reward_type=RewardType.SPARSE,
    survival_reward=10.0,
    death_penalty=-10.0,
    timestep_penalty=0.0,  # No intermediate penalties
)

SHAPED_CONFIG = RewardConfig(
    reward_type=RewardType.SHAPED,
    survival_reward=10.0,
    death_penalty=-10.0,
    timestep_penalty=-0.05,
    vital_stability_bonus=0.1,
    vital_abnormal_penalty=-0.2,
)

SURVIVAL_LOS_CONFIG = RewardConfig(
    reward_type=RewardType.SURVIVAL_LOS,
    survival_reward=10.0,
    death_penalty=-10.0,
    timestep_penalty=-0.05,
    los_penalty_weight=0.1,
)
