"""Shaped reward: Dense rewards based on intermediate vital sign stability."""

import numpy as np
import pandas as pd
from typing import Dict, Optional

from .config import RewardConfig, SHAPED_CONFIG


class ShapedReward:
    """
    Shaped reward function with dense intermediate rewards.

    Rewards:
    - Terminal: +survival_reward or +death_penalty
    - Per timestep: Small penalty to encourage efficiency
    - Vital stability: Bonus for vitals in normal range
    - Vital abnormality: Penalty for vitals outside normal range
    - Improvement: Bonus for improving abnormal vitals

    This provides learning signals throughout the episode.
    """

    def __init__(self, config: RewardConfig = None):
        """
        Initialize shaped reward function.

        Args:
            config: Reward configuration
        """
        self.config = config if config is not None else SHAPED_CONFIG

    def compute_reward(
        self,
        hour: int,
        is_terminal: bool,
        survived: bool,
        vitals_current: Optional[Dict[str, float]] = None,
        vitals_previous: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> float:
        """
        Compute reward for a single time step.

        Args:
            hour: Current hour in episode
            is_terminal: Whether this is the last time step
            survived: Whether patient survived
            vitals_current: Current vital signs {name: value}
            vitals_previous: Previous vital signs (for improvement detection)
            **kwargs: Additional arguments

        Returns:
            Reward value
        """
        reward = 0.0

        # 1. Terminal reward
        if is_terminal:
            if survived:
                reward += self.config.survival_reward
            else:
                reward += self.config.death_penalty

        # 2. Timestep penalty (encourage efficiency)
        if not is_terminal:
            reward += self.config.timestep_penalty

        # 3. Vital sign stability (intermediate rewards)
        if vitals_current is not None:
            vital_reward = self._compute_vital_stability_reward(vitals_current)
            reward += vital_reward

        # 4. Improvement bonus (if vitals improved from previous step)
        if vitals_current is not None and vitals_previous is not None:
            improvement_reward = self._compute_improvement_reward(
                vitals_current, vitals_previous
            )
            reward += improvement_reward

        return reward

    def _compute_vital_stability_reward(self, vitals: Dict[str, float]) -> float:
        """
        Compute reward based on vital sign stability.

        Args:
            vitals: Dictionary of vital sign values

        Returns:
            Reward contribution from vital stability
        """
        reward = 0.0
        normal_ranges = self.config.vital_normal_ranges

        for vital_name, value in vitals.items():
            if vital_name not in normal_ranges:
                continue

            if pd.isna(value):
                # Missing vital = slight penalty
                reward += self.config.vital_abnormal_penalty * 0.5
                continue

            low, high = normal_ranges[vital_name]

            if low <= value <= high:
                # Normal range = small bonus
                reward += self.config.vital_stability_bonus
            else:
                # Outside normal range = penalty
                # Penalty scales with how far outside normal
                if value < low:
                    deviation = (low - value) / low
                elif value > high:
                    deviation = (value - high) / high
                else:
                    deviation = 0.0

                penalty = self.config.vital_abnormal_penalty * min(deviation, 1.0)
                reward += penalty

        return reward

    def _compute_improvement_reward(
        self, vitals_current: Dict[str, float], vitals_previous: Dict[str, float]
    ) -> float:
        """
        Compute reward for improving abnormal vitals.

        Args:
            vitals_current: Current vitals
            vitals_previous: Previous vitals

        Returns:
            Reward contribution from improvement
        """
        reward = 0.0
        normal_ranges = self.config.vital_normal_ranges

        for vital_name in vitals_current.keys():
            if vital_name not in normal_ranges:
                continue

            curr_val = vitals_current.get(vital_name)
            prev_val = vitals_previous.get(vital_name)

            # Skip if either is missing
            if pd.isna(curr_val) or pd.isna(prev_val):
                continue

            low, high = normal_ranges[vital_name]

            # Check if vital improved (moved closer to normal range)
            improved = False

            if prev_val < low and curr_val > prev_val:
                # Was too low, increased
                improved = True
            elif prev_val > high and curr_val < prev_val:
                # Was too high, decreased
                improved = True
            elif prev_val < low and low <= curr_val <= high:
                # Entered normal range from below
                improved = True
            elif prev_val > high and low <= curr_val <= high:
                # Entered normal range from above
                improved = True

            if improved:
                reward += self.config.improvement_bonus

        return reward

    def compute_episode_rewards(
        self,
        episode_length: int,
        survived: bool,
        vitals_sequence: Optional[pd.DataFrame] = None,
    ) -> np.ndarray:
        """
        Compute rewards for an entire episode.

        Args:
            episode_length: Number of time steps
            survived: Whether patient survived
            vitals_sequence: DataFrame with vitals over time (optional)
                           Expected columns: hour, heart_rate, sbp, etc.

        Returns:
            Array of rewards, shape (episode_length,)
        """
        rewards = np.zeros(episode_length)

        for hour in range(episode_length):
            is_terminal = hour == episode_length - 1

            # Extract vitals if available
            vitals_current = None
            vitals_previous = None

            if vitals_sequence is not None:
                curr_row = vitals_sequence[vitals_sequence["hour"] == hour]
                if len(curr_row) > 0:
                    vitals_current = curr_row.iloc[0].to_dict()

                if hour > 0:
                    prev_row = vitals_sequence[vitals_sequence["hour"] == hour - 1]
                    if len(prev_row) > 0:
                        vitals_previous = prev_row.iloc[0].to_dict()

            # Compute reward
            rewards[hour] = self.compute_reward(
                hour=hour,
                is_terminal=is_terminal,
                survived=survived,
                vitals_current=vitals_current,
                vitals_previous=vitals_previous,
            )

        return rewards

    def __repr__(self):
        return (
            f"ShapedReward(survival={self.config.survival_reward}, "
            f"death={self.config.death_penalty}, "
            f"timestep={self.config.timestep_penalty}, "
            f"stability={self.config.vital_stability_bonus})"
        )
