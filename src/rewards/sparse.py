"""Sparse reward: Only terminal reward based on survival outcome."""

import numpy as np
import pandas as pd

from .config import RewardConfig, SPARSE_CONFIG


class SparseReward:
    """
    Sparse binary reward function.

    Rewards:
    - R(terminal) = +survival_reward if patient survives
    - R(terminal) = +death_penalty if patient dies
    - R(t) = 0 for all non-terminal time steps

    This is the simplest reward function but has credit assignment challenges.
    """

    def __init__(self, config: RewardConfig = None):
        """
        Initialize sparse reward function.

        Args:
            config: Reward configuration
        """
        self.config = config if config is not None else SPARSE_CONFIG

    def compute_reward(
        self, hour: int, is_terminal: bool, survived: bool, **kwargs
    ) -> float:
        """
        Compute reward for a single time step.

        Args:
            hour: Current hour in episode
            is_terminal: Whether this is the last time step
            survived: Whether patient survived (only used if is_terminal)
            **kwargs: Additional arguments (ignored for sparse reward)

        Returns:
            Reward value
        """
        if not is_terminal:
            # No reward for intermediate steps
            return 0.0

        # Terminal reward based on survival
        if survived:
            return self.config.survival_reward
        else:
            return self.config.death_penalty

    def compute_episode_rewards(
        self, episode_length: int, survived: bool
    ) -> np.ndarray:
        """
        Compute rewards for an entire episode.

        Args:
            episode_length: Number of time steps in episode
            survived: Whether patient survived

        Returns:
            Array of rewards, shape (episode_length,)
        """
        rewards = np.zeros(episode_length)

        # Only the final step gets a reward
        if survived:
            rewards[-1] = self.config.survival_reward
        else:
            rewards[-1] = self.config.death_penalty

        return rewards

    def __repr__(self):
        return (
            f"SparseReward(survival={self.config.survival_reward}, "
            f"death={self.config.death_penalty})"
        )
