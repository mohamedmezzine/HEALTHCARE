import pytest
import numpy as np
import pandas as pd
from src.rewards.shaped import ShapedReward
from src.rewards.config import RewardConfig, RewardType

def test_shaped_reward_basics():
    config = RewardConfig(
        reward_type=RewardType.SHAPED,
        survival_reward=10.0,
        death_penalty=-10.0,
        timestep_penalty=-0.1,
        vital_stability_bonus=0.1,
        vital_abnormal_penalty=-0.1,
        improvement_bonus=0.5
    )
    reward_fn = ShapedReward(config)

    # Test step reward (non-terminal)
    # Only timestep penalty applies here as vitals are None
    r = reward_fn.compute_reward(
        hour=0,
        is_terminal=False,
        survived=False,
        vitals_current=None,
        vitals_previous=None
    )
    assert r == -0.1

    # Test terminal survival
    # Terminal rewards are ADDED to other rewards.
    # So it should be survival_reward + timestep_penalty (since is_terminal also triggers timestep logic?)
    # Let's check shaped.py logic:
    #
    # if is_terminal:
    #     reward += survival/death
    #
    # if not is_terminal:
    #     reward += timestep_penalty
    #
    # So if is_terminal, NO timestep_penalty.

    r = reward_fn.compute_reward(
        hour=0,
        is_terminal=True,
        survived=True
    )
    assert r == 10.0

    # Test terminal death
    r = reward_fn.compute_reward(
        hour=0,
        is_terminal=True,
        survived=False
    )
    assert r == -10.0

def test_vital_stability_reward():
    config = RewardConfig(
        reward_type=RewardType.SHAPED,
        timestep_penalty=0.0,
        vital_stability_bonus=1.0,
        vital_abnormal_penalty=-1.0,
        vital_normal_ranges={
            "heart_rate": (60, 100)
        }
    )
    reward_fn = ShapedReward(config)

    # Normal HR
    r = reward_fn.compute_reward(
        hour=0, is_terminal=False, survived=False,
        vitals_current={"heart_rate": 80}
    )
    assert r == 1.0 # Only bonus

    # Abnormal HR (Low)
    # Deviation = (60 - 50) / 60 = 0.166...
    # Penalty = -1.0 * min(0.166, 1.0) = -0.166...
    r = reward_fn.compute_reward(
        hour=0, is_terminal=False, survived=False,
        vitals_current={"heart_rate": 50}
    )
    assert abs(r - (-1.0 * (10/60))) < 1e-5

def test_improvement_reward():
    config = RewardConfig(
        reward_type=RewardType.SHAPED,
        timestep_penalty=0.0,
        vital_stability_bonus=0.0, # disable for clarity
        vital_abnormal_penalty=0.0, # disable for clarity
        improvement_bonus=5.0,
        vital_normal_ranges={
            "heart_rate": (60, 100)
        }
    )
    reward_fn = ShapedReward(config)

    # Improved from Low
    # Prev: 50 (Low), Curr: 55 (Still Low, but better)
    r = reward_fn.compute_reward(
        hour=0, is_terminal=False, survived=False,
        vitals_current={"heart_rate": 55},
        vitals_previous={"heart_rate": 50}
    )
    assert r == 5.0

    # Worsened from Low
    # Prev: 50, Curr: 40
    r = reward_fn.compute_reward(
        hour=0, is_terminal=False, survived=False,
        vitals_current={"heart_rate": 40},
        vitals_previous={"heart_rate": 50}
    )
    assert r == 0.0

    # Entered Normal
    # Prev: 50, Curr: 70
    r = reward_fn.compute_reward(
        hour=0, is_terminal=False, survived=False,
        vitals_current={"heart_rate": 70},
        vitals_previous={"heart_rate": 50}
    )
    assert r == 5.0
