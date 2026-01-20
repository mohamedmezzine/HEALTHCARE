# Reward Functions Guide

## Overview

The reward system provides multiple reward function designs for learning ICU treatment policies from offline data.

## Reward Function Types

### 1. **Sparse Reward** (Recommended for baseline)

**Design:**
- R(terminal) = +10 if patient survives
- R(terminal) = -10 if patient dies
- R(t) = 0 for all intermediate steps

**Pros:**
- Simple and interpretable
- Directly optimizes survival
- No reward shaping bias

**Cons:**
- Credit assignment problem (which actions led to survival?)
- Slow learning (sparse signal)

**Use when:** Starting offline RL, want simple baseline

**Example:**
```python
from src.rewards import SPARSE_CONFIG, compute_all_rewards

rewards_df = compute_all_rewards(
    icustays, admissions, config=SPARSE_CONFIG
)
```

---

### 2. **Shaped Reward** (Recommended for better learning)

**Design:**
- R(terminal) = +10 if survived, -10 if died
- R(step) = -0.05 per hour (efficiency penalty)
- R(vitals) = +0.1 for each vital in normal range
- R(vitals) = -0.2 for each vital outside normal range
- R(improvement) = +0.2 for improving abnormal vitals

**Pros:**
- Dense learning signal throughout episode
- Encourages stable vitals
- Rewards clinical improvement
- Faster learning than sparse

**Cons:**
- Requires domain knowledge for shaping
- May introduce bias if ranges wrong
- More complex to tune

**Use when:** Sparse rewards too slow, want intermediate feedback

**Example:**
```python
from src.rewards import SHAPED_CONFIG, compute_all_rewards

# Needs vitals data
rewards_df = compute_all_rewards(
    icustays, admissions,
    config=SHAPED_CONFIG,
    vitals_df=vitals_df  # From state features
)
```

---

### 3. **Survival + LOS** (Balances survival and efficiency)

**Design:**
- R(terminal) = +10 if survived, -10 if died
- R(terminal) -= 0.1 × episode_length (penalize long stays)

**Pros:**
- Encourages both survival AND shorter ICU stays
- Clinically relevant (reduce ICU time)

**Cons:**
- May rush discharge inappropriately
- Trades off survival for efficiency

**Use when:** Want to optimize both outcome and resource use

**Example:**
```python
from src.rewards.config import SURVIVAL_LOS_CONFIG

rewards_df = compute_all_rewards(
    icustays, admissions, config=SURVIVAL_LOS_CONFIG
)
```

---

## Quick Start

### Command Line

```bash
# Sparse rewards (simplest)
python -m src.rewards.builder \
    --data-dir artifacts/cleaned \
    --reward-type sparse \
    --output artifacts/rewards_sparse.parquet

# Shaped rewards (needs vitals)
python -m src.rewards.builder \
    --data-dir artifacts/cleaned \
    --reward-type shaped \
    --vitals artifacts/states.parquet \
    --output artifacts/rewards_shaped.parquet

# Survival + LOS
python -m src.rewards.builder \
    --data-dir artifacts/cleaned \
    --reward-type survival_los \
    --output artifacts/rewards_los.parquet
```

### Python API

```python
from src.rewards import (
    compute_all_rewards,
    SPARSE_CONFIG,
    SHAPED_CONFIG,
)

# Load data
icustays = pd.read_parquet('artifacts/cleaned/ICUSTAYS.parquet')
admissions = pd.read_parquet('artifacts/cleaned/ADMISSIONS.parquet')

# Compute sparse rewards
rewards_df = compute_all_rewards(
    icustays, admissions, config=SPARSE_CONFIG
)

# Result columns:
# - icustay_id
# - hour
# - reward
# - is_terminal
# - survived
```

## Custom Reward Functions

### Create Custom Config

```python
from src.rewards.config import RewardConfig, RewardType

custom_config = RewardConfig(
    reward_type=RewardType.SHAPED,
    survival_reward=20.0,  # Increase survival importance
    death_penalty=-20.0,
    timestep_penalty=-0.1,  # Stronger efficiency pressure
    vital_stability_bonus=0.2,  # Increase stability reward
    vital_abnormal_penalty=-0.3,
    improvement_bonus=0.5,  # Big bonus for improvement
)

rewards_df = compute_all_rewards(
    icustays, admissions, config=custom_config, vitals_df=vitals_df
)
```

### Extend with New Reward Type

```python
from src.rewards.config import RewardConfig

class MyCustomReward:
    def __init__(self, config: RewardConfig):
        self.config = config

    def compute_reward(self, hour, is_terminal, survived, **kwargs):
        # Your custom logic
        reward = 0.0

        if is_terminal:
            reward = self.config.survival_reward if survived else self.config.death_penalty

        # Add your custom intermediate rewards
        # e.g., based on lactate levels, SOFA score, etc.

        return reward
```

## Reward Statistics

From your MIMIC-III demo data (136 episodes, 14,599 hours):

```
Sparse Reward Statistics:
  Mean reward: 0.03 (mostly zeros)
  Std: 0.96
  Range: [-10, +10]
  Non-zero rewards: 136 (only terminal steps)
  Survival rate: ~69% (94 survived, 42 died)
```

Expected shaped reward statistics (with vitals):
```
Shaped Reward Statistics:
  Mean reward: ~-0.5 to +0.5
  Non-zero rewards: ~14,599 (all steps)
  Dense signal throughout episodes
```

## Integration with RL

### In Offline RL Dataset

```python
from src.actions import load_and_extract_actions
from src.features import build_all_state_features
from src.rewards import compute_all_rewards, SPARSE_CONFIG

# Extract components
actions_df = load_and_extract_actions('artifacts/cleaned')
states_df, _ = build_all_state_features('artifacts/cleaned')
rewards_df = compute_all_rewards(icustays, admissions, SPARSE_CONFIG)

# Merge into (s, a, r, s') tuples
rl_data = states_df.merge(
    actions_df[['icustay_id', 'hour', 'action']],
    on=['icustay_id', 'hour']
).merge(
    rewards_df[['icustay_id', 'hour', 'reward', 'is_terminal']],
    on=['icustay_id', 'hour']
)

# Now you have complete MDP tuples!
```

## Validation Checks

### 1. Terminal Reward Distribution

```python
terminal_rewards = rewards_df[rewards_df['is_terminal']]['reward']

survived_rewards = terminal_rewards[rewards_df[rewards_df['is_terminal']]['survived']]
died_rewards = terminal_rewards[~rewards_df[rewards_df['is_terminal']]['survived']]

print(f"Survived: {survived_rewards.mean():.1f}")  # Should be +10
print(f"Died: {died_rewards.mean():.1f}")  # Should be -10
```

### 2. Reward Sum per Episode

```python
episode_returns = rewards_df.groupby('icustay_id')['reward'].sum()

print(f"Average return: {episode_returns.mean():.2f}")
print(f"Return std: {episode_returns.std():.2f}")

# Should see bimodal distribution (survived vs died)
```

### 3. Shaped Reward Sanity

```python
# Check intermediate rewards make sense
intermediate = rewards_df[~rewards_df['is_terminal']]

print(f"Intermediate rewards:")
print(f"  Mean: {intermediate['reward'].mean():.3f}")  # Should be small
print(f"  Non-zero: {(intermediate['reward'] != 0).sum()}")  # Should be many
```

## Tips for Reward Design

1. **Start with Sparse**: Establish baseline performance
2. **Add Shaping Gradually**: One signal at a time (vitals → labs → improvement)
3. **Validate Clinically**: Check if rewards match clinical intuition
4. **Monitor Returns**: Track episode returns during training
5. **Ablation Studies**: Test with/without each reward component

## Common Issues

**Issue**: All rewards are zero
- Check: Are you extracting terminal rewards correctly?
- Fix: Ensure `is_terminal` flag is set

**Issue**: Rewards too large/small
- Check: Magnitude relative to discount factor (γ=0.99)
- Fix: Scale rewards appropriately

**Issue**: Shaped rewards dominate terminal
- Check: Ratio of intermediate to terminal rewards
- Fix: Reduce intermediate reward magnitudes

**Issue**: No learning signal
- Check: Reward variance across episodes
- Fix: Ensure rewards differentiate good/bad policies

## References

- Komorowski et al. (2018): "The Artificial Intelligence Clinician" - Used similar sparse survival rewards
- Raghu et al. (2017): Continuous state-space models for sepsis treatment - Shaped rewards with clinical scores
