# Healthcare Offline Reinforcement Learning

Offline reinforcement learning project using structured ICU-style patient trajectories.

## Overview

This project studies an offline RL workflow built from historical transition data. Each transition contains a patient state, an observed action, a reward, the next state, and a terminal flag.

The repository includes two modeling approaches:

1. **Behavior Cloning (BC)** — a supervised baseline that learns to imitate historical actions.
2. **Deep Q-Network (DQN)** — an offline RL agent that estimates Q-values for discrete actions using a target network.

The code is organized around data preparation, action and reward logic, model classes, evaluation utilities, a training script, and unit tests.

## Technical Scope

- PyTorch model implementation
- Behavior cloning baseline
- DQN with target network updates
- Class imbalance handling
- Dataset loading from Parquet transition files
- Offline evaluation with classification metrics and Q-value inspection
- Unit tests for feature, action, and reward logic

## Repository Structure

```text
.
├── artifacts/
│   └── rl/
│       ├── transitions_train.parquet
│       ├── transitions_val.parquet
│       └── transitions_test.parquet
├── src/
│   ├── actions/
│   ├── datasets/
│   ├── eval/
│   │   └── offline.py
│   ├── features/
│   ├── models/
│   │   ├── bc.py
│   │   └── dqn.py
│   └── rewards/
├── tests/
│   ├── test_actions.py
│   ├── test_features.py
│   └── test_rewards.py
├── train_offline.py
└── README.md
```

## Modeling Approach

### Behavior Cloning

The BC model treats the historical action as a supervised classification target. It uses a neural network policy and cross-entropy loss. The implementation supports class weights for imbalanced action distributions.

### Deep Q-Network

The DQN model estimates action values for each state using:

- a feed-forward Q-network;
- a target network;
- mean squared error loss on Bellman targets;
- Adam optimization;
- a fixed discrete action space.

### Imbalance Handling

The training pipeline includes inverse-frequency class weights for behavior cloning and balanced batch sampling for DQN training.

## Evaluation

The project evaluates models offline:

- **Behavior Cloning:** accuracy and classification report.
- **DQN:** mean Q-value, maximum/minimum Q-values, mean Q-value of the historical action, and greedy-action match rate.

These metrics are mainly used for debugging and comparison. Formal off-policy evaluation can be added in a later version.

## How to Run

### 1. Create a virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare data

The training script expects preprocessed Parquet files inside:

```text
artifacts/rl/
```

Expected files:

```text
transitions_train.parquet
transitions_val.parquet
transitions_test.parquet
```

### 4. Train and evaluate

```bash
python train_offline.py --data-dir artifacts/rl
```

You can also choose a device manually:

```bash
python train_offline.py --data-dir artifacts/rl --device cpu
```

## Notes

- The transition datasets are expected to be prepared before training.
- The DQN implementation is a research prototype.
- Evaluation is diagnostic and can be extended with stronger offline evaluation methods.
