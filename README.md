# Healthcare Offline Reinforcement Learning

Offline reinforcement learning project for healthcare decision-support research using structured ICU-style patient trajectories.

> **Important:** This repository is for research and education only. It is not a medical product and must not be used for real clinical decisions.

## Overview

This project explores how offline reinforcement learning can be applied to historical healthcare data. The pipeline works with preprocessed transition datasets where each row represents a patient state, an observed treatment action, a reward, the next state, and a terminal flag.

The repository includes two modeling approaches:

1. **Behavior Cloning (BC)** — a supervised baseline that learns to imitate historical actions.
2. **Deep Q-Network (DQN)** — an offline RL agent that estimates Q-values for discrete actions using a target network.

The project is structured like a real ML research prototype: data preparation modules, action/reward logic, model classes, evaluation utilities, a training entry point, and unit tests.

## Problem Statement

In healthcare, direct online experimentation is usually not acceptable because actions can affect patient safety. Offline RL studies policies using already-collected data. This project focuses on the technical workflow needed for that type of research:

- represent patient states as feature vectors;
- map treatment decisions into a discrete action space;
- assign rewards from outcomes or shaped clinical signals;
- train models without interacting with a live environment;
- evaluate learned policies offline.

## What This Project Demonstrates

- PyTorch model implementation
- Offline reinforcement learning concepts
- Behavior cloning baseline design
- DQN with target network updates
- Handling imbalanced medical action classes
- Dataset loading from Parquet transition files
- Evaluation with classification metrics and Q-value inspection
- Unit testing for feature, action, and reward logic

## Repository Structure

```text
.
├── artifacts/
│   └── rl/                         # Expected preprocessed transition datasets
│       ├── transitions_train.parquet
│       ├── transitions_val.parquet
│       └── transitions_test.parquet
├── src/
│   ├── actions/                    # Treatment/action discretization logic
│   ├── datasets/                   # Dataset and replay-buffer builders
│   ├── eval/
│   │   └── offline.py              # Offline evaluation utilities
│   ├── features/                   # State feature construction
│   ├── models/
│   │   ├── bc.py                   # Behavior cloning agent
│   │   └── dqn.py                  # Deep Q-Network agent
│   └── rewards/                    # Reward definitions
├── tests/
│   ├── test_actions.py
│   ├── test_features.py
│   └── test_rewards.py
├── train_offline.py                # Main training script
└── README.md
```

## Modeling Approach

### Behavior Cloning Baseline

The BC model treats the historical treatment action as a supervised classification target. It uses a neural network policy and cross-entropy loss. The implementation also supports class weights, which is useful when some treatment actions appear much less often than others.

### Deep Q-Network Agent

The DQN agent estimates action values for each state. It uses:

- a feed-forward Q-network;
- a target network;
- mean squared error loss on Bellman targets;
- Adam optimization;
- a fixed discrete action space.

### Imbalance Handling

Healthcare treatment data is often imbalanced. The training pipeline includes:

- inverse-frequency class weights for behavior cloning;
- a balanced batch sampler for DQN training.

## Evaluation

The project evaluates models offline:

- **Behavior Cloning:** accuracy and classification report.
- **DQN:** mean Q-value, maximum/minimum Q-values, mean Q-value of the historical action, and greedy-action match rate against the observed action.

These metrics are useful for debugging and comparison, but they are not enough to validate a healthcare policy. A stronger version of this project should add formal off-policy evaluation methods.

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

## Current Limitations

- The repository assumes the transition datasets are already prepared.
- The current DQN implementation is a research prototype, not a production medical model.
- Evaluation is mostly diagnostic and should be extended with stronger off-policy evaluation.
- The project needs experiment tracking and clearer dataset documentation.

## Next Improvements

- Add a complete data-preparation guide.
- Add a diagram of the offline RL pipeline.
- Add experiment results with tables and plots.
- Add formal off-policy evaluation such as FQE or importance-sampling methods.
- Add Double DQN, CQL, or conservative offline RL baselines.
- Add CI for tests.
- Rename the repository to `healthcare-offline-rl`.

## Skills Highlighted

**Machine Learning:** offline RL, behavior cloning, DQN, class imbalance handling  
**Python:** modular ML code, training scripts, evaluation utilities  
**PyTorch:** neural networks, optimizers, target networks, model saving/loading  
**Data:** Parquet datasets, feature extraction, transition-based modeling  
**Software Engineering:** tests, project structure, separation of concerns
