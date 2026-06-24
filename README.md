# Healthcare Offline Reinforcement Learning

A machine learning project exploring offline reinforcement learning for healthcare decision-support research using structured ICU-style patient trajectories.

> Research/educational project only. This repository is not intended for clinical use.

## Overview

The project builds an offline RL workflow around patient state features, discrete treatment actions, rewards, and evaluation utilities. It includes behavior cloning and DQN-based modeling components, with code organized into data processing, feature construction, reward logic, models, evaluation, and tests.

## Recruiter Summary

This repository demonstrates practical skills in:

- Python machine learning project structure
- PyTorch model implementation
- Offline reinforcement learning concepts
- Dataset preparation for sequential decision-making
- Evaluation and testing for ML pipelines
- Clean modular code organization

## Project Structure

```text
.
├── src/
│   ├── actions/      # Action mapping and discretization
│   ├── datasets/     # Replay-buffer / dataset builders
│   ├── eval/         # Offline evaluation utilities
│   ├── features/     # Patient state construction
│   ├── models/       # Behavior cloning and DQN agents
│   └── rewards/      # Reward definitions
├── tests/            # Unit tests for core pipeline logic
├── train_offline.py  # Training entry point
└── README.md
```

## Main Components

### Behavior Cloning Baseline

A supervised learning baseline that learns from historical actions and provides a comparison point for RL models.

### Deep Q-Network Agent

A PyTorch-based DQN implementation for learning Q-values over discrete actions in an offline setting.

### Offline Evaluation

Evaluation utilities for comparing model behavior without interacting with a live environment.

## Tech Stack

- Python
- PyTorch
- pandas
- NumPy
- scikit-learn-style ML workflow
- Unit testing with Python test modules

## How to Run

```bash
# 1. Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run training script
python train_offline.py --help
```

## What to Improve Next

- Add a clear architecture diagram
- Add sample data schema documentation
- Add experiment results and plots
- Add exact setup instructions based on the final dependency file
- Rename the repository to something more descriptive, for example `healthcare-offline-rl`

## Status

Documentation cleanup in progress.
