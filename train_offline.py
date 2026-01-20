import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.models.bc import BCAgent
from src.models.dqn import DQNAgent
from src.eval.offline import evaluate_bc, evaluate_dqn

def load_dataset(path):
    df = pd.read_parquet(path)

    # Extract columns
    meta_cols = {'icustay_id', 'hour', 'action', 'reward', 'done', 'survived', 'hour_start', 'hour_end'}
    next_cols = [c for c in df.columns if c.endswith('_next')]
    feature_cols = [c for c in df.columns if c not in meta_cols and c not in next_cols]

    states = df[feature_cols].values.astype(np.float32)
    actions = df['action'].values.astype(np.int64)
    rewards = df['reward'].values.astype(np.float32)
    next_states = df[next_cols].values.astype(np.float32)
    dones = df['done'].values.astype(np.float32)

    # Handle NaNs in next_states
    next_states = np.nan_to_num(next_states)

    return states, actions, rewards, next_states, dones, feature_cols

def compute_class_weights(actions, num_actions):
    """Compute inverse frequency weights."""
    counts = np.bincount(actions, minlength=num_actions)
    total = len(actions)
    # Weights = total / (num_classes * count)
    # Add epsilon to avoid div by zero if a class is empty
    weights = total / (num_actions * (counts + 1e-6))
    return weights

class BalancedBatchSampler:
    """Samples batches ensuring representation from all classes."""
    def __init__(self, actions, batch_size):
        self.actions = actions
        self.batch_size = batch_size
        self.num_actions = len(np.unique(actions))
        if self.num_actions < 4: self.num_actions = 4

        # Store indices for each action
        self.indices_by_action = {}
        for a in range(self.num_actions):
            self.indices_by_action[a] = np.where(actions == a)[0]

    def sample(self):
        # We want approx equal number of samples per class in the batch
        # But some classes might be empty in the whole dataset (unlikely here but possible)
        # Assuming we want to force mix.

        samples_per_class = self.batch_size // self.num_actions
        batch_indices = []

        for a in range(self.num_actions):
            indices = self.indices_by_action[a]
            if len(indices) > 0:
                selected = np.random.choice(indices, samples_per_class, replace=True)
                batch_indices.extend(selected)

        # Fill remaining spots if any (due to integer division)
        remaining = self.batch_size - len(batch_indices)
        if remaining > 0:
            extra = np.random.choice(np.arange(len(self.actions)), remaining)
            batch_indices.extend(extra)

        np.random.shuffle(batch_indices)
        return np.array(batch_indices)

def train_bc(train_data, val_data, args):
    print("\nTraining BC Agent...")
    states, actions, _, _, _, _ = train_data
    val_states, val_actions, _, _, _, _ = val_data

    state_dim = states.shape[1]
    # Ensure num_actions is at least 4 for MIMIC vasopressors
    num_actions = 4

    # Compute weights
    class_weights = compute_class_weights(actions, num_actions)
    print(f"Class weights: {class_weights}")

    agent = BCAgent(
        state_dim,
        num_actions,
        device=args.device,
        lr=1e-3,
        class_weights=class_weights
    )

    batch_size = 64
    num_epochs = 10

    dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(states), torch.LongTensor(actions)
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        total_loss = 0
        for b_states, b_actions in dataloader:
            loss = agent.train_step(b_states.numpy(), b_actions.numpy())
            total_loss += loss

        avg_loss = total_loss / len(dataloader)

        # Validation
        val_res = evaluate_bc(agent, val_states, val_actions)
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_res['accuracy']:.4f}")

    return agent

def train_dqn(train_data, val_data, args):
    print("\nTraining DQN Agent...")
    states, actions, rewards, next_states, dones, _ = train_data
    val_states, val_actions, val_rewards, _, _, _ = val_data

    state_dim = states.shape[1]
    num_actions = 4

    agent = DQNAgent(state_dim, num_actions, device=args.device, lr=3e-4)

    batch_size = 64
    num_steps = 5000

    sampler = BalancedBatchSampler(actions, batch_size)

    pbar = tqdm(range(num_steps))
    losses = []

    for step in pbar:
        batch_idx = sampler.sample()

        b_states = states[batch_idx]
        b_actions = actions[batch_idx]
        b_rewards = rewards[batch_idx]
        b_next_states = next_states[batch_idx]
        b_dones = dones[batch_idx]

        loss = agent.update(b_states, b_actions, b_rewards, b_next_states, b_dones)
        losses.append(loss)

        if step % 100 == 0:
            pbar.set_description(f"Loss: {np.mean(losses[-100:]):.4f}")

    # Eval
    res = evaluate_dqn(agent, val_states, val_actions, val_rewards)
    print(f"Validation results: Mean Q: {res['mean_q']:.2f}, Match Rate: {res['match_rate']:.2f}")

    return agent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="artifacts/rl")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Using device: {args.device}")

    # Load data
    train_path = Path(args.data_dir) / "transitions_train.parquet"
    val_path = Path(args.data_dir) / "transitions_val.parquet"
    test_path = Path(args.data_dir) / "transitions_test.parquet"

    train_data = load_dataset(train_path)
    val_data = load_dataset(val_path)
    test_data = load_dataset(test_path)

    print(f"Train size: {len(train_data[0])}")
    print(f"Val size: {len(val_data[0])}")
    print(f"Test size: {len(test_data[0])}")

    # Train BC
    bc_agent = train_bc(train_data, val_data, args)
    bc_agent.save("artifacts/bc_model.pt")

    # Train DQN
    dqn_agent = train_dqn(train_data, val_data, args)
    dqn_agent.save("artifacts/dqn_model.pt")

    # Final Evaluation
    print("\nFINAL EVALUATION (TEST SET)")

    bc_res = evaluate_bc(bc_agent, test_data[0], test_data[1])
    print("\nBC Results:")
    print(f"Accuracy: {bc_res['accuracy']:.4f}")
    print(bc_res['classification_report'])

    dqn_res = evaluate_dqn(dqn_agent, test_data[0], test_data[1], test_data[2])
    print("\nDQN Results:")
    print(f"Mean Q: {dqn_res['mean_q']:.4f}")
    print(f"Match Rate: {dqn_res['match_rate']:.4f}")

if __name__ == "__main__":
    main()
