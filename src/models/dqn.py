"""Deep Q-Network (DQN) agent for offline RL."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from typing import Tuple

class QNetwork(nn.Module):
    def __init__(self, input_dim: int, num_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    """
    Deep Q-Network Agent.
    """
    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        hidden_dim: int = 256,
        target_update_freq: int = 100,
        device: str = "cpu"
    ):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.device = device

        self.q_net = QNetwork(state_dim, num_actions, hidden_dim).to(device)
        self.target_net = copy.deepcopy(self.q_net).to(device)
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.steps = 0

    def update(self, states, actions, rewards, next_states, dones):
        """
        Update Q-network using a batch of transitions.
        """
        self.steps += 1
        self.q_net.train()

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Compute current Q values
        q_values = self.q_net(states).gather(1, actions)

        # Compute target Q values (Double DQN could be implemented here)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1, keepdim=True)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.loss_fn(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def get_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """Select action using epsilon-greedy policy."""
        if np.random.random() < epsilon:
            return np.random.randint(self.num_actions)

        self.q_net.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_net(state_tensor)
            action = torch.argmax(q_values, dim=1).item()
        return action

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for all actions."""
        self.q_net.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_net(state_tensor)
            return q_values.cpu().numpy()[0]

    def save(self, path: str):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path: str):
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
