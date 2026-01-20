"""Behavior Cloning (BC) agent for offline RL."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from typing import Optional, Tuple, Union

class BCNetwork(nn.Module):
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

class BCAgent:
    """
    Behavior Cloning Agent.
    Learns a policy pi(a|s) via supervised learning on the demonstration dataset.
    """
    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        lr: float = 1e-3,
        hidden_dim: int = 256,
        device: str = "cpu",
        class_weights: Optional[Union[np.ndarray, torch.Tensor]] = None
    ):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.device = device

        self.model = BCNetwork(state_dim, num_actions, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Initialize Loss with weights if provided
        if class_weights is not None:
            if not isinstance(class_weights, torch.Tensor):
                class_weights = torch.FloatTensor(class_weights)
            class_weights = class_weights.to(device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

    def train_step(self, states, actions):
        """
        Perform one gradient update.
        states: (batch_size, state_dim)
        actions: (batch_size,)  -- integer class indices
        """
        self.model.train()

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)

        logits = self.model(states)
        loss = self.criterion(logits, actions)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def predict(self, state: np.ndarray) -> int:
        """Predict best action for a single state."""
        self.model.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            logits = self.model(state_tensor)
            action = torch.argmax(logits, dim=1).item()
        return action

    def get_probs(self, state: np.ndarray) -> np.ndarray:
        """Get action probabilities for a single state."""
        self.model.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            logits = self.model(state_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        return probs

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
