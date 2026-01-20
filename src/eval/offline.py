import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report
from typing import Dict, Any
from src.models.bc import BCAgent
from src.models.dqn import DQNAgent

def evaluate_bc(agent: BCAgent, states: np.ndarray, actions: np.ndarray) -> Dict[str, Any]:
    """
    Evaluate BC agent using classification metrics.
    """
    preds = []
    batch_size = 256

    # Batch prediction to avoid OOM
    for i in range(0, len(states), batch_size):
        batch_states = states[i:i+batch_size]
        batch_preds = []
        for state in batch_states:
            batch_preds.append(agent.predict(state))
        preds.extend(batch_preds)

    preds = np.array(preds)
    acc = accuracy_score(actions, preds)
    report = classification_report(actions, preds, output_dict=True, zero_division=0)

    return {
        "accuracy": acc,
        "classification_report": report
    }

def evaluate_dqn(agent: DQNAgent, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray) -> Dict[str, Any]:
    """
    Evaluate DQN agent by inspecting Q-values on the dataset.
    """
    q_values_list = []
    batch_size = 256

    for i in range(0, len(states), batch_size):
        batch_states = states[i:i+batch_size]
        # We can't batch predict with get_q_values efficiently without changing the method,
        # but for evaluation let's just loop or implement batching in agent if needed.
        # Let's trust the loop for small datasets.
        # Actually, let's just use the Q-net directly for efficiency.
        agent.q_net.eval()
        with torch.no_grad():
            batch_states_tensor = torch.FloatTensor(batch_states).to(agent.device)
            qs = agent.q_net(batch_states_tensor).cpu().numpy()
            q_values_list.append(qs)

    all_q_values = np.concatenate(q_values_list, axis=0)

    # Metrics
    mean_q = np.mean(all_q_values)
    max_q = np.max(all_q_values)
    min_q = np.min(all_q_values)

    # Q-value of the action actually taken
    taken_action_q = all_q_values[np.arange(len(actions)), actions]
    mean_taken_q = np.mean(taken_action_q)

    # Greedy action match rate (how often does DQN agree with clinician?)
    greedy_actions = np.argmax(all_q_values, axis=1)
    match_rate = np.mean(greedy_actions == actions)

    return {
        "mean_q": mean_q,
        "max_q": max_q,
        "min_q": min_q,
        "mean_q_taken": mean_taken_q,
        "match_rate": match_rate
    }
