import pandas as pd

from src.datasets.builder import build_transitions, split_by_icustay


def test_build_transitions_creates_next_state():
    states = pd.DataFrame(
        {
            "icustay_id": [1, 1, 2],
            "hour": [0, 1, 0],
            "feature_a": [10.0, 11.0, 20.0],
        }
    )
    actions = pd.DataFrame(
        {"icustay_id": [1, 1, 2], "hour": [0, 1, 0], "action": [0, 1, 0]}
    )
    rewards = pd.DataFrame(
        {
            "icustay_id": [1, 1, 2],
            "hour": [0, 1, 0],
            "reward": [0.0, 1.0, 0.0],
            "is_terminal": [False, True, True],
            "survived": [True, True, False],
        }
    )

    transitions = build_transitions(states, actions, rewards)
    row0 = transitions[(transitions["icustay_id"] == 1) & (transitions["hour"] == 0)]
    assert row0.iloc[0]["feature_a_next"] == 11.0

    row1 = transitions[(transitions["icustay_id"] == 1) & (transitions["hour"] == 1)]
    assert pd.isna(row1.iloc[0]["feature_a_next"])


def test_split_by_icustay_no_leakage():
    transitions = pd.DataFrame(
        {
            "icustay_id": [1, 1, 2, 2, 3],
            "hour": [0, 1, 0, 1, 0],
            "feature_a": [0, 1, 2, 3, 4],
            "action": [0, 0, 1, 1, 2],
            "reward": [0, 0, 1, 1, -1],
            "done": [False, True, False, True, True],
            "survived": [True, True, True, True, False],
            "feature_a_next": [1, None, 3, None, None],
        }
    )

    train_df, val_df, test_df = split_by_icustay(transitions, seed=1)
    all_ids = set(train_df["icustay_id"]) | set(val_df["icustay_id"]) | set(
        test_df["icustay_id"]
    )
    assert all_ids == {1, 2, 3}
