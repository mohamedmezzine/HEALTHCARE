import pandas as pd
import numpy as np
import pytest
from src.features.vitals import extract_vitals_for_episode
from src.features.labs import extract_labs_for_episode
from src.features.config import StateFeatureConfig, DEFAULT_CONFIG

@pytest.fixture
def mock_icustay():
    return {
        "icustay_id": 100,
        "hadm_id": 1000,
        "intime": pd.Timestamp("2100-01-01 10:00:00"),
        "outtime": pd.Timestamp("2100-01-01 14:00:00")
    }

@pytest.fixture
def mock_chartevents():
    # Item ID 211 is Heart Rate (configured in config.py usually, let's assume or verify)
    # DEFAULT_CONFIG has vital_itemids. Let's look up Heart Rate.
    # We will mock the config to specific itemids to be safe.

    data = {
        "icustay_id": [100, 100, 100],
        "charttime": [
            pd.Timestamp("2100-01-01 10:30:00"),
            pd.Timestamp("2100-01-01 11:30:00"),
            pd.Timestamp("2100-01-01 12:30:00")
        ],
        "itemid": [211, 211, 211], # 211 is HR in CareVue
        "valuenum": [80, 85, 90]
    }
    return pd.DataFrame(data)

@pytest.fixture
def mock_labevents():
    # Item ID 50912 is Creatinine
    data = {
        "hadm_id": [1000, 1000],
        "charttime": [
            pd.Timestamp("2100-01-01 10:30:00"),
            pd.Timestamp("2100-01-01 12:30:00")
        ],
        "itemid": [50912, 50912],
        "valuenum": [0.8, 0.9]
    }
    return pd.DataFrame(data)

def test_extract_vitals(mock_icustay, mock_chartevents):
    config = DEFAULT_CONFIG
    # Ensure HR is in config
    assert "heart_rate" in config.vital_features

    vitals_df = extract_vitals_for_episode(
        icustay_id=mock_icustay["icustay_id"],
        intime=mock_icustay["intime"],
        outtime=mock_icustay["outtime"],
        chartevents=mock_chartevents,
        config=config
    )

    # 4 hours duration (10 to 14) -> 0, 1, 2, 3
    assert len(vitals_df) == 4
    assert "heart_rate" in vitals_df.columns
    # Check values
    # hour 0 (10-11): 80
    assert vitals_df.loc[0, "heart_rate"] == 80
    # hour 1 (11-12): 85
    assert vitals_df.loc[1, "heart_rate"] == 85
    # hour 2 (12-13): 90
    assert vitals_df.loc[2, "heart_rate"] == 90
    # hour 3 (13-14): ffill -> 90
    assert vitals_df.loc[3, "heart_rate"] == 90

def test_extract_labs(mock_icustay, mock_labevents):
    config = DEFAULT_CONFIG
    assert "creatinine" in config.lab_features

    labs_df = extract_labs_for_episode(
        icustay_id=mock_icustay["icustay_id"],
        hadm_id=mock_icustay["hadm_id"],
        intime=mock_icustay["intime"],
        outtime=mock_icustay["outtime"],
        labevents=mock_labevents,
        config=config
    )

    assert len(labs_df) == 4
    assert "creatinine" in labs_df.columns

    # hour 0 (10-11): 0.8
    assert labs_df.loc[0, "creatinine"] == 0.8
    # hour 1 (11-12): ffill 0.8
    assert labs_df.loc[1, "creatinine"] == 0.8
    # hour 2 (12-13): 0.9
    assert labs_df.loc[2, "creatinine"] == 0.9
    # hour 3 (13-14): ffill 0.9
    assert labs_df.loc[3, "creatinine"] == 0.9
