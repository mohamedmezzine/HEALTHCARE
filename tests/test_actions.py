import pandas as pd
import numpy as np
import pytest
from src.actions.discretizer import discretize_vasopressor_rate, VasopressorActionSpace

def test_discretize_vasopressor_rate():
    # Test scalar
    assert discretize_vasopressor_rate(0.0) == VasopressorActionSpace.NO_PRESSOR
    assert discretize_vasopressor_rate(0.05) == VasopressorActionSpace.LOW_DOSE
    assert discretize_vasopressor_rate(0.15) == VasopressorActionSpace.MEDIUM_DOSE
    assert discretize_vasopressor_rate(0.5) == VasopressorActionSpace.HIGH_DOSE

    # Test array
    rates = np.array([0.0, 0.05, 0.2, 0.4])
    actions = discretize_vasopressor_rate(rates)
    expected = np.array([0, 1, 2, 3])
    np.testing.assert_array_equal(actions, expected)

    # Test Series
    s_rates = pd.Series([0.0, 0.2])
    s_actions = discretize_vasopressor_rate(s_rates)
    assert isinstance(s_actions, pd.Series)
    assert s_actions.iloc[0] == 0
    assert s_actions.iloc[1] == 2

def test_action_bounds():
    bounds = VasopressorActionSpace.get_action_bounds()
    assert bounds[0] == (0.0, 0.0)
    assert bounds[1] == (0.0, 0.1)
    assert bounds[2] == (0.1, 0.3)
    assert bounds[3] == (0.3, float("inf"))
