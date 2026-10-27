"""Tests for De Finetti reserve funds and premium models.

Topic 2: De Finetti, Games, and Ruin.
Author: Maria Lucia Monticelli D’Elias
"""

import numpy as np
import pytest
from therisktheory.de_finetti.reserves import de_finetti_premium, de_finetti_profit, reserve_fund


def test_de_finetti_premium():
    """Test De Finetti premium calculation using expected value principle."""
    ex = 200.0
    theta = 0.15
    expected = 200.0 * 1.15
    res = de_finetti_premium(ex, theta)
    np.testing.assert_allclose(res, expected)


def test_de_finetti_profit():
    """Test periodic profit calculation."""
    premiums = np.array([100.0, 100.0])
    claims = np.array([60.0, 120.0])
    expected = np.array([40.0, -20.0])
    res = de_finetti_profit(premiums, claims)
    np.testing.assert_allclose(res, expected)


def test_reserve_fund_trajectory():
    """Test accumulated reserve fund trajectory over time."""
    initial = 100.0
    premiums = np.array([50.0, 50.0, 50.0])
    claims = np.array([40.0, 60.0, 30.0])
    expected = np.array([110.0, 100.0, 120.0])
    res = reserve_fund(initial, premiums, claims)
    np.testing.assert_allclose(res, expected)


def test_reserve_fund_negative_states():
    """Test reserve fund tracking allowing negative values (ruin detection)."""
    initial = 50.0
    premiums = np.array