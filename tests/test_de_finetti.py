"""Tests for De Finetti reserve fund, profit, and premium models."""

import numpy as np

from therisktheory.de_finetti import (
    de_finetti_premium,
    de_finetti_profit,
    reserve_fund,
)


def test_reserve_fund_scalar():
    """Test reserve fund calculation with scalar values."""
    u_0 = 100.0
    premiums = 50.0
    claims = 30.0
    expected = 120.0
    res = reserve_fund(u_0, premiums, claims)
    np.testing.assert_allclose(res, expected)


def test_reserve_fund_time_series():
    """Test reserve fund path using vector cumulative sum over time."""
    u_0 = 100.0
    premiums = np.array([50.0, 50.0, 50.0])
    claims = np.array([30.0, 40.0, 60.0])
    expected = np.array([120.0, 130.0, 120.0])
    res = reserve_fund(u_0, premiums, claims)
    np.testing.assert_allclose(res, expected)


def test_de_finetti_profit():
    """Test De Finetti profit computation for array inputs."""
    premiums = np.array([100.0, 150.0, 200.0])
    claims = np.array([80.0, 150.0, 220.0])
    expected = np.array([20.0, 0.0, -20.0])
    np.testing.assert_allclose(de_finetti_profit(premiums, claims), expected)


def test_de_finetti_premium():
    """Test De Finetti premium calculation."""
    exp_claims = 500.0
    loading = 0.15
    expected = 575.0
    res = de_finetti_premium(exp_claims, loading)
    np.testing.assert_allclose(res, expected)
