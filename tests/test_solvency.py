"""Comprehensive tests for solvency and guarantee capital models with 100+ cases.

Topic 1: Solvency and Guarantee Capital.
Author: Maria Lucia Monticelli D’Elias
"""

import numpy as np
import pytest
from therisktheory.solvency_pricing import guarantee_capital, solvency_margin


# ==========================================
# 1. PARAMETRIZED TESTS (100+ Combinations)
# ==========================================

# Generamos una matriz de combinaciones para guarantee_capital:
# 5 capitales * 5 siniestros * 5 márgenes = 125 casos de prueba unitarios independientes.
INITIAL_CAPITALS = [0.0, 50.0, 100.0, 250.0, 500.0]
MAX_CLAims = [100.0, 200.0, 300.0, 400.0, 600.0]
SAFETY_MARGINS = [0.0, 0.05, 0.1, 0.15, 0.2]


@pytest.mark.parametrize("u", INITIAL_CAPITALS)
@pytest.mark.parametrize("max_claim", MAX_CLAims)
@pytest.mark.parametrize("margin", SAFETY_MARGINS)
def test_guarantee_capital_matrix(u, max_claim, margin):
    """Parametrized test suite generating 125 distinct scenarios for guarantee capital."""
    expected = max(0.0, max_claim - u) * (1.0 + margin)
    res = guarantee_capital(u, max_claim, margin)
    np.testing.assert_allclose(res, expected)


# Generamos una matriz de combinaciones para solvency_margin:
# 4 primas * 4 siniestros * 5 coeficientes = 80 casos de prueba unitarios independientes.
PREMIUMS = [500.0, 1000.0, 2500.0, 5000.0]
CLAIMS = [400.0, 1200.0, 2000.0, 4500.0]
COEFFICIENTS = [0.1, 0.15, 0.18, 0.2, 0.25]


@pytest.mark.parametrize("prem", PREMIUMS)
@pytest.mark.parametrize("claim", CLAIMS)
@pytest.mark.parametrize("coeff", COEFFICIENTS)
def test_solvency_margin_matrix(prem, claim, coeff):
    """Parametrized test suite generating 80 distinct scenarios for solvency margin."""
    base = max(prem, claim)
    expected = base * coeff
    res = solvency_margin(prem, claim, coeff)
    np.testing.assert_allclose(res, expected)


# ==========================================
# 2. EDGE CASES & VECTORIZED ARRAYS
# ==========================================

def test_guarantee_capital_vectorized_arrays():
    """Test guarantee capital with full NumPy arrays using vectorization."""
    u = np.array([0.0, 100.0, 200.0])
    max_claim = np.array([150.0, 150.0, 150.0])
    margin = np.array([0.1, 0.1, 0.1])
    
    expected = np.array([150.0, 50.0, 0.0]) * 1.1
    res = guarantee_capital(u, max_claim, margin)
    np.testing.assert_allclose(res, expected)


def test_solvency_margin_vectorized_arrays():
    """Test solvency margin with full NumPy arrays."""
    premiums = np.array([1000.0, 3000.0])
    claims = np.array([2000.0, 1500.0])
    coefficients = np.array([0.2, 0.15])
    
    # max(1000, 2000)*0.2 = 400; max(3000, 1500)*0.15 = 450
    expected = np.array([400.0, 450.0])
    res = solvency_margin(premiums, claims, coefficients)
    np.testing.assert_allclose(res, expected)


# ==========================================
# 3. EXCEPTION & ERROR VALIDATION TESTS
# ==========================================

@pytest.mark.parametrize("invalid_u, invalid_claim, invalid_margin", [
    (-10.0, 100.0, 0.1),
    (100.0, -50.0, 0.1),
    (100.0, 100.0, -1.5),
    (np.nan, 100.0, 0.1),
    (100.0, np.inf, 0.1),
])
def test_guarantee_capital_raises_value_error(invalid_u, invalid_claim, invalid_margin):
    """Test that invalid inputs (negative values, NaNs, infs) raise ValueError."""
    with pytest.raises(ValueError):
        guarantee_capital(invalid_u, invalid_claim, invalid_margin)


@pytest.mark.parametrize("invalid_prem, invalid_claim, invalid_coeff", [
    (-500.0, 100.0, 0.2),
    (500.0, -100.0, 0.2),
    (500.0, 100.0, -0.1),
    (np.nan, 100.0, 0.2),
    (500.0, np.inf, 0.2),
])
def test_solvency_margin_raises_value_error(invalid_prem, invalid_claim, invalid_coeff):
    """Test that invalid inputs in solvency margin raise ValueError."""
    with pytest.raises(ValueError):
        solvency_margin(invalid_prem, invalid_claim, invalid_coeff)