"""De Finetti reserve fund, profit, and premium models for TheRiskTheory.

Topic 2: De Finetti, game and ruin.
Author: Maria Lucia Monticelli D’Elias
"""

import numpy as np


def reserve_fund(
    initial_capital: float | np.ndarray,
    premiums: float | np.ndarray,
    claims: float | np.ndarray,
) -> np.ndarray:
    """Calculate the insurer's reserve fund over time in the De Finetti framework.

    Args:
        initial_capital: Initial reserve or capital level (u).
        premiums: Collected premium amounts.
        claims: Incurred claim amounts.

    Returns:
        np.ndarray: Resulting reserve fund path or final reserve levels.
    """
    cap_arr = np.asarray(initial_capital, dtype=np.float64)
    prem_arr = np.asarray(premiums, dtype=np.float64)
    claim_arr = np.asarray(claims, dtype=np.float64)

    if np.ndim(prem_arr) > 0:
        reserve = cap_arr + np.cumsum(prem_arr - claim_arr, axis=0)
    else:
        reserve = cap_arr + prem_arr - claim_arr

    return np.asarray(reserve, dtype=np.float64)


def de_finetti_profit(
    premiums: float | np.ndarray,
    claims: float | np.ndarray,
) -> np.ndarray:
    """Calculate the De Finetti financial profit over a portfolio.

    Args:
        premiums: Collected premium income.
        claims: Incurred claims.

    Returns:
        np.ndarray: De Finetti profit values.
    """
    prem_arr = np.asarray(premiums, dtype=np.float64)
    claim_arr = np.asarray(claims, dtype=np.float64)

    profit = prem_arr - claim_arr
    return np.asarray(profit, dtype=np.float64)


def de_finetti_premium(
    expected_claims: float | np.ndarray,
    safety_loading: float | np.ndarray,
) -> np.ndarray:
    """Calculate the De Finetti premium based on expected claims and safety loading.

    Args:
        expected_claims: Expected claim amount.
        safety_loading: Safety loading coefficient.

    Returns:
        np.ndarray: De Finetti premium.
    """
    exp_c = np.asarray(expected_claims, dtype=np.float64)
    loading = np.asarray(safety_loading, dtype=np.float64)

    premium = exp_c * (1.0 + loading)
    return np.asarray(premium, dtype=np.float64)
