"""De Finetti reserve funds and surplus trajectory models.

Topic 2: De Finetti, Games, and Ruin.
Author: Maria Lucia Monticelli D’Elias
"""

import numpy as np


def de_finetti_premium(
    expected_severity: float | np.ndarray,
    safety_loading: float | np.ndarray,
) -> np.ndarray:
    """Calculate the premium rate under the De Finetti surplus framework.

    Theoretical Context:
        Within De Finetti's discrete risk model, the premium collected per period 
        is structured using the expected value premium principle, incorporating a 
        safety loading theta over the expected claim severity E[X]:
            P = E[X] * (1 + theta)

    Args:
        expected_severity: Expected claim amount per period (E[X]). Must be non-negative.
        safety_loading: Safety loading parameter (theta). Must be >= 0.

    Returns:
        np.ndarray: Calculated De Finetti premium amount.
    """
    ex = np.asarray(expected_severity, dtype=np.float64)
    theta = np.asarray(safety_loading, dtype=np.float64)

    if np.any(ex < 0.0) or np.any(theta < 0.0):
        raise ValueError("Expected severity and safety loading must be non-negative.")

    return np.asarray(ex * (1.0 + theta), dtype=np.float64)


def de_finetti_profit(
    premiums: np.ndarray,
    claims: np.ndarray,
) -> np.ndarray:
    """Calculate the periodic profit (premiums minus claims) under De Finetti framework."""
    prem_seq = np.asarray(premiums, dtype=np.float64)
    claim_seq = np.asarray(claims, dtype=np.float64)
    return np.asarray(prem_seq - claim_seq, dtype=np.float64)


def reserve_fund(
    initial_reserve: float | np.ndarray,
    premiums: np.ndarray,
    claims: np.ndarray,
) -> np.ndarray:
    """Calculate the accumulated reserve fund trajectory over discrete time periods.

    Theoretical Context:
        The surplus process U_t tracks the capital of the insurer over time t.
        The periodic profit is P_t - C_t, and the accumulated reserve fund at time t is:
            U_t = U_{0} + sum_{k=1}^{t} (P_k - C_k)
        This formulation explicitly allows for negative reserve states to evaluate ruin.

    Args:
        initial_reserve: Starting capital or initial surplus level (U_0).
        premiums: Array of premium income sequence per period (P_t).
        claims: Array of claim payments sequence per period (C_t).

    Returns:
        np.ndarray: Trajectory of accumulated reserves across the time horizon.
    """
    u0 = np.asarray(initial_reserve, dtype=np.float64)
    prem_seq = np.asarray(premiums, dtype=np.float64)
    claim_seq = np.asarray(claims, dtype=np.float64)

    if np.any(prem_seq < 0.0) or np.any(claim_seq < 0.0):
        raise ValueError("Premiums and claims sequences must be non-negative amounts.")

    periodic_profit = prem_seq - claim_seq
    trajectory = u0 + np.cumsum(periodic_profit)

    return np.asarray(trajectory, dtype=np.float64)