from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import cvxpy as cp

def derive_diminishing_params(theta: np.ndarray, total_impressions: int, half_sat_share: float = 0.04, gamma: float = 1.0):
    """
    Utility per network: U_i(x) = a_i * (1 - exp(-b_i x))
    Match initial slope: U_i'(0) = a_i * b_i = theta_i
    Choose b_i relative to theta for heterogeneity.
    """
    theta = np.asarray(theta, dtype=float)
    N = float(total_impressions)
    b_base = np.log(2.0) / (half_sat_share * N)
    scale = (theta / np.clip(theta.mean(), 1e-12, None)) ** gamma
    b = b_base * scale
    a = theta / np.clip(b, 1e-12, None)
    return a, b

def optimize_allocation_diminishing_returns(
    preds_df: pd.DataFrame,
    company_id: int = 1,
    total_impressions: int = 10_000_000,
    half_sat_share: float = 0.04,
    max_share: float = 0.30,
    hhi_max: float = 0.18,
    min_share: float = 0.0,
    tau_log: float = 0.0,
    eps: float = 1.0,
    gamma: float = 1.0,
    solver: str = "SCS",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    dfc = preds_df[preds_df["company_id"] == company_id].copy()
    if dfc.empty:
        raise ValueError(f"No predictions found for company_id={company_id}.")
    if "theta_hat" not in dfc.columns:
        raise ValueError("preds_df must contain theta_hat.")

    networks = dfc["network_id"].to_numpy()
    theta = dfc["theta_hat"].to_numpy(dtype=float)

    a, b = derive_diminishing_params(theta, total_impressions, half_sat_share=half_sat_share, gamma=gamma)
    k = len(networks)
    N = float(total_impressions)

    x = cp.Variable(k, nonneg=True)  # impressions per network

    concave_terms = cp.multiply(a, (1 - cp.exp(-cp.multiply(b, x))))
    obj = cp.sum(concave_terms)

    if tau_log and tau_log > 0:
        obj += tau_log * cp.sum(cp.log(x + eps))

    constraints = [cp.sum(x) == N]

    if max_share is not None:
        constraints.append(x <= max_share * N)
    if min_share is not None and min_share > 0:
        constraints.append(x >= min_share * N)
    if hhi_max is not None:
        constraints.append(cp.sum_squares(x / N) <= hhi_max)

    problem = cp.Problem(cp.Maximize(obj), constraints)

    if solver.upper() == "SCS":
        problem.solve(solver=cp.SCS, verbose=False)
    else:
        problem.solve(verbose=False)

    if x.value is None:
        raise RuntimeError(f"Optimization failed (status={problem.status}). Try relaxing constraints.")

    alloc = np.maximum(0, np.array(x.value).ravel())
    share = alloc / N
    exp_conversions = float(np.sum(a * (1 - np.exp(-b * alloc))))
    hhi = float(np.sum(share ** 2))

    out = pd.DataFrame({
        "network_id": networks,
        "theta_hat": theta,
        "allocation_impressions": alloc,
        "allocation_share": share,
    }).sort_values("allocation_impressions", ascending=False)

    info = {
        "total_impressions": int(N),
        "expected_conversions": exp_conversions,
        "hhi": hhi,
        "max_share_used": float(share.max()),
        "status": problem.status,
        "objective_value": float(problem.value),
        "solver": solver,
    }
    return out, info
