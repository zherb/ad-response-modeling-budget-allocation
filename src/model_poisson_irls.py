from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd

@dataclass
class PoissonRidgeFit:
    beta: np.ndarray
    columns: List[str]
    pearson_phi: float
    design: Dict[str, Any]  # dummy column names for companies/networks

def _build_design_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    company_dummies = pd.get_dummies(df["company_id"].astype("category"), drop_first=True)
    network_dummies = pd.get_dummies(df["network_id"].astype("category"), drop_first=True)

    X = pd.concat([company_dummies, network_dummies], axis=1)
    X.insert(0, "intercept", 1.0)

    columns = list(X.columns)
    design = {
        "companies": company_dummies.columns.to_numpy(),
        "networks": network_dummies.columns.to_numpy(),
    }
    return X.to_numpy(dtype=float), columns, design

def fit_poisson_ridge(
    df_agg: pd.DataFrame,
    lambda_: float = 10.0,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> PoissonRidgeFit:
    """
    Fits: conversions ~ Poisson(exp(X beta + offset)), offset = log(impressions)
    Ridge penalty on beta (excluding intercept).
    """
    required = {"company_id", "network_id", "conversions", "impressions"}
    missing = required - set(df_agg.columns)
    if missing:
        raise ValueError(f"df_agg missing columns: {missing}")

    X, columns, design = _build_design_matrix(df_agg)
    y = df_agg["conversions"].to_numpy(dtype=float)
    impressions = df_agg["impressions"].to_numpy(dtype=float)
    offset = np.log(np.clip(impressions, 1.0, None))

    n, p = X.shape

    # penalty matrix: intercept not penalized
    P = np.eye(p)
    P[0, 0] = 0.0

    beta = np.zeros(p, dtype=float)

    for _ in range(max_iter):
        eta = X @ beta + offset
        mu = np.exp(eta)  # expected conversions

        # IRLS working response z
        # z = eta + (y - mu)/mu
        z = eta + (y - mu) / np.clip(mu, 1e-12, None)

        # Weighted least squares without forming diag(W):
        # Solve (X^T W X + λP) β = X^T W (z - offset)
        w = mu  # W = diag(mu)
        WX = X * w[:, None]                 # each row weighted by w_i
        LHS = X.T @ WX + lambda_ * P
        RHS = X.T @ (w * (z - offset))

        beta_new = np.linalg.solve(LHS, RHS)

        if np.linalg.norm(beta_new - beta) < tol:
            beta = beta_new
            break
        beta = beta_new

    # fitted mu and dispersion
    eta_hat = X @ beta + offset
    mu_hat = np.exp(eta_hat)
    pearson_phi = float(np.sum(((y - mu_hat) ** 2) / np.clip(mu_hat, 1e-12, None)) / max(n - p, 1))

    return PoissonRidgeFit(beta=beta, columns=columns, pearson_phi=pearson_phi, design=design)

def predict_theta_for_pairs(
    fit: PoissonRidgeFit,
    company_ids: np.ndarray,
    network_ids: np.ndarray,
) -> pd.DataFrame:
    """
    Predicts theta (conversion rate per impression) for all cartesian pairs
    using the fitted fixed effects model:
      theta = exp(intercept + company_effect + network_effect)
    """
    companies = list(company_ids)
    networks = list(network_ids)

    comp_cols = list(fit.design["companies"])
    net_cols = list(fit.design["networks"])

    rows = []
    for c in companies:
        for n in networks:
            x = [1.0]  # intercept
            for cname in comp_cols:
                x.append(1.0 if cname == c else 0.0)
            for nname in net_cols:
                x.append(1.0 if nname == n else 0.0)
            x = np.asarray(x, dtype=float)
            eta = float(x @ fit.beta)
            theta = float(np.exp(eta))
            rows.append((c, n, theta))

    return pd.DataFrame(rows, columns=["company_id", "network_id", "theta_hat"])
