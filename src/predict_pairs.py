import numpy as np
import pandas as pd

from .features import aggregate_impressions
from .model_poisson_irls import fit_poisson_ridge, predict_theta_for_pairs

def train_and_predict(
    impressions_df: pd.DataFrame,
    companies_to_predict: np.ndarray,
    networks_to_predict: np.ndarray,
    lambda_: float = 10.0,
) -> tuple[pd.DataFrame, float]:
    agg = aggregate_impressions(impressions_df)
    fit = fit_poisson_ridge(agg, lambda_=lambda_)
    preds = predict_theta_for_pairs(fit, companies_to_predict, networks_to_predict)
    return preds, fit.pearson_phi
