import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Paths, ensure_dirs
from .data_ingest import ingest
from .features import aggregate_impressions
from .model_poisson_irls import fit_poisson_ridge, predict_theta_for_pairs
from .optimize_budget import optimize_allocation_diminishing_returns

def run(
    gdrive_url: str | None,
    lambda_: float,
    company_cold_start: int,
    companies_to_predict: list[int],
    networks_to_predict: list[int],
    total_impressions: int,
):
    paths = Paths()
    ensure_dirs(paths)

    events_df, impressions_df = ingest(gdrive_url, remove_tar_after=False)
    agg = aggregate_impressions(impressions_df)

    fit = fit_poisson_ridge(agg, lambda_=lambda_)
    preds = predict_theta_for_pairs(fit, np.array(companies_to_predict), np.array(networks_to_predict))

    alloc_df, info = optimize_allocation_diminishing_returns(
        preds_df=preds,
        company_id=company_cold_start,
        total_impressions=total_impressions,
    )

    # Save outputs
    preds_out = paths.outputs_dir / "predicted_theta_all_pairs.csv"
    alloc_out = paths.outputs_dir / "budget_allocation_company1.csv"
    meta_out = paths.outputs_dir / "run_metadata.json"

    preds.to_csv(preds_out, index=False)
    alloc_df.to_csv(alloc_out, index=False)
    meta = {
        "pearson_phi": fit.pearson_phi,
        "lambda": lambda_,
        **info,
    }
    meta_out.write_text(json.dumps(meta, indent=2))

    print("Saved:")
    print(" -", preds_out)
    print(" -", alloc_out)
    print(" -", meta_out)
    print("\nDiagnostics:")
    print("Pearson dispersion:", round(fit.pearson_phi, 3))
    print("Optimization status:", info["status"])
    print("HHI:", round(info["hhi"], 4))
    print("Max share used:", round(info["max_share_used"], 4))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="End-to-end pipeline: ingest -> fit -> predict -> optimize.")
    parser.add_argument("--gdrive-url", type=str, default=None, help="Google Drive share link for tarball.")
    parser.add_argument("--lambda", dest="lambda_", type=float, default=10.0, help="Ridge penalty strength.")
    parser.add_argument("--company", type=int, default=1, help="Cold-start company_id to allocate budget for.")
    parser.add_argument("--companies", type=str, default="0-18", help="Company IDs to predict, e.g. '0-18' or '0,1,2'.")
    parser.add_argument("--networks", type=str, default="0-99", help="Network IDs to predict, e.g. '0-99' or '0,5,10'.")
    parser.add_argument("--total-impressions", type=int, default=10_000_000, help="Total impressions to allocate.")
    args = parser.parse_args()

    def parse_range(s: str) -> list[int]:
        s = s.strip()
        if "-" in s and "," not in s:
            a, b = s.split("-")
            return list(range(int(a), int(b) + 1))
        return [int(x) for x in s.split(",") if x.strip()]

    companies = parse_range(args.companies)
    networks = parse_range(args.networks)

    run(
        gdrive_url=args.gdrive_url,
        lambda_=args.lambda_,
        company_cold_start=args.company,
        companies_to_predict=companies,
        networks_to_predict=networks,
        total_impressions=args.total_impressions,
    )
