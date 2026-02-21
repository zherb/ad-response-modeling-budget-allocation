import pandas as pd

def aggregate_impressions(impressions_df: pd.DataFrame) -> pd.DataFrame:
    required = {"company_id", "network_id", "conv", "lift"}
    missing = required - set(impressions_df.columns)
    if missing:
        raise ValueError(f"Missing columns in impressions_df: {missing}")

    agg = (
        impressions_df
        .groupby(["company_id", "network_id"], as_index=False)
        .agg(
            impressions=("conv", "size"),
            conversions=("conv", "sum"),
            lifts=("lift", "sum"),
        )
    )
    return agg
