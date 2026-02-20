# Streaming Ad Conversion Modeling & Budget Optimization (Cold Start Scenario)

## Overview

This project develops a predictive modeling and optimization framework for streaming TV advertising performance.

### Objectives

1. **Predict conversion efficiency** for any `(company_id, network_id)` pair — including combinations that have not historically occurred.
2. **Recommend optimal budget allocation** for a new advertiser (`company_id = 1`) with a $100,000 budget (10,000,000 impressions at $0.01 per impression).

The system combines:

- Regularized Poisson Generalized Linear Modeling (GLM)
- Cold-start inference via partial pooling
- Overdispersion diagnostics
- Constrained convex optimization with diminishing returns
- Diversification constraints (HHI cap, max share limits)

This project demonstrates applied statistical modeling, sparse data generalization, and decision-focused optimization.

---

## Data Description

The tarball contains two parquet datasets:

### 1. `events`
Represents meaningful user interactions with companies.

**Columns:**
- `user_id`
- `company_id`

No timestamps provided.

---

### 2. `impressions`
Represents streaming ad exposures and outcomes.

**Columns:**
- `user_id`
- `company_id`
- `network_id`
- `lift` — whether a website visit occurred
- `conv` — whether a purchase occurred

**Key constraints:**

- Not every company advertises on every network.
- No time-series information is available.
- Conversion events are sparse.
- Website visits and impressions occur in non-overlapping time intervals.

---

## Modeling Strategy

### Step 1 — Aggregation

Impression-level data is aggregated to `(company_id, network_id)` level:

- `impressions`
- `conversions`
- `lifts`

We model total conversions using an exposure-adjusted Poisson regression:

\[
Y_{c,n} \sim \text{Poisson}(\mu_{c,n})
\]

\[
\log(\mu_{c,n}) = \log(\text{impressions}_{c,n}) + \beta_0 + \beta_c + \gamma_n
\]

Where:

- Offset term: `log(impressions)`
- Company fixed effects
- Network fixed effects
- Ridge regularization applied to prevent overfitting

---

## Why Poisson?

- Conversion is a rare event.
- Impression counts are large.
- In rare-event regimes, Binomial approximates Poisson.
- Exposure offset naturally models rate per impression.
- Model remains interpretable and scalable.

---

## Regularization

Ridge penalty prevents:

- Extreme estimates for sparse `(company, network)` pairs
- Overfitting to low-volume combinations
- Poor generalization to unseen pairs

The intercept is not penalized.

---

## Overdispersion Diagnostics

Pearson dispersion statistic is computed:

\[
\phi = \frac{\sum (y - \mu)^2 / \mu}{n - p}
\]

If φ > 1:
- Indicates overdispersion
- Suggests potential Negative Binomial extension
- Future improvement: hierarchical Bayesian model

---

## Cold-Start Inference

Company `company_id = 1` has not yet run ads.

Cold-start prediction is enabled via:

- Learned network-level effects
- Global intercept
- Regularized company effects (shrunk toward baseline)

Predicted marginal conversion rate:

\[
\hat{\theta}_{c,n} = \exp(\beta_0 + \beta_c + \gamma_n)
\]

This allows prediction for any `(company, network)` pair.

---

## Budget Optimization

### Business Objective

Allocate 10,000,000 impressions to maximize expected conversions under:

- Diminishing returns
- Diversification constraints
- Risk controls

---

### Utility Function

For each network \( i \):

\[
U_i(x_i) = a_i \left(1 - e^{-b_i x_i}\right)
\]

Where:

- Initial slope matches predicted marginal conversion rate
- \( b_i \) controls saturation rate
- Function is concave (diminishing returns)

---

### Constraints

- Total impressions = 10,000,000
- Maximum share per network (e.g., 30%)
- Herfindahl-Hirschman Index (HHI) cap
- Optional diversification bonus

Optimization solved using `cvxpy`.

---

## Results

The optimized allocation:

- Prioritizes high-performing networks
- Avoids excessive concentration
- Balances expected return and diversification
- Remains stable under reasonable dispersion assumptions

Sensitivity analysis confirms allocation robustness across penalty parameters.

---

## Repository Structure

```
ad-response-modeling-budget-allocation/
│
├── src/
│   ├── data_ingest.py
│   ├── features.py
│   ├── model_poisson_irls.py
│   ├── predict_pairs.py
│   ├── optimize_budget.py
│   └── run_pipeline.py
│
├── notebooks/
│   └── 01_end_to_end_demo.ipynb
│
├── requirements.txt
└── README.md
```

---

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run full pipeline:

```bash
python -m src.run_pipeline \
    --gdrive-url "YOUR_GOOGLE_DRIVE_LINK" \
    --lambda 10 \
    --companies 0-18 \
    --networks 0-99
```

Outputs will be saved to:

- `outputs/predicted_theta_all_pairs.csv`
- `outputs/budget_allocation_company1.csv`
- `outputs/run_metadata.json`

---

## Skills Demonstrated

- Generalized Linear Models (Poisson with exposure offset)
- Regularization and sparse data handling
- Cold-start prediction strategies
- Overdispersion diagnostics
- Convex optimization
- Decision-focused modeling
- Reproducible Python data pipelines

---

## Limitations & Future Work

- No timestamp data prevents modeling sequential exposure effects
- Repeated user exposure not explicitly modeled
- Overdispersion suggests exploring Negative Binomial extension
- Joint modeling of lift and conversion
- Bayesian uncertainty intervals for allocation robustness

---

## Author

Zofia (Zoe) Herbermann  
M.S. Mathematics (Expected 2025)  
Focus: Statistical Modeling, Experimentation, Optimization, Data Science
