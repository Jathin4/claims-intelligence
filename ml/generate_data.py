"""Generate a synthetic, insurance-style claims dataset (seeded, so the same seed gives the same files).

    python ml/generate_data.py --seed 42

Three years of claims: the first year (before MODEL_START) is history only, so features such as
"previous claims in the last 365 days" are complete for every modelled claim.

Writes:
  data/claims.csv        what a company would have: claims + the historical review outcome (the label)
  data/claims_truth.csv  generator-only truth for evaluation: planted anomalies, hidden risk probability.
                         NEVER used as model features.

All data is synthetic. The label means "needed manual review", not "fraud".
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from features import CATEGORIES, MODEL_START, build_features, category_medians

ROOT = Path(__file__).resolve().parent.parent
START, END = pd.Timestamp("2023-11-01"), pd.Timestamp("2026-10-31")   # first year = history only (MODEL_START)

# Per category: share of claims, log-normal amount (median, sigma), mean supporting documents
CATEGORY_SPEC = {
    "vehicle": (0.30, 3500, 0.80, 4.0),
    "home":    (0.22, 2500, 0.90, 3.5),
    "health":  (0.20, 1200, 0.70, 3.0),
    "travel":  (0.15,  400, 0.60, 2.0),
    "device":  (0.13,  300, 0.50, 1.5),
}
REGIONS = ["north", "south", "east", "west", "central"]
TARGET_RATE = 0.12          # share of claims that needed review
ANOMALY_RATE = 0.01         # planted unusual claims


def make_customers(rng, n):
    claim_rate = rng.gamma(shape=1.2, scale=1.0, size=n)              # a few customers claim often
    return pd.DataFrame({"customer_id": [f"CUS-{i:05d}" for i in range(n)],
                         "region": rng.choice(REGIONS, size=n, p=[0.24, 0.22, 0.2, 0.18, 0.16]),
                         "claim_rate": claim_rate})


def claim_dates(rng, n):
    days = (END - START).days
    month_weight = np.array([1.25, 1.2, 1.0, 0.95, 0.9, 0.95, 1.05, 1.05, 0.9, 0.9, 1.0, 1.2])  # winter peak
    out = []
    while len(out) < n:                                              # rejection sampling for seasonality
        d = START + pd.to_timedelta(rng.integers(0, days + 1, size=n), unit="D")
        keep = rng.random(n) < month_weight[d.month - 1] / month_weight.max()
        out.extend(d[keep])
    return pd.DatetimeIndex(out[:n])


def make_claims(rng, customers, n_claims):
    p = customers["claim_rate"] / customers["claim_rate"].sum()
    who = rng.choice(len(customers), size=n_claims, p=p)
    cats = rng.choice(CATEGORIES, size=n_claims, p=[CATEGORY_SPEC[c][0] for c in CATEGORIES])
    med = np.array([CATEGORY_SPEC[c][1] for c in cats], dtype=float)
    sig = np.array([CATEGORY_SPEC[c][2] for c in cats])
    docs_mean = np.array([CATEGORY_SPEC[c][3] for c in cats])
    df = pd.DataFrame({
        "customer_id": customers["customer_id"].to_numpy()[who],
        "region": customers["region"].to_numpy()[who],
        "claim_date": claim_dates(rng, n_claims),
        "claim_category": cats,
        "claim_amount": np.round(med * rng.lognormal(0, sig), 2),
        "document_count": rng.poisson(docs_mean).astype(float),
    })
    # Policy starts 30 days to ~4 years before the customer's first claim
    first = df.groupby("customer_id")["claim_date"].transform("min")
    start = first.groupby(df["customer_id"]).transform(
        lambda s: s.iloc[0] - pd.Timedelta(days=int(rng.integers(30, 1500))))
    df["policy_start_date"] = start
    # A few claims made very soon after the policy started (a common review trigger)
    early = rng.random(len(df)) < 0.03
    df.loc[early, "policy_start_date"] = df.loc[early, "claim_date"] - pd.to_timedelta(rng.integers(1, 60, early.sum()), unit="D")
    # ...but a policy can't start after an earlier claim of the same customer
    df["policy_start_date"] = df.groupby("customer_id")["policy_start_date"].transform("min")
    return df


def plant_anomalies(rng, df):
    """Mark ~1% of claims as unusual, in three known ways, so the detector can be measured later."""
    n = int(len(df) * ANOMALY_RATE)
    idx = rng.choice(df.index, size=n, replace=False)
    kinds = rng.choice(["huge_amount", "burst", "no_documents"], size=n, p=[0.45, 0.30, 0.25])
    df["is_planted_anomaly"], df["anomaly_kind"] = 0, ""
    for i, kind in zip(idx, kinds):
        df.loc[i, ["is_planted_anomaly", "anomaly_kind"]] = 1, kind
        cat = df.loc[i, "claim_category"]
        if kind == "huge_amount":
            df.loc[i, "claim_amount"] = round(CATEGORY_SPEC[cat][1] * rng.uniform(8, 20), 2)
        elif kind == "burst":                                       # 2 extra claims a few days earlier
            extra = df.loc[[i, i]].copy()
            extra["claim_date"] = df.loc[i, "claim_date"] - pd.to_timedelta(rng.integers(1, 10, 2), unit="D")
            extra["is_planted_anomaly"], extra["anomaly_kind"] = 1, "burst"
            df = pd.concat([df, extra], ignore_index=True)
        else:                                                       # high amount, no documents
            df.loc[i, "claim_amount"] = round(CATEGORY_SPEC[cat][1] * rng.uniform(4, 8), 2)
            df.loc[i, "document_count"] = 0
    df["claim_date"] = df["claim_date"].clip(lower=START)
    df["policy_start_date"] = df.groupby("customer_id")["policy_start_date"].transform("min")
    df["policy_start_date"] = df[["policy_start_date", "claim_date"]].min(axis=1)
    return df


def make_label(rng, df):
    """A hidden risk score from several factors plus noise, turned into a probability, then DRAWN.
    So the label is probabilistic: no model can (or should) reach 100%."""
    ratio = df["amount_vs_category_median"].clip(lower=0.05)
    recent = df["days_since_previous_claim"].fillna(9999) < 60
    risk = (1.1 * np.log(ratio)
            + 0.9 * recent
            + 0.25 * df["claim_history_count"].clip(upper=5)
            - 0.45 * np.log1p(df["document_count"].fillna(df["document_count"].median()))
            + 0.6 * (df["policy_age_days"] < 90)
            + df["claim_category"].map({"vehicle": 0.2, "home": 0.1, "health": 0.0, "travel": -0.1, "device": 0.1})
            + 1.0 * df["is_planted_anomaly"]
            + rng.normal(0, 0.8, len(df)))                              # noise the model never sees
    lo, hi = -10.0, 10.0                                               # intercept so the mean rate ≈ TARGET_RATE
    for _ in range(60):
        mid = (lo + hi) / 2
        rate = (1 / (1 + np.exp(-(risk + mid)))).mean()
        lo, hi = (mid, hi) if rate < TARGET_RATE else (lo, mid)
    prob = 1 / (1 + np.exp(-(risk + mid)))
    return (rng.random(len(df)) < prob).astype(int), prob


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--claims", type=int, default=30_000)
    ap.add_argument("--customers", type=int, default=10_000)
    ap.add_argument("--out", type=Path, default=ROOT / "data", help="output folder (tests write to a temp folder)")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    customers = make_customers(rng, args.customers)
    df = make_claims(rng, customers, args.claims)
    df = plant_anomalies(rng, df)
    df = df.sort_values(["claim_date", "customer_id"]).reset_index(drop=True)
    df["claim_id"] = [f"CLM-{i:06d}" for i in range(1, len(df) + 1)]

    feats = build_features(df, category_medians(df))   # the generator's own truth may use all data
    label, prob = make_label(rng, feats)
    df["manual_review_required"] = label

    # Messiness: some missing values, as in real exports
    df.loc[rng.random(len(df)) < 0.03, "document_count"] = np.nan
    df.loc[rng.random(len(df)) < 0.01, "region"] = np.nan

    out = args.out
    out.mkdir(exist_ok=True)
    cols = ["claim_id", "customer_id", "region", "policy_start_date", "claim_date",
            "claim_category", "claim_amount", "document_count", "manual_review_required"]
    df[cols].to_csv(out / "claims.csv", index=False, date_format="%Y-%m-%d")
    truth = df[["claim_id", "is_planted_anomaly", "anomaly_kind"]].assign(hidden_risk_probability=prob.round(4))
    truth.to_csv(out / "claims_truth.csv", index=False)

    print(f"claims: {len(df):,}  customers: {df['customer_id'].nunique():,}  "
          f"dates: {df['claim_date'].min():%Y-%m-%d} → {df['claim_date'].max():%Y-%m-%d}")
    model = df[df["claim_date"] >= MODEL_START]
    print(f"modelling window (from {MODEL_START}): {len(model):,} claims  review rate: {model['manual_review_required'].mean():.3f}  "
          f"planted anomalies: {int(model['is_planted_anomaly'].sum())}")
    print("review rate per category:", model.groupby("claim_category")["manual_review_required"].mean().round(3).to_dict())


if __name__ == "__main__":
    main()
