"""Point-in-time features, shared by the data generator, training and the API.

Every feature uses only information available on the claim's own date: history features look at the
same customer's EARLIER claims only, and category medians must come from training data.
"""
import numpy as np
import pandas as pd

CATEGORIES = ["vehicle", "home", "health", "travel", "device"]
NUMERIC = ["claim_amount", "policy_age_days", "claim_history_count",
           "days_since_previous_claim", "document_count", "amount_vs_category_median"]
CATEGORICAL = ["claim_category"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "manual_review_required"
HISTORY_DAYS = 365          # "previous claims" = the customer's claims in the 365 days before this one
BURST_DAYS = 14             # claims in the 14 days before: a direct "burst" signal for the anomaly detector
MODEL_START = "2024-11-01"  # claims before this date are history only (a warm-up year), never modelled


ANOMALY_FEATURES = ["log_amount_ratio", "claim_history_count", "log_days_since_previous",
                    "document_count", "log_policy_age", "claims_last_14d"]
NO_PREVIOUS_CLAIM_DAYS = 3 * 365    # "no earlier claim" is treated as a very long gap, not as missing


def anomaly_matrix(feats: pd.DataFrame, document_median: float) -> pd.DataFrame:
    """Inputs for the anomaly detector, built from the model features. Logs tame the long tails,
    so the detector isn't dominated by raw amounts. `document_median` comes from training data."""
    return pd.DataFrame({
        "log_amount_ratio": np.log(feats["amount_vs_category_median"].clip(lower=1e-3)),
        "claim_history_count": feats["claim_history_count"].astype(float),
        "log_days_since_previous": np.log1p(feats["days_since_previous_claim"].fillna(NO_PREVIOUS_CLAIM_DAYS)),
        "document_count": feats["document_count"].fillna(document_median),
        "log_policy_age": np.log1p(feats["policy_age_days"].clip(lower=0)),
        "claims_last_14d": feats["claims_last_14d"].astype(float),
    }, index=feats.index)[ANOMALY_FEATURES]


def category_medians(train: pd.DataFrame) -> dict[str, float]:
    """Median claim amount per category. Compute on TRAINING data only, then reuse everywhere."""
    return train.groupby("claim_category")["claim_amount"].median().to_dict()


def build_features(claims: pd.DataFrame, medians: dict[str, float]) -> pd.DataFrame:
    """Add the model features to a copy of the claims table (row order is kept)."""
    df = claims.copy()
    order = df.sort_values(["customer_id", "claim_date", "claim_id"]).index
    s = df.loc[order]
    by_customer = s.groupby("customer_id", sort=False)
    # Earlier claims within HISTORY_DAYS. A rolling window, not an all-time count: an all-time count
    # only ever grows, which would make the data drift upwards over time (found in EDA).
    days = (s["claim_date"] - pd.Timestamp("1970-01-01")).dt.days.to_numpy()
    counts, burst = np.empty(len(s), dtype=int), np.empty(len(s), dtype=int)
    for _, pos in by_customer.indices.items():
        d = days[pos]                                       # this customer's dates, already sorted
        counts[pos] = np.arange(len(d)) - np.searchsorted(d, d - HISTORY_DAYS, side="left")
        burst[pos] = np.arange(len(d)) - np.searchsorted(d, d - BURST_DAYS, side="left")
    df.loc[order, "claim_history_count"] = counts
    df.loc[order, "claims_last_14d"] = burst
    gaps = by_customer["claim_date"].diff().dt.days                               # NaN for a first claim
    df.loc[order, "days_since_previous_claim"] = gaps.to_numpy()
    df["policy_age_days"] = (df["claim_date"] - df["policy_start_date"]).dt.days
    median = df["claim_category"].map(medians).astype(float)
    df["amount_vs_category_median"] = df["claim_amount"] / median
    df["claim_history_count"] = df["claim_history_count"].astype(int)
    df["claims_last_14d"] = df["claims_last_14d"].astype(int)
    return df
