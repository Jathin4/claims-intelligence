"""Training logic: threshold choice, feature contract, determinism."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ml"))
import train  # noqa: E402
from features import CATEGORICAL, FEATURES, NUMERIC  # noqa: E402


def test_features_never_include_identifiers_region_or_generator_truth():
    banned = {"claim_id", "customer_id", "region", "manual_review_required",
              "is_planted_anomaly", "anomaly_kind", "hidden_risk_probability", "claim_date", "policy_start_date"}
    assert not banned & set(FEATURES)
    assert FEATURES == NUMERIC + CATEGORICAL


def test_threshold_minimises_cost_within_the_review_cap():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.13).astype(int)
    p = np.clip(0.13 + 0.25 * y + rng.normal(0, 0.12, 2000), 0, 1)
    best = train.choose_threshold(y, p)
    assert best["review_rate"] <= train.REVIEW_CAP
    # no other allowed threshold is cheaper
    for t in np.unique(np.round(p, 4))[::25]:
        r = train.at_threshold(y, p, t)
        if r["review_rate"] <= train.REVIEW_CAP:
            assert r["cost"] >= best["cost"]


def test_at_threshold_counts():
    r = train.at_threshold(np.array([1, 1, 0, 0]), np.array([0.9, 0.2, 0.8, 0.1]), 0.5)
    assert (r["tp"], r["fn"], r["fp"], r["tn"]) == (1, 1, 1, 1)
    assert r["cost"] == train.COST_FN * 1 + train.COST_FP * 1 and r["review_rate"] == 0.5


def small_frame(seed=1, n=1500):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "claim_amount": rng.lognormal(7, 1, n), "policy_age_days": rng.integers(1, 1500, n),
        "claim_history_count": rng.poisson(1, n), "days_since_previous_claim": rng.choice([np.nan, 30.0, 200.0], n),
        "document_count": rng.poisson(3, n).astype(float), "amount_vs_category_median": rng.lognormal(0, 0.7, n),
        "claim_category": rng.choice(["vehicle", "home", "travel"], n)})
    y = (rng.random(n) < 0.1 + 0.2 * (df["amount_vs_category_median"] > 2)).astype(int)
    return df[FEATURES], y


def test_same_seed_same_model():
    X, y = small_frame()
    a = train.boosted().fit(X, y).predict_proba(X)[:, 1]
    b = train.boosted().fit(X, y).predict_proba(X)[:, 1]
    assert np.array_equal(a, b)
    lr = train.baseline().fit(X, y)          # baseline handles missing values and unknown categories
    X2 = X.head(3).assign(claim_category="device", days_since_previous_claim=np.nan)
    assert lr.predict_proba(X2).shape == (3, 2)
