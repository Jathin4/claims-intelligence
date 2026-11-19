"""Features must use only information available on each claim's own date."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ml"))
from features import build_features, category_medians  # noqa: E402


def claims(rows):
    df = pd.DataFrame(rows, columns=["claim_id", "customer_id", "claim_date", "claim_category", "claim_amount"])
    df["claim_date"] = pd.to_datetime(df["claim_date"])
    df["policy_start_date"] = pd.Timestamp("2025-06-01")
    df["document_count"] = 2.0
    return df


BASE = [("C1", "A", "2026-01-01", "home", 1000), ("C2", "A", "2026-02-01", "home", 3000),
        ("C3", "B", "2026-01-15", "travel", 400)]
MEDIANS = {"home": 2000.0, "travel": 400.0}


def test_history_counts_only_earlier_claims():
    f = build_features(claims(BASE), MEDIANS).set_index("claim_id")
    assert f.loc["C1", "claim_history_count"] == 0 and np.isnan(f.loc["C1", "days_since_previous_claim"])
    assert f.loc["C2", "claim_history_count"] == 1 and f.loc["C2", "days_since_previous_claim"] == 31
    assert f.loc["C3", "claim_history_count"] == 0


def test_future_claims_do_not_change_todays_features():
    with_future = BASE + [("C4", "A", "2026-06-01", "home", 9000)]
    a = build_features(claims(BASE), MEDIANS).set_index("claim_id").loc["C2"]
    b = build_features(claims(with_future), MEDIANS).set_index("claim_id").loc["C2"]
    for col in ("claim_history_count", "days_since_previous_claim", "amount_vs_category_median", "policy_age_days"):
        assert a[col] == b[col], col


def test_history_only_counts_the_last_365_days():
    rows = [("C0", "A", "2024-12-01", "home", 1000)] + BASE          # 396 days before C2: too old
    f = build_features(claims(rows), MEDIANS).set_index("claim_id")
    assert f.loc["C1", "claim_history_count"] == 0                  # C0 is 396 days before C1
    assert f.loc["C2", "claim_history_count"] == 1                  # only C1 counts
    assert f.loc["C1", "days_since_previous_claim"] == 396          # the gap itself is still known


def test_burst_counts_only_the_previous_14_days():
    rows = BASE + [("C5", "A", "2026-01-25", "home", 500), ("C6", "A", "2026-01-28", "home", 500)]
    f = build_features(claims(rows), MEDIANS).set_index("claim_id")
    assert f.loc["C6", "claims_last_14d"] == 1                      # C5 (3 days before); C1 is 27 days before
    assert f.loc["C2", "claims_last_14d"] == 2                      # C5, C6 within 14 days of 1 Feb
    assert f.loc["C1", "claims_last_14d"] == 0


def test_ratio_uses_the_given_medians_and_row_order_is_kept():
    df = claims(BASE).iloc[::-1].reset_index(drop=True)          # unsorted input
    f = build_features(df, MEDIANS)
    assert list(f["claim_id"]) == list(df["claim_id"])
    assert f.set_index("claim_id").loc["C2", "amount_vs_category_median"] == 1.5
    assert category_medians(claims(BASE)) == {"home": 2000.0, "travel": 400.0}
