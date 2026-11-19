"""PSI arithmetic and the drift report."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ml"))
from drift import categorical_psi, drift_report, level, psi_from_shares  # noqa: E402
from features import NUMERIC  # noqa: E402


def test_worked_example_from_the_topic_article():
    # income bins: expected 40/30/30 %, actual 35/25/40 %  →  PSI ≈ 0.0446 (stable)
    psi = psi_from_shares(np.array([0.40, 0.30, 0.30]), np.array([0.35, 0.25, 0.40]))
    assert abs(psi - 0.04456) < 1e-4 and level(psi) == "stable"


def test_empty_bins_do_not_explode():
    assert np.isfinite(psi_from_shares(np.array([0.5, 0.5, 0.0]), np.array([0.0, 0.5, 0.5])))
    assert categorical_psi(pd.Series(["a"] * 50 + ["b"] * 50), pd.Series(["a"] * 50 + ["b"] * 50)) == 0


def frame(rng, n, amount_scale=1.0, missing_docs=0.03, device_share=0.13):
    cats = rng.choice(["vehicle", "home", "device"], n, p=[0.5 - device_share / 2, 0.5 - device_share / 2, device_share])
    docs = rng.poisson(3, n).astype(float)
    docs[rng.random(n) < missing_docs] = np.nan
    return pd.DataFrame({"claim_amount": rng.lognormal(7, 0.8, n) * amount_scale, "policy_age_days": rng.integers(1, 1500, n),
                         "claim_history_count": rng.poisson(1.5, n), "days_since_previous_claim": rng.exponential(120, n),
                         "document_count": docs, "amount_vs_category_median": rng.lognormal(0, 0.8, n) * amount_scale,
                         "claim_category": cats})


def test_same_distribution_is_stable_and_shifted_data_is_flagged():
    rng = np.random.default_rng(7)
    ref = frame(rng, 5000)
    same = drift_report(ref, frame(rng, 1500))
    assert same["checked"] and same["worst"] < 0.1 and same["warnings"] == []
    shifted = drift_report(ref, frame(rng, 1500, amount_scale=1.8, missing_docs=0.30, device_share=0.4))
    assert shifted["levels"]["claim_amount"] == "large" and shifted["levels"]["amount_vs_category_median"] == "large"
    assert any("document_count missing" in w for w in shifted["warnings"])
    assert shifted["psi"]["claim_category"] >= 0.1
    assert set(shifted["psi"]) == set(NUMERIC) | {"claim_category"}


def test_small_batches_are_not_judged():
    rng = np.random.default_rng(1)
    r = drift_report(frame(rng, 2000), frame(rng, 50))
    assert r["checked"] is False and "200" in r["reason"]
