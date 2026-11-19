"""Contracts the project must keep: reproducible data with a fixed schema, and saved models that clear
minimum quality floors. The model checks read model-artifacts/ (git-ignored), so they are skipped on a
fresh clone until `python ml/train.py` and `python ml/anomaly.py` have been run."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "model-artifacts"
COLUMNS = ["claim_id", "customer_id", "region", "policy_start_date", "claim_date",
           "claim_category", "claim_amount", "document_count", "manual_review_required"]


def meta(name: str) -> dict:
    path = ARTIFACTS / f"{name}.json"
    if not path.exists():
        pytest.skip(f"{path.name} not found: train the models first")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def generated(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("data")
    subprocess.run([sys.executable, str(ROOT / "ml" / "generate_data.py"), "--seed", "42", "--out", str(out)],
                   check=True, capture_output=True)
    return out / "claims.csv"


def test_generated_data_has_the_agreed_schema(generated):
    df = pd.read_csv(generated, parse_dates=["policy_start_date", "claim_date"])
    assert list(df.columns) == COLUMNS
    assert df["claim_id"].is_unique and df["claim_id"].str.fullmatch(r"CLM-\d{6}").all()
    assert df["customer_id"].str.fullmatch(r"CUS-\d{5}").all()
    assert set(df["claim_category"]) <= {"vehicle", "home", "health", "travel", "device"}
    assert (df["claim_amount"] > 0).all() and (df["policy_start_date"] <= df["claim_date"]).all()
    docs = df["document_count"].dropna()
    assert (docs >= 0).all() and (docs == docs.round()).all()
    assert set(df["manual_review_required"].unique()) == {0, 1}
    assert 0.10 <= df.loc[df["claim_date"] >= "2024-11-01", "manual_review_required"].mean() <= 0.15


def test_same_seed_gives_exactly_the_data_the_model_was_trained_on(generated):
    trained_on = meta("claims-review-v1")["data"]["sha256"]
    assert hashlib.sha256(generated.read_bytes()).hexdigest() == trained_on


def test_review_model_clears_the_minimum_floors():
    m = meta("claims-review-v1")
    assert hashlib.sha256((ARTIFACTS / m["file"]).read_bytes()).hexdigest() == m["sha256"]
    t = m["test"]                                      # held-out period, never used for any choice
    base = t["positive_rate"]
    assert t["pr_auc"] >= 2 * base                     # at least twice as good as random ordering
    assert t["roc_auc"] >= 0.70
    assert t["brier"] < base * (1 - base)              # better than always predicting the base rate
    assert t["review_rate"] <= m["review_cap"] + 0.02  # the workload the team agreed to (small drift allowed)
    for region, g in m["fairness_by_region_test"].items():
        assert abs(g["flag_rate"] - t["review_rate"]) <= 0.03, region
        assert abs(g["tpr"] - t["recall"]) <= 0.10, region


def test_anomaly_detector_beats_random_flagging():
    m = meta("claims-anomaly-v1")
    assert hashlib.sha256((ARTIFACTS / m["file"]).read_bytes()).hexdigest() == m["sha256"]
    t = m["evaluation"]["test"]
    assert t["precision_vs_planted"] >= 3 * t["random_precision"]
    assert t["flag_rate"] <= 0.03
