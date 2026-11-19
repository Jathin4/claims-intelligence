"""Make a CSV of NEW claims (for existing customers) to import, optionally with deliberate drift.

    python ml/make_batch.py --out data/batch_normal.csv --n 1500 --seed 7
    python ml/make_batch.py --out data/batch_drifted.csv --n 1500 --seed 8 --drift

Drift = amounts inflated by 70%, device claims up from 13% to 40%, 25% of document counts missing.
New claims have no review outcome yet (the model predicts it).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from generate_data import CATEGORY_SPEC, ROOT

START, END = pd.Timestamp("2026-11-01"), pd.Timestamp("2026-11-15")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--drift", action="store_true")
    ap.add_argument("--first-id", type=int, default=None, help="first claim number (default: after the existing data)")
    ap.add_argument("--customer-group", type=int, choices=[0, 1], default=None,
                    help="use only even (0) or odd (1) customer numbers, so two batches don't share customers")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    existing = pd.read_csv(ROOT / "data" / "claims.csv")
    customers = existing.groupby("customer_id").agg(region=("region", "first"), policy_start_date=("policy_start_date", "min"),
                                                    past_claims=("claim_id", "count")).reset_index()
    first = a.first_id or int(existing["claim_id"].str[4:].astype(int).max()) + 1

    cats = list(CATEGORY_SPEC)
    p = np.array([CATEGORY_SPEC[c][0] for c in cats])
    if a.drift:                                   # device claims surge to 40%
        p = np.where(np.array(cats) == "device", 0.40, p * (0.60 / p[np.array(cats) != "device"].sum()))
    cat = rng.choice(cats, size=a.n, p=p / p.sum())
    amount = np.array([CATEGORY_SPEC[c][1] for c in cat]) * rng.lognormal(0, [CATEGORY_SPEC[c][2] for c in cat])
    if a.drift:
        amount *= 1.7                             # inflation
    docs = rng.poisson([CATEGORY_SPEC[c][3] for c in cat]).astype(float)
    docs[rng.random(a.n) < (0.25 if a.drift else 0.03)] = np.nan
    if a.customer_group is not None:
        customers = customers[customers["customer_id"].str[4:].astype(int) % 2 == a.customer_group]
    # Frequent claimers file most claims: pick customers in proportion to their past claims
    who = customers.sample(a.n, replace=True, weights="past_claims", random_state=a.seed).reset_index(drop=True)
    dates = START + pd.to_timedelta(rng.integers(0, (END - START).days + 1, a.n), unit="D")
    df = pd.DataFrame({"claim_id": [f"CLM-{i:06d}" for i in range(first, first + a.n)],
                       "customer_id": who["customer_id"], "region": who["region"],
                       "policy_start_date": who["policy_start_date"], "claim_date": dates.strftime("%Y-%m-%d"),
                       "claim_category": cat, "claim_amount": amount.round(2), "document_count": docs})
    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    df.to_csv(out, index=False)
    print(f"wrote {len(df):,} claims to {out.name} ({'drifted' if a.drift else 'normal'}), ids {df['claim_id'].iloc[0]}…{df['claim_id'].iloc[-1]}")


if __name__ == "__main__":
    main()
