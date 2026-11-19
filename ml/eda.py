"""Exploratory data analysis on the TRAINING period only (oldest 70% of the modelled claims by date).

    python ml/eda.py

Writes charts and a written summary to docs/eda/. Nothing here is used to train the model;
it's how we understand the data (and check for leakage) before training.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                       # files only, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from features import CATEGORIES, MODEL_START, NUMERIC, TARGET, build_features, category_medians

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "eda"


def load():
    df = pd.read_csv(ROOT / "data" / "claims.csv", parse_dates=["claim_date", "policy_start_date"])
    df = df.sort_values(["claim_date", "claim_id"]).reset_index(drop=True)
    model = df[df["claim_date"] >= MODEL_START]                       # the warm-up year is history only
    train_ids = model["claim_id"].iloc[: int(len(model) * 0.70)]
    medians = category_medians(df[df["claim_id"].isin(train_ids)])    # training data only
    feats = build_features(df, medians)                               # history may look into the warm-up year
    return model, feats[feats["claim_id"].isin(train_ids)].copy()


def band(series, edges, labels):
    return pd.cut(series, edges, labels=labels, right=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    full, tr = load()
    lines = ["# EDA: training period", "",
             f"Explored the oldest 70% of the modelled claims only ({len(tr):,} of {len(full):,} claims from {MODEL_START}; the year before is history only), "
             f"{tr['claim_date'].min():%Y-%m-%d} to {tr['claim_date'].max():%Y-%m-%d}. "
             "Validation and test periods were not looked at.", ""]

    def section(title, obj):
        lines.extend([f"## {title}", "", "```", str(obj), "```", ""])

    # 1. Shape, types, duplicates
    section("Shape and types", f"rows {tr.shape[0]:,} · columns {tr.shape[1]}\n"
            f"duplicate claim_id: {tr['claim_id'].duplicated().sum()}\n\n{tr.dtypes.to_string()}")
    # 2. Target balance
    rate = tr[TARGET].mean()
    section("Target balance", f"{TARGET}: {rate:.3f} positive ({tr[TARGET].sum():,} of {len(tr):,})\n"
            f"-> imbalanced: use PR-AUC, recall, precision; class weights; never accuracy")
    # 3. Missing values
    missing = tr.isna().mean().round(4)
    section("Missing values (share of rows)", missing[missing > 0].to_string())
    # 4. Amounts per category
    amounts = tr.groupby("claim_category")["claim_amount"].describe(percentiles=[0.5, 0.95])[["count", "50%", "95%", "max"]]
    section("Claim amount per category", amounts.round(0).to_string())
    # 5. Target rate by category, region, bands of each engineered feature
    by_cat = tr.groupby("claim_category")[TARGET].agg(["mean", "count"]).round(3)
    section("Review rate by category", by_cat.to_string())
    by_region = tr.groupby(tr["region"].fillna("(missing)"))[TARGET].agg(["mean", "count"]).round(3)
    section("Review rate by region (fairness check: large gaps would need a reason)", by_region.to_string())
    bands = {
        "amount_vs_category_median": ([0, 0.5, 1, 2, 4, 1e9], ["<0.5", "0.5-1", "1-2", "2-4", "4+"]),
        "days_since_previous_claim": ([0, 30, 60, 180, 1e9], ["<30", "30-60", "60-180", "180+"]),
        "claim_history_count": ([0, 1, 2, 4, 1e9], ["0", "1", "2-3", "4+"]),
        "document_count": ([0, 1, 3, 5, 1e9], ["0", "1-2", "3-4", "5+"]),
        "policy_age_days": ([0, 90, 365, 1e9], ["<90", "90-365", "365+"]),
    }
    for col, (edges, labels) in bands.items():
        g = tr.groupby(band(tr[col], edges, labels), observed=False)[TARGET].agg(["mean", "count"]).round(3)
        section(f"Review rate by {col}", g.to_string())
    # 6. Over time
    monthly = tr.groupby(tr["claim_date"].dt.to_period("M"))[TARGET].agg(["mean", "count"]).round(3)
    section("Review rate and volume per month", monthly.to_string())
    # 7. Leakage check: how well does each feature ALONE rank the target?
    aucs = {}
    for col in NUMERIC:
        x = tr[col].fillna(tr[col].median())
        auc = roc_auc_score(tr[TARGET], x)
        aucs[col] = round(max(auc, 1 - auc), 3)              # direction doesn't matter
    flagged = [c for c, a in aucs.items() if a > 0.9]
    section("Single-feature ROC-AUC (leakage check: > 0.9 would be suspicious)",
            pd.Series(aucs).sort_values(ascending=False).to_string()
            + f"\n\nflagged: {flagged or 'none'}")

    # Charts
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for cat in CATEGORIES:
        ax.hist(np.log10(tr.loc[tr["claim_category"] == cat, "claim_amount"]), bins=40, alpha=0.5, label=cat)
    ax.set(title="Claim amount per category (log10 scale)", xlabel="log10(amount)", ylabel="claims")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "amounts_by_category.png", dpi=110); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    monthly["mean"].plot(ax=ax, marker="o")
    ax.axhline(rate, ls="--", color="grey", label=f"overall {rate:.3f}")
    ax.set(title="Review rate per month (training period)", ylabel="share needing review"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "review_rate_by_month.png", dpi=110); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    g = tr.groupby(band(tr["amount_vs_category_median"], *bands["amount_vs_category_median"]), observed=False)[TARGET].mean()
    g.plot.bar(ax=ax, color="#7b5cff")
    ax.set(title="Review rate by amount vs category median", xlabel="amount ÷ category median", ylabel="share needing review")
    fig.tight_layout(); fig.savefig(OUT / "review_rate_by_amount_ratio.png", dpi=110); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    missing[missing > 0].plot.barh(ax=ax, color="#ff4fd8")
    ax.set(title="Missing values (share of rows)", xlabel="share"); fig.tight_layout()
    fig.savefig(OUT / "missing_values.png", dpi=110); plt.close(fig)

    lines += ["## Charts", "", "![amounts](amounts_by_category.png)", "![monthly](review_rate_by_month.png)",
              "![ratio](review_rate_by_amount_ratio.png)", "![missing](missing_values.png)", ""]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"training rows {len(tr):,} · review rate {rate:.3f} · single-feature AUC flagged: {flagged or 'none'}")
    print(pd.Series(aucs).sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
