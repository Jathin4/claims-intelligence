"""Train, evaluate and save a versioned review-priority model.

    python ml/train.py --version 1

Order matters (every choice on validation data, the test set used exactly once):
  1. features (point-in-time; category medians from training data only)
  2. time-based split 70 / 15 / 15 of the modelled claims
  3. baseline: logistic regression        → recorded first
  4. candidate: histogram gradient boosting
  5. calibration check (calibrators fitted with time-series CV on TRAINING data)
  6. threshold by cost (10 × missed + 1 × unnecessary) with a review-rate cap, on validation
  7. final test, once
  8. save model + metadata (checksum, metrics, threshold, data checksum, library versions) + report
"""
import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, log_loss,
                             precision_recall_curve, roc_auc_score)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import CATEGORICAL, FEATURES, MODEL_START, NUMERIC, TARGET, build_features, category_medians

ROOT = Path(__file__).resolve().parent.parent
COST_FN, COST_FP = 10, 1        # a missed review-worthy claim assumed 10× worse than an unnecessary review
REVIEW_CAP = 0.15               # the team can review at most 15% of claims
SEED = 42


def load_split():
    path = ROOT / "data" / "claims.csv"
    df = pd.read_csv(path, parse_dates=["claim_date", "policy_start_date"])
    df = df.sort_values(["claim_date", "claim_id"]).reset_index(drop=True)
    model = df[df["claim_date"] >= MODEL_START]
    n = len(model)
    ids = {"train": model["claim_id"].iloc[: int(n * 0.70)],
           "valid": model["claim_id"].iloc[int(n * 0.70): int(n * 0.85)],
           "test": model["claim_id"].iloc[int(n * 0.85):]}
    medians = category_medians(df[df["claim_id"].isin(ids["train"])])
    feats = build_features(df, medians)                       # history may use the warm-up year
    parts = {k: feats[feats["claim_id"].isin(v)].copy() for k, v in ids.items()}
    assert parts["train"]["claim_date"].max() <= parts["valid"]["claim_date"].min() <= parts["test"]["claim_date"].min()
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    return parts, medians, checksum


def baseline():
    prep = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])
    return Pipeline([("prep", prep), ("clf", LogisticRegression(class_weight="balanced", max_iter=2000))])


def boosted():
    prep = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL)],
                             remainder="passthrough", verbose_feature_names_out=False)
    clf = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=500, max_leaf_nodes=15, min_samples_leaf=40,
                                         l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
                                         class_weight="balanced", random_state=SEED)
    return Pipeline([("prep", prep), ("clf", clf)])


def metrics(y, p):
    return {"pr_auc": round(average_precision_score(y, p), 4), "roc_auc": round(roc_auc_score(y, p), 4),
            "brier": round(brier_score_loss(y, p), 4), "log_loss": round(log_loss(y, p), 4),
            "positive_rate": round(float(np.mean(y)), 4)}


def at_threshold(y, p, t):
    pred = p >= t
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"threshold": round(float(t), 4), "review_rate": round(float(pred.mean()), 4),
            "precision": round(float(tp / (tp + fp)), 4) if tp + fp else 0.0, "recall": round(float(tp / (tp + fn)), 4),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn), "cost": int(COST_FN * fn + COST_FP * fp)}


def choose_threshold(y, p):
    """Lowest cost on VALIDATION data among thresholds that respect the review cap."""
    best = None
    for t in np.unique(np.round(p, 4)):                       # every distinct score is a candidate
        r = at_threshold(y, p, t)
        if r["review_rate"] <= REVIEW_CAP and (best is None or r["cost"] < best["cost"]):
            best = r
    return best


def group_rates(y, p, t, groups):
    out = {}
    for g in sorted(groups.dropna().unique()):
        m = (groups == g).to_numpy()
        yt, yp = y[m], (p[m] >= t)
        out[g] = {"n": int(m.sum()), "flag_rate": round(float(yp.mean()), 3),
                  "tpr": round(float(yp[yt == 1].mean()), 3), "fpr": round(float(yp[yt == 0].mean()), 3)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="1")
    args = ap.parse_args()
    parts, medians, data_sha = load_split()
    X = {k: v[FEATURES] for k, v in parts.items()}
    y = {k: v[TARGET].to_numpy() for k, v in parts.items()}
    print({k: (len(v), round(float(y[k].mean()), 3)) for k, v in X.items()})

    results = {}
    lr = baseline().fit(X["train"], y["train"])
    results["logistic_regression"] = {"model": lr, "valid": metrics(y["valid"], lr.predict_proba(X["valid"])[:, 1])}
    print("baseline (logistic regression) valid:", results["logistic_regression"]["valid"])

    hgb = boosted().fit(X["train"], y["train"])
    results["hist_gradient_boosting"] = {"model": hgb, "valid": metrics(y["valid"], hgb.predict_proba(X["valid"])[:, 1])}
    print("boosted trees valid:              ", results["hist_gradient_boosting"]["valid"])

    # Calibrated version of the boosted model: calibrators learned on time-ordered folds of TRAINING data
    cal = CalibratedClassifierCV(boosted(), method="isotonic", cv=TimeSeriesSplit(n_splits=5)).fit(X["train"], y["train"])
    results["hist_gradient_boosting_calibrated"] = {"model": cal, "valid": metrics(y["valid"], cal.predict_proba(X["valid"])[:, 1])}
    print("boosted + isotonic calibration:   ", results["hist_gradient_boosting_calibrated"]["valid"])

    # Pick the model on validation PR-AUC. Prefer the calibrated version when its PR-AUC is within 0.01:
    # people will read these scores, and class weighting inflates raw scores (validation Brier 0.184 vs 0.097).
    best_name = max(results, key=lambda k: (results[k]["valid"]["pr_auc"], -results[k]["valid"]["brier"]))
    if results["hist_gradient_boosting_calibrated"]["valid"]["pr_auc"] >= results["hist_gradient_boosting"]["valid"]["pr_auc"] - 0.01 \
            and best_name == "hist_gradient_boosting":
        best_name = "hist_gradient_boosting_calibrated"
    model = results[best_name]["model"]
    p_valid = model.predict_proba(X["valid"])[:, 1]
    chosen = choose_threshold(y["valid"], p_valid)
    for name, r in results.items():                      # every model at its own best threshold, for comparison
        r["valid_at_threshold"] = choose_threshold(y["valid"], r["model"].predict_proba(X["valid"])[:, 1])
    print("chosen:", best_name, "threshold on validation:", chosen)

    # 7. The test set, once.
    p_test = model.predict_proba(X["test"])[:, 1]
    test = {**metrics(y["test"], p_test), **at_threshold(y["test"], p_test, chosen["threshold"])}
    fairness = group_rates(y["test"], p_test, chosen["threshold"], parts["test"]["region"])
    # Ceiling: how well the generator's own hidden probability ranks the labels (evaluation only, never a feature)
    truth = pd.read_csv(ROOT / "data" / "claims_truth.csv").set_index("claim_id")["hidden_risk_probability"]
    oracle = {k: round(average_precision_score(y[k], truth.loc[parts[k]["claim_id"]].to_numpy()), 4) for k in ("valid", "test")}
    print("oracle PR-AUC (best any model could do):", oracle)
    print("TEST:", test)

    # 8. Save the model and its metadata
    art = ROOT / "model-artifacts"
    art.mkdir(exist_ok=True)
    name = f"claims-review-v{args.version}"
    model_path = art / f"{name}.joblib"
    joblib.dump(model, model_path)
    meta = {
        "name": "claims-review", "version": args.version, "estimator": best_name,
        "file": model_path.name, "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "features": FEATURES, "category_medians": {k: round(v, 2) for k, v in medians.items()},
        "threshold": chosen["threshold"], "cost": {"fn": COST_FN, "fp": COST_FP}, "review_cap": REVIEW_CAP,
        "data": {"file": "data/claims.csv", "sha256": data_sha,
                 **{f"{k}_period": [str(v["claim_date"].min().date()), str(v["claim_date"].max().date())] for k, v in parts.items()},
                 **{f"{k}_rows": len(v) for k, v in parts.items()}},
        "validation": {k: {"metrics": r["valid"], "at_threshold": r["valid_at_threshold"]} for k, r in results.items()},
        "test": test, "fairness_by_region_test": fairness, "oracle_pr_auc": oracle,
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
    }
    (art / f"{name}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    loaded = joblib.load(model_path)                                    # round-trip check
    assert np.allclose(loaded.predict_proba(X["test"])[:, 1], p_test)
    write_report(meta, results, X, y, model, p_valid)
    print(f"saved {model_path.name} (sha256 {meta['sha256'][:12]}…) and {name}.json")


def write_report(meta, results, X, y, model, p_valid):
    out = ROOT / "docs" / "model"
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name, r in results.items():
        pr, rc, _ = precision_recall_curve(y["valid"], r["model"].predict_proba(X["valid"])[:, 1])
        ax.plot(rc, pr, label=f"{name} (AP {r['valid']['pr_auc']:.3f})")
    ax.axhline(y["valid"].mean(), ls="--", color="grey", label=f"random ({y['valid'].mean():.3f})")
    ax.set(xlabel="recall", ylabel="precision", title="Precision–recall, validation"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "pr_curves.png", dpi=110); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], ls="--", color="grey", label="perfectly calibrated")
    for name, r in results.items():
        frac, mean = calibration_curve(y["valid"], r["model"].predict_proba(X["valid"])[:, 1], n_bins=10, strategy="quantile")
        ax.plot(mean, frac, marker="o", label=f"{name} (Brier {r['valid']['brier']:.3f})")
    ax.set(xlabel="mean predicted score", ylabel="actual share needing review", title="Calibration, validation")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out / "calibration.png", dpi=110); plt.close(fig)

    ts = np.linspace(0.02, 0.98, 97)
    rates = [(p_valid >= t).mean() for t in ts]
    recalls = [((p_valid >= t) & (y["valid"] == 1)).sum() / max(1, (y["valid"] == 1).sum()) for t in ts]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(ts, rates, label="review rate (workload)"); ax.plot(ts, recalls, label="recall")
    ax.axvline(meta["threshold"], color="#ff4fd8", label=f"chosen threshold {meta['threshold']:.3f}")
    ax.axhline(REVIEW_CAP, ls=":", color="grey", label=f"cap {REVIEW_CAP:.0%}")
    ax.set(xlabel="threshold", title="Workload and recall by threshold (validation)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "threshold.png", dpi=110); plt.close(fig)

    rows = ["| Model | PR-AUC | ROC-AUC | Brier | Precision* | Recall* | Review rate* |", "|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        v, a = r["valid"], r["valid_at_threshold"]
        rows.append(f"| {name} | {v['pr_auc']} | {v['roc_auc']} | {v['brier']} | {a['precision']} | {a['recall']} | {a['review_rate']} |")
    t = meta["test"]
    text = [f"# Model report: {meta['name']} v{meta['version']}", "",
            "All results are on **synthetic** data. The score means \"needs manual review\", not fraud.", "",
            f"Data: `{meta['data']['file']}` (sha256 {meta['data']['sha256'][:12]}…). "
            f"Train {meta['data']['train_period']} ({meta['data']['train_rows']:,} claims), "
            f"validation {meta['data']['valid_period']} ({meta['data']['valid_rows']:,}), "
            f"test {meta['data']['test_period']} ({meta['data']['test_rows']:,}).", "",
            "## Validation (all choices made here)", "", *rows, "",
            f"\\* at each model's own lowest-cost threshold (cost = {COST_FN} × missed + {COST_FP} × unnecessary, review rate ≤ {REVIEW_CAP:.0%}).", "",
            f"Chosen: **{meta['estimator']}**, threshold **{meta['threshold']}**.", "",
            f"Ceiling: the generator's hidden probability scores PR-AUC {meta['oracle_pr_auc']['valid']} on validation and "
            f"{meta['oracle_pr_auc']['test']} on test: the best any model could do, because the labels contain random noise.", "",
            "## Test (used once)", "",
            f"PR-AUC {t['pr_auc']} · ROC-AUC {t['roc_auc']} · Brier {t['brier']} · at threshold {t['threshold']}: "
            f"precision {t['precision']}, recall {t['recall']}, review rate {t['review_rate']} "
            f"(TP {t['tp']}, FP {t['fp']}, FN {t['fn']}, TN {t['tn']}).", "",
            "## Fairness check by region (test, at the threshold)", "",
            "| Region | Claims | Flag rate | TPR (recall) | FPR |", "|---|---|---|---|---|",
            *[f"| {g} | {v['n']} | {v['flag_rate']} | {v['tpr']} | {v['fpr']} |" for g, v in meta["fairness_by_region_test"].items()], "",
            "![PR curves](pr_curves.png)", "![Calibration](calibration.png)", "![Threshold](threshold.png)", ""]
    (out / "README.md").write_text("\n".join(text), encoding="utf-8")


if __name__ == "__main__":
    main()
