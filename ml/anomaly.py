"""Train and evaluate the anomaly detector (Isolation Forest), versioned separately from the classifier.

    python ml/anomaly.py --version 1

Unsupervised: it never sees the review label. Evaluated against the anomalies the generator planted
(claims_truth.csv, evaluation only). The cut-off (flag the top 2%) was fixed before looking at results;
the score cut-off itself comes from the TRAINING period.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest

from features import ANOMALY_FEATURES, anomaly_matrix
from train import ROOT, SEED, load_split

FLAG_SHARE = 0.02          # flag the most unusual 2% (a review-capacity choice, fixed in advance)


def evaluate(scores, cutoff, truth):
    flagged = scores >= cutoff
    planted = truth["is_planted_anomaly"].to_numpy() == 1
    out = {"claims": int(len(scores)), "flag_rate": round(float(flagged.mean()), 4),
           "planted": int(planted.sum()), "caught": int((flagged & planted).sum()),
           "detection_rate": round(float((flagged & planted).sum() / max(1, planted.sum())), 3),
           "precision_vs_planted": round(float((flagged & planted).sum() / max(1, flagged.sum())), 3),
           "random_precision": round(float(planted.mean()), 3), "by_kind": {}}
    for kind, g in truth[planted].groupby("anomaly_kind"):
        idx = truth.index.get_indexer(g.index)
        out["by_kind"][kind] = f"{int(flagged[idx].sum())}/{len(g)}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="1")
    args = ap.parse_args()
    parts, medians, data_sha = load_split()
    truth = pd.read_csv(ROOT / "data" / "claims_truth.csv").set_index("claim_id")
    doc_median = float(parts["train"]["document_count"].median())
    X = {k: anomaly_matrix(v, doc_median) for k, v in parts.items()}
    T = {k: truth.loc[v["claim_id"]] for k, v in parts.items()}

    iso = IsolationForest(n_estimators=300, max_samples=256, contamination="auto", random_state=SEED)
    iso.fit(X["train"])
    score = {k: -iso.score_samples(v) for k, v in X.items()}        # higher = more unusual
    cutoff = float(np.quantile(score["train"], 1 - FLAG_SHARE))

    report = {"train": evaluate(score["train"], cutoff, T["train"]),
              "valid": evaluate(score["valid"], cutoff, T["valid"])}
    report["valid_other_cutoffs"] = {f"top {int(s * 100)}%": evaluate(score["valid"], float(np.quantile(score["train"], 1 - s)), T["valid"])
                                     for s in (0.01, 0.05)}
    report["test"] = evaluate(score["test"], cutoff, T["test"])
    for k in ("train", "valid", "test"):
        r = report[k]
        print(f"{k:5}: flagged {r['flag_rate']:.1%}  caught {r['caught']}/{r['planted']} planted "
              f"(detection {r['detection_rate']:.0%}, precision {r['precision_vs_planted']:.0%} vs random {r['random_precision']:.0%})  {r['by_kind']}")
    for name, r in report["valid_other_cutoffs"].items():
        print(f"valid {name}: detection {r['detection_rate']:.0%}, precision {r['precision_vs_planted']:.0%}")

    art = ROOT / "model-artifacts"
    name = f"claims-anomaly-v{args.version}"
    path = art / f"{name}.joblib"
    joblib.dump(iso, path)
    meta = {"name": "claims-anomaly", "version": args.version, "estimator": "isolation_forest", "file": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "features": ANOMALY_FEATURES, "category_medians": {k: round(v, 2) for k, v in medians.items()},
            "document_median": doc_median, "threshold": round(cutoff, 6), "flag_share": FLAG_SHARE,
            "score": "-IsolationForest.score_samples (higher = more unusual)",
            "data": {"file": "data/claims.csv", "sha256": data_sha}, "evaluation": report,
            "versions": {"scikit-learn": sklearn.__version__}}
    (art / f"{name}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    assert np.allclose(-joblib.load(path).score_samples(X["test"]), score["test"])
    print(f"saved {path.name} (cut-off {cutoff:.4f}) and {name}.json")


if __name__ == "__main__":
    main()
