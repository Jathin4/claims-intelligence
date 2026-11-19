"""Global SHAP importance and two example explanations for the report (validation data).

    python ml/shap_report.py --version 1
"""
import argparse
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from explain import LABELS, Explainer, explain_one
from features import FEATURES
from train import ROOT, load_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="1")
    v = ap.parse_args().version
    model = joblib.load(ROOT / "model-artifacts" / f"claims-review-v{v}.joblib")
    parts, _, _ = load_split()
    X = parts["valid"][FEATURES]
    ex = Explainer(model, FEATURES)
    vals, base = ex.explain(X)
    assert np.abs(base + vals.sum(axis=1) - ex.log_odds(X)).max() < 1e-6, "SHAP values don't add up"
    imp = dict(sorted(zip(FEATURES, np.abs(vals).mean(axis=0).round(4)), key=lambda t: -t[1]))
    out = ROOT / "docs" / "model"
    fig, ax = plt.subplots(figsize=(7, 3.8))
    names = list(imp)[::-1]
    ax.barh([LABELS[n] for n in names], [imp[n] for n in names], color="#7b5cff")
    ax.set(title="Global importance: mean |SHAP| (log-odds), validation", xlabel="average size of contribution")
    fig.tight_layout(); fig.savefig(out / "shap_importance.png", dpi=110); plt.close(fig)

    scores = model.predict_proba(X)[:, 1]
    examples = {}
    for label, i in (("highest score", int(np.argmax(scores))), ("lowest score", int(np.argmin(scores)))):
        e = explain_one(ex, X.iloc[[i]], top=4)
        examples[label] = {"claim_id": parts["valid"].iloc[i]["claim_id"], "score": round(float(scores[i]), 3), **e}
    (out / "shap.json").write_text(json.dumps({"importance": {k: float(v) for k, v in imp.items()}, "examples": examples},
                                              indent=2, default=float), encoding="utf-8")
    print("importance:", imp)
    for label, e in examples.items():
        print(f"{label}: {e['claim_id']} score {e['score']}: " + ", ".join(f"{c['label']} {c['contribution']:+.2f}" for c in e["contributions"]))


if __name__ == "__main__":
    main()
