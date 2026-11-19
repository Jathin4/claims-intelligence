"""SHAP explanations for the calibrated review model, shared by the report script and the API.

The active model is CalibratedClassifierCV: 5 boosted-tree pipelines (one per time-series fold), each
followed by an isotonic calibrator. We explain each fold's trees exactly with TreeExplainer in log-odds
(the internal risk score), then average the 5. Calibration only re-maps that score onto honest
probabilities, keeping the order within each fold, so the contributions show what pushed a claim's risk up or down.
"""
import numpy as np
import pandas as pd
import shap

LABELS = {"claim_amount": "Claim amount", "policy_age_days": "Policy age (days)",
          "claim_history_count": "Claims in the last 12 months", "days_since_previous_claim": "Days since previous claim",
          "document_count": "Supporting documents", "amount_vs_category_median": "Amount vs category median",
          "claim_category": "Claim category"}


def _parts(model):
    """(preprocessing, tree model) for every fold of a calibrated model, or the single pipeline."""
    pipes = [c.estimator for c in model.calibrated_classifiers_] if hasattr(model, "calibrated_classifiers_") else [model]
    return [(p.named_steps["prep"], p.named_steps["clf"]) for p in pipes]


class Explainer:
    def __init__(self, model, features: list[str]):
        self.features = features
        self.folds = [(prep, clf, shap.TreeExplainer(clf)) for prep, clf in _parts(model)]

    def explain(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Contributions per ORIGINAL feature (one-hot columns summed back into claim_category),
        averaged over folds: (values [n, features], base values [n])."""
        total, base = np.zeros((len(X), len(self.features))), np.zeros(len(X))
        for prep, _, ex in self.folds:
            Xt = prep.transform(X[self.features])
            names = list(prep.get_feature_names_out())
            sv = ex(Xt)
            vals = sv.values if sv.values.ndim == 2 else sv.values[..., 1]
            b = np.asarray(sv.base_values, dtype=float)
            base += b if b.ndim == 1 else b[..., 1]
            for j, f in enumerate(self.features):
                cols = [i for i, n in enumerate(names) if n == f or n.startswith(f + "_")]
                total[:, j] += vals[:, cols].sum(axis=1)
        k = len(self.folds)
        return total / k, base / k

    def log_odds(self, X: pd.DataFrame) -> np.ndarray:
        """The averaged internal risk score that base + contributions must add up to."""
        return np.mean([clf.decision_function(prep.transform(X[self.features])) for prep, clf, _ in self.folds], axis=0)


def explain_one(explainer: Explainer, row: pd.DataFrame, top: int | None = None) -> dict:
    values, base = explainer.explain(row)
    items = [{"feature": f, "label": LABELS.get(f, f), "value": None if pd.isna(row.iloc[0][f]) else row.iloc[0][f],
              "contribution": round(float(v), 4)} for f, v in zip(explainer.features, values[0])]
    items.sort(key=lambda d: -abs(d["contribution"]))
    return {"base_value": round(float(base[0]), 4), "risk_log_odds": round(float(base[0] + values[0].sum()), 4),
            "contributions": items[:top] if top else items}
