"""Scoring claims with the ACTIVE model versions: the review classifier (required) and the anomaly
detector (optional, separately versioned).

Features come from the same ml/features.py used in training, with the settings saved alongside each
model (category medians, document median), so training and serving can't drift apart.
"""
import hashlib
import logging
import sys
from datetime import date
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from fastapi import HTTPException
from sqlalchemy import delete, or_, select
from sqlalchemy.orm import Session

from app.config import PROJECT_DIR, settings
from app.models import AnomalyResult, Claim, ModelVersion, RiskPrediction

sys.path.insert(0, str(PROJECT_DIR))
from ml.features import FEATURES, anomaly_matrix, build_features  # noqa: E402

MODEL_NAME = "claims-review"
ANOMALY_NAME = "claims-anomaly"


def find_active(db: Session, name: str) -> ModelVersion | None:
    return db.scalar(select(ModelVersion).where(ModelVersion.name == name, ModelVersion.status == "active"))


def active_model(db: Session) -> ModelVersion:
    mv = find_active(db, MODEL_NAME)
    if mv is None:
        raise HTTPException(409, "No active model. An admin must activate a model version first.")
    return mv


@lru_cache(maxsize=8)
def _load(path: str, sha256: str):
    """Load a model file only if it is exactly the file that was registered (unpickling runs code)."""
    data = open(path, "rb").read()
    if hashlib.sha256(data).hexdigest() != sha256:
        raise RuntimeError(f"Model file {path} doesn't match its registered checksum; refusing to load it.")
    return joblib.load(path)


def load_model(mv: ModelVersion):
    return _load(str(settings.artifacts_dir / mv.file), mv.sha256)


def claims_frame(db: Session, customer_ids) -> pd.DataFrame:
    rows = db.execute(select(Claim.claim_id, Claim.customer_id, Claim.claim_date, Claim.policy_start_date,
                             Claim.claim_category, Claim.claim_amount, Claim.document_count)
                      .where(Claim.customer_id.in_(customer_ids))).all()
    df = pd.DataFrame(rows, columns=["claim_id", "customer_id", "claim_date", "policy_start_date",
                                     "claim_category", "claim_amount", "document_count"])
    df["claim_date"] = pd.to_datetime(df["claim_date"])
    df["policy_start_date"] = pd.to_datetime(df["policy_start_date"])
    df["document_count"] = df["document_count"].astype(float)
    return df


@lru_cache(maxsize=4)
def _explainer(path: str, sha256: str):
    from ml.explain import Explainer          # SHAP is imported only when an explanation is asked for
    return Explainer(_load(path, sha256), FEATURES)


def explain_claim(db: Session, claim_id: str) -> dict:
    """SHAP contributions for a claim's stored prediction, using the exact inputs that were scored."""
    from ml.explain import explain_one
    mv = active_model(db)
    pred = db.scalar(select(RiskPrediction).where(RiskPrediction.claim_id == claim_id, RiskPrediction.model_version_id == mv.id))
    if pred is None:
        raise HTTPException(404, "This claim has no score from the active model yet")
    row = pd.DataFrame([{f: pred.features.get(f) for f in FEATURES}])
    for f in FEATURES:
        if f != "claim_category":
            row[f] = row[f].astype(float)
    result = explain_one(_explainer(str(settings.artifacts_dir / mv.file), mv.sha256), row)
    return {"claim_id": claim_id, "model": f"{mv.name} v{mv.version}", "score": pred.score, "threshold": pred.threshold,
            "review_recommended": pred.review_recommended, **result,
            "note": "Contributions are in log-odds of the model's internal risk score (positive = raises risk). "
                    "The score shown is that risk after calibration into a probability. This explains the model, "
                    "not the real-world cause, and a high score is a reason to review, not proof of wrongdoing."}


_reference_cache: dict[int, pd.DataFrame] = {}


def training_reference(db: Session, mv: ModelVersion) -> pd.DataFrame:
    """Features of the model's training-period claims (from its metadata), rebuilt from the database."""
    if mv.id not in _reference_cache:
        start, end = (date.fromisoformat(d) for d in mv.metadata_json["data"]["train_period"])
        ids = list(db.scalars(select(Claim.claim_id).where(Claim.claim_date.between(start, end))))
        customers = list(db.scalars(select(Claim.customer_id).where(Claim.claim_id.in_(ids)).distinct()))
        feats = build_features(claims_frame(db, customers), mv.category_medians)
        _reference_cache[mv.id] = feats[feats["claim_id"].isin(ids)]
    return _reference_cache[mv.id]


def drift_for_claims(db: Session, claim_ids: list[str]) -> dict | None:
    """Drift report for newly imported claims vs the active model's training data (None if no model)."""
    from ml.drift import drift_report
    mv = find_active(db, MODEL_NAME)
    if mv is None or "train_period" not in mv.metadata_json.get("data", {}):
        return None
    if not claim_ids:
        return {"checked": False, "reason": "no rows were imported"}
    try:
        customers = list(db.scalars(select(Claim.customer_id).where(Claim.claim_id.in_(claim_ids)).distinct()))
        feats = build_features(claims_frame(db, customers), mv.category_medians)
        report = drift_report(training_reference(db, mv), feats[feats["claim_id"].isin(claim_ids)])
    except Exception as e:                     # a failed check must never undo or fail an import
        logging.getLogger(__name__).exception("drift check failed")
        db.rollback()
        report = {"checked": False, "reason": f"drift check failed: {type(e).__name__}"}
    return {"model": f"{mv.name} v{mv.version}", **report}


def plain(v):
    return None if pd.isna(v) else (v.item() if isinstance(v, np.generic) else v)


def unscored_ids(db: Session, mv: ModelVersion, anomaly: ModelVersion | None) -> list[str]:
    have_pred = select(RiskPrediction.claim_id).where(RiskPrediction.model_version_id == mv.id)
    conditions = [Claim.claim_id.not_in(have_pred)]
    if anomaly is not None:
        conditions.append(Claim.claim_id.not_in(select(AnomalyResult.claim_id).where(AnomalyResult.model_version_id == anomaly.id)))
    return list(db.scalars(select(Claim.claim_id).where(or_(*conditions))))


def score_claims(db: Session, claim_ids: list[str] | None, user_id: int | None, batch: int = 2000) -> dict:
    """Score the given claims (or every claim missing a score from an active model)."""
    mv = active_model(db)
    model = load_model(mv)
    anomaly = find_active(db, ANOMALY_NAME)
    detector = load_model(anomaly) if anomaly is not None else None
    if claim_ids is None:
        claim_ids = unscored_ids(db, mv, anomaly)
    flagged = unusual = 0
    for i in range(0, len(claim_ids), batch):
        ids = claim_ids[i:i + batch]
        customers = list(db.scalars(select(Claim.customer_id).where(Claim.claim_id.in_(ids)).distinct()))
        history = claims_frame(db, customers)                             # full history per customer
        feats = build_features(history, mv.category_medians)
        feats = feats[feats["claim_id"].isin(ids)]
        scores = model.predict_proba(feats[FEATURES])[:, 1]
        db.execute(delete(RiskPrediction).where(RiskPrediction.model_version_id == mv.id, RiskPrediction.claim_id.in_(ids)))
        for (_, r), s in zip(feats.iterrows(), scores):
            db.add(RiskPrediction(claim_id=r["claim_id"], model_version_id=mv.id, score=round(float(s), 4),
                                  threshold=mv.threshold, review_recommended=bool(s >= mv.threshold),
                                  features={f: plain(r[f]) for f in FEATURES + ["claims_last_14d"]}, created_by=user_id))
            flagged += int(s >= mv.threshold)
        if detector is not None:
            afeats = feats if anomaly.category_medians == mv.category_medians else \
                build_features(history, anomaly.category_medians).pipe(lambda d: d[d["claim_id"].isin(ids)])
            a = -detector.score_samples(anomaly_matrix(afeats, anomaly.metadata_json["document_median"]))
            db.execute(delete(AnomalyResult).where(AnomalyResult.model_version_id == anomaly.id, AnomalyResult.claim_id.in_(ids)))
            db.add_all(AnomalyResult(claim_id=c, model_version_id=anomaly.id, score=round(float(s), 4),
                                     flagged=bool(s >= anomaly.threshold)) for c, s in zip(afeats["claim_id"], a))
            unusual += int((a >= anomaly.threshold).sum())
        db.commit()
    return {"model": f"{mv.name} v{mv.version}", "anomaly_model": f"{anomaly.name} v{anomaly.version}" if anomaly else None,
            "scored": len(claim_ids), "review_recommended": flagged, "unusual": unusual}
