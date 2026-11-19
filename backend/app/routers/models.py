"""Model versions and predictions. Scoring: analysts and admins. Activating a model: admins only."""
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import Integer, and_, func, select
from sqlalchemy.orm import Session

from app import scoring
from app.db import get_db
from app.models import AnomalyResult, Claim, ClaimImport, ModelVersion, RiskPrediction, User
from app.security import audit, get_current_user, require_role

router = APIRouter(prefix="/api", tags=["models"])


class ModelOut(BaseModel):
    id: int
    name: str
    version: str
    estimator: str
    status: str
    threshold: float
    created_at: datetime
    activated_at: datetime | None
    validation: dict
    test: dict
    fairness_by_region_test: dict


class ScoreRequest(BaseModel):
    claim_ids: list[str] | None = None      # omit to score every claim the active model hasn't scored yet


def flat(metrics: dict, estimator: str) -> dict:
    """Validation may be keyed by candidate estimator ({name: {metrics, at_threshold}}); keep the chosen one, flattened."""
    chosen = metrics.get(estimator)
    if isinstance(chosen, dict) and "metrics" in chosen:
        return {**chosen["metrics"], **chosen.get("at_threshold", {})}
    return metrics


def to_out(mv: ModelVersion) -> ModelOut:
    m = mv.metadata_json
    ev = m.get("evaluation", {})                   # the anomaly detector stores {train, valid, test} here
    return ModelOut(id=mv.id, name=mv.name, version=mv.version, estimator=mv.estimator, status=mv.status,
                    threshold=mv.threshold, created_at=mv.created_at, activated_at=mv.activated_at,
                    validation=flat(m.get("validation") or ev.get("valid", {}), mv.estimator),
                    test=m.get("test") or ev.get("test", {}),
                    fairness_by_region_test=m.get("fairness_by_region_test", {}))


@router.get("/models", response_model=list[ModelOut])
def list_models(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return [to_out(m) for m in db.scalars(select(ModelVersion).order_by(ModelVersion.created_at.desc()))]


def activate(db: Session, mv: ModelVersion, user_id: int | None) -> None:
    """Switch the active model deliberately: retire the current one, check the file, then activate."""
    scoring._load(str(scoring.settings.artifacts_dir / mv.file), mv.sha256)      # refuses a changed file
    current = db.scalar(select(ModelVersion).where(ModelVersion.name == mv.name, ModelVersion.status == "active"))
    if current is not None and current.id != mv.id:
        current.status = "retired"
        db.flush()
    mv.status, mv.activated_at, mv.activated_by = "active", datetime.now(timezone.utc), user_id
    audit(db, user_id, "model.activate", "model_version", mv.id, name=mv.name, version=mv.version,
          replaced=current.version if current is not None and current.id != mv.id else None)
    db.commit()


@router.post("/models/{model_id}/activate", response_model=ModelOut)
def activate_model(model_id: int, user: User = Depends(require_role()), db: Session = Depends(get_db)):
    mv = db.get(ModelVersion, model_id)
    if mv is None:
        raise HTTPException(404, "Model version not found")
    try:
        activate(db, mv, user.id)
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return to_out(mv)


@router.post("/predictions")
def predict(req: ScoreRequest, user: User = Depends(require_role("analyst")), db: Session = Depends(get_db)):
    result = scoring.score_claims(db, req.claim_ids, user.id)
    audit(db, user.id, "predictions.run", "model_version", None, **result)
    db.commit()
    return result


@router.get("/stats")
def stats(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Numbers for the dashboard: totals, score histogram, per-category rates, recent imports."""
    mv, amv = scoring.find_active(db, scoring.MODEL_NAME), scoring.find_active(db, scoring.ANOMALY_NAME)
    pj = and_(RiskPrediction.claim_id == Claim.claim_id, RiskPrediction.model_version_id == (mv.id if mv else -1))
    aj = and_(AnomalyResult.claim_id == Claim.claim_id, AnomalyResult.model_version_id == (amv.id if amv else -1))
    rec, unusual = RiskPrediction.review_recommended.cast(Integer), AnomalyResult.flagged.cast(Integer)
    totals = db.execute(select(func.count(Claim.claim_id), func.count(RiskPrediction.id), func.sum(rec), func.sum(unusual))
                        .select_from(Claim).outerjoin(RiskPrediction, pj).outerjoin(AnomalyResult, aj)).one()
    bins = 20
    bucket = func.least(func.width_bucket(RiskPrediction.score, 0.0, 1.0, bins), bins)
    hist = dict(db.execute(select(bucket, func.count()).where(RiskPrediction.model_version_id == (mv.id if mv else -1))
                           .group_by(bucket)).all())
    by_cat = db.execute(select(Claim.claim_category, func.count(), func.avg(rec), func.avg(unusual),
                               func.avg(Claim.manual_review_required))
                        .select_from(Claim).outerjoin(RiskPrediction, pj).outerjoin(AnomalyResult, aj)
                        .group_by(Claim.claim_category).order_by(func.avg(rec).desc())).all()
    imports = db.scalars(select(ClaimImport).order_by(ClaimImport.id.desc()).limit(5)).all()
    return {
        "model": f"{mv.name} v{mv.version}" if mv else None, "anomaly_model": f"{amv.name} v{amv.version}" if amv else None,
        "threshold": mv.threshold if mv else None,
        "claims": totals[0], "scored": totals[1], "review_recommended": int(totals[2] or 0), "unusual": int(totals[3] or 0),
        "histogram": [{"from": round(i / bins, 2), "to": round((i + 1) / bins, 2), "count": int(hist.get(i + 1, 0))} for i in range(bins)],
        "by_category": [{"category": c, "claims": n, "recommended_rate": round(float(r or 0), 4),
                         "unusual_rate": round(float(u or 0), 4), "historical_review_rate": None if h is None else round(float(h), 4)}
                        for c, n, r, u, h in by_cat],
        "recent_imports": [{"id": i.id, "filename": i.filename, "created_at": i.created_at, "rows_imported": i.rows_imported,
                            "rows_rejected": i.rows_rejected, "drift_checked": bool(i.drift and i.drift.get("checked")),
                            "drift_worst": (i.drift or {}).get("worst"), "drift_warnings": (i.drift or {}).get("warnings", [])}
                           for i in imports],
    }