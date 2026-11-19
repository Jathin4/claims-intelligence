"""Claims: CSV import, search, detail. Every user can read claims; only analysts and admins import."""
from datetime import date
from typing import Literal

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel
from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session

from app import scoring
from app.config import settings
from app.db import get_db
from app.importer import ImportRejected, import_claims
from app.models import CATEGORIES, AnomalyResult, Claim, ModelVersion, RiskPrediction, User
from app.security import get_current_user, require_role

router = APIRouter(prefix="/api/claims", tags=["claims"])


class ImportOut(BaseModel):
    import_id: int
    rows_total: int
    rows_imported: int
    rows_rejected: int
    errors: list[str]
    ignored_columns: list[str]
    drift: dict | None = None


class PredictionOut(BaseModel):
    score: float
    threshold: float
    review_recommended: bool
    model: str


class AnomalyOut(BaseModel):
    score: float
    flagged: bool
    model: str


class ClaimOut(BaseModel):
    claim_id: str
    customer_id: str
    region: str | None
    policy_start_date: date
    claim_date: date
    claim_category: str
    claim_amount: float
    document_count: int | None
    manual_review_required: int | None
    prediction: PredictionOut | None = None
    anomaly: AnomalyOut | None = None


class ClaimPage(BaseModel):
    total: int
    items: list[ClaimOut]


class ClaimDetail(ClaimOut):
    features: dict | None = None
    history: list[ClaimOut]          # the same customer's earlier claims, newest first


def to_out(c: Claim, p: RiskPrediction | None, mv: ModelVersion | None,
           a: AnomalyResult | None = None, amv: ModelVersion | None = None) -> ClaimOut:
    pred = PredictionOut(score=p.score, threshold=p.threshold, review_recommended=p.review_recommended,
                         model=f"{mv.name} v{mv.version}") if p is not None and mv is not None else None
    anom = AnomalyOut(score=a.score, flagged=a.flagged, model=f"{amv.name} v{amv.version}") if a is not None and amv is not None else None
    return ClaimOut(**{k: getattr(c, k) for k in ClaimOut.model_fields if k not in ("prediction", "anomaly")},
                    prediction=pred, anomaly=anom)


def active_or_none(db: Session) -> ModelVersion | None:
    return scoring.find_active(db, scoring.MODEL_NAME)


@router.post("/import", response_model=ImportOut)
async def import_csv(request: Request, file: UploadFile = File(...),
                     user: User = Depends(require_role("analyst")), db: Session = Depends(get_db)):
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(415, "Upload a .csv file")
    data = await file.read(settings.max_import_mb * 1024 * 1024 + 1)
    if len(data) > settings.max_import_mb * 1024 * 1024:
        raise HTTPException(413, f"File is larger than {settings.max_import_mb} MB")
    try:
        imp = import_claims(db, user.id, file.filename or "upload.csv", data)
    except ImportRejected as e:
        raise HTTPException(400, str(e))
    return ImportOut(import_id=imp.id, rows_total=imp.rows_total, rows_imported=imp.rows_imported,
                     rows_rejected=imp.rows_rejected, errors=imp.errors, ignored_columns=imp.ignored_columns,
                     drift=imp.drift)


@router.get("", response_model=ClaimPage)
def list_claims(category: Literal[CATEGORIES] | None = None,
                recommended: bool | None = Query(None, description="only claims the model recommends for review (or not)"),
                unusual: bool | None = Query(None, description="only claims the anomaly detector flags (or not)"),
                search: str | None = Query(None, max_length=20, description="claim or customer ID, or its beginning"),
                sort: Literal["score", "date", "amount"] = "score",
                limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0),
                user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    mv, amv = active_or_none(db), scoring.find_active(db, scoring.ANOMALY_NAME)
    join_on = and_(RiskPrediction.claim_id == Claim.claim_id, RiskPrediction.model_version_id == (mv.id if mv else -1))
    join_a = and_(AnomalyResult.claim_id == Claim.claim_id, AnomalyResult.model_version_id == (amv.id if amv else -1))
    q = select(Claim, RiskPrediction, AnomalyResult).outerjoin(RiskPrediction, join_on).outerjoin(AnomalyResult, join_a)
    if category:
        q = q.where(Claim.claim_category == category)
    if search and search.strip():
        prefix = search.strip().upper().replace("%", "").replace("_", r"\_") + "%"
        q = q.where(or_(Claim.claim_id.like(prefix), Claim.customer_id.like(prefix)))
    if recommended is not None:
        q = q.where(RiskPrediction.review_recommended == recommended)
    if unusual is not None:
        q = q.where(AnomalyResult.flagged == unusual)
    order = {"score": RiskPrediction.score.desc().nulls_last(), "date": Claim.claim_date.desc(),
             "amount": Claim.claim_amount.desc()}[sort]
    total = db.scalar(select(func.count()).select_from(q.subquery()))
    rows = db.execute(q.order_by(order, Claim.claim_id).limit(limit).offset(offset)).all()
    return ClaimPage(total=total, items=[to_out(c, p, mv, a, amv) for c, p, a in rows])


@router.get("/{claim_id}", response_model=ClaimDetail)
def get_claim(claim_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    c = db.get(Claim, claim_id)
    if c is None:
        raise HTTPException(404, "Claim not found")
    mv, amv = active_or_none(db), scoring.find_active(db, scoring.ANOMALY_NAME)
    p = db.scalar(select(RiskPrediction).where(RiskPrediction.claim_id == claim_id,
                                               RiskPrediction.model_version_id == mv.id)) if mv else None
    a = db.scalar(select(AnomalyResult).where(AnomalyResult.claim_id == claim_id,
                                              AnomalyResult.model_version_id == amv.id)) if amv else None
    history = db.scalars(select(Claim).where(Claim.customer_id == c.customer_id, Claim.claim_date < c.claim_date)
                         .order_by(Claim.claim_date.desc()).limit(20)).all()
    out = to_out(c, p, mv, a, amv)
    return ClaimDetail(**out.model_dump(), features=p.features if p else None,
                       history=[to_out(h, None, None) for h in history])


@router.get("/{claim_id}/explanation")
def explain(claim_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Why did this claim get its score? SHAP contributions per feature (see `note` in the response)."""
    if db.get(Claim, claim_id) is None:
        raise HTTPException(404, "Claim not found")
    return scoring.explain_claim(db, claim_id)