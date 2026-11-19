"""Investigation cases. Analysts (and admins) open cases; reviewers (and admins) decide them, never their own."""
from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import cases
from app.db import get_db
from app.models import CASE_STATUSES, Case, CaseEvent, Claim, User
from app.security import get_current_user, require_role

router = APIRouter(prefix="/api/cases", tags=["cases"])


class OpenCase(BaseModel):
    claim_id: str


class Decision(BaseModel):
    decision: Literal["escalate", "clear"]
    reason: str = Field(min_length=10, max_length=2000, description="why: required for the audit trail")


class CaseOut(BaseModel):
    id: int
    claim_id: str
    status: str
    priority: str
    current_step: str | None
    opened_by: str
    decision: str | None
    created_at: datetime
    updated_at: datetime


class EventOut(BaseModel):
    step: str
    message: str
    created_at: datetime


class CaseDetail(CaseOut):
    evidence: list | None
    report: dict | None
    error: str | None
    decision_reason: str | None
    decided_by: str | None
    decided_at: datetime | None
    can_decide: bool
    events: list[EventOut]


def emails(db: Session, ids) -> dict[int, str]:
    ids = {i for i in ids if i is not None}
    return dict(db.execute(select(User.id, User.email).where(User.id.in_(ids))).all()) if ids else {}


def to_out(c: Case, who: dict[int, str]) -> dict:
    return {k: getattr(c, k) for k in CaseOut.model_fields if k != "opened_by"} | {"opened_by": who.get(c.opened_by, "?")}


def get_case(db: Session, case_id: int) -> Case:
    c = db.get(Case, case_id)
    if c is None:
        raise HTTPException(404, "Case not found")
    return c


def detail(db: Session, c: Case, user: User) -> CaseDetail:
    who = emails(db, [c.opened_by, c.decided_by])
    events = db.scalars(select(CaseEvent).where(CaseEvent.case_id == c.id).order_by(CaseEvent.id)).all()
    return CaseDetail(**to_out(c, who), evidence=c.evidence, report=c.report, error=c.error,
                      decision_reason=c.decision_reason, decided_by=who.get(c.decided_by), decided_at=c.decided_at,
                      can_decide=cases.can_decide(user, c),
                      events=[EventOut(step=e.step, message=e.message, created_at=e.created_at) for e in events])


@router.post("", response_model=CaseDetail, status_code=201)
def open_case(body: OpenCase, user: User = Depends(require_role("analyst")), db: Session = Depends(get_db)):
    if db.get(Claim, body.claim_id) is None:
        raise HTTPException(404, "Claim not found")
    return detail(db, cases.open_case(db, user, body.claim_id), user)


@router.get("", response_model=list[CaseOut])
def list_cases(status: Literal[CASE_STATUSES] | None = None, claim_id: str | None = None,
               user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    q = select(Case).order_by(Case.id.desc()).limit(200)
    if status:
        q = q.where(Case.status == status)
    if claim_id:
        q = q.where(Case.claim_id == claim_id)
    rows = db.scalars(q).all()
    who = emails(db, [c.opened_by for c in rows])
    return [to_out(c, who) for c in rows]


@router.get("/{case_id}", response_model=CaseDetail)
def read_case(case_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return detail(db, get_case(db, case_id), user)


@router.post("/{case_id}/retry", response_model=CaseDetail)
def retry_case(case_id: int, user: User = Depends(require_role("analyst")), db: Session = Depends(get_db)):
    c = get_case(db, case_id)
    cases.retry(db, user, c)
    return detail(db, c, user)


@router.post("/{case_id}/decision", response_model=CaseDetail)
def decide_case(case_id: int, body: Decision, user: User = Depends(require_role("reviewer")), db: Session = Depends(get_db)):
    c = get_case(db, case_id)
    cases.decide(db, user, c, body.decision, body.reason.strip())
    return detail(db, c, user)
