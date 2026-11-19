"""Runs investigations in the background and records reviewers' decisions.

All database writes about a case happen here, not in graph nodes. One worker thread: the CPU can only
usefully run one LLM call at a time. Without a checkpointer, a case interrupted by a server restart is
simply run again from the start at the next start-up: the steps only read data, so that's safe.
"""
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import agents
from app import db as dbmod
from app import scoring
from app.models import AnomalyResult, Case, CaseEvent, RiskPrediction, User
from app.security import audit

log = logging.getLogger(__name__)
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="investigation")
_graph = None
RUNNING = ("queued", "investigating")


def graph():
    global _graph
    if _graph is None:
        _graph = agents.build_graph()
    return _graph


def enqueue(case_id: int) -> None:
    _executor.submit(drive, case_id)


def add_event(db: Session, case_id: int, step: str, message: str) -> None:
    db.add(CaseEvent(case_id=case_id, step=step, message=message))


def describe(step: str, update: dict) -> str:
    if step == "plan":
        return "Lookups planned: " + ", ".join(update["plan"]) + "."
    if step == "gather_evidence":
        return f"Collected {len(update['evidence'])} evidence items from the database."
    if step == "analysis":
        d = update["draft"]
        return f"AI analysis wrote {len(d['findings'])} finding(s) and {len(d['questions'])} question(s)."
    if step == "verification":
        msg = f"Checked in code: {len(update['findings'])} passed, {len(update['unverified'])} failed."
        return msg + (" Sending the failed ones back for one rewrite." if update["feedback"] else "")
    return "Report ready for a reviewer's decision."


def drive(case_id: int) -> None:
    """Run the graph for one case, saving each step's progress."""
    with dbmod.SessionLocal() as db:
        case = db.get(Case, case_id)
        if case is None or case.status not in RUNNING:
            return
        try:     # everything inside: an error in a worker thread would otherwise vanish silently
            case.status = "investigating"
            db.commit()
            for chunk in graph().stream({"case_id": case.id, "claim_id": case.claim_id}, stream_mode="updates"):
                for step, update in chunk.items():
                    case.current_step = step
                    if step == "gather_evidence":
                        case.evidence = update["evidence"]
                    if step == "report":
                        case.report = update["report_doc"]
                    add_event(db, case.id, step, describe(step, update))
                    db.commit()
            case.status, case.current_step = "awaiting_decision", "decision"
            add_event(db, case.id, "decision", "Waiting for a reviewer's decision.")
            db.commit()
        except Exception as e:
            log.exception("Investigation %s failed", case_id)
            db.rollback()
            case = db.get(Case, case_id)
            case.status, case.error = "failed", f"{type(e).__name__}: {e}"[:500]
            add_event(db, case.id, "failed", "The investigation stopped because of an error. It can be retried.")
            db.commit()


def priority(db: Session, claim_id: str) -> str:
    """high = recommended for review AND flagged unusual; medium = either one; low = neither."""
    mv, amv = scoring.find_active(db, scoring.MODEL_NAME), scoring.find_active(db, scoring.ANOMALY_NAME)
    rec = mv and db.scalar(select(RiskPrediction.review_recommended).where(
        RiskPrediction.claim_id == claim_id, RiskPrediction.model_version_id == mv.id))
    unusual = amv and db.scalar(select(AnomalyResult.flagged).where(
        AnomalyResult.claim_id == claim_id, AnomalyResult.model_version_id == amv.id))
    return ("low", "medium", "high")[bool(rec) + bool(unusual)]


def open_case(db: Session, user: User, claim_id: str) -> Case:
    existing = db.scalar(select(Case).where(Case.claim_id == claim_id, Case.status != "closed"))
    if existing is not None:
        raise HTTPException(409, f"Case {existing.id} is already open for this claim")
    case = Case(claim_id=claim_id, opened_by=user.id, priority=priority(db, claim_id))
    db.add(case)
    db.flush()
    add_event(db, case.id, "opened", f"Case opened by {user.email} (priority {case.priority}).")
    audit(db, user.id, "case.open", "case", case.id, claim_id=claim_id, priority=case.priority)
    db.commit()
    enqueue(case.id)
    return case


def retry(db: Session, user: User, case: Case) -> None:
    if case.status != "failed":
        raise HTTPException(409, "Only a failed investigation can be retried")
    case.status, case.error, case.current_step, case.evidence, case.report = "queued", None, None, None, None
    add_event(db, case.id, "retry", f"Retried by {user.email}.")
    audit(db, user.id, "case.retry", "case", case.id)
    db.commit()
    enqueue(case.id)


def can_decide(user: User, case: Case) -> bool:
    """Reviewers and admins, never the person who opened the case (four-eyes rule, also a DB constraint)."""
    return user.role in ("reviewer", "admin") and user.id != case.opened_by and case.status == "awaiting_decision"


def decide(db: Session, user: User, case: Case, decision: str, reason: str) -> None:
    if not can_decide(user, case):
        raise HTTPException(403 if case.status == "awaiting_decision" else 409,
                            "You can't decide this case" if case.status == "awaiting_decision" else "This case isn't waiting for a decision")
    case.decision, case.decision_reason, case.decided_by = decision, reason, user.id
    case.decided_at = datetime.now(timezone.utc)
    case.status, case.current_step = "closed", None
    label = {"escalate": "Escalated for full manual review", "clear": "Cleared: no further review"}[decision]
    add_event(db, case.id, "decision", f"{label}, by {user.email}. Reason: {reason}")
    audit(db, user.id, f"case.{decision}", "case", case.id, claim_id=case.claim_id, reason=reason)
    db.commit()


def resume_unfinished() -> list[int]:
    """At server start: re-run cases a previous process left queued or half-done."""
    with dbmod.SessionLocal() as db:
        cases = db.scalars(select(Case).where(Case.status.in_(RUNNING))).all()
        for c in cases:
            c.status = "queued"
            add_event(db, c.id, "recovery", "Server restarted: running the investigation again from the start.")
        db.commit()
        for c in cases:
            enqueue(c.id)
        return [c.id for c in cases]
