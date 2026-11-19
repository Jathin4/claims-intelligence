"""Agent benchmark: run the investigation graph on a fixed, rule-picked set of claims and record what happened.

The claims are chosen by rules (not by hand), so the set is the same on every run with the same data:
the hardest cases for an explanation are borderline scores, signals that disagree, missing data and no history.
The graph runs directly: no cases are created, nothing is written to the database.
Output: docs/agent-benchmark.json, which a person then grades finding by finding (docs/agent-benchmark.md).
"""
import json
import time
from datetime import datetime, timezone

from sqlalchemy import and_, exists, select
from sqlalchemy.orm import Session, aliased

from app import agents, scoring
from app.config import PROJECT_DIR, settings
from app.db import SessionLocal
from app.models import AnomalyResult, Claim, RiskPrediction


def pick_claims(db: Session, per_rule: int = 1, exclude: tuple[str, ...] = ()) -> list[tuple[str, str]]:
    mv, amv = scoring.find_active(db, scoring.MODEL_NAME), scoring.find_active(db, scoring.ANOMALY_NAME)
    P, A = RiskPrediction, AnomalyResult
    base = (select(Claim.claim_id).join(P, and_(P.claim_id == Claim.claim_id, P.model_version_id == mv.id))
            .join(A, and_(A.claim_id == Claim.claim_id, A.model_version_id == amv.id)))
    earlier = aliased(Claim)
    no_history = ~exists().where(earlier.customer_id == Claim.customer_id, earlier.claim_date < Claim.claim_date)
    rules = [
        ("highest risk score", base.order_by(P.score.desc())),
        ("just above the threshold", base.where(P.score >= P.threshold).order_by(P.score)),
        ("just below the threshold", base.where(P.score < P.threshold).order_by(P.score.desc())),
        ("unusual but not recommended", base.where(A.flagged, ~P.review_recommended).order_by(A.score.desc())),
        ("documents missing, recommended", base.where(Claim.document_count.is_(None), P.review_recommended).order_by(P.score.desc())),
        ("customer's first claim", base.where(no_history).order_by(P.score.desc())),
        ("from the drifted batch", base.where(Claim.claim_id >= "CLM-040001").order_by(P.score.desc())),
    ]
    picked = []
    for name, q in rules:
        taken = list(exclude) + [c for _, c in picked]
        for cid in db.scalars(q.where(Claim.claim_id.notin_(taken)).order_by(Claim.claim_id).limit(per_rule)):
            picked.append((name, cid))
    return picked


def run(heldout: bool = False) -> dict:
    """heldout: 2 new claims per rule, excluding the 7 the prompt and evidence fixes were made on."""
    with SessionLocal() as db:
        picked = pick_claims(db, per_rule=2, exclude=tuple(c for _, c in pick_claims(db))) if heldout else pick_claims(db)
    graph, results = agents.build_graph(), []
    for kind, claim_id in picked:
        t, steps = time.time(), []
        state: dict = {}
        try:
            for chunk in graph.stream({"claim_id": claim_id}, stream_mode="updates"):
                for step, update in chunk.items():
                    steps.append(step)
                    state |= update
            r = state["report_doc"]
            results.append({"kind": kind, "claim_id": claim_id, "seconds": round(time.time() - t), "steps": steps,
                            "retries": r["retries"], "passed": len(r["findings"]), "failed": len(r["unverified"]),
                            "summary": r["summary"], "findings": r["findings"], "unverified": r["unverified"],
                            "questions": r["reviewer_questions"], "evidence": state["evidence"]})
        except Exception as e:
            results.append({"kind": kind, "claim_id": claim_id, "error": f"{type(e).__name__}: {e}"})
        print(f"{kind:32} {claim_id}  {results[-1].get('seconds', '-')}s  passed {results[-1].get('passed', '-')}  "
              f"failed {results[-1].get('failed', '-')}  retries {results[-1].get('retries', '-')}", flush=True)
    done = [r for r in results if "error" not in r]
    out = {"run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "model": settings.chat_model,
           "claims": len(results), "errors": len(results) - len(done),
           "findings_passed": sum(r["passed"] for r in done), "findings_failed": sum(r["failed"] for r in done),
           "cases_with_retry": sum(r["retries"] > 0 for r in done),
           "median_seconds": sorted(r["seconds"] for r in done)[len(done) // 2] if done else None, "results": results}
    path = PROJECT_DIR / "docs" / ("agent-benchmark-heldout.json" if heldout else "agent-benchmark.json")
    path.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(f"saved {path}")
    return out
