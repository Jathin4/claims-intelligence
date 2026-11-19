"""Investigation cases end to end, with the LLM replaced by a stub (the real one takes minutes on CPU)."""
from sqlalchemy import select

from app import agents, cases
from app.models import AuditEvent
from tests.conftest import auth, register_tiny_model, upload
from tests.test_api import GOOD


def setup_scored(client, tiny_model, monkeypatch):
    analyst, admin = auth(client, "analyst"), auth(client, "admin")
    upload(client, analyst, GOOD)
    client.post(f"/api/models/{register_tiny_model(client, tiny_model)}/activate", headers=admin)
    client.post("/api/predictions", headers=analyst, json={})
    monkeypatch.setattr(cases, "enqueue", cases.drive)          # run investigations inline, not in a thread
    return analyst, admin


def stub_llm(monkeypatch, findings, fail=False):
    calls = []

    def fake(prompt, schema):
        calls.append(prompt)
        if fail:
            raise ConnectionError("Ollama is not running")
        return agents.Draft(findings=[agents.Finding(text=t, evidence=e) for t, e in findings],
                            reviewer_questions=["Do the documents match the amount?", "Is 99 relevant?"])
    monkeypatch.setattr(agents, "generate_json", fake)
    return calls


def test_investigation_checks_numbers_and_needs_a_second_person(client, tiny_model, monkeypatch):
    analyst, admin = setup_scored(client, tiny_model, monkeypatch)
    reviewer = auth(client, "reviewer")
    calls = stub_llm(monkeypatch, [("The claim is in the vehicle category with amount 30,000.00.", ["E1"]),   # true
                                   ("The amount 31,000 shows this is fraud.", ["E1"]),                         # wrong number + accusation
                                   ("The customer has 1 earlier claim.", ["E9"])])                            # citation doesn't exist
    case = client.post("/api/cases", headers=analyst, json={"claim_id": "CLM-000002"}).json()
    case = client.get(f"/api/cases/{case['id']}", headers=reviewer).json()
    assert case["status"] == "awaiting_decision" and case["priority"] in ("low", "medium", "high")
    assert len(calls) == 2 and "FAILED THESE CHECKS" in calls[1]                # one rewrite with the problems listed
    r = case["report"]
    assert [f["text"] for f in r["findings"]] == ["The claim is in the vehicle category with amount 30,000.00."]
    problems = " | ".join(p for f in r["unverified"] for p in f["problems"])
    assert "31,000" in problems and "fraud" in problems and "E9" in problems
    assert r["reviewer_questions"] == ["Do the documents match the amount?"]   # "99" isn't in the evidence
    tools = [e["tool"] for e in case["evidence"]]
    assert tools[:2] == ["claim", "risk"] and "history" in tools                # CLM-000001 is an earlier claim
    assert "Risk score" in r["summary"] and case["can_decide"]

    # one open case per claim; only a reviewer/admin who didn't open it may decide; a reason is required
    assert client.post("/api/cases", headers=analyst, json={"claim_id": "CLM-000002"}).status_code == 409
    assert client.post("/api/cases", headers=reviewer, json={"claim_id": "CLM-000003"}).status_code == 403
    assert client.post(f"/api/cases/{case['id']}/decision", headers=analyst, json={"decision": "clear", "reason": "looks fine to me"}).status_code == 403
    assert client.post(f"/api/cases/{case['id']}/decision", headers=reviewer, json={"decision": "clear", "reason": "ok"}).status_code == 422
    done = client.post(f"/api/cases/{case['id']}/decision", headers=reviewer,
                       json={"decision": "escalate", "reason": "Amount is 8x the category median; needs documents."}).json()
    assert (done["status"], done["decision"], done["decided_by"]) == ("closed", "escalate", "reviewer@example.com")
    assert client.post(f"/api/cases/{case['id']}/decision", headers=reviewer, json={"decision": "clear", "reason": "changed my mind"}).status_code == 409
    own = client.post("/api/cases", headers=admin, json={"claim_id": "CLM-000001"}).json()
    assert client.post(f"/api/cases/{own['id']}/decision", headers=admin, json={"decision": "clear", "reason": "my own case, should fail"}).status_code == 403
    with client.db() as db:
        assert db.scalar(select(AuditEvent.details).where(AuditEvent.action == "case.escalate"))["reason"].startswith("Amount")
    assert client.post("/api/cases", headers=analyst, json={"claim_id": "CLM-000002"}).status_code == 201    # closed → may reopen


def test_failed_investigation_can_be_retried(client, tiny_model, monkeypatch):
    analyst, _ = setup_scored(client, tiny_model, monkeypatch)
    stub_llm(monkeypatch, [], fail=True)
    case = client.post("/api/cases", headers=analyst, json={"claim_id": "CLM-000001"}).json()
    case = client.get(f"/api/cases/{case['id']}", headers=analyst).json()
    assert case["status"] == "failed" and "Ollama is not running" in case["error"]
    stub_llm(monkeypatch, [("Category is vehicle.", ["E1"])])
    assert client.post(f"/api/cases/{case['id']}/retry", headers=analyst).status_code == 200
    case = client.get(f"/api/cases/{case['id']}", headers=analyst).json()
    assert case["status"] == "awaiting_decision" and case["error"] is None and len(case["report"]["findings"]) == 1
