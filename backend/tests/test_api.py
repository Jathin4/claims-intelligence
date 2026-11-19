"""Login, roles, CSV import, scoring and model activation."""
from tests.conftest import auth, register_tiny_model, upload

GOOD = ("CLM-000001,CUS-00001,north,2025-01-10,2026-01-05,vehicle,4200,3\n"
        "CLM-000002,CUS-00001,north,2025-01-10,2026-02-01,vehicle,30000,0\n"
        "CLM-000003,CUS-00002,,2024-06-01,2026-01-20,travel,350,\n")


def test_login_and_rate_limit(client):
    bad = {"username": "analyst@example.com", "password": "wrong-password"}
    assert [client.post("/api/auth/login", data=bad).status_code for _ in range(6)] == [401] * 5 + [429]
    assert client.get("/api/auth/me", headers=auth(client, "reviewer")).json()["role"] == "reviewer"
    assert client.get("/api/claims").status_code == 401


def test_every_endpoint_except_login_and_health_needs_a_token(client):
    from app.main import app
    open_paths = {"/api/auth/login", "/api/health"}
    checked = 0
    for path, methods in app.openapi()["paths"].items():      # the public list of endpoints
        if path in open_paths:
            continue
        for method in methods:
            url = path.replace("{claim_id}", "CLM-000001").replace("{model_id}", "1").replace("{case_id}", "1")
            assert client.request(method, url).status_code == 401, (method, path)
            checked += 1
    assert checked >= 14                                 # a new router that forgot auth would show up here


def test_import_keeps_good_rows_and_reports_bad_ones(client):
    analyst = auth(client, "analyst")
    body = GOOD + ("CLM-000004,CUS-00003,east,2025-01-01,2026-01-01,boat,100,1\n"         # unknown category
                   "CLM-000005,CUS-00003,east,2026-05-01,2026-01-01,home,100,1\n"         # policy after claim
                   "CLM-000006,CUS-00003,east,2025-01-01,2026-01-01,home,-5,1\n"          # negative amount
                   "CLM-000007,CUS-00003,east,2025-01-01,2026-01-01,home,100,2.5\n"       # half a document
                   "CLM-000001,CUS-00009,east,2025-01-01,2026-01-01,home,100,1\n")        # duplicate id
    r = upload(client, analyst, body).json()
    assert (r["rows_total"], r["rows_imported"], r["rows_rejected"]) == (8, 3, 5)
    joined = " | ".join(r["errors"])
    for expected in ("row 5: claim_category", "row 6: policy_start_date", "row 7: claim_amount",
                     "row 8: document_count", "row 9: claim_id: CLM-000001 already exists"):
        assert expected in joined, (expected, joined)
    claim = client.get("/api/claims/CLM-000003", headers=analyst).json()
    assert claim["region"] is None and claim["document_count"] is None       # blanks are missing, not errors


def test_whole_file_problems_and_roles(client):
    analyst, reviewer = auth(client, "analyst"), auth(client, "reviewer")
    r = client.post("/api/claims/import", headers=analyst, files={"file": ("x.csv", b"claim_id,claim_amount\nCLM-000001,5\n")})
    assert r.status_code == 400 and "customer_id" in r.json()["detail"]
    assert client.post("/api/claims/import", headers=analyst, files={"file": ("x.txt", b"hi")}).status_code == 415
    assert upload(client, reviewer, GOOD).status_code == 403                  # reviewers can't import
    assert client.post("/api/predictions", headers=reviewer, json={}).status_code == 403


def test_scoring_needs_an_active_model_and_records_the_version(client, tiny_model):
    analyst, admin = auth(client, "analyst"), auth(client, "admin")
    upload(client, analyst, GOOD)
    assert client.post("/api/predictions", headers=analyst, json={}).status_code == 409   # nothing active yet
    model_id = register_tiny_model(client, tiny_model)
    assert client.post(f"/api/models/{model_id}/activate", headers=analyst).status_code == 403
    assert client.post(f"/api/models/{model_id}/activate", headers=admin).json()["status"] == "active"
    result = client.post("/api/predictions", headers=analyst, json={}).json()
    assert result["scored"] == 3 and result["model"] == "claims-review v1"
    page = client.get("/api/claims?sort=score", headers=analyst).json()
    scores = [c["prediction"]["score"] for c in page["items"]]
    assert scores == sorted(scores, reverse=True) and page["total"] == 3
    detail = client.get("/api/claims/CLM-000002", headers=analyst).json()
    assert detail["features"]["claim_history_count"] == 1                     # the earlier January claim
    assert [h["claim_id"] for h in detail["history"]] == ["CLM-000001"]       # earlier claims only
    assert client.post("/api/predictions", headers=analyst, json={}).json()["scored"] == 0   # nothing left to score
    assert [c["claim_id"] for c in client.get("/api/claims?search=clm-000002", headers=analyst).json()["items"]] == ["CLM-000002"]
    assert client.get("/api/claims?search=CUS-00001", headers=analyst).json()["total"] == 2
    assert client.get("/api/claims?search=%25", headers=analyst).json()["total"] == 3            # "%" is not a wildcard
    stats = client.get("/api/stats", headers=auth(client, "reviewer")).json()                # dashboard numbers agree
    assert stats["claims"] == stats["scored"] == 3 and sum(b["count"] for b in stats["histogram"]) == 3
    assert stats["review_recommended"] == sum(c["prediction"]["review_recommended"] for c in page["items"])
    assert stats["recent_imports"][0]["rows_imported"] == 3


def test_activation_refuses_a_changed_model_file_and_keeps_one_active(client, tiny_model):
    admin = auth(client, "admin")
    first = register_tiny_model(client, tiny_model, version="1")
    second = register_tiny_model(client, tiny_model, version="2")
    client.post(f"/api/models/{first}/activate", headers=admin)
    client.post(f"/api/models/{second}/activate", headers=admin)
    statuses = {m["version"]: m["status"] for m in client.get("/api/models", headers=admin).json()}
    assert statuses == {"1": "retired", "2": "active"}
    with client.db() as db:                       # someone swaps the registered file's checksum
        from app.models import ModelVersion
        db.get(ModelVersion, first).sha256 = "0" * 64
        db.commit()
    from app import scoring
    scoring._load.cache_clear()
    r = client.post(f"/api/models/{first}/activate", headers=admin)
    assert r.status_code == 409 and "checksum" in r.json()["detail"]


def test_explanation_adds_up_and_needs_a_score(client, tiny_model):
    analyst, admin = auth(client, "analyst"), auth(client, "admin")
    upload(client, analyst, GOOD)
    client.post(f"/api/models/{register_tiny_model(client, tiny_model)}/activate", headers=admin)
    assert client.get("/api/claims/CLM-000002/explanation", headers=analyst).status_code == 404   # not scored yet
    client.post("/api/predictions", headers=analyst, json={})
    e = client.get("/api/claims/CLM-000002/explanation", headers=analyst).json()
    assert len(e["contributions"]) == 7 and "not proof" in e["note"]
    total = e["base_value"] + sum(c["contribution"] for c in e["contributions"])
    assert abs(total - e["risk_log_odds"]) < 0.01
    assert e["contributions"][0]["feature"] == "amount_vs_category_median"     # the tiny model only learned this
    assert client.get("/api/claims/CLM-999999/explanation", headers=analyst).status_code == 404

def test_import_reports_drift_against_the_active_model(client, tiny_model):
    analyst, admin = auth(client, "analyst"), auth(client, "admin")
    assert upload(client, analyst, GOOD).json()["drift"] is None                    # no active model: nothing to compare
    meta = {"validation": {}, "test": {}, "data": {"train_period": ["2026-01-01", "2026-01-31"]}}
    client.post(f"/api/models/{register_tiny_model(client, tiny_model, metadata=meta)}/activate", headers=admin)
    r = upload(client, analyst, "CLM-000010,CUS-00005,south,2025-01-01,2026-03-01,home,900,2\n").json()
    assert r["rows_imported"] == 1
    assert r["drift"]["checked"] is False and "200" in r["drift"]["reason"]       # too few rows to judge