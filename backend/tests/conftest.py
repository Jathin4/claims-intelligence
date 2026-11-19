"""API tests on the separate database claims_intelligence_test, with a tiny model trained on the fly."""
import hashlib
import json
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app import db as dbmod
from app import scoring
from app.config import PROJECT_DIR, settings
from app.db import Base
from app.models import ModelVersion, User
from app.security import hash_password

sys.path.insert(0, str(PROJECT_DIR / "ml"))
import train  # noqa: E402
from features import FEATURES  # noqa: E402

PASSWORD = "correct-horse-battery"
HEADER = "claim_id,customer_id,region,policy_start_date,claim_date,claim_category,claim_amount,document_count\n"


@pytest.fixture(scope="session")
def engine():
    assert settings.test_database_url, "TEST_DATABASE_URL missing from .env"
    eng = create_engine(settings.test_database_url)
    Base.metadata.drop_all(eng)
    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory):
    """A real (small) boosted pipeline saved to a temp folder, like a registered model file."""
    rng = np.random.default_rng(3)
    n = 800
    X = pd.DataFrame({"claim_amount": rng.lognormal(7, 1, n), "policy_age_days": rng.integers(1, 900, n),
                      "claim_history_count": rng.poisson(1, n), "days_since_previous_claim": rng.choice([np.nan, 20.0, 300.0], n),
                      "document_count": rng.poisson(3, n).astype(float), "amount_vs_category_median": rng.lognormal(0, 0.8, n),
                      "claim_category": rng.choice(["vehicle", "home", "travel"], n)})[FEATURES]
    y = (X["amount_vs_category_median"] > 2).astype(int).to_numpy()
    folder = tmp_path_factory.mktemp("artifacts")
    path = folder / "tiny-v1.joblib"
    joblib.dump(train.boosted().fit(X, y), path)
    return folder, path


@pytest.fixture
def client(engine, monkeypatch, tiny_model):
    with engine.begin() as c:
        c.execute(text("TRUNCATE " + ", ".join(t.name for t in Base.metadata.sorted_tables) + " RESTART IDENTITY CASCADE"))
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(dbmod, "SessionLocal", session_factory)
    monkeypatch.setattr(settings, "artifacts_dir", tiny_model[0])
    scoring._load.cache_clear()
    with session_factory() as db:
        db.add_all([User(email=f"{r}@example.com", password_hash=hash_password(PASSWORD), role=r)
                    for r in ("analyst", "reviewer", "admin")])
        db.commit()
    from app.main import app
    app.dependency_overrides[dbmod.get_db] = lambda: (yield from _session(session_factory))
    with TestClient(app) as c:
        c.db = session_factory
        yield c
    app.dependency_overrides.clear()


def _session(factory):
    db = factory()
    try:
        yield db
    finally:
        db.close()


def register_tiny_model(client, tiny_model, status="candidate", version="1", metadata=None):
    path = tiny_model[1]
    with client.db() as db:
        mv = ModelVersion(name=scoring.MODEL_NAME, version=version, estimator="hist_gradient_boosting", file=path.name,
                          sha256=hashlib.sha256(path.read_bytes()).hexdigest(), threshold=0.5, features=FEATURES,
                          category_medians={"vehicle": 3500.0, "home": 2500.0, "travel": 400.0, "health": 1200.0, "device": 300.0},
                          metadata_json=metadata or json.loads('{"validation": {}, "test": {}}'), status=status)
        db.add(mv)
        db.commit()
        return mv.id


def auth(client, role):
    r = client.post("/api/auth/login", data={"username": f"{role}@example.com", "password": PASSWORD})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def upload(client, headers, body, name="claims.csv"):
    return client.post("/api/claims/import", headers=headers, files={"file": (name, (HEADER + body).encode())})
