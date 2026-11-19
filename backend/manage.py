"""Admin commands.

    python manage.py create-user anna@example.com --role analyst      # asks for a password
    python manage.py register-model ../model-artifacts/claims-review-v1.json [--activate]
    python manage.py import-csv ../data/claims.csv
    python manage.py score                                            # every claim the active model hasn't scored
    python manage.py list
    python manage.py benchmark-agents                                 # ~15 min on CPU; writes docs/agent-benchmark.json
"""
import argparse
import getpass
import json
from pathlib import Path

from sqlalchemy import func, select

from app import benchmark
from app.db import SessionLocal
from app.importer import import_claims
from app.models import ROLES, Claim, ModelVersion, RiskPrediction, User
from app.routers.models import activate
from app.scoring import score_claims
from app.security import audit, hash_password


def create_user(email: str, role: str, password: str | None):
    password = password or getpass.getpass("Password (12+ characters): ")
    if len(password) < 12:
        raise SystemExit("Password must be at least 12 characters.")
    with SessionLocal() as db:
        if db.scalar(select(User).where(User.email == email.lower())):
            raise SystemExit(f"{email} already exists")
        u = User(email=email.lower(), password_hash=hash_password(password), role=role)
        db.add(u); db.flush()
        audit(db, None, "user.create", "user", u.id, email=u.email, role=role)
        db.commit()
        print(f"created {u.email} ({role})")


def register_model(meta_path: str, make_active: bool):
    path = Path(meta_path).resolve()
    meta = json.loads(path.read_text(encoding="utf-8"))
    if not (path.parent / meta["file"]).exists():
        raise SystemExit(f"model file {meta['file']} not found next to {path.name}")
    with SessionLocal() as db:
        if db.scalar(select(ModelVersion).where(ModelVersion.name == meta["name"], ModelVersion.version == meta["version"])):
            raise SystemExit(f"{meta['name']} v{meta['version']} is already registered (versions are immutable)")
        mv = ModelVersion(name=meta["name"], version=meta["version"], estimator=meta["estimator"], file=meta["file"],
                          sha256=meta["sha256"], threshold=meta["threshold"], features=meta["features"],
                          category_medians=meta["category_medians"], metadata_json=meta, status="candidate")
        db.add(mv); db.flush()
        audit(db, None, "model.register", "model_version", mv.id, name=mv.name, version=mv.version, sha256=mv.sha256)
        db.commit()
        print(f"registered {mv.name} v{mv.version} as candidate (id {mv.id})")
        if make_active:
            activate(db, mv, None)
            print(f"activated {mv.name} v{mv.version}")


def import_csv(path: str):
    p = Path(path)
    with SessionLocal() as db:
        imp = import_claims(db, None, p.name, p.read_bytes())
        print(f"{imp.rows_imported:,} imported, {imp.rows_rejected:,} rejected of {imp.rows_total:,}")
        for e in imp.errors[:10]:
            print("  ", e)


def score():
    with SessionLocal() as db:
        print(score_claims(db, None, None))


def list_all():
    with SessionLocal() as db:
        for u in db.scalars(select(User).order_by(User.id)):
            print(f"user  {u.email:<28} {u.role}")
        for m in db.scalars(select(ModelVersion).order_by(ModelVersion.id)):
            print(f"model {m.name} v{m.version:<4} {m.status:<9} threshold {m.threshold}  {m.estimator}")
        n_claims = db.scalar(select(func.count()).select_from(Claim))
        n_pred = db.scalar(select(func.count()).select_from(RiskPrediction))
        print(f"claims {n_claims:,} · predictions {n_pred:,}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Claims Intelligence admin commands")
    sub = ap.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("create-user"); u.add_argument("email"); u.add_argument("--role", choices=ROLES, required=True)
    u.add_argument("--password", help="omit to be prompted (safer: not saved in shell history)")
    r = sub.add_parser("register-model"); r.add_argument("meta_json"); r.add_argument("--activate", action="store_true")
    sub.add_parser("import-csv").add_argument("path")
    sub.add_parser("score")
    sub.add_parser("list")
    sub.add_parser("benchmark-agents")
    a = ap.parse_args()
    {"create-user": lambda: create_user(a.email, a.role, a.password),
     "register-model": lambda: register_model(a.meta_json, a.activate),
     "import-csv": lambda: import_csv(a.path), "score": score, "list": list_all,
     "benchmark-agents": benchmark.run}[a.cmd]()
