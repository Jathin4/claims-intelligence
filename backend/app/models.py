"""Database tables: claims and models (Stage 2), anomaly results (Stage 3), investigation cases (Stage 4)."""
from datetime import date, datetime

from sqlalchemy import CheckConstraint, ForeignKey, Index, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base

ROLES = ("analyst", "reviewer", "admin")
CATEGORIES = ("vehicle", "home", "health", "travel", "device")
MODEL_STATUSES = ("candidate", "active", "retired")
CASE_STATUSES = ("queued", "investigating", "awaiting_decision", "closed", "failed")
PRIORITIES = ("low", "medium", "high")
DECISIONS = ("escalate", "clear")


class User(Base):
    __tablename__ = "users"
    __table_args__ = (CheckConstraint(f"role IN {ROLES}", name="ck_users_role"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True)
    password_hash: Mapped[str]
    role: Mapped[str] = mapped_column(String(20))
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class ClaimImport(Base):
    """One CSV upload: how many rows came in, how many were rejected and why."""
    __tablename__ = "claim_imports"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    filename: Mapped[str] = mapped_column(String(255))
    rows_total: Mapped[int]
    rows_imported: Mapped[int]
    rows_rejected: Mapped[int]
    errors: Mapped[list] = mapped_column(JSONB, default=list)    # first N row errors
    drift: Mapped[dict | None] = mapped_column(JSONB)            # PSI vs the active model's training data
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class Claim(Base):
    __tablename__ = "claims"
    __table_args__ = (
        CheckConstraint(f"claim_category IN {CATEGORIES}", name="ck_claims_category"),
        CheckConstraint("claim_amount > 0", name="ck_claims_amount_positive"),
        CheckConstraint("document_count IS NULL OR document_count >= 0", name="ck_claims_documents"),
        CheckConstraint("policy_start_date <= claim_date", name="ck_claims_policy_before_claim"),
        Index("ix_claims_customer_date", "customer_id", "claim_date"),
    )

    claim_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    customer_id: Mapped[str] = mapped_column(String(20))
    region: Mapped[str | None] = mapped_column(String(30))
    policy_start_date: Mapped[date]
    claim_date: Mapped[date] = mapped_column(index=True)
    claim_category: Mapped[str] = mapped_column(String(20))
    claim_amount: Mapped[float]
    document_count: Mapped[int | None]
    # The historical outcome, if known. New claims have none; the model predicts it.
    manual_review_required: Mapped[int | None]
    import_id: Mapped[int | None] = mapped_column(ForeignKey("claim_imports.id", ondelete="SET NULL"))
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class ModelVersion(Base):
    """A trained model file plus everything needed to trust it. Only one can be active."""
    __tablename__ = "model_versions"
    __table_args__ = (
        UniqueConstraint("name", "version", name="uq_model_versions_name_version"),
        CheckConstraint(f"status IN {MODEL_STATUSES}", name="ck_model_versions_status"),
        Index("uq_model_versions_one_active", "name", unique=True, postgresql_where="status = 'active'"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(60))
    version: Mapped[str] = mapped_column(String(20))
    estimator: Mapped[str] = mapped_column(String(60))
    file: Mapped[str] = mapped_column(String(255))
    sha256: Mapped[str] = mapped_column(String(64))
    threshold: Mapped[float]
    features: Mapped[list] = mapped_column(JSONB)
    category_medians: Mapped[dict] = mapped_column(JSONB)
    metadata_json: Mapped[dict] = mapped_column(JSONB)          # metrics, data checksum, fairness, versions
    status: Mapped[str] = mapped_column(String(20), default="candidate")
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    activated_at: Mapped[datetime | None]
    activated_by: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))


class RiskPrediction(Base):
    """A model's score for a claim, with the exact model version and threshold that produced it."""
    __tablename__ = "risk_predictions"
    __table_args__ = (UniqueConstraint("claim_id", "model_version_id", name="uq_risk_predictions_claim_model"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    claim_id: Mapped[str] = mapped_column(ForeignKey("claims.claim_id", ondelete="CASCADE"), index=True)
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id"))
    score: Mapped[float]
    threshold: Mapped[float]
    review_recommended: Mapped[bool]
    features: Mapped[dict] = mapped_column(JSONB)               # the exact inputs used (for explanations and audits)
    created_by: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class AnomalyResult(Base):
    """The anomaly detector's score for a claim. A signal to look closer, never an accusation."""
    __tablename__ = "anomaly_results"
    __table_args__ = (UniqueConstraint("claim_id", "model_version_id", name="uq_anomaly_results_claim_model"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    claim_id: Mapped[str] = mapped_column(ForeignKey("claims.claim_id", ondelete="CASCADE"), index=True)
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id"))
    score: Mapped[float]                     # higher = more unusual
    flagged: Mapped[bool]
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class AuditEvent(Base):
    __tablename__ = "audit_events"
    __table_args__ = (Index("ix_audit_events_created_at", "created_at"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    actor_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    action: Mapped[str] = mapped_column(String(50))
    object_type: Mapped[str | None] = mapped_column(String(50))
    object_id: Mapped[str | None] = mapped_column(String(50))
    ip: Mapped[str | None] = mapped_column(String(45))
    details: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class Case(Base):
    """An investigation of one claim: the agents' evidence and report, then a reviewer's decision.

    `evidence` is a frozen copy of what the agents saw, each item pointing at the record it came from,
    so the report can still be checked after the underlying data or model changes.
    """
    __tablename__ = "cases"
    __table_args__ = (
        CheckConstraint(f"status IN {CASE_STATUSES}", name="ck_cases_status"),
        CheckConstraint(f"priority IN {PRIORITIES}", name="ck_cases_priority"),
        CheckConstraint(f"decision IS NULL OR decision IN {DECISIONS}", name="ck_cases_decision"),
        CheckConstraint("(decision IS NULL) = (decided_by IS NULL)", name="ck_cases_decision_by"),
        CheckConstraint("decided_by IS NULL OR decided_by <> opened_by", name="ck_cases_four_eyes"),
        Index("uq_cases_one_open_per_claim", "claim_id", unique=True, postgresql_where="status <> 'closed'"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    claim_id: Mapped[str] = mapped_column(ForeignKey("claims.claim_id", ondelete="CASCADE"), index=True)
    status: Mapped[str] = mapped_column(String(20), default="queued")
    priority: Mapped[str] = mapped_column(String(10))
    opened_by: Mapped[int] = mapped_column(ForeignKey("users.id"))      # the case owner
    current_step: Mapped[str | None] = mapped_column(String(30))
    evidence: Mapped[list | None] = mapped_column(JSONB)
    report: Mapped[dict | None] = mapped_column(JSONB)
    error: Mapped[str | None] = mapped_column(String(500))
    decision: Mapped[str | None] = mapped_column(String(20))
    decision_reason: Mapped[str | None]
    decided_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    decided_at: Mapped[datetime | None]
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())


class CaseEvent(Base):
    """Progress messages from the agents and people working on a case, in order."""
    __tablename__ = "case_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    case_id: Mapped[int] = mapped_column(ForeignKey("cases.id", ondelete="CASCADE"), index=True)
    step: Mapped[str] = mapped_column(String(30))
    message: Mapped[str]
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())