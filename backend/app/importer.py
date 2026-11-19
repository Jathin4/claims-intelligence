"""CSV import with row-level errors: good rows are imported, bad rows are reported with the reason."""
import csv
import io
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import CATEGORIES, Claim, ClaimImport
from app.security import audit

REQUIRED = ["claim_id", "customer_id", "policy_start_date", "claim_date", "claim_category", "claim_amount"]
OPTIONAL = ["region", "document_count", "manual_review_required"]
REGIONS = ("north", "south", "east", "west", "central")


class ClaimRow(BaseModel):
    claim_id: str = Field(pattern=r"^CLM-\d{6}$")
    customer_id: str = Field(pattern=r"^CUS-\d{5}$")
    region: Literal[REGIONS] | None = None
    policy_start_date: date
    claim_date: date
    claim_category: Literal[CATEGORIES]
    claim_amount: float = Field(gt=0, le=10_000_000)
    document_count: int | None = Field(default=None, ge=0, le=200)
    manual_review_required: Literal[0, 1] | None = None

    @field_validator("*", mode="before")
    @classmethod
    def blank_is_missing(cls, v):
        return None if isinstance(v, str) and v.strip() == "" else v

    @field_validator("document_count", "manual_review_required", mode="before")
    @classmethod
    def whole_number(cls, v):
        """Exports often write integers as '3.0'. Accept that, reject '3.5'."""
        if isinstance(v, str):
            if v.strip() == "":                 # validators can run in either order: handle blanks here too
                return None
            f = float(v)
            if not f.is_integer():
                raise ValueError("must be a whole number")
            return int(f)
        return v

    @field_validator("claim_date")
    @classmethod
    def not_in_future(cls, v):
        if v > date.today():
            raise ValueError("claim date is in the future")
        return v


class ImportRejected(Exception):
    """The whole file is unusable (wrong type, missing columns)."""


def import_claims(db: Session, user_id: int | None, filename: str, data: bytes) -> ClaimImport:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ImportRejected("The file isn't UTF-8 text. Export it as CSV (UTF-8).")
    reader = csv.DictReader(io.StringIO(text))
    header = [h.strip() for h in (reader.fieldnames or [])]
    missing = [c for c in REQUIRED if c not in header]
    if missing:
        raise ImportRejected(f"Missing required column(s): {', '.join(missing)}")
    unknown = [c for c in header if c not in REQUIRED + OPTIONAL]

    existing = set(db.scalars(select(Claim.claim_id)))
    seen: set[str] = set()
    good: list[Claim] = []
    errors: list[str] = []
    total = 0
    for line_no, raw in enumerate(reader, start=2):              # line 1 is the header
        total += 1
        row = {k.strip(): v for k, v in raw.items() if k and k.strip() in REQUIRED + OPTIONAL}
        try:
            r = ClaimRow.model_validate(row)
        except ValidationError as e:
            for err in e.errors():
                field = err["loc"][0] if err["loc"] else "row"
                errors.append(f"row {line_no}: {field}: {err['msg']}")
            continue
        if r.policy_start_date > r.claim_date:
            errors.append(f"row {line_no}: policy_start_date: is after the claim date")
            continue
        if r.claim_id in existing or r.claim_id in seen:
            errors.append(f"row {line_no}: claim_id: {r.claim_id} already exists")
            continue
        seen.add(r.claim_id)
        good.append(Claim(**r.model_dump()))

    rejected = total - len(good)
    imp = ClaimImport(user_id=user_id, filename=filename[:255], rows_total=total, rows_imported=len(good),
                      rows_rejected=rejected, errors=errors[: settings.max_row_errors_reported])
    db.add(imp)
    db.flush()
    for c in good:
        c.import_id = imp.id
    db.add_all(good)
    audit(db, user_id, "claims.import", "claim_import", imp.id, filename=filename,
          imported=len(good), rejected=rejected, ignored_columns=unknown)
    db.commit()
    # Drift check against the active model's training data (after the rows are saved, so history counts)
    from app.scoring import drift_for_claims
    imp.drift = drift_for_claims(db, [c.claim_id for c in good])
    if imp.drift and imp.drift.get("warnings"):
        audit(db, user_id, "claims.import_drift", "claim_import", imp.id, warnings=imp.drift["warnings"])
    db.commit()
    imp.ignored_columns = unknown           # not stored; returned to the caller
    return imp
