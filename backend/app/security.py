"""Authentication (who are you?), authorisation (may you do this?) and the audit log."""
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from pwdlib import PasswordHash
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.models import AuditEvent, User

password_hasher = PasswordHash.recommended()  # Argon2id
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")
# Checked when the email doesn't exist, so a failed login takes the same time either way.
_DUMMY_HASH = password_hasher.hash("not-a-real-password")


def hash_password(password: str) -> str:
    return password_hasher.hash(password)


def authenticate(db: Session, email: str, password: str) -> User | None:
    user = db.scalar(select(User).where(User.email == email.strip().lower()))
    if user is None:
        password_hasher.verify(password, _DUMMY_HASH)
        return None
    return user if password_hasher.verify(password, user.password_hash) else None


def create_access_token(user_id: int) -> str:
    expires = datetime.now(timezone.utc) + timedelta(minutes=settings.access_token_minutes)
    return jwt.encode({"sub": str(user_id), "exp": expires}, settings.secret_key, algorithm="HS256")


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    unauthorized = HTTPException(401, "Invalid or expired token", headers={"WWW-Authenticate": "Bearer"})
    try:
        user_id = int(jwt.decode(token, settings.secret_key, algorithms=["HS256"])["sub"])
    except (jwt.InvalidTokenError, KeyError, ValueError):
        raise unauthorized
    user = db.get(User, user_id)
    if user is None:
        raise unauthorized
    return user


def require_role(*roles: str):
    """Dependency factory: only these roles (admins always pass)."""
    def check(user: User = Depends(get_current_user)) -> User:
        if user.role != "admin" and user.role not in roles:
            raise HTTPException(403, "Your role can't do this")
        return user
    return check


def audit(db: Session, actor_id: int | None, action: str, object_type: str | None = None,
          object_id: object = None, ip: str | None = None, **details) -> None:
    """Add an audit row to the current transaction; the caller commits it with the action."""
    db.add(AuditEvent(actor_id=actor_id, action=action, object_type=object_type,
                      object_id=None if object_id is None else str(object_id), ip=ip, details=details))
