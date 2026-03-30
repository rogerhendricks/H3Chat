import hashlib
import os
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, Header, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

# Load .env for local development
load_dotenv()

# ---------------------------------------------------------------------------
# Configuration (env-based)
# ---------------------------------------------------------------------------
AUTH_DB_PATH = os.getenv("AUTH_DB_PATH", "auth.db")

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change-me")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "15"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))

# Simple admin token for bootstrapping users (admin-created users only)
ADMIN_BOOTSTRAP_TOKEN = os.getenv("ADMIN_BOOTSTRAP_TOKEN", "")

# ---------------------------------------------------------------------------
# Security / hashing
# ---------------------------------------------------------------------------
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class RefreshRequest(BaseModel):
    refresh_token: str


class UserCreate(BaseModel):
    username: str
    password: str
    email: Optional[str] = None
    is_active: bool = True


class UserOut(BaseModel):
    id: int
    username: str
    email: Optional[str] = None
    is_active: bool


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------
def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(AUTH_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_auth_db() -> None:
    conn = _get_db()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            email TEXT,
            password_hash TEXT NOT NULL,
            is_active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS refresh_tokens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            token_hash TEXT NOT NULL UNIQUE,
            family_id TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            revoked INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            last_used_at TEXT,
            replaced_at TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        );
        """
    )

    # Migrations for older auth.db schemas (add missing columns)
    for stmt in (
        "ALTER TABLE refresh_tokens ADD COLUMN family_id TEXT",
        "ALTER TABLE refresh_tokens ADD COLUMN last_used_at TEXT",
        "ALTER TABLE refresh_tokens ADD COLUMN replaced_at TEXT",
    ):
        try:
            cur.execute(stmt)
        except sqlite3.OperationalError:
            pass

    # Backfill family_id for legacy rows if needed
    try:
        cur.execute(
            "UPDATE refresh_tokens SET family_id = token_hash WHERE family_id IS NULL"
        )
    except sqlite3.OperationalError:
        pass

    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Password + token helpers
# ---------------------------------------------------------------------------
def _hash_password(password: str) -> str:
    return pwd_context.hash(password)


def _verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def _hash_refresh_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _create_access_token(subject: str, expires_delta: timedelta) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": subject,
        "type": "access",
        "iat": int(now.timestamp()),
        "exp": int((now + expires_delta).timestamp()),
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def _create_refresh_token(
    subject: str, expires_delta: timedelta, family_id: str
) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": subject,
        "type": "refresh",
        "fid": family_id,
        "iat": int(now.timestamp()),
        "exp": int((now + expires_delta).timestamp()),
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


# ---------------------------------------------------------------------------
# User CRUD
# ---------------------------------------------------------------------------
def get_user_by_username(username: str) -> Optional[sqlite3.Row]:
    conn = _get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    return row


def get_user_by_id(user_id: int) -> Optional[sqlite3.Row]:
    conn = _get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    row = cur.fetchone()
    conn.close()
    return row


def create_user(user: UserCreate) -> UserOut:
    conn = _get_db()
    cur = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()
    password_hash = _hash_password(user.password)

    try:
        cur.execute(
            """
            INSERT INTO users (username, email, password_hash, is_active, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user.username, user.email, password_hash, 1 if user.is_active else 0, now),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already exists.",
        )
    user_id = cur.lastrowid
    conn.close()
    return UserOut(
        id=user_id,
        username=user.username,
        email=user.email,
        is_active=user.is_active,
    )


def authenticate_user(username: str, password: str) -> Optional[sqlite3.Row]:
    user = get_user_by_username(username)
    if not user:
        return None
    if not user["is_active"]:
        return None
    if not _verify_password(password, user["password_hash"]):
        return None
    return user


# ---------------------------------------------------------------------------
# Refresh token storage
# ---------------------------------------------------------------------------
def store_refresh_token(
    user_id: int, token: str, expires_at: datetime, family_id: str
) -> None:
    conn = _get_db()
    cur = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()
    cur.execute(
        """
        INSERT INTO refresh_tokens
        (user_id, token_hash, family_id, expires_at, revoked, created_at, last_used_at)
        VALUES (?, ?, ?, ?, 0, ?, ?)
        """,
        (
            user_id,
            _hash_refresh_token(token),
            family_id,
            expires_at.isoformat(),
            now,
            now,
        ),
    )
    conn.commit()
    conn.close()


def revoke_refresh_token(token: str) -> None:
    conn = _get_db()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE refresh_tokens
        SET revoked = 1, replaced_at = ?
        WHERE token_hash = ?
        """,
        (datetime.now(timezone.utc).isoformat(), _hash_refresh_token(token)),
    )
    conn.commit()
    conn.close()


def revoke_refresh_token_family(family_id: str) -> None:
    conn = _get_db()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE refresh_tokens
        SET revoked = 1, replaced_at = ?
        WHERE family_id = ?
        """,
        (datetime.now(timezone.utc).isoformat(), family_id),
    )
    conn.commit()
    conn.close()


def get_refresh_token_record(token: str) -> Optional[sqlite3.Row]:
    conn = _get_db()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT user_id, family_id, expires_at, revoked
        FROM refresh_tokens
        WHERE token_hash = ?
        """,
        (_hash_refresh_token(token),),
    )
    row = cur.fetchone()
    conn.close()
    return row


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------
def get_current_user(token: str = Depends(oauth2_scheme)) -> UserOut:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        if payload.get("type") != "access":
            raise credentials_exception
        username: str = payload.get("sub")
        if not username:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = get_user_by_username(username)
    if not user:
        raise credentials_exception
    if not user["is_active"]:
        raise HTTPException(status_code=403, detail="User is inactive.")
    return UserOut(
        id=user["id"],
        username=user["username"],
        email=user["email"],
        is_active=bool(user["is_active"]),
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
def login(form_data: OAuth2PasswordRequestForm = Depends()) -> TokenResponse:
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect username or password.")

    access_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    refresh_expires = timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)

    access_token = _create_access_token(user["username"], access_expires)
    family_id = str(uuid.uuid4())
    refresh_token = _create_refresh_token(user["username"], refresh_expires, family_id)

    store_refresh_token(
        user["id"],
        refresh_token,
        datetime.now(timezone.utc) + refresh_expires,
        family_id,
    )

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=int(access_expires.total_seconds()),
    )


@router.post("/refresh", response_model=TokenResponse)
def refresh_tokens(body: RefreshRequest) -> TokenResponse:
    # Verify JWT signature/type
    try:
        payload = jwt.decode(
            body.refresh_token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM]
        )
        if payload.get("type") != "refresh":
            raise HTTPException(status_code=401, detail="Invalid refresh token.")
        username = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid refresh token.")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid refresh token.")

    # Verify token is stored and not revoked (reuse detection)
    record = get_refresh_token_record(body.refresh_token)
    if not record:
        raise HTTPException(status_code=401, detail="Invalid refresh token.")

    if record["revoked"]:
        revoke_refresh_token_family(record["family_id"])
        raise HTTPException(
            status_code=401,
            detail="Refresh token reuse detected. Please re-authenticate.",
        )

    expires_at = datetime.fromisoformat(record["expires_at"])
    if expires_at < datetime.now(timezone.utc):
        revoke_refresh_token_family(record["family_id"])
        raise HTTPException(status_code=401, detail="Refresh token expired.")

    user = get_user_by_id(int(record["user_id"]))
    if not user or not user["is_active"]:
        raise HTTPException(status_code=403, detail="User is inactive.")

    # Rotate refresh token (same family)
    revoke_refresh_token(body.refresh_token)

    access_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    refresh_expires = timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)

    access_token = _create_access_token(user["username"], access_expires)
    new_refresh_token = _create_refresh_token(
        user["username"], refresh_expires, record["family_id"]
    )
    store_refresh_token(
        user["id"],
        new_refresh_token,
        datetime.now(timezone.utc) + refresh_expires,
        record["family_id"],
    )

    return TokenResponse(
        access_token=access_token,
        refresh_token=new_refresh_token,
        expires_in=int(access_expires.total_seconds()),
    )


@router.get("/me", response_model=UserOut)
def me(current_user: UserOut = Depends(get_current_user)) -> UserOut:
    return current_user


@router.post("/admin/create_user", response_model=UserOut)
def admin_create_user(
    user: UserCreate,
    x_admin_token: Optional[str] = Header(default=None, alias="X-Admin-Token"),
) -> UserOut:
    if not ADMIN_BOOTSTRAP_TOKEN or x_admin_token != ADMIN_BOOTSTRAP_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid admin token.")
    return create_user(user)


# Convenience function to include in startup if desired
def setup_auth() -> None:
    init_auth_db()
