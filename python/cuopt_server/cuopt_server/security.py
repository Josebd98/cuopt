# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Security module for cuOpt server.
Handles JWT token generation/validation, password hashing, and user authentication,
reading users from a real SQL Server database using SQLAlchemy.
"""

from __future__ import annotations
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

# --- Load environment variables ---
from dotenv import load_dotenv
load_dotenv()  # Loads .env into os.environ

# --- SQLAlchemy ---
from sqlalchemy import create_engine, func, or_, Boolean, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SECRET_KEY = os.environ.get("CUOPT_SECRET_KEY", "dev_secret_key")  # fallback para entorno local
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("CUOPT_TOKEN_EXPIRE_MINUTES", "60"))

# Database connection (SQLAlchemy)
# Usa la cadena ya compatible con SQLAlchemy de tu .env
DB_URL = os.environ.get("ConnectionString")
if not DB_URL:
    raise RuntimeError("ConnectionString must be set in .env")

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# HTTP Bearer token scheme
security_scheme = HTTPBearer()

# ---------------------------------------------------------------------------
# SQLAlchemy setup
# ---------------------------------------------------------------------------

engine = create_engine(DB_URL, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()


class DbUser(Base):
    """
    ORM mapping for [dbo].[User]
    Only essential fields for authentication are included.
    """
    __tablename__ = "User"
    __table_args__ = {"schema": "dbo"}

    UserId = Column(Integer, primary_key=True)
    Identifier = Column(String, nullable=True)
    Email = Column(String, nullable=True)
    PasswordHash = Column(String, nullable=False)
    Enabled = Column(Boolean, nullable=True)
    ActiveChk = Column(Boolean, nullable=True)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None


class User(BaseModel):
    username: str
    disabled: Optional[bool] = None


class UserInDB(User):
    hashed_password: str


class LoginRequest(BaseModel):
    username: str
    password: str

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password with bcrypt"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Generate bcrypt hash"""
    return pwd_context.hash(password)


def _row_to_userindb(row: DbUser) -> UserInDB:
    username = row.Email.strip()
    is_disabled = False
    if row.Enabled is not None and row.Enabled is False:
        is_disabled = True
    if row.ActiveChk is not None and row.ActiveChk is False:
        is_disabled = True
    return UserInDB(username=username, disabled=is_disabled, hashed_password=row.PasswordHash)


def get_user(username: str) -> Optional[UserInDB]:
    """Fetch user by Identifier or Email"""
    uname = (username or "").strip().lower()
    if not uname:
        return None

    with SessionLocal() as db:
        row = (
            db.query(DbUser)
            .filter(
                or_(
                    func.lower(DbUser.Email) == uname,
                )
            )
            .first()
        )
        if not row:
            return None
        return _row_to_userindb(row)


def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    """Authenticate user against database"""
    user = get_user(username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode = data.copy()
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security_scheme),
) -> User:
    """Get the current authenticated user from JWT"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: Optional[str] = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception

    user = get_user(username=token_data.username or "")
    if user is None:
        raise credentials_exception

    if user.disabled:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Inactive user")

    return User(username=user.username, disabled=user.disabled)


async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """Ensure user is active"""
    if current_user.disabled:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Inactive user")
    return current_user
