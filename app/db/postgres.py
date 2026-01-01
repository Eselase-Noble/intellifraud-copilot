from __future__ import annotations
import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine, AsyncSession
from sqlalchemy.orm import sessionmaker
from app.utils.config import settings


#author: Noble Eselase Vulley
#version: 1.0.0


def make_engine() -> AsyncEngine:
    # We use SQLite for local MVP but keep file name as postgres.py to match blueprint.
    db_path = settings.sqlite_path
    url = f"sqlite+aiosqlite:///{db_path}"
    return create_async_engine(url, echo=False, future=True)

engine = make_engine()
AsyncSessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

async def get_session() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session
