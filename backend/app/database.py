
# backend/app/database.py - Database Configuration
import os
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from typing import AsyncGenerator

# Database URL from environment
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./multimodal_ai.db")

# Create async engine
engine = create_async_engine(
    DATABASE_URL,
    echo=True if os.getenv("DEBUG") == "true" else False,
    future=True
)

# Create async session factory
async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False
)

Base = declarative_base()

# Dependency to get database session
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Get database session"""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

async def create_tables():
    """Create database tables"""
    from .models import Base  # Import here to avoid circular imports
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def check_database_health() -> bool:
    """Check database connection health"""
    try:
        async with engine.begin() as conn:
            await conn.execute("SELECT 1")
        return True
    except Exception:
        return False