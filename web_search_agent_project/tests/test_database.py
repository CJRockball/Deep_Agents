# tests/test_database.py
"""
Database integration tests
Tests PostgreSQL, Redis, and ChromaDB connections
"""

import pytest
import asyncio
from database.manager import DatabaseManager, get_database_manager
from config.settings import get_settings


@pytest.fixture
async def db_manager():
    """Fixture to provide database manager for tests."""
    settings = get_settings()
    manager = get_database_manager(
        postgres_uri=settings.POSTGRES_URI,
        redis_uri=settings.REDIS_URI,
        chroma_path=settings.CHROMA_PATH
    )
    yield manager
    await manager.close_all()


@pytest.mark.asyncio
async def test_database_manager_singleton():
    """Test that DatabaseManager is a singleton."""
    manager1 = DatabaseManager.get_instance()
    manager2 = DatabaseManager.get_instance()
    assert manager1 is manager2


@pytest.mark.asyncio
async def test_postgres_connection(db_manager):
    """Test PostgreSQL connection."""
    async with db_manager.get_postgres_session() as session:
        from sqlalchemy import text
        result = await session.execute(text("SELECT 1"))
        assert result.scalar() == 1


@pytest.mark.asyncio
async def test_redis_connection(db_manager):
    """Test Redis connection."""
    redis_client = await db_manager.get_redis()
    await redis_client.set("test_key", "test_value")
    value = await redis_client.get("test_key")
    assert value == "test_value"
    await redis_client.delete("test_key")


@pytest.mark.asyncio
async def test_chroma_collection(db_manager):
    """Test ChromaDB collection creation."""
    collection = db_manager.get_chroma_collection("test_collection")
    assert collection is not None
    assert collection.name == "test_collection"


@pytest.mark.asyncio
async def test_health_check(db_manager):
    """Test database health check."""
    health = await db_manager.health_check()
    assert isinstance(health, dict)
    assert "postgres" in health or "redis" in health or "chroma" in health


@pytest.mark.asyncio
async def test_postgres_session_rollback(db_manager):
    """Test that PostgreSQL sessions rollback on error."""
    try:
        async with db_manager.get_postgres_session() as session:
            from sqlalchemy import text
            # This should fail
            await session.execute(text("SELECT * FROM nonexistent_table"))
    except Exception:
        pass  # Expected to fail
    
    # Session should still work after error
    async with db_manager.get_postgres_session() as session:
        from sqlalchemy import text
        result = await session.execute(text("SELECT 1"))
        assert result.scalar() == 1
