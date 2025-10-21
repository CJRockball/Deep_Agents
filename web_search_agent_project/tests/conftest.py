# tests/conftest.py
"""
Pytest configuration and fixtures for testing
"""

import pytest
import asyncio
from typing import Generator, AsyncGenerator

from config.settings import get_settings
from database.manager import get_database_manager, DatabaseManager


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def settings():
    """Provide settings for tests."""
    return get_settings()


@pytest.fixture(scope="function")
async def db_manager(settings) -> AsyncGenerator[DatabaseManager, None]:
    """
    Provide database manager for tests.
    Creates fresh instance for each test.
    """
    manager = get_database_manager(
        postgres_uri=settings.POSTGRES_URI,
        redis_uri=settings.REDIS_URI,
        chroma_path=settings.CHROMA_PATH
    )
    
    yield manager
    
    # Cleanup after test
    await manager.close_all()


@pytest.fixture(scope="function")
async def clean_redis(db_manager):
    """Clean Redis before test."""
    redis = await db_manager.get_redis()
    await redis.flushdb()
    yield
    await redis.flushdb()


@pytest.fixture(scope="function")
def clean_chroma(db_manager):
    """Clean ChromaDB collection before test."""
    # Get test collection
    collection = db_manager.get_chroma_collection("test_collection")
    
    # Delete all items
    try:
        collection.delete()
    except:
        pass
    
    yield collection
