# database/manager.py
"""
Multi-Database Manager for PostgreSQL, ChromaDB, and Redis
Implements singleton pattern with connection pooling
"""

import logging
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool
import chromadb
import redis.asyncio as redis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Centralized database manager for PostgreSQL, ChromaDB, and Redis.
    Singleton pattern ensures single instance across application.
    """
    _instance: Optional['DatabaseManager'] = None
    _initialized: bool = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        postgres_uri: Optional[str] = None,
        redis_uri: Optional[str] = None,
        chroma_path: Optional[str] = None
    ):
        """
        Initialize database connections (only once via singleton).
        
        Args:
            postgres_uri: PostgreSQL connection string
            redis_uri: Redis connection string
            chroma_path: ChromaDB persistent storage path
        """
        if self._initialized:
            return

        logger.info("Initializing DatabaseManager...")

        # PostgreSQL setup with async SQLAlchemy
        self.postgres_engine = None
        self.postgres_session_factory = None
        if postgres_uri:
            self.postgres_engine = create_async_engine(
                postgres_uri,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=False
            )
            self.postgres_session_factory = async_sessionmaker(
                self.postgres_engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            logger.info("✓ PostgreSQL engine initialized")

        # Redis setup
        self.redis_client = None
        if redis_uri:
            self.redis_client = redis.from_url(
                redis_uri,
                encoding="utf-8",
                decode_responses=True,
                max_connections=50
            )
            logger.info("✓ Redis client initialized")

        # ChromaDB setup (persistent)
        self.chroma_client = None
        if chroma_path:
            self.chroma_client = chromadb.PersistentClient(path=chroma_path)
            logger.info(f"✓ ChromaDB initialized at {chroma_path}")

        self._initialized = True
        logger.info("DatabaseManager fully initialized")

    @classmethod
    def get_instance(cls) -> 'DatabaseManager':
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @asynccontextmanager
    async def get_postgres_session(self):
        """
        Async context manager for PostgreSQL sessions.
        Automatically commits on success, rolls back on error.
        
        Usage:
            async with db_manager.get_postgres_session() as session:
                result = await session.execute(query)
        """
        if not self.postgres_session_factory:
            raise RuntimeError("PostgreSQL not configured")

        session = self.postgres_session_factory()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"PostgreSQL session error: {e}")
            raise
        finally:
            await session.close()

    async def get_redis(self) -> redis.Redis:
        """Get Redis client instance."""
        if not self.redis_client:
            raise RuntimeError("Redis not configured")
        return self.redis_client

    def get_chroma_collection(self, collection_name: str):
        """
        Get or create a ChromaDB collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            ChromaDB collection object
        """
        if not self.chroma_client:
            raise RuntimeError("ChromaDB not configured")
        return self.chroma_client.get_or_create_collection(name=collection_name)

    async def health_check(self) -> Dict[str, bool]:
        """
        Check health of all database connections.
        
        Returns:
            Dictionary with health status for each database
        """
        health = {}

        # PostgreSQL health
        if self.postgres_engine:
            try:
                async with self.get_postgres_session() as session:
                    from sqlalchemy import text
                    await session.execute(text("SELECT 1"))
                health['postgres'] = True
            except Exception as e:
                logger.error(f"PostgreSQL health check failed: {e}")
                health['postgres'] = False

        # Redis health
        if self.redis_client:
            try:
                await self.redis_client.ping()
                health['redis'] = True
            except Exception as e:
                logger.error(f"Redis health check failed: {e}")
                health['redis'] = False

        # ChromaDB health (always True if client exists)
        if self.chroma_client:
            try:
                self.chroma_client.heartbeat()
                health['chroma'] = True
            except Exception as e:
                logger.error(f"ChromaDB health check failed: {e}")
                health['chroma'] = False

        return health

    async def close_all(self):
        """Gracefully close all database connections."""
        logger.info("Closing all database connections...")

        if self.postgres_engine:
            await self.postgres_engine.dispose()
            logger.info("✓ PostgreSQL engine disposed")

        if self.redis_client:
            await self.redis_client.close()
            logger.info("✓ Redis client closed")

        # ChromaDB doesn't need explicit closing
        logger.info("All database connections closed")


# Global instance getter
_db_manager_instance: Optional[DatabaseManager] = None


def get_database_manager(
    postgres_uri: Optional[str] = None,
    redis_uri: Optional[str] = None,
    chroma_path: Optional[str] = None
) -> DatabaseManager:
    """
    Get or create the global DatabaseManager instance.
    First call initializes with config, subsequent calls return existing instance.
    """
    global _db_manager_instance
    if _db_manager_instance is None:
        _db_manager_instance = DatabaseManager(
            postgres_uri=postgres_uri,
            redis_uri=redis_uri,
            chroma_path=chroma_path
        )
    return _db_manager_instance
