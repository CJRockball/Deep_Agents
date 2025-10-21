# database/models.py
"""
SQLAlchemy models for PostgreSQL database
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class SearchHistory(Base):
    """Search history table - tracks all search queries."""
    __tablename__ = "search_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(255), nullable=False, index=True)
    query = Column(Text, nullable=False)
    result_count = Column(Integer, default=0)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    # Relationship to results
    results = relationship("SearchResult", back_populates="search", cascade="all, delete-orphan")


class SearchResult(Base):
    """Individual search results linked to search history."""
    __tablename__ = "search_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    search_id = Column(Integer, ForeignKey("search_history.id"), nullable=False)
    url = Column(Text, nullable=False)
    title = Column(Text)
    snippet = Column(Text)
    position = Column(Integer)
    score = Column(Float, default=0.0)

    # Relationship back to search
    search = relationship("SearchHistory", back_populates="results")


class AgentSession(Base):
    """Agent session tracking."""
    __tablename__ = "agent_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(255), unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    metadata = Column(Text)  # JSON string for additional data


# Database initialization function
async def init_database(engine):
    """Initialize database tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
