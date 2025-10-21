# api/routes/health_routes.py
"""
Health check endpoints for API and databases
"""

from fastapi import APIRouter, Depends
from typing import Dict, Any
from database.manager import get_database_manager, DatabaseManager

router = APIRouter()


@router.get("/")
async def health_check() -> Dict[str, str]:
    """Basic health check endpoint."""
    return {"status": "healthy", "service": "web_search_agent"}


@router.get("/databases")
async def database_health() -> Dict[str, Any]:
    """
    Check health of all database connections.
    
    Returns:
        Dictionary with health status for PostgreSQL, Redis, and ChromaDB
    """
    db_manager = get_database_manager()
    health = await db_manager.health_check()
    
    overall_status = "healthy" if all(health.values()) else "degraded"
    
    return {
        "status": overall_status,
        "databases": health
    }


@router.get("/ping")
async def ping() -> Dict[str, str]:
    """Simple ping endpoint."""
    return {"message": "pong"}
