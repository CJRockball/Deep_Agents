# api/main.py
"""
FastAPI application entry point (prepared, not fully implemented)
Ready for backend/frontend integration
"""

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from config.settings import get_settings
from database.manager import get_database_manager
from api.routes import agent_routes, health_routes

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup/shutdown events.
    Initializes databases on startup, closes on shutdown.
    """
    logger.info("Starting up application...")
    
    # Initialize database manager
    settings = get_settings()
    db_manager = get_database_manager(
        postgres_uri=settings.POSTGRES_URI,
        redis_uri=settings.REDIS_URI,
        chroma_path=settings.CHROMA_PATH
    )
    
    # Health check
    health = await db_manager.health_check()
    logger.info(f"Database health: {health}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")
    await db_manager.close_all()


# Create FastAPI app
app = FastAPI(
    title="Web Search Agent API",
    description="Multi-database web search agent with BM25, RRF, and cross-encoder ranking",
    version="0.1.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_routes.router, prefix="/health", tags=["Health"])
app.include_router(agent_routes.router, prefix="/agent", tags=["Agent"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Web Search Agent API",
        "version": "0.1.0",
        "status": "ready"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
