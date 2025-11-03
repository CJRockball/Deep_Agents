# services/agent-core/src/main.py

import boot

import logging
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from agents.react_agent import FinancialReActAgent
from memory.persistence import init_db
from db.mongo_client import db
from db.neo4j_client import driver



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager using modern FastAPI async context manager.
    Handles startup and shutdown events for database initialization and cleanup.
    """
    # Startup
    logger.info("🚀 Starting IntelliFinQ Agent Platform")
    
    try:
        # Initialize PostgreSQL database schema
        logger.info("Initializing PostgreSQL database...")
        init_db()
        
        # Test MongoDB connection
        logger.info("Testing MongoDB connection...")
        _ = db.list_collection_names()  # Test connection
        logger.info("MongoDB connected successfully")
        
        # Test Neo4j connection
        logger.info("Testing Neo4j connection...")
        with driver.session() as session:
            session.run("RETURN 1")
        logger.info("Neo4j connected successfully")
        
        # Initialize agent (could pre-load models here)
        logger.info("Initializing ReAct agent...")
        # The agent will be initialized when first requested
        
        logger.info("✅ All services initialized successfully")
        
    except Exception as e:
        logger.error("❌ Failed to initialize services: %s", e)
        raise
    
    # Yield control to FastAPI to handle requests
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down IntelliFinQ Agent Platform")
    
    try:
        # Close Neo4j driver
        logger.info("Closing Neo4j connection...")
        driver.close()
        
        # Close MongoDB client (if needed)
        # MongoDB client usually handles this automatically
        
        logger.info("✅ Cleanup completed successfully")
        
    except Exception as e:
        logger.error("❌ Error during shutdown: %s", e)

# Create FastAPI app with lifespan
app = FastAPI(
    title="IntelliFinQ Chat API",
    description="A financial Q&A agent with memory, web search, and chat interface",
    version="1.0.0",
    lifespan=lifespan  # Use the modern lifespan parameter
)

# Mount static files and templates
app.mount("/static", 
          StaticFiles(directory=str(BASE_DIR /"static")), 
          name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# JSON schema for AJAX chat requests
class QueryRequest(BaseModel):
    user_id: str
    query: str

class QueryResponse(BaseModel):
    response: str

# Global agent instance (initialized lazily)
_agent = None

def get_agent():
    """Get or create the ReAct agent instance."""
    global _agent
    if _agent is None:
        logger.info("Creating FinancialReActAgent instance")
        _agent = FinancialReActAgent()
    return _agent

# Routes

@app.get("/", response_class=HTMLResponse)
async def get_chat_interface(request: Request):
    """Serve the main chat UI."""
    logger.info("Serving chat interface")
    return templates.TemplateResponse("chat.html", {"request": request})

@app.post("/api/chat", response_model=QueryResponse)
async def chat_endpoint(req: QueryRequest):
    """
    Handle chat requests from the web interface.
    
    Processes user queries through the ReAct agent workflow:
    - Saves user message to memory
    - Retrieves and filters relevant context
    - Performs web search if needed
    - Generates response using LLM
    - Saves agent response to memory
    """
    try:
        logger.info("🔵 Chat request from user: %s", req.user_id)
        logger.debug("Query: %s", req.query[:100])
        
        agent = get_agent()
        answer = agent.handle_query(req.user_id, req.query)
        
        logger.info("✅ Successfully processed chat request")
        return QueryResponse(response=answer)
        
    except Exception as e:
        logger.exception("❌ Error processing chat request")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/api/health")
async def health_check():
    """Health check endpoint to verify service status."""
    try:
        # Test database connections
        db_status = {}
        
        # Test PostgreSQL
        try:
            from db.postgres_client import SessionLocal
            with SessionLocal() as session:
                session.execute("SELECT 1")
            db_status["postgresql"] = "healthy"
        except Exception as e:
            db_status["postgresql"] = f"error: {str(e)}"
        
        # Test MongoDB
        try:
            _ = db.list_collection_names()
            db_status["mongodb"] = "healthy"
        except Exception as e:
            db_status["mongodb"] = f"error: {str(e)}"
        
        # Test Neo4j
        try:
            with driver.session() as session:
                session.run("RETURN 1")
            db_status["neo4j"] = "healthy"
        except Exception as e:
            db_status["neo4j"] = f"error: {str(e)}"
        
        return {
            "status": "healthy",
            "service": "IntelliFinQ Agent Platform",
            "databases": db_status
        }
        
    except Exception as e:
        logger.exception("Health check failed")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@app.get("/api/user/{user_id}/history")
async def get_user_history(user_id: str, limit: int = 50):
    """Get conversation history for a specific user."""
    try:
        logger.info("Fetching history for user: %s", user_id)
        
        from memory.persistence import load_messages
        messages = load_messages(user_id)
        
        # Convert to API format
        history = []
        for msg in messages[-limit:]:  # Get last N messages
            history.append({
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat()
            })
        
        return {"user_id": user_id, "history": history}
        
    except Exception as e:
        logger.exception("Error fetching user history")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers

@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Custom 404 handler."""
    return templates.TemplateResponse(
        "chat.html", 
        {"request": request}, 
        status_code=404
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    """Custom 500 handler."""
    logger.exception("Internal server error")
    return {"error": "Internal server error"}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting development server")
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )
