# tools/web_search/langchain_tool.py - ENHANCED FOR FULL FUNCTIONALITY
"""
LangChain-compatible wrapper for the complete WebSearchTool.
Fully functional with Tavily API, content processing, and optional databases.
"""

import logging
from typing import Optional, List
from langchain_core.tools import tool

from config.settings import get_settings
from database.manager import DatabaseManager, get_database_manager
from .search_tool import WebSearchTool

logger = logging.getLogger(__name__)

# Global instances
_web_search_tool_instance: Optional[WebSearchTool] = None
_db_manager_instance: Optional[DatabaseManager] = None


def get_web_search_tool_instance() -> WebSearchTool:
    """Get or create the global WebSearchTool instance with full configuration."""
    global _web_search_tool_instance, _db_manager_instance
    
    if _web_search_tool_instance is None:
        settings = get_settings()
        
        # Initialize database manager if configured
        if settings.POSTGRES_URI or settings.REDIS_URI or settings.CHROMA_PATH:
            _db_manager_instance = get_database_manager(
                postgres_uri=settings.POSTGRES_URI,
                redis_uri=settings.REDIS_URI,
                chroma_path=settings.CHROMA_PATH
            )
            logger.info("✓ Database manager initialized for web search tool")
        else:
            _db_manager_instance = None
            logger.info("⚠ Web search tool running without database (standalone mode)")
        
        # Create tool instance
        _web_search_tool_instance = WebSearchTool(
            tavily_api_key=settings.TAVILY_API_KEY,
            openai_api_key=settings.OPENAI_API_KEY,
            db_manager=_db_manager_instance,
            enable_summarization=True,  # Use LLM summarization
            enable_offload=True,        # Save full content to disk
            offload_dir=str(settings.OFFLOAD_DIR)
        )
        logger.info("✓ WebSearchTool instance created and ready")
    
    return _web_search_tool_instance


@tool
async def web_search(
    query: str,
    max_results: int = 5,
    retrieval_method: str = "bm25"
) -> str:
    """
    Search the web for current information using Tavily API with full content processing.
    
    This tool:
    - Searches the web via Tavily API
    - Fetches full HTML content from each result
    - Converts HTML to markdown
    - Summarizes content using LLM (if OpenAI key provided)
    - Saves full content to disk for later reference
    - Optionally stores metadata in PostgreSQL
    - Optionally caches results in Redis
    - Optionally stores embeddings in ChromaDB
    - Applies advanced ranking (BM25, RRF, or cross-encoder)
    
    Args:
        query: The search query string
        max_results: Maximum number of results to return (1-10, default: 5)
        retrieval_method: Ranking method - "bm25", "rrf", or "cross_encoder" (default: "bm25")
    
    Returns:
        Formatted string with search results including titles, URLs, summaries, and file paths
    
    Examples:
        "What are the latest AI developments in 2025?"
        "How does LangGraph work?"
        "Python async programming best practices"
    """
    try:
        logger.info(f"🔍 Web search invoked: query='{query}', method={retrieval_method}")
        
        # Get tool instance (initializes on first call)
        search_tool = get_web_search_tool_instance()
        
        # Execute search with full processing
        results = await search_tool.search(
            query=query,
            session_id="langgraph_session",
            max_results=min(max_results, 10),  # Cap at 10
            use_cache=True,
            retrieval_method=retrieval_method,
            search_depth="advanced"
        )
        
        # Format results for agent consumption
        if not results:
            return f"No results found for query: {query}"
        
        formatted_results = [f"**Search Results for: {query}**\n"]
        
        for i, result in enumerate(results, 1):
            formatted_results.append(
                f"\n**{i}. {result.get('title', 'No title')}**\n"
                f"URL: {result.get('url', 'No URL')}\n"
                f"Summary: {result.get('snippet', 'No summary')}\n"
            )
            
            if result.get('offload_path'):
                formatted_results.append(f"Full content saved to: {result['offload_path']}\n")
        
        formatted_output = "\n".join(formatted_results)
        logger.info(f"✓ Search complete: returned {len(results)} results to agent")
        
        return formatted_output
        
    except Exception as e:
        logger.error(f"❌ Web search error: {e}", exc_info=True)
        return f"Error performing search: {str(e)}"


def get_web_search_tools() -> List:
    """
    Get list of web search tools for LangGraph agent.
    
    Returns:
        List containing the web_search tool
    """
    return [web_search]


# Standalone function for direct use (non-LangChain)
async def standalone_web_search(
    query: str,
    max_results: int = 5,
    tavily_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None
) -> List[dict]:
    """
    Standalone web search function that doesn't require LangChain.
    Useful for direct integration without agents.
    
    Args:
        query: Search query
        max_results: Maximum results
        tavily_api_key: Tavily API key (required)
        openai_api_key: OpenAI API key (optional, for summarization)
        
    Returns:
        List of result dictionaries
    """
    if not tavily_api_key:
        raise ValueError("tavily_api_key is required for standalone search")
    
    # Create standalone tool instance (no database)
    tool = WebSearchTool(
        tavily_api_key=tavily_api_key,
        openai_api_key=openai_api_key,
        db_manager=None,  # Standalone mode
        enable_summarization=bool(openai_api_key),
        enable_offload=True
    )
    
    results = await tool.search(
        query=query,
        max_results=max_results,
        use_cache=False  # No cache in standalone
    )
    
    return results
