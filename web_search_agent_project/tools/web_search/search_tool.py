# tools/web_search/search_tool.py - ENHANCED WITH FULL TAVILY + DATABASE INTEGRATION
"""
Complete web search tool with Tavily API, content processing, and optional database storage.
Can work standalone OR with database integration - fully modular.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import requests

from database.manager import DatabaseManager
from .retrieval import bm25_score, rrf_score, cross_encoder_rerank
from .storage import SearchStorage
from .processors import process_search_results

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebSearchTool:
    """
    Complete web search tool with Tavily API integration and optional database storage.
    
    Features:
    - Tavily API for web search
    - Full content fetching and processing
    - LLM-based summarization
    - Content offloading to disk
    - Optional PostgreSQL metadata storage
    - Optional Redis caching
    - Optional ChromaDB embedding storage
    - BM25, RRF, and cross-encoder ranking
    """

    def __init__(
        self,
        tavily_api_key: str,
        openai_api_key: Optional[str] = None,
        db_manager: Optional[DatabaseManager] = None,
        enable_summarization: bool = True,
        enable_offload: bool = True,
        offload_dir: str = "./data/offloaded_pages"
    ):
        """
        Initialize web search tool.
        
        Args:
            tavily_api_key: Tavily API key (required)
            openai_api_key: OpenAI API key for summarization (optional)
            db_manager: DatabaseManager for storage (optional - tool works without it)
            enable_summarization: Use LLM to summarize content
            enable_offload: Save full content to disk
            offload_dir: Directory for offloaded content
        """
        self.tavily_api_key = tavily_api_key
        self.openai_api_key = openai_api_key
        self.db_manager = db_manager
        self.enable_summarization = enable_summarization
        self.enable_offload = enable_offload
        self.offload_dir = offload_dir
        
        # Storage only if db_manager provided
        self.storage = SearchStorage(db_manager) if db_manager else None
        
        logger.info(f"WebSearchTool initialized (DB: {bool(db_manager)}, Summarization: {enable_summarization}, Offload: {enable_offload})")

    async def search(
        self,
        query: str,
        session_id: str = "default",
        max_results: int = 5,
        use_cache: bool = True,
        retrieval_method: str = "bm25",
        search_depth: str = "advanced"
    ) -> List[Dict[str, Any]]:
        """
        Execute complete web search with Tavily API.
        
        Args:
            query: Search query
            session_id: Session identifier
            max_results: Maximum results to return
            use_cache: Use Redis cache (if db_manager available)
            retrieval_method: Ranking method (bm25, rrf, cross_encoder)
            search_depth: Tavily search depth (basic, advanced)
            
        Returns:
            List of processed search results
        """
        logger.info(f"🔍 Search: '{query}' | method: {retrieval_method} | depth: {search_depth}")

        # Check cache if available
        if use_cache and self.db_manager:
            cached = await self._get_from_cache(query)
            if cached:
                logger.info("✓ Cache hit - returning cached results")
                return cached

        # Execute Tavily search
        raw_results = self._tavily_search(query, max_results, search_depth)
        
        if not raw_results:
            logger.warning(f"No results from Tavily for: {query}")
            return []

        # Process results (fetch content, summarize, offload)
        processed_results = process_search_results(
            raw_results,
            enable_summarization=self.enable_summarization,
            enable_offload=self.enable_offload,
            offload_dir=self.offload_dir,
            openai_api_key=self.openai_api_key
        )

        # Apply ranking algorithm
        ranked_results = await self._rank_results(query, processed_results, retrieval_method)

        # Store in databases if available
        if self.storage:
            await self._store_results(query, ranked_results, session_id)

        # Cache if available
        if use_cache and self.db_manager:
            await self._cache_results(query, ranked_results)

        logger.info(f"✓ Search complete: {len(ranked_results)} results")
        return ranked_results

    def _tavily_search(
        self,
        query: str,
        max_results: int,
        search_depth: str
    ) -> List[Dict[str, Any]]:
        """
        Execute actual Tavily API search.
        
        Args:
            query: Search query
            max_results: Number of results
            search_depth: Search depth (basic or advanced)
            
        Returns:
            Raw Tavily results
        """
        url = "https://api.tavily.com/search"
        
        payload = {
            "api_key": self.tavily_api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": search_depth,
            "include_answer": True,
            "include_raw_content": False,  # We fetch content ourselves
            "include_images": False
        }
        
        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            results = data.get("results", [])
            
            logger.info(f"✓ Tavily returned {len(results)} results")
            return results
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Tavily API error: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in Tavily search: {e}")
            return []

    async def _rank_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        method: str
    ) -> List[Dict[str, Any]]:
        """Apply ranking algorithm to results."""
        logger.info(f"Ranking with: {method}")
        
        if method == "bm25":
            return bm25_score(query, results)
        elif method == "rrf":
            return rrf_score([results])
        elif method == "cross_encoder":
            return cross_encoder_rerank(query, results)
        else:
            logger.warning(f"Unknown ranking method: {method}, returning as-is")
            return results

    async def _store_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        session_id: str
    ):
        """Store results in PostgreSQL and ChromaDB (if available)."""
        if not self.storage:
            return
            
        logger.info("Storing results in databases...")
        
        try:
            # Store metadata in PostgreSQL
            await self.storage.store_search_metadata(query, results, session_id)
            
            # Store embeddings in ChromaDB
            await self.storage.store_search_embeddings(query, results, session_id)
            
        except Exception as e:
            logger.error(f"Error storing results: {e}")

    async def _get_from_cache(self, query: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached results from Redis."""
        if not self.db_manager:
            return None
            
        try:
            redis_client = await self.db_manager.get_redis()
            cache_key = f"search:{query}"
            cached = await redis_client.get(cache_key)
            
            if cached:
                return json.loads(cached)
            return None
        except Exception as e:
            logger.error(f"Cache retrieval error: {e}")
            return None

    async def _cache_results(self, query: str, results: List[Dict[str, Any]]):
        """Cache results in Redis with TTL."""
        if not self.db_manager:
            return
            
        try:
            redis_client = await self.db_manager.get_redis()
            cache_key = f"search:{query}"
            
            # Cache for 1 hour
            await redis_client.setex(
                cache_key,
                3600,
                json.dumps(results, default=str)
            )
            logger.info(f"✓ Cached results for: {query}")
        except Exception as e:
            logger.error(f"Cache storage error: {e}")
