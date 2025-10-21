# tools/web_search/storage.py
"""
Database storage operations for web search results
Handles PostgreSQL, Redis, and ChromaDB storage
"""

import logging
from typing import List, Dict, Any
from datetime import datetime
from sqlalchemy import text

logger = logging.getLogger(__name__)


class SearchStorage:
    """Handles storage operations for search results across databases."""

    def __init__(self, db_manager):
        self.db_manager = db_manager

    async def store_search_metadata(
        self,
        query: str,
        results: List[Dict[str, Any]],
        session_id: str
    ):
        """
        Store search metadata in PostgreSQL.
        
        Schema:
        - search_history: search_id, session_id, query, timestamp
        - search_results: result_id, search_id, url, title, snippet, position
        """
        try:
            async with self.db_manager.get_postgres_session() as session:
                # Insert search record
                search_query = text("""
                    INSERT INTO search_history (session_id, query, result_count, timestamp)
                    VALUES (:session_id, :query, :count, :timestamp)
                    RETURNING id
                """)
                
                result = await session.execute(
                    search_query,
                    {
                        "session_id": session_id,
                        "query": query,
                        "count": len(results),
                        "timestamp": datetime.utcnow()
                    }
                )
                search_id = result.scalar()

                # Insert individual results
                for idx, item in enumerate(results):
                    result_query = text("""
                        INSERT INTO search_results 
                        (search_id, url, title, snippet, position, score)
                        VALUES (:search_id, :url, :title, :snippet, :position, :score)
                    """)
                    
                    await session.execute(
                        result_query,
                        {
                            "search_id": search_id,
                            "url": item.get("url", ""),
                            "title": item.get("title", ""),
                            "snippet": item.get("snippet", ""),
                            "position": idx,
                            "score": item.get("score", 0.0)
                        }
                    )

                logger.info(f"✓ Stored search metadata (search_id: {search_id})")

        except Exception as e:
            logger.error(f"PostgreSQL storage error: {e}")

    async def store_search_embeddings(
        self,
        query: str,
        results: List[Dict[str, Any]],
        session_id: str
    ):
        """
        Store search result embeddings in ChromaDB for semantic retrieval.
        
        Collections:
        - search_results: Stores result snippets with metadata
        """
        try:
            collection = self.db_manager.get_chroma_collection("search_results")
            
            # Prepare documents and metadata
            documents = []
            metadatas = []
            ids = []
            
            for idx, result in enumerate(results):
                doc_id = f"{session_id}_{hash(query)}_{idx}"
                documents.append(result.get("snippet", ""))
                metadatas.append({
                    "session_id": session_id,
                    "query": query,
                    "url": result.get("url", ""),
                    "title": result.get("title", ""),
                    "position": idx,
                    "timestamp": datetime.utcnow().isoformat()
                })
                ids.append(doc_id)
            
            # Add to ChromaDB
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"✓ Stored {len(documents)} embeddings in ChromaDB")

        except Exception as e:
            logger.error(f"ChromaDB storage error: {e}")

    async def retrieve_similar_searches(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve similar past searches using ChromaDB semantic search.
        
        Args:
            query: Query to find similar searches for
            top_k: Number of similar results to return
            
        Returns:
            List of similar search results
        """
        try:
            collection = self.db_manager.get_chroma_collection("search_results")
            
            results = collection.query(
                query_texts=[query],
                n_results=top_k
            )
            
            similar = []
            for i in range(len(results["ids"][0])):
                similar.append({
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i]
                })
            
            logger.info(f"Retrieved {len(similar)} similar searches")
            return similar

        except Exception as e:
            logger.error(f"ChromaDB retrieval error: {e}")
            return []
