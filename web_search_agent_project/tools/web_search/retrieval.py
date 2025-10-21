# tools/web_search/retrieval.py
"""
Advanced retrieval algorithms: BM25, RRF, Cross-Encoder
Bare bones implementations with logging for extensibility
"""

import logging
from typing import List, Dict, Any, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


from rank_bm25 import BM25Okapi
import numpy as np

def bm25_score(query: str, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Real BM25 scoring using rank-bm25 library.
    """
    logger.info(f"BM25 ranking for query: '{query}'")
    
    if not docs:
        return docs
    
    # Prepare corpus (tokenize documents)
    corpus = []
    for doc in docs:
        text = doc.get("snippet", "") + " " + doc.get("content", "") + " " + doc.get("title", "")
        # Simple tokenization (lowercase + split)
        tokens = text.lower().split()
        corpus.append(tokens)
    
    # Create BM25 model
    bm25 = BM25Okapi(corpus)
    
    # Tokenize query
    query_tokens = query.lower().split()
    
    # Get BM25 scores
    scores = bm25.get_scores(query_tokens)
    
    # Add scores to documents
    for i, doc in enumerate(docs):
        doc["bm25_score"] = float(scores[i])
    
    # Sort by BM25 score
    ranked = sorted(docs, key=lambda x: x.get("bm25_score", 0), reverse=True)
    
    logger.info(f"BM25 ranked {len(ranked)} documents")
    return ranked


def rrf_score(
    results_lists: List[List[Dict[str, Any]]],
    k: int = 60
) -> List[Dict[str, Any]]:
    """
    Real Reciprocal Rank Fusion implementation.
    Formula: score(d) = sum(1 / (k + rank(d))) for each ranking
    """
    logger.info(f"RRF fusion of {len(results_lists)} result lists")
    
    # Map document ID to RRF score and document
    rrf_scores = defaultdict(float)
    doc_map = {}
    
    # Calculate RRF scores
    for result_list in results_lists:
        for rank, doc in enumerate(result_list, start=1):
            # Use URL as unique document identifier
            doc_id = doc.get("url", str(hash(str(doc))))
            
            # RRF formula: 1 / (k + rank)
            rrf_scores[doc_id] += 1.0 / (k + rank)
            
            # Store document (keep first occurrence)
            if doc_id not in doc_map:
                doc_map[doc_id] = doc.copy()
    
    # Create merged results with RRF scores
    merged = []
    for doc_id, score in rrf_scores.items():
        doc = doc_map[doc_id]
        doc["rrf_score"] = score
        merged.append(doc)
    
    # Sort by RRF score (higher is better)
    ranked = sorted(merged, key=lambda x: x["rrf_score"], reverse=True)
    
    logger.info(f"RRF merged into {len(ranked)} unique documents")
    return ranked

from sentence_transformers import CrossEncoder

# Global cross-encoder model (lazy loaded)
_cross_encoder_model = None

def get_cross_encoder():
    """Lazy load cross-encoder model."""
    global _cross_encoder_model
    if _cross_encoder_model is None:
        logger.info("Loading cross-encoder model...")
        _cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return _cross_encoder_model

def cross_encoder_rerank(
    query: str,
    docs: List[Dict[str, Any]],
    top_k: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Real cross-encoder reranking using sentence-transformers.
    """
    logger.info(f"Cross-encoder reranking for query: '{query}'")
    
    if not docs:
        return docs
    
    # Get cross-encoder model
    model = get_cross_encoder()
    
    # Prepare query-document pairs
    pairs = []
    for doc in docs:
        text = doc.get("snippet", "") + " " + doc.get("title", "")
        pairs.append([query, text])
    
    # Get relevance scores
    scores = model.predict(pairs)
    
    # Add scores to documents
    for i, doc in enumerate(docs):
        doc["cross_encoder_score"] = float(scores[i])
    
    # Sort by score
    ranked = sorted(docs, key=lambda x: x.get("cross_encoder_score", 0), reverse=True)
    
    # Return top-k if specified
    if top_k:
        ranked = ranked[:top_k]
    
    logger.info(f"Cross-encoder ranked {len(ranked)} documents")
    return ranked



def hybrid_search(
    query: str,
    docs: List[Dict[str, Any]],
    bm25_weight: float = 0.5,
    semantic_weight: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Hybrid search combining BM25 and semantic ranking.
    
    Args:
        query: Search query
        docs: Documents to rank
        bm25_weight: Weight for BM25 score
        semantic_weight: Weight for semantic score
        
    Returns:
        Hybrid ranked documents
    """
    logger.info("Hybrid search: BM25 + Semantic")
    
    # Get BM25 scores
    bm25_ranked = bm25_score(query, docs)
    
    # Get semantic scores
    semantic_ranked = cross_encoder_rerank(query, bm25_ranked)
    
    # Combine scores
    for doc in semantic_ranked:
        bm25_norm = doc.get("bm25_score", 0)
        semantic_norm = doc.get("cross_encoder_score", 0)
        
        doc["hybrid_score"] = (bm25_weight * bm25_norm) + (semantic_weight * semantic_norm)
    
    # Sort by hybrid score
    ranked = sorted(semantic_ranked, key=lambda x: x.get("hybrid_score", 0), reverse=True)
    logger.info(f"Hybrid ranked {len(ranked)} documents")
    
    return ranked


def vector_search(
    query: str,
    collection,
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """
    Search ChromaDB collection using semantic similarity.
    """
    logger.info(f"Vector search in ChromaDB for: '{query}'")
    
    try:
        results = collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        # Convert to standard format
        docs = []
        for i in range(len(results['ids'][0])):
            doc = {
                'id': results['ids'][0][i],
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i],
                'vector_score': 1.0 - results['distances'][0][i]  # Convert distance to similarity
            }
            docs.append(doc)
        
        logger.info(f"Vector search found {len(docs)} results")
        return docs
        
    except Exception as e:
        logger.error(f"Vector search error: {e}")
        return []


def hybrid_search_pipeline(
    query: str,
    docs: List[Dict[str, Any]],
    use_bm25: bool = True,
    use_cross_encoder: bool = True
) -> List[Dict[str, Any]]:
    """
    Complete hybrid search pipeline: BM25 → Cross-encoder reranking.
    """
    logger.info("Running hybrid search pipeline")
    
    if not docs:
        return docs
    
    # Stage 1: BM25 ranking
    if use_bm25:
        docs = bm25_score(query, docs)
    
    # Stage 2: Cross-encoder reranking on top results
    if use_cross_encoder:
        # Rerank top 20 results (or all if fewer)
        top_n = min(20, len(docs))
        top_docs = docs[:top_n]
        remaining_docs = docs[top_n:]
        
        reranked = cross_encoder_rerank(query, top_docs)
        docs = reranked + remaining_docs
    
    logger.info("Hybrid pipeline complete")
    return docs
