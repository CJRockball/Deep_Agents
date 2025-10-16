# services/agent-core/src/memory/embeddings.py

import logging
import time
import numpy as np
import faiss
import openai
from sentence_transformers import SentenceTransformer
from langchain_openai import OpenAIEmbeddings
from google import genai
from google.genai import types
from google.api_core import retry, exceptions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure retry with exponential backoff for rate limits
custom_retry = retry.Retry(
    initial=1.0,              # Start with 1 second delay
    maximum=60.0,             # Max 60 seconds between retries
    multiplier=2.0,           # Double delay each retry
    timeout=300.0,            # Give up after 5 minutes total
    predicate=retry.if_exception_type(
        exceptions.ResourceExhausted,  # 429 rate limit errors
        exceptions.ServiceUnavailable,  # 503 temporary unavailable
        exceptions.DeadlineExceeded,   # Timeout errors
        exceptions.InternalServerError # 500 errors
    )
)

# Initialize Gemini client with retry config
client = genai.Client()

# FAISS index and in-memory store
EMBEDDING_MODEL = 'models/text-embedding-004'
DIMENSION = 768
index = faiss.IndexFlatL2(DIMENSION)
documents = []

@custom_retry
def get_embedding(text: str) -> np.ndarray:
    """
    Get embedding with automatic retry on rate limits.
    The @custom_retry decorator handles exponential backoff.
    """
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            task_type='SEMANTIC_SIMILARITY'
        )
    )
    return np.array(response.embeddings[0].values, dtype="float32")

def add_embedding(text: str, metadata: dict):
    """Compute embedding with automatic retry handling."""
    try:
        emb = get_embedding(text)  # Retry decorator applies here
        
        index.add(emb.reshape(1, -1))
        documents.append({"text": text, "meta": metadata})
        
        logger.info(f"Added embedding: {text[:50]}... (total: {len(documents)})")
        
    except Exception as e:
        logger.error(f"Error adding embedding: {e}")
        raise

def query_embeddings(query: str, top_k: int = 5) -> list:
    """Search with automatic retry handling."""
    try:
        if not documents:
            return []
        
        q_emb = get_embedding(query)  # Retry decorator applies here
        
        D, I = index.search(q_emb.reshape(1, -1), min(top_k, len(documents)))
        
        results = []
        for distance, idx in zip(D[0], I[0]):
            if idx < len(documents):
                doc = documents[idx].copy()
                doc["similarity_distance"] = float(distance)
                doc["similarity_score"] = 1.0 / (1.0 + distance)
                results.append(doc)
        
        logger.info(f"Retrieved {len(results)} embeddings")
        return results
        
    except Exception as e:
        logger.error(f"Error querying: {e}")
        return []