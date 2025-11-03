# services/agent-core/src/memory/embeddings.py

import logging
import numpy as np
import faiss
import os
import pickle
from pathlib import Path
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

# Initialize Gemini client
client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

# Configuration
EMBEDDING_MODEL = 'models/text-embedding-004'
DIMENSION = 768
FAISS_INDEX_PATH = Path(__file__).parent.parent.parent.parent / 'data' / 'faiss_index.bin'
FAISS_DOCS_PATH = Path(__file__).parent.parent.parent.parent / 'data' / 'faiss_docs.pkl'

# Ensure data directory exists
FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

# Initialize or load FAISS index
def _load_or_create_index():
    """Load existing FAISS index or create new one"""
    global index, documents
    
    if FAISS_INDEX_PATH.exists() and FAISS_DOCS_PATH.exists():
        try:
            logger.info(f"Loading existing FAISS index from {FAISS_INDEX_PATH}")
            index = faiss.read_index(str(FAISS_INDEX_PATH))
            
            with open(FAISS_DOCS_PATH, 'rb') as f:
                documents = pickle.load(f)
            
            logger.info(f"Loaded FAISS index with {index.ntotal} vectors and {len(documents)} documents")
            return index, documents
        except Exception as e:
            logger.warning(f"Failed to load FAISS index: {e}. Creating new one.")
    
    logger.info("Creating new FAISS index")
    index = faiss.IndexFlatL2(DIMENSION)
    documents = []
    return index, documents

# Load or create
index, documents = _load_or_create_index()

def save_index():
    """Save FAISS index and documents to disk"""
    try:
        faiss.write_index(index, str(FAISS_INDEX_PATH))
        
        with open(FAISS_DOCS_PATH, 'wb') as f:
            pickle.dump(documents, f)
        
        logger.info(f"Saved FAISS index with {index.ntotal} vectors to {FAISS_INDEX_PATH}")
    except Exception as e:
        logger.error(f"Failed to save FAISS index: {e}")

@custom_retry
def get_embedding(text: str) -> np.ndarray:
    """Get embedding with retry"""
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(task_type='SEMANTIC_SIMILARITY')
    )
    return np.array(response.embeddings[0].values, dtype="float32")

def add_embedding(text: str, metadata: dict):
    """Add embedding and auto-save"""
    try:
        emb = get_embedding(text)
        
        index.add(emb.reshape(1, -1))
        documents.append({"text": text, "meta": metadata})
        
        # Auto-save every 100 embeddings
        if len(documents) % 100 == 0:
            save_index()
            logger.info(f"Auto-saved FAISS index at {len(documents)} documents")
        
        logger.info(f"Added embedding: {text[:50]}... (total: {len(documents)})")
        
    except Exception as e:
        logger.error(f"Error adding embedding: {e}")
        raise

def query_embeddings(query: str, top_k: int = 5) -> list:
    """Query FAISS for similar documents"""
    try:
        if not documents:
            logger.warning("No documents in FAISS index")
            return []
        
        q_emb = get_embedding(query)
        
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
        logger.error(f"Error querying embeddings: {e}")
        return []

def clear_index():
    """Clear all embeddings"""
    global index, documents
    index = faiss.IndexFlatL2(DIMENSION)
    documents = []
    save_index()
    logger.info("Cleared FAISS index")

def get_index_stats():
    """Get FAISS index statistics"""
    return {
        "total_vectors": index.ntotal,
        "total_documents": len(documents),
        "dimension": DIMENSION,
        "index_path": str(FAISS_INDEX_PATH),
        "docs_path": str(FAISS_DOCS_PATH)
    }
