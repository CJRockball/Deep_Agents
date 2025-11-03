# Context filtering logic - to be implemented
# context_filter.py
import logging
import time
from memory.embeddings import query_embeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def filter_context(user_query, past_messages, top_k=20):
    """
    1. Load past_messages (list of strings with timestamp metadata).
    2. Query embeddings for relevance.
    3. Apply decay: older than 1 day get 0.5 score penalty.
    """
    # Prepare embedding store
    for msg in past_messages:
        # assume embeddings already added at save time

        pass

    # Retrieve best matches
    hits = query_embeddings(user_query, top_k=top_k)
    # Extract text
    context = [h["text"] for h in hits]
    logger.info("Filtered to %d context messages", len(context))
    return context
