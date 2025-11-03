# services/agent-core/src/utils/prompt_builder.py

import logging
from typing import List, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_react_prompt(query: str, context: List[str] = None) -> str:
    """
    Build an enriched prompt for the ReAct agent that includes relevant context.
    
    Args:
        query: The user's current question
        context: List of relevant previous conversation snippets
        
    Returns:
        Enriched prompt string with context and current query
    """
    logger.info(f"Building ReAct prompt for query: {query[:50]}...")
    
    prompt_parts = []
    
    # Add context if available
    if context and len(context) > 0:
        prompt_parts.append("📋 **Relevant Context from Previous Conversations:**")
        for i, ctx in enumerate(context, 1):
            prompt_parts.append(f"{i}. {ctx}")
        prompt_parts.append("")  # Empty line separator
    
    # Add the current query
    prompt_parts.append("🔍 **Current Question:**")
    prompt_parts.append(query)
    
    # Add instructions for using context
    if context:
        prompt_parts.append("")
        prompt_parts.append("Please consider the context above when answering, but focus primarily on the current question. Use web search to find the most current information available.")
    
    final_prompt = "\n".join(prompt_parts)
    logger.info(f"Built prompt with {len(context) if context else 0} context items")
    
    return final_prompt

# Legacy function - kept for backward compatibility
def build_prompt(context: List[str], search_results: List[dict], query: str) -> str:
    """
    Legacy prompt builder - kept for backward compatibility.
    Note: The new ReAct agent handles search internally, so this is mainly for reference.
    """
    context_block = "\n".join(f"- {c}" for c in context) if context else "None"
    search_block = "\n".join(f"{i+1}. {r['snippet']} ({r['url']})"
                             for i, r in enumerate(search_results)) if search_results else "None"
    
    prompt = f"""
Context from previous conversations:
{context_block}

Search Results:
{search_block}

User Query:
{query}

Please provide a comprehensive answer using the context and search results above.
"""
    
    logger.info("Built legacy prompt")
    return prompt
