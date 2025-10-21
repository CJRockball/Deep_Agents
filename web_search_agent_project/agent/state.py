# agent/state.py
"""
State definitions for LangGraph agents
"""

from typing import Annotated, List, Dict, Any, Optional
from typing_extensions import TypedDict
from langgraph.graph import MessagesState
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """
    State for LangGraph agent with database-backed web search.
    
    Extends MessagesState to include conversation history and adds
    custom fields for search tracking.
    """
    # Core conversation
    messages: Annotated[List[BaseMessage], "Conversation messages"]
    
    # Session tracking
    session_id: str
    
    # Search tracking (optional)
    last_query: Optional[str]
    search_results: Optional[List[Dict[str, Any]]]
    
    # Agent control
    next_step: Optional[str]


class SearchState(TypedDict):
    """
    Simplified state for search-only agents.
    Use this for agents that only need search functionality.
    """
    # User query
    query: str
    
    # Session identifier
    session_id: str
    
    # Search results
    results: List[Dict[str, Any]]
    
    # Retrieval method used
    retrieval_method: str
