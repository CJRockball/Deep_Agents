# agent/react_agent.py - FIXED: ToolNode deprecation
"""
ReAct agent implementation using LangGraph with web search tool
UPDATED: Uses create_react_agent instead of deprecated ToolNode
"""

import logging
from langchain_openai import ChatOpenAI
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai       import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage

from .state import AgentState
from tools.web_search.langchain_tool import get_web_search_tools
from database.manager import DatabaseManager, get_database_manager
from config.settings import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReactAgent:
    """
    ReAct (Reasoning + Acting) agent with web search capabilities.
    Uses LangGraph's create_react_agent for orchestration and database-backed tools.
    """
    
    def __init__(self, db_manager: DatabaseManager = None):
        """
        Initialize ReAct agent with database-backed tools.
        
        Args:
            db_manager: DatabaseManager instance (creates if None)
        """
        self.settings = get_settings()
        self.db_manager = db_manager or get_database_manager(
            postgres_uri=self.settings.POSTGRES_URI,
            redis_uri=self.settings.REDIS_URI,
            chroma_path=self.settings.CHROMA_PATH
        )
        
        # Initialize LLM
        # Rate limiter
        rate_limiter = InMemoryRateLimiter(
        requests_per_second=0.1,  # <-- Super slow! We can only make a request once every 10 seconds!!
        check_every_n_seconds=0.1,  # Wake up every 100 ms to check whether allowed to make a request,
        max_bucket_size=10,  # Controls the maximum burst size.
        )
        
        self.llm = ChatGoogleGenerativeAI(model=self.settings.LLM_MODEL, 
                               temperature=self.settings.LLM_TEMPERATURE, 
                               api_key=self.settings.GOOGLE_API_KEY,
                               rate_limiter=rate_limiter)
        
        # Get tools with database access
        self.tools = get_web_search_tools()
        
        # Create agent using LangGraph's built-in factory (replaces deprecated ToolNode)
        # Enhanced system prompt
        system_prompt = system_prompt = """You are an expert research assistant with web search capabilities.

REASONING PROCESS:
1. Analyze the user's question carefully
2. Determine what information you need
3. Use web_search to gather current, accurate information
4. Synthesize findings into a comprehensive answer
5. Always cite sources with URLs

CRITICAL RULES:
- ALWAYS use web_search for questions requiring external knowledge
- Search with 2-3 different query variations for thorough coverage
- After searching, analyze ALL results before answering
- Provide detailed answers with specific facts and citations
- If initial results are insufficient, search again with refined queries

ANSWER FORMAT:
- Start with a direct answer to the question
- Provide supporting details from search results
- Include relevant examples or explanations
- End with key takeaways
- Always cite sources [URL]"""


        self.graph = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=system_prompt
        )
        
        logger.info("ReAct agent initialized with create_react_agent")

    async def run(self, query: str, session_id: str = "default") -> dict:
        """
        Run the ReAct agent with a query.
        
        Args:
            query: User query
            session_id: Session identifier for database tracking
            
        Returns:
            Dictionary with agent response and metadata
        """
        logger.info(f"Running ReAct agent: query='{query}', session={session_id}")
        
        # Create initial state with messages
        initial_state = {
            "messages": [{"role": "user", "content": query}]
        }
        
        # Run graph
        final_state = await self.graph.ainvoke(initial_state)
        
        # Extract response
        messages = final_state.get("messages", [])
        
        # Get the final AI message
        final_message = None
        for msg in reversed(messages):
            if hasattr(msg, "content") and msg.content:
                final_message = msg
                break
        
        # Check if any tools were used
        tool_calls_used = any(
            hasattr(msg, "tool_calls") and msg.tool_calls 
            for msg in messages
        )
        response = {
            "answer": final_message.content if final_message else "No response generated",
            "session_id": session_id,
            "message_count": len(messages),
            "search_used": tool_calls_used
        }
        
        logger.info(f"ReAct agent completed: {response['message_count']} messages exchanged")
        
        return response

    async def stream(self, query: str, session_id: str = "default"):
        """
        Stream the ReAct agent execution.
        
        Args:
            query: User query
            session_id: Session identifier
            
        Yields:
            State updates as agent executes
        """
        logger.info(f"Streaming ReAct agent: query='{query}'")
        
        initial_state = {
            "messages": [{"role": "user", "content": query}]
        }
        
        async for state in self.graph.astream(initial_state):
            yield state


# Convenience function to create and run agent
async def run_react_agent(
    query: str,
    session_id: str = "default",
    db_manager: DatabaseManager = None
) -> dict:
    """
    Convenience function to create and run a ReAct agent.
    
    Args:
        query: User query
        session_id: Session identifier
        db_manager: Optional DatabaseManager instance
        
    Returns:
        Agent response dictionary
    """
    agent = ReactAgent(db_manager=db_manager)
    return await agent.run(query, session_id)
