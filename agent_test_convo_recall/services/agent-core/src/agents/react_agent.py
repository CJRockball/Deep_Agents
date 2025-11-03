# services/agent-core/src/agents/react_agent.py

import logging
from typing import List, Dict, Any
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai       import ChatGoogleGenerativeAI
from langchain_core.tools import tool

from tools.web_search import tavily_search_tool
from memory.persistence import save_message, load_messages
from memory.embeddings import add_embedding
from utils.context_filter import filter_context
from utils.prompt_builder import build_react_prompt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialReActAgent:
    """
    A financial Q&A agent using LangGraph's prebuilt create_react_agent.
    Integrates web search, memory persistence, and context filtering.
    """
    
    def __init__(self, model_name="gemini-2.0-flash", temperature=0.1):
        """Initialize the ReAct agent with LLM and tools."""
        logger.info("Initializing FinancialReActAgent")
        
        # Rate limiter
        rate_limiter = InMemoryRateLimiter(
        requests_per_second=0.1,  # <-- Super slow! We can only make a request once every 10 seconds!!
        check_every_n_seconds=0.1,  # Wake up every 100 ms to check whether allowed to make a request,
        max_bucket_size=10,  # Controls the maximum burst size.
        )
        # Initialize the language model
        # self.llm = ChatOpenAI(
        #     model=model_name,
        #     temperature=temperature
        # )
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature, rate_limiter=rate_limiter)
        
        # Define tools available to the agent
        self.tools = [tavily_search_tool]
        
        # Create the ReAct agent using LangGraph prebuilt
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=self._get_system_prompt(),
            debug=False  # Enable debug mode for development
        )
        
        logger.info("FinancialReActAgent initialized successfully")
    
    def _get_system_prompt(self) -> str:
        """Returns the system prompt for the financial agent."""
        return """You are IntelliFinQ, an expert financial research assistant.

Your role is to help users with financial questions by:
1. Using web search to find current, accurate financial information
2. Analyzing and synthesizing information from multiple sources
3. Providing clear, well-reasoned answers with proper citations
4. Considering relevant context from previous conversations

Guidelines:
- Always search for current information when dealing with market data, stock prices, or economic indicators
- Cite your sources with URLs when providing factual information  
- Be transparent about limitations and uncertainties
- Focus on providing actionable insights while noting associated risks
- Use previous conversation context when relevant to the current question

Remember: Financial markets change rapidly, so prioritize recent, authoritative sources."""

    def handle_query(self, user_id: str, query: str) -> str:
        """
        Process a user query through the complete ReAct workflow:
        1. Save user message and create embedding
        2. Load and filter relevant context from memory
        3. Build enriched prompt with context
        4. Execute ReAct agent (which handles tool calls internally)
        5. Save agent response and create embedding
        """
        try:
            logger.info(f"Processing query for user {user_id}: {query[:50]}...")
            
            # 1. Save user message
            save_message(user_id, "user", query)
            add_embedding(query, {"user_id": user_id, "role": "user"})
            
            # 2. Load and filter context from previous conversations
            past_messages = load_messages(user_id)
            relevant_context = filter_context(query, past_messages)
            
            # 3. Build the input with context
            enriched_query = build_react_prompt(query, relevant_context)
            
            # 4. Prepare input for the ReAct agent
            agent_input = {
                "messages": [
                    {"role": "user", "content": enriched_query}
                ]
            }
            
            # 5. Execute the ReAct agent
            logger.info("Invoking ReAct agent")
            result = self.agent.invoke(agent_input)
            
            # 6. Extract the final response from the agent output
            response = self._extract_response(result)
            
            # 7. Save agent response
            save_message(user_id, "agent", response)
            add_embedding(response, {"user_id": user_id, "role": "agent"})
            
            logger.info(f"Successfully processed query for user {user_id}")
            return response
            
        except Exception as e:
            logger.exception(f"Error handling query for user {user_id}")
            error_msg = f"I encountered an error while processing your request: {str(e)}"
            save_message(user_id, "agent", error_msg)
            return error_msg
    
    def _extract_response(self, agent_result: Dict[str, Any]) -> str:
        """
        Extract the final response from the ReAct agent output.
        The agent returns a dict with 'messages' key containing the conversation.
        """
        messages = agent_result.get("messages", [])
        if not messages:
            return "I apologize, but I couldn't generate a response."
        
        # Get the last AI message
        for message in reversed(messages):
            if hasattr(message, 'content') and message.content:
                # If it's an AIMessage with content
                if not getattr(message, 'tool_calls', None):
                    return message.content
        
        # Fallback - return content from the last message
        last_message = messages[-1]
        return getattr(last_message, 'content', str(last_message))

    def stream_query(self, user_id: str, query: str):
        """
        Stream the agent's response for real-time interaction.
        Useful for web interfaces that want to show partial responses.
        """
        try:
            logger.info(f"Streaming query for user {user_id}: {query[:50]}...")
            
            # Save user message and get context (same as handle_query)
            save_message(user_id, "user", query)
            add_embedding(query, {"user_id": user_id, "role": "user"})
            
            past_messages = load_messages(user_id)
            relevant_context = filter_context(query, past_messages)
            enriched_query = build_react_prompt(query, relevant_context)
            
            agent_input = {
                "messages": [
                    {"role": "user", "content": enriched_query}
                ]
            }
            
            # Stream the agent execution
            full_response = ""
            for chunk in self.agent.stream(agent_input):
                logger.debug(f"Agent chunk: {chunk}")
                # Extract content from chunk and yield
                if "messages" in chunk:
                    messages = chunk["messages"]
                    if messages:
                        last_msg = messages[-1]
                        if hasattr(last_msg, 'content') and last_msg.content:
                            chunk_content = last_msg.content
                            full_response += chunk_content
                            yield chunk_content
            
            # Save the complete response
            save_message(user_id, "agent", full_response)
            add_embedding(full_response, {"user_id": user_id, "role": "agent"})
            
        except Exception as e:
            logger.exception(f"Error streaming query for user {user_id}")
            error_msg = f"Error: {str(e)}"
            yield error_msg
