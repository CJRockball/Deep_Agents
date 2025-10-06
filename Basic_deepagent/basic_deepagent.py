#%%
from deepagents import create_deep_agent
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

import os
from dotenv import load_dotenv
from tavily import TavilyClient
import textwrap

# ==== LOAD ENVIRONMENT ====
load_dotenv()

# ==== LOAD MODEL CONFIGS ====
os.environ["GOOGLE_API_KEY"] = 'Put your API key'
os.environ['TAVILY_API_KEY'] = "Put your Tavily key"
RECURSION_LIMIT = os.getenv("RECURSION_LIMIT", 50)

rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.1,
    check_every_n_seconds=0.1,
    max_bucket_size=10,
)

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1,rate_limiter=rate_limiter)

#%% ===== WEB SEARCH TOOL =====
tavily_client = TavilyClient()

@tool
def search_web(query: str):
    """Simple web search using Tavily"""
    return tavily_client.search(query, max_results=3)

#%% ===== SUB-AGENTS CONFIGURATION WITH REFLECTION =====

# Built-in general-purpose sub-agent is always available for reflection
# Let's add a specialized reflection sub-agent

reflection_subagent = {
    'name': "reflector",
    'description': "Specialized agent for critically evaluating and reflecting on work quality, completeness, and accuracy",
    'prompt': """You are a critical reflection and evaluation specialist. Your job is to:
    
    1. ANALYZE: Carefully examine the provided work, findings, or responses
    2. EVALUATE: Assess quality, completeness, accuracy, and potential gaps
    3. CRITIQUE: Identify strengths and weaknesses objectively
    4. SUGGEST: Provide specific recommendations for improvement
    5. VALIDATE: Check if the work meets the original requirements
    
    Be thorough, honest, and constructive in your reflections. Focus on:
    - Factual accuracy and evidence quality
    - Logical reasoning and coherence  
    - Completeness of coverage
    - Clarity and presentation
    - Potential biases or gaps
    
    Always provide specific, actionable feedback.
    Reflect on your final answer no more than twice. If you are satisfied with the answer’s completeness and correctness, stop reflecting and finalize.
    """
}

# Research sub-agent that works with the reflector
researcher = {
    'name': "researcher",
    'description': "Conducts thorough research and gathers information for analysis",
    'prompt': """You are a thorough researcher. Your job is to:
    1. Search for comprehensive information using available tools
    2. Use the built-in planning tool (write_to_dos) to structure your research
    3. Save findings to files using the built-in file system
    4. Provide well-organized, factual summaries
    
    Always save your research to files and create a clear plan before starting."""
}

#%% ===== CREATE DEEP AGENT WITH REFLECTION CAPABILITIES =====

# Create agent with built-in reflection capabilities via sub-agents
agent = create_deep_agent(
    tools=[search_web],
    instructions='''You are a research agent with advanced reflection and self-evaluation capabilities.

    BUILT-IN REFLECTION WORKFLOW:
    1. PLAN: Use write_to_dos to create a structured plan
    2. RESEARCH: Delegate to the researcher sub-agent for thorough investigation  
    3. REFLECT: Use the reflector sub-agent to critically evaluate your work
    4. ITERATE: Based on reflection, improve or complete the work
    5. FINALIZE: Provide a comprehensive final answer
    
    REFLECTION PROCESS:
    - Always use the reflector sub-agent to evaluate major findings
    - Ask critical questions about completeness and accuracy
    - Consider alternative perspectives and potential gaps
    - Use the built-in file system to track iterations and improvements
    
    The built-in tools (planning, file system, sub-agents) enable deep reflection and iterative improvement.''',
    
    subagents=[researcher, reflection_subagent],
    model=llm,
).with_config({"recursion_limit": int(RECURSION_LIMIT)})

#%%

# Run a deep search
result = agent.invoke({
    'messages': [{'role':'user', 'content': """Research the current state of AI agents in finance, 
                  then reflect on the quality and completeness of your findings. 
                  Use your reflection to improve the final response."""}]
})

# %%
# Pretty print the answer
main_answer = result["messages"][-1].content
formatted_answer = textwrap.fill(main_answer, width=80)
print(formatted_answer)


#%%
# Pretty print the full conversation

def pretty_print_conversation(result, width: int = 90) -> None:
    """
    Formats and prints all messages from the agent conversation.
    
    Handles:
    - HumanMessage, AIMessage, ToolMessage objects
    - Tool calls made by the agent
    - Multi-line content with proper wrapping
    
    Args:
        result: The result dictionary from agent.invoke()
        width: Maximum line width for text wrapping (default: 90)
    
    Example:
        >>> result = agent.invoke({'messages': [...]})
        >>> pretty_print_conversation(result)
    """
    
    border = "=" * 60
    divider = "-" * 60
    
    print(f"\n{border}")
    print("🤖 AGENT CONVERSATION TRANSCRIPT")
    print(f"{border}\n")
    
    messages = result.get("messages", [])
    
    if not messages:
        print("❌ No messages found in result")
        return
    
    for idx, msg in enumerate(messages, start=1):
        # Determine message type and role
        msg_type = type(msg).__name__
        
        # Get role - different message types have different ways to access this
        if isinstance(msg, HumanMessage):
            role = "USER"
            icon = "👤"
        elif isinstance(msg, AIMessage):
            role = "ASSISTANT"
            icon = "🤖"
        elif isinstance(msg, ToolMessage):
            role = "TOOL"
            icon = "🔧"
        else:
            role = "SYSTEM"
            icon = "⚙️"
        
        # Get content - use attribute access for LangChain objects
        content = msg.content if hasattr(msg, 'content') else str(msg)
        
        # Print message header
        print(f"{idx:>2} | {icon} {role} [{msg_type}]")
        
        # Handle tool calls if present (AIMessage with tool_calls)
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            print(f"    📞 Tool Calls:")
            for tool_call in msg.tool_calls:
                tool_name = tool_call.get('name', 'unknown')
                tool_args = tool_call.get('args', {})
                print(f"       • {tool_name}({tool_args})")
            
            # If there's also content, print it
            if content and content.strip():
                wrapped = textwrap.fill(content.strip(), width=width)
                print(f"    💭 Reasoning:")
                for line in wrapped.split('\n'):
                    print(f"       {line}")
        
        # Handle tool message (result from tool execution)
        elif isinstance(msg, ToolMessage):
            tool_name = getattr(msg, 'name', 'unknown_tool')
            wrapped = textwrap.fill(content.strip(), width=width-8)
            print(f"    🔧 [{tool_name}] Result:")
            for line in wrapped.split('\n'):
                print(f"       {line}")
        
        # Handle regular content
        else:
            if content and content.strip():
                wrapped = textwrap.fill(content.strip(), width=width)
                for line in wrapped.split('\n'):
                    print(f"    {line}")
            else:
                print(f"    (empty message)")
        
        print()  # Blank line between messages
    
    print(divider)
    print(f"📊 Total messages: {len(messages)}")


pretty_print_conversation(result)

# %%
