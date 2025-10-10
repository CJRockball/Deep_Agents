"""
Fixed Minimal LangGraph ReAct Agent for Academic Research
Handles proper data flow between tools and provides detailed logging
"""

import os
import sys
from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv
import logging

# Add project root to sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai       import ChatGoogleGenerativeAI

# Configure logging to see academic_paper_tool logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Agent State Definition
class AgentState(TypedDict):
    """State for the academic research agent"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    subject: str
    query: str
    search_results: str
    processing_results: str
    final_answer: str
    current_step: str

# Initialize the model
os.environ["GOOGLE_API_KEY"] =  ''

rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.1,  # <-- Super slow! We can only make a request once every 10 seconds!!
    check_every_n_seconds=0.1,  # Wake up every 100 ms to check whether allowed to make a request,
    max_bucket_size=10,  # Controls the maximum burst size.
)

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1,)

# Try to import real academic paper tools
TOOLS_AVAILABLE = False
try:
    from academic_paper_tool.wrappers.search_and_select import search_and_select_papers as real_search
    from academic_paper_tool.wrappers.download_and_process import download_and_process_papers as real_process
    from academic_paper_tool.wrappers.query_tools import query_processed_data as real_query
    
    TOOLS_AVAILABLE = True
    print("✅ Academic paper tool successfully imported")
    
except ImportError as e:
    print(f"⚠️ Academic paper tool not found: {e}")
    TOOLS_AVAILABLE = False

# Global variable to store ArxivPaper objects between tool calls
_paper_objects = []

@tool
def search_and_select_papers(subject: str) -> str:
    """
    Search for academic papers on a given subject.
    Returns information about found papers.
    """
    global _paper_objects
    
    if TOOLS_AVAILABLE:
        print(f"🔍 REAL TOOL: Searching for papers about '{subject}'")
        print(f"📋 Parameters: max_papers=3, max_results=20")
        
        try:
            # Call the real search function with proper parameters
            papers = real_search(
                search_terms=subject,
                max_papers=3,  # Limit to 3 papers for processing
                max_results=20  # Search through 20 to find best 3
            )
            
            # Store the actual ArxivPaper objects for later use
            _paper_objects = papers
            
            # Create detailed summary for the agent
            total_found = len(papers)
            html_available = sum(1 for p in papers if hasattr(p, 'html_url') and p.html_url)
            
            summary = f"""📊 SEARCH RESULTS SUMMARY:
• Total papers found: {total_found}
• Papers with HTML available: {html_available}
• Ready for processing: {html_available}

📚 PAPERS SELECTED:"""
            
            for i, paper in enumerate(papers[:5], 1):  # Show first 5
                title = getattr(paper, 'title', 'Unknown Title')[:80]
                authors = getattr(paper, 'authors', ['Unknown'])
                has_html = "✅" if (hasattr(paper, 'html_url') and getattr(paper, 'html_url')) else "❌"
                summary += f"\n{i}. {has_html} {title}"
                if len(authors) > 0:
                    summary += f"\n   Authors: {', '.join(authors[:2])}{'...' if len(authors) > 2 else ''}"
            
            if len(papers) > 5:
                summary += f"\n... and {len(papers) - 5} more papers"
            
            print(summary)
            return summary
            
        except Exception as e:
            print(f"❌ Real search tool failed: {e}")
            import traceback
            traceback.print_exc()
            return f"Search failed: {str(e)}"
    else:
        # Mock implementation
        print(f"🔍 MOCK TOOL: Searching for papers about '{subject}'")
        _paper_objects = []  # Reset for mock
        result = f"📊 MOCK SEARCH RESULTS:\n• Total papers found: 3\n• Papers with HTML available: 3\n• Ready for processing: 3\n\n📚 PAPERS SELECTED:\n1. ✅ Introduction to {subject}\n2. ✅ Advanced {subject} Methods\n3. ✅ {subject} Applications"
        print(result)
        return result

@tool
def download_and_process_papers(search_results: str) -> str:
    """
    Download and process papers from search results.
    Runs the complete 9-stage pipeline.
    """
    global _paper_objects
    
    if TOOLS_AVAILABLE:
        print("⚙️ REAL TOOL: Processing papers through 9-stage pipeline")
        
        if not _paper_objects:
            return "❌ Error: No paper objects available. Run search_and_select_papers first."
        
        print(f"📊 Processing {len(_paper_objects)} papers...")
        
        # Show which papers are being processed
        for i, paper in enumerate(_paper_objects, 1):
            title = getattr(paper, 'title', 'Unknown Title')[:60]
            print(f"   {i}. {title}...")
        
        try:
            # Call the real processing function with ArxivPaper objects
            result = real_process(
                papers=_paper_objects,  # Pass the actual ArxivPaper objects
                enable_database_optimization=True,
                skip_failed_papers=True
            )
            
            # Extract and display key information from the result
            if hasattr(result, 'total_papers'):
                total = getattr(result, 'total_papers', 0)
                successful = getattr(result, 'successful_papers', 0)
                failed = getattr(result, 'failed_papers', 0)
                quality_score = getattr(result, 'average_quality_score', 0)
                
                summary = f"""✅ PROCESSING COMPLETE:
• Total papers: {total}
• Successfully processed: {successful}
• Failed: {failed}
• Average quality score: {quality_score:.2f}
• Database updated with content chunks, entities, and citations
• Papers ready for querying"""
                
                print(summary)
                return summary
            else:
                return f"✅ Processing completed successfully: {str(result)}"
                
        except Exception as e:
            print(f"❌ Real processing tool failed: {e}")
            import traceback
            traceback.print_exc()
            return f"Processing failed: {str(e)}"
    else:
        # Mock implementation
        print("🔄 MOCK TOOL: Simulating 9-stage pipeline processing")
        print("   - Stage 1: Content downloading...")
        print("   - Stage 2: Document structure parsing...")
        print("   - Stage 3: Mathematical content processing...")
        print("   - Stage 4: Citation reference parsing...")
        print("   - Stage 5: Content chunking...")
        print("   - Stage 6: Entity extraction...")
        print("   - Stage 7: Metadata enrichment...")
        print("   - Stage 8: Quality validation...")
        print("   - Stage 9: Index building...")
        
        result = """✅ MOCK PROCESSING COMPLETE:
• Total papers: 3
• Successfully processed: 3
• Failed: 0
• Average quality score: 0.85
• Database updated with content chunks, entities, and citations
• Papers ready for querying"""
        
        print(result)
        return result

@tool
def query_processed_data(query: str) -> str:
    """
    Query the processed academic papers to answer a specific question.
    """
    if TOOLS_AVAILABLE:
        print(f"🔍 REAL TOOL: Querying processed data for: '{query}'")
        print(f"📋 Parameters: query_type='semantic', max_results=5")
        
        try:
            result = real_query(
                query=query,
                query_type="semantic",
                max_results=5
            )
            
            # Format the result nicely
            if isinstance(result, str):
                formatted_result = f"""🎯 QUERY RESULTS:

{result}

📚 Sources: Based on processed academic papers in database
🔍 Query type: Semantic search
📊 Confidence: High (from processed paper content)"""
            else:
                formatted_result = f"🎯 Query completed: {str(result)}"
                
            print("📝 Query results generated successfully")
            return formatted_result
            
        except Exception as e:
            print(f"❌ Real query tool failed: {e}")
            import traceback
            traceback.print_exc()
            return f"Query failed: {str(e)}"
    else:
        # Mock implementation
        print(f"❓ MOCK TOOL: Generating answer for: '{query}'")
        
        result = f"""🎯 MOCK QUERY RESULTS:

Based on the processed papers, here's the answer to '{query}':

The research shows that [detailed answer based on paper analysis]. Key findings include:
1. Important result A from recent studies
2. Significant finding B with practical implications  
3. Notable conclusion C that advances the field

📚 Sources: Paper A (Section 2), Paper B (Figure 3), Paper C (Conclusion)
🔍 Query type: Semantic search  
📊 Confidence: High (from processed paper content)"""
        
        print("📝 Mock query results generated")
        return result

# Create tools list
tools = [search_and_select_papers, download_and_process_papers, query_processed_data]

# Bind tools to model
model_with_tools = model.bind_tools(tools)

# Node Functions
def call_model(state: AgentState):
    """Call the LLM to decide what to do next"""
    print(f"\n🤖 AGENT NODE: call_model (Step: {state.get('current_step', 'unknown')})")
    
    messages = state["messages"]
    response = model_with_tools.invoke(messages)
    
    print(f"🧠 AGENT RESPONSE: {response.content if response.content else 'Tool call requested'}")
    
    return {"messages": [response]}

def execute_tools(state: AgentState):
    """Execute the tools that the LLM wants to call"""
    print(f"\n🔧 AGENT NODE: execute_tools")
    
    tool_calls = state["messages"][-1].tool_calls
    tools_by_name = {tool.name: tool for tool in tools}
    
    results = []
    for tool_call in tool_calls:
        print(f"⚡ Executing tool: {tool_call['name']}")
        print(f"📝 Tool arguments: {tool_call['args']}")
        
        tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
        
        # Create tool message
        results.append(
            ToolMessage(
                content=str(tool_result),
                name=tool_call["name"],
                tool_call_id=tool_call["id"]
            )
        )
    
    return {"messages": results}

# Edge Functions
def should_continue(state: AgentState):
    """Decide whether to continue with tools or end"""
    messages = state["messages"]
    last_message = messages[-1]
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        print(f"➡️ ROUTING: Continuing to execute tools")
        return "continue"
    else:
        print(f"🏁 ROUTING: Ending - final answer ready")
        return "end"

# Create the graph
def create_research_agent():
    """Create the research agent graph"""
    print("🏗️ BUILDING AGENT: Creating LangGraph ReAct agent")
    
    # Build the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", execute_tools)
    
    # Add edges
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END
        }
    )
    workflow.add_edge("tools", "agent")
    
    # Compile the graph
    app = workflow.compile()
    
    print("✅ AGENT BUILT: Ready to process research queries")
    return app

# Main execution function
def run_research_agent(subject: str, query: str):
    """
    Run the research agent with a subject and query
    
    Args:
        subject: Topic to search for papers (e.g., "machine learning")
        query: Specific question to answer (e.g., "What are the latest trends?")
    """
    print("=" * 80)
    print("🚀 STARTING ACADEMIC RESEARCH AGENT")
    print("=" * 80)
    print(f"📋 Subject: {subject}")
    print(f"❓ Query: {query}")
    print(f"🛠️ Tools Available: {'Real academic tools' if TOOLS_AVAILABLE else 'Mock tools only'}")
    print("-" * 80)
    
    # Reset global state
    global _paper_objects
    _paper_objects = []
    
    # Create the agent
    agent = create_research_agent()
    
    # Create initial state
    initial_message = HumanMessage(
        content=f"""I need you to research academic papers about "{subject}" and answer this question: "{query}"

Please follow these steps exactly:
1. First, use search_and_select_papers to find relevant papers about "{subject}" (limit to 3 papers)
2. Then, use download_and_process_papers to process those papers through the 9-stage pipeline
3. Finally, use query_processed_data to answer the specific query: "{query}"

Execute each tool in sequence and provide me with a comprehensive answer at the end."""
    )
    
    initial_state = {
        "messages": [initial_message],
        "subject": subject,
        "query": query,
        "search_results": "",
        "processing_results": "",
        "final_answer": "",
        "current_step": "starting"
    }
    
    # Run the agent
    print(f"\n🎯 AGENT EXECUTION: Starting research process...")
    
    try:
        result = agent.invoke(initial_state)
        
        print("\n" + "=" * 80)
        print("🎉 RESEARCH COMPLETE!")
        print("=" * 80)
        
        # Extract final answer
        final_message = result["messages"][-1]
        if hasattr(final_message, 'content') and final_message.content:
            print(f"📋 FINAL ANSWER:\n{final_message.content}")
        
        return result
        
    except Exception as e:
        print(f"\n❌ ERROR: Agent execution failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Example usage
    print("Testing the Academic Research Agent...")
    
    # Test with a simple query
    subject = "transformer neural networks"
    query = "What are the main architectural improvements in recent transformer models?"
    
    result = run_research_agent(subject, query)