# run_agent.py
"""
Main entry point to run LangGraph ReAct agent with web search tool and databases.
No API/web interface - direct agent execution.
"""

import asyncio
import logging
from pathlib import Path

from config.settings import get_settings
from database.manager import get_database_manager
from agent.react_agent import ReactAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def initialize_databases():
    """Initialize database connections and verify health."""
    logger.info("Initializing databases...")
    
    settings = get_settings()
    
    # Create data directories
    Path(settings.DATA_DIR).mkdir(exist_ok=True)
    Path(settings.CHROMA_PATH).mkdir(parents=True, exist_ok=True)
    Path(settings.LOG_DIR).mkdir(exist_ok=True)
    Path(settings.OFFLOAD_DIR).mkdir(parents=True, exist_ok=True)
    
    # Initialize database manager
    db_manager = get_database_manager(
        postgres_uri=settings.POSTGRES_URI,
        redis_uri=settings.REDIS_URI,
        chroma_path=settings.CHROMA_PATH
    )
    
    # Check database health
    health = await db_manager.health_check()
    logger.info(f"Database health status: {health}")
    
    if not all(health.values()):
        logger.warning("Some databases are not healthy. Agent may have limited functionality.")
    
    return db_manager


async def run_single_query(agent: ReactAgent, query: str, session_id: str = "cli_session"):
    """Run a single query through the agent."""
    logger.info(f"\n{'='*70}")
    logger.info(f"QUERY: {query}")
    logger.info(f"{'='*70}\n")
    
    try:
        result = await agent.run(query=query, session_id=session_id)
        
        logger.info(f"\n{'='*70}")
        logger.info("AGENT RESPONSE:")
        logger.info(f"{'='*70}")
        logger.info(f"\n{result['answer']}\n")
        logger.info(f"Session: {result['session_id']}")
        logger.info(f"Messages exchanged: {result['message_count']}")
        logger.info(f"Search used: {result['search_used']}")
        logger.info(f"{'='*70}\n")
        
        return result
        
    except Exception as e:
        logger.error(f"Error running agent: {e}", exc_info=True)
        raise


async def interactive_mode(agent: ReactAgent):
    """Run agent in interactive CLI mode."""
    logger.info("\n🤖 Interactive Agent Mode")
    logger.info("Type 'exit' or 'quit' to stop\n")
    
    session_id = "interactive_session"
    
    while True:
        try:
            # Get user input
            query = input("\nYou: ").strip()
            
            if not query:
                continue
                
            if query.lower() in ['exit', 'quit', 'q']:
                logger.info("Exiting interactive mode...")
                break
            
            # Run query
            await run_single_query(agent, query, session_id)
            
        except KeyboardInterrupt:
            logger.info("\nInterrupted by user. Exiting...")
            break
        except Exception as e:
            logger.error(f"Error in interactive mode: {e}")
            continue


async def demo_mode(agent: ReactAgent):
    """Run agent with demo queries."""
    logger.info("\n🎯 Demo Mode - Running sample queries\n")
    
    demo_queries = [
        "What are the latest developments in AI agents?",
        "Search for information about LangGraph and LangChain",
        "What is BM25 ranking algorithm?"
    ]
    
    for i, query in enumerate(demo_queries, 1):
        logger.info(f"\n--- Demo Query {i}/{len(demo_queries)} ---")
        await run_single_query(agent, query, session_id=f"demo_session_{i}")
        
        # Pause between queries
        if i < len(demo_queries):
            await asyncio.sleep(2)


async def main():
    """Main execution function."""
    logger.info("=" * 70)
    logger.info("WEB SEARCH AGENT WITH LANGGRAPH")
    logger.info("=" * 70)
    
    try:
        # Initialize databases
        db_manager = await initialize_databases()
        
        # Create ReAct agent
        logger.info("\nInitializing ReAct agent...")
        agent = ReactAgent(db_manager=db_manager)
        logger.info("✓ Agent ready\n")
        
        # Choose mode
        import sys
        
        if len(sys.argv) > 1:
            mode = sys.argv[1].lower()
            
            if mode == "demo":
                # Run demo queries
                await demo_mode(agent)
                
            elif mode == "query":
                # Run single query from command line
                if len(sys.argv) < 3:
                    logger.error("Usage: python run_agent.py query 'your question here'")
                    return
                query = " ".join(sys.argv[2:])
                await run_single_query(agent, query)
                
            elif mode == "interactive":
                # Interactive mode
                await interactive_mode(agent)
                
            else:
                logger.error(f"Unknown mode: {mode}")
                logger.info("Available modes: demo, query, interactive")
                return
        else:
            # Default: interactive mode
            await interactive_mode(agent)
        
        # Cleanup
        logger.info("\nCleaning up...")
        await db_manager.close_all()
        logger.info("✓ Databases closed")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    """
    Usage examples:
    
    1. Interactive mode (default):
       python run_agent.py
       or
       python run_agent.py interactive
    
    2. Single query:
       python run_agent.py query "What is machine learning?"
    
    3. Demo mode:
       python run_agent.py demo
    """
    asyncio.run(main())
