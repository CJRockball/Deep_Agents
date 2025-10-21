# run_agent_simple.py
"""
Simplified main entry point - runs a single query through the agent.
Usage: python run_agent_simple.py "Your query here"
"""

import asyncio
import logging
import sys
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


async def run_single_query(query: str, session_id: str = "simple_session"):
    """
    Run a single query through the agent.
    
    Args:
        query: User query
        session_id: Session identifier
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"QUERY: {query}")
    logger.info(f"{'='*70}\n")
    
    try:
        # Initialize databases
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
        
        # Create agent
        logger.info("Initializing ReAct agent...")
        agent = ReactAgent(db_manager=db_manager)
        logger.info("✓ Agent ready\n")
        
        # Run query
        result = await agent.run(query=query, session_id=session_id)
        
        # Display results
        logger.info(f"\n{'='*70}")
        logger.info("AGENT RESPONSE:")
        logger.info(f"{'='*70}")
        logger.info(f"\n{result['answer']}\n")
        logger.info(f"Session: {result['session_id']}")
        logger.info(f"Messages exchanged: {result['message_count']}")
        logger.info(f"Search used: {result['search_used']}")
        logger.info(f"{'='*70}\n")

        # Print clean final answer to console (separate from logs)
        print("\n" + "="*70)
        print("FINAL ANSWER:")
        print("="*70)
        print(f"\n{result['answer']}\n")
        print("="*70 + "\n")
                
        # Cleanup
        await db_manager.close_all()
        
        return result
        
    except Exception as e:
        logger.error(f"Error running agent: {e}", exc_info=True)
        raise


async def main():
    """Main execution function."""
    # Get query from command line argument
    if len(sys.argv) < 2:
        print("Usage: python run_agent_simple.py 'Your query here'")
        print("\nExample:")
        print("  python run_agent_simple.py 'What are AI agents?'")
        sys.exit(1)
    
    query = " ".join(sys.argv[1:])
    
    await run_single_query(query)


if __name__ == "__main__":
    asyncio.run(main())
