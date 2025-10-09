"""Search and selection wrapper functions for the paper search tool."""

import logging
from typing import List

# Import from relative core modules
from ..core.arxiv_search import arxiv_search, ArxivPaper
from ..core.paper_selection import (
    select_papers_for_analysis, 
    SelectionCriteria, 
    SelectionStrategy,
    get_selection_summary
)

# Setup logging
logger = logging.getLogger(__name__)


def search_and_select_papers(
    search_terms: str, 
    max_results: int = 20, 
    max_papers: int = 3,
    start: int = 0
) -> List[ArxivPaper]:
    """
    Search ArXiv and select the top HTML-available papers for processing.
    
    This function combines arxiv_search and paper_selection to provide
    a simple interface for the ReAct agent.
    
    Args:
        search_terms: Search query string (e.g., "quantum machine learning")
        max_results: Maximum search results to retrieve from ArXiv (default: 20)
        max_papers: Maximum papers to select for processing (default: 3)
        start: Starting index for search pagination (default: 0)
        
    Returns:
        List[ArxivPaper]: Selected papers with HTML URLs available
        
    Example:
        papers = search_and_select_papers("fintech blockchain", max_papers=3)
        # Returns up to 3 ArxivPaper objects with HTML content available
    """
    
    logger.info(f"Searching for papers: '{search_terms}' (max_results={max_results})")
    
    try:
        # Step 1: Search ArXiv with HTML link fetching
        search_results = arxiv_search(
            query=search_terms,
            max_results=max_results,
            start=start,
            fetch_html=True  # Important: enables HTML URL detection
        )
        
        logger.info(f"ArXiv search returned {len(search_results)} papers")
        
        if not search_results:
            logger.warning("No papers found for search terms")
            return []
        
        # Step 2: Apply selection criteria (HTML-only strategy)
        criteria = SelectionCriteria(
            max_papers=max_papers,
            strategy=SelectionStrategy.HTML_ONLY,  # Only papers with HTML URLs
            require_abstract=True,  # Ensure papers have abstracts
            min_authors=1  # At least one author
        )
        
        # Step 3: Select top papers based on criteria
        selected_papers = select_papers_for_analysis(search_results, criteria)
        
        logger.info(f"Selected {len(selected_papers)} papers with HTML content")
        
        # Log selection details for debugging
        for i, paper in enumerate(selected_papers, 1):
            logger.info(f"Selected paper {i}: {paper.id} - {paper.title[:50]}...")
            logger.info(f"  HTML URL: {paper.html_url}")
            logger.info(f"  Authors: {', '.join(paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}")
        
        return selected_papers
        
    except Exception as e:
        logger.error(f"Error in search_and_select_papers: {str(e)}")
        return []


def get_search_summary(papers: List[ArxivPaper]) -> dict:
    """
    Generate a summary of search and selection results.
    
    Args:
        papers: List of selected ArxivPaper objects
        
    Returns:
        dict: Summary statistics for the ReAct agent
    """
    
    if not papers:
        return {
            "success": False,
            "count": 0,
            "message": "No papers selected"
        }
    
    # Get detailed selection summary
    summary = get_selection_summary(papers)
    
    # Add success status and paper IDs for agent use
    return {
        "success": True,
        "count": len(papers),
        "paper_ids": [paper.id for paper in papers],
        "paper_titles": [paper.title[:100] + "..." if len(paper.title) > 100 else paper.title 
                        for paper in papers],
        "html_available": summary.get("html_available", 0),
        "categories": summary.get("categories", []),
        "message": f"Successfully selected {len(papers)} papers with HTML content"
    }

# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     papers = search_and_select_papers("langgraph")
#     print(f"\nFound {len(papers)} papers:")
#     for paper in papers:
#         print(f"- {paper.id}: {paper.title}")