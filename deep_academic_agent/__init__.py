"""
Academic Paper Search Tool

A comprehensive tool for searching, downloading, and processing academic papers
from ArXiv with support for HTML content analysis, citation parsing, and 
mathematical content extraction.

This tool is designed to be:
- Standalone: Works independently for development and testing
- Agent-ready: Provides clean wrapper functions for LangGraph agents
- Future-compatible: Easily integrates into multi-tool ecosystems
"""

# Import wrapper functions (agent-facing API)
from .wrappers.search_and_select import search_and_select_papers
from .wrappers.download_and_process import download_and_process_papers
#from .wrappers.query_tools import query_processed_data, get_paper_summary

# Import core classes for advanced usage
from .core.arxiv_search import ArxivPaper, arxiv_search
from .core.paper_selection import SelectionCriteria, SelectionStrategy
from .config.settings import PaperSearchConfig

# Tool metadata for future tool registry
__tool_name__ = "paper_search_tool"
__version__ = "1.0.0"
__description__ = "Academic paper search and analysis tool"
__author__ = "Your Name"
__requires__ = ["requests", "beautifulsoup4", "sqlite3"]

# Public API - what agents will import
__all__ = [
    # Agent wrapper functions
    'search_and_select_papers',
    'download_and_process_papers', 
    'query_processed_data',
    'get_paper_summary',
    
    # Core classes for advanced usage
    'ArxivPaper',
    'SelectionCriteria',
    'SelectionStrategy',
    'PaperSearchConfig',
    
    # Metadata
    '__tool_name__',
    '__version__',
    '__description__'
]
