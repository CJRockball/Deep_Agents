"""Configuration settings for the paper search tool."""

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class PaperSearchConfig:
    """Configuration class for paper search tool"""
    
    # Storage paths
    data_path: str = "data"
    database_path: str = "data/databases/papers.db"
    cache_path: str = "data/cache"
    
    # ArXiv API settings
    arxiv_rate_limit: float = 1.0
    arxiv_timeout: int = 30
    max_search_results: int = 50
    
    # Processing settings
    default_max_papers: int = 3
    require_html: bool = True
    enable_caching: bool = True
    
    # Pipeline settings
    enable_math_processing: bool = True
    enable_citation_parsing: bool = True
    enable_entity_extraction: bool = True
    
    @classmethod
    def from_env(cls) -> 'PaperSearchConfig':
        """Create config from environment variables"""
        return cls(
            data_path=os.getenv('PAPER_TOOL_DATA_PATH', cls.data_path),
            arxiv_rate_limit=float(os.getenv('ARXIV_RATE_LIMIT', cls.arxiv_rate_limit)),
            max_search_results=int(os.getenv('MAX_SEARCH_RESULTS', cls.max_search_results)),
            # Add more env variable mappings as needed
        )
    
    def validate(self) -> bool:
        """Validate configuration settings"""
        # Add validation logic
        return True

# Default configuration instance
DEFAULT_CONFIG = PaperSearchConfig()
