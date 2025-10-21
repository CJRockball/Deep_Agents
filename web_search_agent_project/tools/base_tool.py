# tools/base_tool.py
"""
Base tool class with database manager access
All tools inherit from this base class
"""

import logging
from typing import Optional, Any, Dict
from database.manager import DatabaseManager


class BaseTool:
    """
    Base class for all tools with database access.
    Provides common functionality for logging and DB manager injection.
    """
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """
        Initialize base tool with database manager.
        
        Args:
            db_manager: DatabaseManager instance (uses singleton if None)
        """
        self.db_manager = db_manager or DatabaseManager.get_instance()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"{self.__class__.__name__} initialized")

    async def execute(self, *args, **kwargs) -> Any:
        """
        Execute tool logic. Override in subclass.
        
        Returns:
            Tool execution result
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement execute()")

    def get_tool_metadata(self) -> Dict[str, Any]:
        """
        Get tool metadata for LangChain/LangGraph integration.
        Override in subclass to provide specific metadata.
        """
        return {
            "name": self.__class__.__name__,
            "description": self.__doc__ or "No description provided"
        }
