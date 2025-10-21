# Web Search Agent Project

A production-ready LangGraph ReAct agent with plug-and-play web search capabilities and professional multi-database architecture.

## Overview

This project provides a modular web search agent built with LangGraph that can perform intelligent web searches, store results across multiple databases, and deliver well-researched answers. The architecture is designed to be **extensible** and **plug-and-play**, making it easy to add new tools and agents.

## Features

- **LangGraph ReAct Agent**: Intelligent reasoning and action loop with tool integration
- **Web Search Tool**: Tavily API integration with content fetching and summarization
- **Multi-Database Support**: 
  - **PostgreSQL**: Structured metadata, search history, session tracking
  - **Redis**: High-speed caching with TTL
  - **ChromaDB**: Vector embeddings for semantic search
- **Advanced Retrieval**: BM25, RRF (Reciprocal Rank Fusion), and Cross-Encoder reranking
- **Modular Architecture**: Clean separation of concerns for easy extension
- **Production-Ready**: Docker orchestration, health checks, comprehensive logging

## Project Structure

```
web_search_agent_project/
├── agent/
│   ├── state.py              # Agent state definition
│   ├── react_agent.py        # LangGraph ReAct agent implementation
│   └── __init__.py
├── config/
│   ├── settings.py           # Pydantic configuration management
│   └── __init__.py
├── database/
│   ├── manager.py            # Centralized database manager (singleton)
│   ├── models.py             # SQLAlchemy models
│   └── __init__.py
├── tools/
│   ├── base_tool.py          # Base class for all tools
│   └── web_search/
│       ├── search_tool.py    # Main web search implementation
│       ├── langchain_tool.py # LangChain tool wrapper
│       ├── storage.py        # Database storage operations
│       ├── retrieval.py      # Ranking algorithms (BM25, RRF, Cross-Encoder)
│       ├── processors.py     # Content processing and summarization
│       └── __init__.py
├── tests/
│   ├── test_database.py      # Database integration tests
│   ├── conftest.py           # Pytest fixtures
│   └── __init__.py
├── run_agent.py              # CLI runner (interactive/query/demo modes)
├── docker-compose.yml        # PostgreSQL and Redis orchestration
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variables template
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.9+
- Docker and Docker Compose
- OpenAI API key
- Tavily API key (for web search)

### Installation

1. **Clone the repository**
   ```bash
   cd web_search_agent_project
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys:
   # - OPENAI_API_KEY
   # - TAVILY_API_KEY
   # - Database connection strings (optional, defaults provided)
   ```

5. **Start databases**
   ```bash
   docker-compose up -d
   ```

### Usage

The agent supports three execution modes:

#### Interactive Mode (Chat)
```bash
python run_agent.py
```
Type your questions and get intelligent responses with web search capabilities.

#### Single Query Mode
```bash
python run_agent.py query "What are the latest developments in LangGraph?"
```

#### Demo Mode
```bash
python run_agent.py demo
```
Runs sample queries to demonstrate capabilities.

## Architecture Highlights

### Database Coordination

The **DatabaseManager** (singleton pattern) coordinates all database connections:
- Connection pooling for efficiency
- Session-per-request pattern for safety
- Health monitoring for all databases
- Graceful shutdown and cleanup

### Tool Architecture

All tools inherit from **BaseTool** and receive the database manager via dependency injection:
```python
class WebSearchTool(BaseTool):
    def __init__(self, db_manager=None):
        super().__init__(db_manager)
        # Tool-specific initialization
```

### Retrieval Pipeline

The web search tool implements a multi-stage retrieval pipeline:
1. **Search**: Tavily API returns initial results
2. **Ranking**: BM25 keyword-based scoring
3. **Fusion**: RRF combines multiple result sources
4. **Reranking**: Cross-encoder for semantic relevance
5. **Storage**: Results cached in Redis, metadata in PostgreSQL, embeddings in ChromaDB

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# LLM Configuration
OPENAI_API_KEY=your-key-here
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.2

# Search Configuration
TAVILY_API_KEY=your-key-here
MAX_SEARCH_RESULTS=10
SEARCH_DEPTH=advanced

# Database Configuration
POSTGRES_URI=postgresql+asyncpg://agent:password@localhost:5432/agentdb
REDIS_URI=redis://localhost:6379/0
CHROMA_PATH=./data/chroma

# Processing Options
ENABLE_SUMMARIZATION=true
ENABLE_CONTENT_OFFLOAD=true
OFFLOAD_DIRECTORY=./offloaded_pages
```

### Custom Configuration

You can programmatically customize the agent:

```python
from web_search_tool import WebSearchConfig, set_config

config = WebSearchConfig(
    max_results=15,
    enable_summarization=True,
    search_depth="advanced"
)
set_config(config)
```

## Extending the Agent

### Adding New Tools

1. Create a new tool inheriting from `BaseTool`:
```python
# tools/my_tool/my_tool.py
from tools.base_tool import BaseTool

class MyTool(BaseTool):
    async def execute(self, *args, **kwargs):
        # Your tool logic here
        pass
```

2. Wrap it with `@tool` decorator for LangChain compatibility:
```python
from langchain_core.tools import tool

@tool
async def my_tool(query: str) -> str:
    """Tool description for the LLM."""
    tool_instance = MyTool(db_manager=get_database_manager())
    return await tool_instance.execute(query)
```

3. Add to agent:
```python
from langgraph.prebuilt import create_react_agent
agent = create_react_agent(model, tools=[web_search_tool, my_tool])
```

### Adding New Databases

Extend `DatabaseManager` in `database/manager.py`:
```python
def __init__(self, ..., new_db_uri: Optional[str] = None):
    # ... existing code ...
    if new_db_uri:
        self.new_db_client = initialize_new_db(new_db_uri)
```

## Testing

Run the test suite:
```bash
pytest tests/ -v
```

Run specific test:
```bash
pytest tests/test_database.py -v
```

## Database Management

### Check Database Health
```bash
docker-compose ps
```

### View Logs
```bash
docker-compose logs -f postgres
docker-compose logs -f redis
```

### Stop Databases
```bash
docker-compose down
```

### Reset Databases (WARNING: Deletes all data)
```bash
docker-compose down -v
docker-compose up -d
```

## Troubleshooting

### Import Errors
If you encounter import errors when running from subdirectories, ensure you're running from the project root:
```bash
cd web_search_agent_project
python run_agent.py
```

Or use the module flag:
```bash
python -m run_agent
```

### Database Connection Issues
1. Ensure Docker containers are running: `docker-compose ps`
2. Check database URIs in `.env`
3. Verify network connectivity: `docker-compose logs`

### API Key Errors
Ensure your `.env` file contains valid API keys:
- `OPENAI_API_KEY` - Get from https://platform.openai.com/api-keys
- `TAVILY_API_KEY` - Get from https://tavily.com

## Performance Optimization

### Caching Strategy
- Redis caches search results with 1-hour TTL
- ChromaDB stores embeddings for semantic search across sessions
- PostgreSQL indexes on `session_id` and `timestamp` for fast queries

### Connection Pooling
- PostgreSQL: 10 base connections, 20 overflow
- MongoDB: 10-50 connections (if enabled)
- Neo4j: 20-50 connections (if enabled)

## Roadmap

Future enhancements planned:
- [ ] Multi-query expansion for comprehensive searches
- [ ] Source quality scoring and verification
- [ ] Plan-and-Execute agent pattern for complex queries
- [ ] RAG integration for document-based Q&A
- [ ] FastAPI backend for web interface
- [ ] Additional tools (memory, document processing, etc.)

## Contributing

This is a personal project structure designed for extensibility. To add features:
1. Follow the existing modular structure
2. Ensure tools inherit from `BaseTool`
3. Add tests in `tests/`
4. Update documentation

## License

This project structure is provided as-is for educational and development purposes.

## Acknowledgments

Built with:
- [LangGraph](https://langchain-ai.github.io/langgraph/) - Agent orchestration framework
- [LangChain](https://python.langchain.com/) - LLM application framework
- [Tavily](https://tavily.com/) - Web search API
- [OpenAI](https://openai.com/) - Language models
- [PostgreSQL](https://www.postgresql.org/) - Relational database
- [Redis](https://redis.io/) - In-memory cache
- [ChromaDB](https://www.trychroma.com/) - Vector database

---

**Note**: This is a development framework for building production-grade LangGraph agents with professional database architecture. The plug-and-play design makes it easy to extend with additional tools and capabilities.
