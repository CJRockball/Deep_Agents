# IntelliFinQ: Financial Q&A Agent with Memory and Context

A production-ready **LangGraph ReAct agent** for financial research that combines **persistent conversation memory**, **semantic context filtering**, and **live web search** to provide intelligent, context-aware responses.

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-0.0.32-green.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-teal.svg)
![License](https://img.shields.io/badge/license-MIT-purple.svg)

---

## 🎯 Key Features

### 🧠 Multi-Database Memory System
- **PostgreSQL**: Stores structured conversation history with full session management
- **MongoDB**: Stores unstructured document data and search results
- **Neo4j**: Graph-based conversation relationships (optional)
- **FAISS**: Vector embeddings for semantic similarity search

### 🔍 Intelligent Context Filtering
- **Semantic Search**: Retrieves relevant past conversations using embedding similarity
- **Relevance Scoring**: Ranks historical context by semantic relevance to current query
- **Smart Context Window**: Only includes top-k most relevant past messages to avoid context overflow

### 🌐 Live Web Search with Content Offloading
- **Tavily Integration**: Real-time financial data and news retrieval
- **Content Summarization**: Uses GPT-4o-mini to generate concise webpage summaries
- **Context Offloading**: Saves full webpage content to disk, returns only summaries to agent
- **HTML-to-Markdown**: Converts web content for clean, readable storage

### 💬 FastAPI Chat Interface
- **Modern Web UI**: Jinja2-based chat interface with real-time responses
- **RESTful API**: Clean HTTP endpoints for integration with other applications
- **Session Management**: Persistent conversations across multiple sessions

---

## 🏗️ Architecture

```
┌─────────────────┐
│   FastAPI UI    │  ← Chat interface (HTML/JS)
└────────┬────────┘
         │
┌────────▼────────┐
│  ReAct Agent    │  ← LangGraph create_react_agent
│  (Financial)    │
└────────┬────────┘
         │
    ┌────┴────────────────────┐
    │                         │
┌───▼──────┐        ┌────────▼────────┐
│  Memory  │        │  Tavily Search  │
│ + Context│        │  + Summarizer   │
└───┬──────┘        └─────────────────┘
    │
┌───┴───────────────────────┐
│  PostgreSQL  │  MongoDB   │  Neo4j  │  FAISS
│  (Sessions)  │ (Docs)     │ (Graph) │ (Vectors)
└──────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- OpenAI API key
- Tavily API key (or Google Gemini API key)

### 1. Clone Repository

```bash
git clone https://github.com/CJRockball/Deep_Agents.git
cd Deep_Agents/agent_chat_web_context
```

### 2. Set Up Environment

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
cd services/agent-core
pip install -r src/requirements.txt
```

### 3. Configure API Keys

Edit `configs/default.env`:

```ini
# Database URIs
PG_URI=postgresql://agent_user:agent_pass_2025@localhost:5432/agent_platform
MONGO_URI=mongodb://admin:admin_pass_2025@localhost:27017/agent_memory?authSource=admin
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASS=neo4j_pass_2025

# API Keys
OPENAI_API_KEY=sk-your-openai-key-here
TAVILY_KEY=tvly_your-tavily-key-here

# Optional: Use Gemini for embeddings (free tier)
GOOGLE_API_KEY=your-google-api-key-here
```

### 4. Start Databases

```bash
# From project root
docker-compose -f infra/docker-compose.yaml up -d

# Verify all services are healthy
docker-compose -f infra/docker-compose.yaml ps
```

### 5. Run the Agent

```bash
# From project root
cd services/agent-core/src
python main.py
```

### 6. Access the Chat Interface

Open your browser to **http://localhost:8000**

---

## 📁 Project Structure

```
AgentPlatform/
├── infra/                          # Database infrastructure
│   ├── docker-compose.yaml         # Multi-database setup
│   ├── init-sql/                   # PostgreSQL schemas
│   ├── init-mongo/                 # MongoDB collections
│   └── init-neo4j/                 # Neo4j constraints
│
├── services/
│   └── agent-core/
│       ├── src/
│       │   ├── main.py             # FastAPI application
│       │   ├── boot.py             # Environment loader
│       │   ├── config.py           # Configuration management
│       │   │
│       │   ├── agents/
│       │   │   └── react_agent.py  # ReAct agent with context
│       │   │
│       │   ├── tools/
│       │   │   └── web_search.py   # Tavily search + summarization
│       │   │
│       │   ├── memory/
│       │   │   ├── persistence.py  # PostgreSQL ORM
│       │   │   └── embeddings.py   # FAISS + OpenAI/Gemini embeddings
│       │   │
│       │   ├── db/
│       │   │   ├── postgres_client.py
│       │   │   ├── mongo_client.py
│       │   │   └── neo4j_client.py
│       │   │
│       │   ├── utils/
│       │   │   ├── context_filter.py   # Semantic filtering
│       │   │   └── prompt_builder.py   # Dynamic prompt assembly
│       │   │
│       │   ├── templates/
│       │   │   └── chat.html           # Chat UI
│       │   │
│       │   └── static/
│       │       ├── css/style.css
│       │       └── js/chat.js
│       │
│       └── requirements.txt
│
├── configs/
│   ├── default.env                 # Local development
│   └── prod.env                    # Production settings
│
└── README.md
```

---

## 🔧 Configuration Options

### Embedding Models

**Option 1: OpenAI (Paid)**
```python
# embeddings.py
from openai import OpenAI
client = OpenAI(max_retries=5)
model = "text-embedding-3-small"  # $0.02 per 1M tokens
```

**Option 2: Google Gemini (Free Tier)**
```python
# embeddings.py
from google import genai
client = genai.Client(api_key=GOOGLE_API_KEY)
model = "models/text-embedding-004"  # Free: 1,500 requests/day
```

**Option 3: Local Models (Completely Free)**
```python
# embeddings.py
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')  # 384 dims, fast
```

### Rate Limiting

Built-in retry with exponential backoff using `google.api_core.retry`:

```python
from google.api_core import retry, exceptions

custom_retry = retry.Retry(
    initial=1.0,
    maximum=60.0,
    multiplier=2.0,
    predicate=retry.if_exception_type(exceptions.ResourceExhausted)
)

@custom_retry
def get_embedding(text):
    # Your embedding call
```

---

## 🧪 Testing Memory & Context

### Build Knowledge Base (5-7 queries)

1. "I'm interested in investing in tech stocks with moderate risk tolerance."
2. "What factors should I consider when evaluating tech companies?"
3. "Tell me about NVIDIA's recent performance."
4. "What are the risks facing semiconductor companies?"
5. "How do interest rates affect tech valuations?"

### Test Context Retrieval

6. "Based on what we discussed, would NVIDIA fit my portfolio?"  
   → *Should recall your risk profile + NVIDIA discussion*

7. "What risks did you mention earlier that apply to this company?"  
   → *Should connect semiconductor risks to NVIDIA*

### Test Temporal Updates

8. "Has there been recent news about NVIDIA this week?"  
   → *Should use web search + reference old context*

9. "Summarize everything we've discussed and give 3 recommendations."  
   → *Should synthesize full conversation history*

---

## 📊 Database Schema

### PostgreSQL (Structured Conversations)

```sql
-- Sessions table
CREATE TABLE sessions (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Messages table
CREATE TABLE messages (
    id SERIAL PRIMARY KEY,
    session_id INTEGER REFERENCES sessions(id),
    role VARCHAR(10) NOT NULL,  -- 'user' or 'agent'
    content TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT NOW()
);
```

### MongoDB (Unstructured Data)

```javascript
// Collections
db.documents          // Offloaded webpage content
db.search_results     // Tavily search results
db.embeddings         // Cached embedding metadata
```

### FAISS (Vector Index)

```python
# In-memory vector store
dimension = 1536  # OpenAI
index = faiss.IndexFlatL2(dimension)
documents = []  # Metadata store
```

---

## 🐳 Docker Services

The `docker-compose.yaml` includes:

- **PostgreSQL**: Relational data (port 5432)
- **MongoDB**: Document store (port 27017)
- **Neo4j**: Graph database (ports 7474, 7687)
- **pgAdmin**: PostgreSQL UI (port 5050)
- **Mongo Express**: MongoDB UI (port 8081)

Access management UIs:
- Neo4j Browser: http://localhost:7474
- pgAdmin: http://localhost:5050
- Mongo Express: http://localhost:8081

---

## 🔌 API Endpoints

### Chat

```bash
POST /api/chat
Content-Type: application/json

{
  "user_id": "alice123",
  "query": "What's the outlook for AI stocks in 2025?"
}

# Response
{
  "response": "Based on recent analysis... [citations]"
}
```

### Health Check

```bash
GET /api/health

# Response
{
  "status": "healthy",
  "databases": {
    "postgresql": "healthy",
    "mongodb": "healthy",
    "neo4j": "healthy"
  }
}
```

### User History

```bash
GET /api/user/{user_id}/history?limit=50

# Response
{
  "user_id": "alice123",
  "history": [
    {"role": "user", "content": "...", "timestamp": "..."},
    {"role": "agent", "content": "...", "timestamp": "..."}
  ]
}
```

---

## 🛠️ Development

### Run Locally (Without Docker)

```bash
# Start databases
docker-compose -f infra/docker-compose.yaml up -d

# Run agent locally
cd services/agent-core/src
python main.py
```

### Debug Mode

```python
# In react_agent.py
self.agent = create_react_agent(
    model=self.llm,
    tools=self.tools,
    debug=True  # Enable detailed logging
)
```

### View Logs

```bash
# Database logs
docker-compose -f infra/docker-compose.yaml logs -f postgres
docker-compose -f infra/docker-compose.yaml logs -f mongo
docker-compose -f infra/docker-compose.yaml logs -f neo4j

# Application logs (stdout)
```

---

## 📚 Key Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Agent Framework | LangGraph | ReAct agent orchestration |
| LLM | OpenAI GPT-4 | Primary reasoning model |
| Embeddings | OpenAI / Gemini | Semantic similarity |
| Web Search | Tavily API | Real-time data retrieval |
| Web Framework | FastAPI | HTTP API & chat interface |
| SQL Database | PostgreSQL 15 | Structured conversations |
| Document Store | MongoDB 7 | Unstructured content |
| Graph Database | Neo4j 5 | Conversation relationships |
| Vector Store | FAISS | Fast similarity search |

---

## 🎓 Learning Objectives

This project demonstrates:

1. **Conversation Persistence**: Save/load chat history across sessions
2. **Context Filtering**: Retrieve only relevant past messages via embeddings
3. **Relevance Scoring**: Rank historical context by semantic similarity
4. **Multi-Database Integration**: Coordinate PostgreSQL, MongoDB, Neo4j, FAISS
5. **Content Offloading**: Separate full content storage from agent context
6. **Rate Limiting**: Handle API constraints with exponential backoff
7. **Tool Integration**: Combine web search + summarization in ReAct loop
8. **Production Patterns**: Docker deployment, health checks, error handling

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Deep Agents from Scratch](https://github.com/langchain-ai/deep-agents-from-scratch)
- [Tavily Search API](https://tavily.com)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)

---

## 📧 Contact

**Author**: CJRockball  
**Repository**: [Deep_Agents/agent_chat_web_context](https://github.com/CJRockball/Deep_Agents/tree/main/agent_chat_web_context)  
**Issues**: [GitHub Issues](https://github.com/CJRockball/Deep_Agents/issues)

---

**Built with ❤️ for learning LangGraph agent architectures**
