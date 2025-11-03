# Configuration module - to be implemented
# config.py
import os

# PostgreSQL URI, e.g. postgresql://user:pass@postgres:5432/agent_platform
PG_URI = os.getenv("PG_URI")

# MongoDB URI, e.g. mongodb://mongo:27017
MONGO_URI = os.getenv("MONGO_URI")

# Neo4j Bolt URI, e.g. bolt://neo4j:7687
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASS = os.getenv("NEO4J_PASS")

# Tavily web search API key
TAVILY_KEY = os.getenv("TAVILY_KEY")
