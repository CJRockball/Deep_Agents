# Neo4j client - to be implemented
# services/agent-core/src/db/neo4j_client.py

import logging
from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASS)
)
logger.info("Connected to Neo4j at %s", NEO4J_URI)

def get_neo4j_session():
    """Yield a Neo4j session."""
    session = driver.session()
    try:
        yield session
    finally:
        session.close()
        logger.info("Closed Neo4j session")
