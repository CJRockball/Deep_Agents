# PostgreSQL client - to be implemented
# services/agent-core/src/db/postgres_client.py

import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from config import PG_URI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create engine and session factory
engine = create_engine(PG_URI, echo=False)
SessionLocal = sessionmaker(bind=engine)

def get_pg_session():
    """Yield a new database session."""
    logger.info("Opening PostgreSQL session")
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
        logger.info("Closed PostgreSQL session")
