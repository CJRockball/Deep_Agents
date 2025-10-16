# Database persistence layer - to be implemented
# persistence.py
import logging
from sqlalchemy import Column, Integer, String, Text, DateTime, func, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

from db.postgres_client import engine, SessionLocal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
Base = declarative_base()

class Session(Base):
    __tablename__ = "sessions"
    id = Column(Integer, primary_key=True)
    user_id = Column(String, nullable=False)
    created_at = Column(DateTime, default=func.now())

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    role = Column(String)  # "user" or "agent"
    content = Column(Text)
    timestamp = Column(DateTime, default=func.now())

    session = relationship("Session", back_populates="messages")

Session.messages = relationship("Message", back_populates="session")

def init_db():
    """Create tables."""
    Base.metadata.create_all(bind=engine)
    logger.info("Initialized PostgreSQL schema")

def save_message(user_id, role, content):
    """Save a message in the DB."""
    db = SessionLocal()
    # Get or create session
    sess = db.query(Session).filter_by(user_id=user_id).first()
    if not sess:
        sess = Session(user_id=user_id)
        db.add(sess)
        db.commit()
    msg = Message(session_id=sess.id, role=role, content=content)
    db.add(msg)
    db.commit()
    db.close()
    logger.info("Saved message: %s", content)

def load_messages(user_id):
    """Load all messages for a user session."""
    db = SessionLocal()
    sess = db.query(Session).filter_by(user_id=user_id).first()
    if not sess:
        return []
    messages = db.query(Message).filter_by(session_id=sess.id).order_by(Message.timestamp).all()
    db.close()
    logger.info("Loaded %d messages", len(messages))
    return messages
