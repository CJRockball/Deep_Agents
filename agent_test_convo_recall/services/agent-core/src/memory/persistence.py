# services/agent-core/src/memory/persistence.py

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import func
from datetime import datetime
import logging
import os
from pathlib import Path

# Load environment variables if not already loaded
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent.parent.parent / 'configs' / 'default.env'
load_dotenv(dotenv_path=env_path)

logger = logging.getLogger(__name__)

# Get database URI from environment
PG_URI = os.getenv('PG_URI')
if not PG_URI:
    raise ValueError("PG_URI environment variable not set. Check configs/default.env")

# Database setup
Base = declarative_base()
engine = create_engine(PG_URI)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Models
class Session(Base):
    __tablename__ = "sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, nullable=False, index=True)
    created_at = Column(DateTime, default=func.now())
    
    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan")

class Message(Base):
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=func.now())
    
    session = relationship("Session", back_populates="messages")

def init_db():
    """Create all tables"""
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialized")

# Helper functions
def get_or_create_session(user_id: str, db_session=None):
    """Get or create session for user"""
    should_close = False
    if db_session is None:
        db_session = SessionLocal()
        should_close = True
    
    try:
        session = db_session.query(Session).filter(
            Session.user_id == user_id
        ).first()
        
        if not session:
            session = Session(
                user_id=user_id,
                created_at=datetime.now()
            )
            db_session.add(session)
            db_session.commit()
            db_session.refresh(session)
            logger.info(f"Created session for {user_id}: session_id={session.id}")
        
        return session
        
    finally:
        if should_close:
            db_session.close()

def save_message(
    user_id: str,
    role: str,
    content: str,
    timestamp: datetime = None,
    db_session=None
):
    """Save message to database"""
    should_close = False
    if db_session is None:
        db_session = SessionLocal()
        should_close = True
    
    try:
        session = get_or_create_session(user_id, db_session)
        
        if timestamp is None:
            timestamp = datetime.now()
        
        message = Message(
            session_id=session.id,
            role=role,
            content=content,
            timestamp=timestamp
        )
        
        db_session.add(message)
        db_session.commit()
        
        logger.debug(f"Saved {role} message for {user_id}")
        
    except Exception as e:
        db_session.rollback()
        logger.error(f"Error saving message: {e}")
        raise
    finally:
        if should_close:
            db_session.close()

def load_messages(user_id: str, limit: int = None, db_session=None):
    """Load messages for user"""
    should_close = False
    if db_session is None:
        db_session = SessionLocal()
        should_close = True
    
    try:
        session = db_session.query(Session).filter(
            Session.user_id == user_id
        ).first()
        
        if not session:
            return []
        
        query = db_session.query(Message).filter(
            Message.session_id == session.id
        ).order_by(Message.timestamp.asc())
        
        if limit:
            query = query.limit(limit)
        
        return query.all()
        
    finally:
        if should_close:
            db_session.close()

def clear_test_users(user_prefix: str = 'test_user', db_session=None):
    """Clear all test users"""
    should_close = False
    if db_session is None:
        db_session = SessionLocal()
        should_close = True
    
    try:
        test_sessions = db_session.query(Session).filter(
            Session.user_id.like(f'{user_prefix}%')
        ).all()
        
        count = len(test_sessions)
        
        for session in test_sessions:
            db_session.query(Message).filter(
                Message.session_id == session.id
            ).delete()
        
        db_session.query(Session).filter(
            Session.user_id.like(f'{user_prefix}%')
        ).delete()
        
        db_session.commit()
        
        logger.info(f"Cleared {count} test sessions")
        return count
        
    except Exception as e:
        db_session.rollback()
        logger.error(f"Error clearing test users: {e}")
        raise
    finally:
        if should_close:
            db_session.close()
            
            
def get_user_stats(user_id: str, db_session=None):
    """Get statistics for user"""
    should_close = False
    if db_session is None:
        db_session = SessionLocal()
        should_close = True
    
    try:
        session = db_session.query(Session).filter(
            Session.user_id == user_id
        ).first()
        
        if not session:
            return {
                'user_id': user_id,
                'session_exists': False,
                'message_count': 0
            }
        
        total_messages = db_session.query(Message).filter(
            Message.session_id == session.id
        ).count()
        
        user_messages = db_session.query(Message).filter(
            Message.session_id == session.id,
            Message.role == 'user'
        ).count()
        
        agent_messages = db_session.query(Message).filter(
            Message.session_id == session.id,
            Message.role == 'agent'
        ).count()
        
        # FIX: Use 'is not None' instead of truthy check
        created_at_str = None
        if session.created_at is not None:  # Changed from 'if session.created_at:'
            if isinstance(session.created_at, datetime):
                created_at_str = session.created_at.isoformat()
            else:
                created_at_str = str(session.created_at)
        
        return {
            'user_id': user_id,
            'session_id': session.id,
            'session_exists': True,
            'created_at': created_at_str,
            'message_count': total_messages,
            'user_messages': user_messages,
            'agent_messages': agent_messages
        }
        
    finally:
        if should_close:
            db_session.close()


def delete_user_data(user_id: str, db_session=None):
    """Delete all user data (GDPR compliance)"""
    should_close = False
    if db_session is None:
        db_session = SessionLocal()
        should_close = True
    
    try:
        sessions = db_session.query(Session).filter(
            Session.user_id == user_id
        ).all()
        
        count = len(sessions)
        
        for session in sessions:
            db_session.query(Message).filter(
                Message.session_id == session.id
            ).delete()
        
        db_session.query(Session).filter(
            Session.user_id == user_id
        ).delete()
        
        db_session.commit()
        
        logger.info(f"Deleted {count} sessions for {user_id}")
        return count
        
    except Exception as e:
        db_session.rollback()
        logger.error(f"Error deleting user data: {e}")
        raise
    finally:
        if should_close:
            db_session.close()
