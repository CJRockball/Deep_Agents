#!/usr/bin/env python3
"""
Load LONGMEMEVAL test data - Now much cleaner!
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Path setup
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
AGENT_SRC = PROJECT_ROOT / 'services' / 'agent-core' / 'src'
sys.path.insert(0, str(AGENT_SRC))

# Import from your enhanced persistence module
import boot
from memory.persistence import (
    SessionLocal,
    save_message,
    clear_test_users,
    get_user_stats,
    init_db
)
from memory.embeddings import add_embedding

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_single_test(test_instance, test_index, user_id_prefix='test_user'):
    """Load one test instance's conversation history"""
    
    user_id = f"{user_id_prefix}_{test_index:04d}"
    sessions = test_instance.get('haystack_sessions', [])
    dates = test_instance.get('haystack_dates', [])
    question = test_instance.get('question', 'N/A')
    
    if not dates:
        dates = [None] * len(sessions)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Loading test {test_index}: {question[:50]}...")
    logger.info(f"User ID: {user_id}")
    logger.info(f"Sessions: {len(sessions)}")
    logger.info(f"{'='*60}")
    
    total_turns = 0
    failed_dates = 0
    
    for sess_idx, (date_str, turns) in enumerate(zip(dates, sessions)):
        
        # Parse timestamp with multiple fallbacks
        timestamp = None
        if date_str:
            # Try ISO format
            try:
                timestamp = datetime.fromisoformat(date_str)
            except:
                pass
            
            # Try LONGMEMEVAL format: "2023/05/20 (Sat) 02:21"
            if not timestamp:
                try:
                    import re
                    clean_date = re.sub(r'\([^)]+\)', '', date_str).strip()
                    timestamp = datetime.strptime(clean_date, '%Y/%m/%d %H:%M')
                except:
                    pass
            
            # Try without time
            if not timestamp:
                try:
                    import re
                    just_date = re.search(r'\d{4}/\d{2}/\d{2}', date_str)
                    if just_date:
                        timestamp = datetime.strptime(just_date.group(), '%Y/%m/%d')
                except:
                    pass
        
        # Fallback to synthetic timestamp
        if not timestamp:
            if date_str and failed_dates == 0:
                logger.warning(f"Using synthetic timestamps (couldn't parse: '{date_str}')")
            failed_dates += 1
            base_date = datetime(2024, 1, 1)
            timestamp = base_date + timedelta(days=sess_idx * 7)
        
        # Process turns in this session
        for turn_idx, turn in enumerate(turns):
            role = turn.get('role')
            content = turn.get('content')
            
            if not content:
                continue
            
            turn_timestamp = timestamp + timedelta(minutes=turn_idx * 2)
            
            try:
                save_message(
                    user_id=user_id,
                    role=role,
                    content=content,
                    timestamp=turn_timestamp
                )
                
                add_embedding(content, {
                    'user_id': user_id,
                    'role': role,
                    'session_idx': sess_idx,
                    'turn_idx': turn_idx
                })
                
                total_turns += 1
                
            except Exception as e:
                logger.error(f"Error saving turn {sess_idx}-{turn_idx}: {e}")
                continue
        
        # Log progress every 10 sessions
        if (sess_idx + 1) % 10 == 0:
            logger.info(f"  Processed {sess_idx+1}/{len(sessions)} sessions...")
    
    if failed_dates > 0:
        logger.info(f"  Used synthetic timestamps for {failed_dates}/{len(sessions)} sessions")
    
    logger.info(f"✓ Loaded {total_turns} turns for {user_id}")
    return user_id

def load_test_data(
    json_file: str = 'data/longmemeval_s_cleaned.json',
    max_tests: int = None,
    clear_existing: bool = True,
    user_id_prefix: str = 'test_user'
):
    """Load all test data from LONGMEMEVAL into database"""
    
    # STEP 0: Initialize database tables if they don't exist
    logger.info("Initializing database tables...")
    init_db()
    
    # Load JSON
    logger.info(f"Loading test data from {json_file}")
    with open(json_file, 'r') as f:
        test_data = json.load(f)
    
    if max_tests:
        test_data = test_data[:max_tests]
        logger.info(f"Limited to first {max_tests} tests")
    
    logger.info(f"Total tests to load: {len(test_data)}")
    
    # Clear existing test data
    if clear_existing:
        clear_test_users(user_prefix=user_id_prefix)
    
    # Load each test
    tests_completed = 0
    for idx, test_instance in enumerate(test_data):
        try:
            user_id = load_single_test(test_instance, idx, user_id_prefix)
            tests_completed += 1
            logger.info(f"✓ Loaded test {idx+1}/{len(test_data)}")
            
        except Exception as e:
            logger.error(f"Failed to load test {idx}: {e}")
            continue
    
    logger.info(f"\n{'='*60}")
    logger.info(f"✓ Data loading complete!")
    logger.info(f"Loaded {tests_completed} test users")
    logger.info(f"{'='*60}\n")
    
    # IMPORTANT: Save FAISS index after loading all data
    try:
        from memory.embeddings import save_index, get_index_stats
        
        logger.info("Saving FAISS index to disk...")
        save_index()
        
        stats = get_index_stats()
        logger.info(f"✓ FAISS index saved:")
        logger.info(f"  - Vectors: {stats['total_vectors']}")
        logger.info(f"  - Documents: {stats['total_documents']}")
        logger.info(f"  - Index path: {stats['index_path']}")
        
    except Exception as e:
        logger.error(f"Failed to save FAISS index: {e}")
    
    return tests_completed


if __name__ == '__main__':
    # Default values
    json_file = 'data/longmemeval_s_cleaned.json'
    max_tests = 10
    
    # Simple command-line override
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    
    if len(sys.argv) > 2:
        max_tests = int(sys.argv[2])
    
    print(f"Loading: {json_file}")
    print(f"Max tests: {max_tests}")
    
    load_test_data(json_file, max_tests=max_tests)