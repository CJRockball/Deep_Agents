#!/usr/bin/env python3
"""
Verify that test data was loaded correctly into both PostgreSQL and FAISS

Usage:
    python verify_data.py
    python verify_data.py --user test_user_0005
"""

import sys
import argparse
from pathlib import Path

# Add agent source to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
AGENT_SRC = PROJECT_ROOT / 'services' / 'agent-core' / 'src'
sys.path.insert(0, str(AGENT_SRC))

# Import your modules
import boot
from memory.embeddings import get_index_stats
from memory.persistence import get_user_stats

def verify_faiss():
    """Check FAISS index"""
    print("\n" + "="*60)
    print("FAISS INDEX CHECK")
    print("="*60)
    
    stats = get_index_stats()
    
    print(f"Total vectors: {stats['total_vectors']}")
    print(f"Total documents: {stats['total_documents']}")
    print(f"Dimension: {stats['dimension']}")
    print(f"Index path: {stats['index_path']}")
    print(f"Docs path: {stats['docs_path']}")
    
    # Check file exists
    import os
    if os.path.exists(stats['index_path']):
        size_mb = os.path.getsize(stats['index_path']) / (1024*1024)
        print(f"✓ Index file exists: {size_mb:.2f} MB")
    else:
        print("✗ Index file NOT found!")
        return False
    
    if stats['total_vectors'] == 0:
        print("✗ WARNING: FAISS index is empty!")
        return False
    
    print("✓ FAISS index looks good")
    return True

def verify_postgresql(user_id='test_user_0000'):
    """Check PostgreSQL data"""
    print("\n" + "="*60)
    print(f"POSTGRESQL CHECK: {user_id}")
    print("="*60)
    
    stats = get_user_stats(user_id)
    
    if not stats['session_exists']:
        print(f"✗ User {user_id} not found in database!")
        return False
    
    print(f"Session ID: {stats['session_id']}")
    print(f"Created at: {stats['created_at']}")
    print(f"Total messages: {stats['message_count']}")
    print(f"User messages: {stats['user_messages']}")
    print(f"Agent messages: {stats['agent_messages']}")
    
    if stats['message_count'] == 0:
        print(f"✗ WARNING: No messages found for {user_id}!")
        return False
    
    print(f"✓ PostgreSQL data looks good for {user_id}")
    return True

def verify_consistency():
    """Check if FAISS and PostgreSQL are roughly consistent"""
    print("\n" + "="*60)
    print("CONSISTENCY CHECK")
    print("="*60)
    
    faiss_stats = get_index_stats()
    pg_stats = get_user_stats('test_user_0000')
    
    faiss_count = faiss_stats['total_vectors']
    pg_count = pg_stats.get('message_count', 0)
    
    print(f"FAISS vectors: {faiss_count}")
    print(f"PostgreSQL messages (test_user_0000): {pg_count}")
    
    # They should be close (FAISS has all users, PG stats is just one user)
    if faiss_count > 0 and pg_count > 0:
        print("✓ Both databases have data")
        
        # Rough check: FAISS should have more vectors than one user's messages
        if faiss_count >= pg_count:
            print("✓ FAISS has at least as many vectors as one user's messages")
            return True
        else:
            print("✗ WARNING: FAISS has fewer vectors than expected")
            return False
    else:
        print("✗ One or both databases are empty")
        return False

def main():
    """Main verification"""
    parser = argparse.ArgumentParser(description='Verify test data was loaded')
    parser.add_argument('--user', default='test_user_0000', help='User ID to check')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("INTELLIFINQ TEST DATA VERIFICATION")
    print("="*60)
    
    # Run checks
    faiss_ok = verify_faiss()
    pg_ok = verify_postgresql(args.user)
    consistency_ok = verify_consistency()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if faiss_ok and pg_ok and consistency_ok:
        print("✓ ALL CHECKS PASSED")
        print("\nYour test data is loaded correctly!")
        print("You can now run: python run_benchmark.py --max-tests 2")
        return 0
    else:
        print("✗ SOME CHECKS FAILED")
        print("\nPlease re-run: python load_test_data.py --max-tests 2")
        return 1

if __name__ == '__main__':
    sys.exit(main())
