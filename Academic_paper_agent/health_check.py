# health_check.py
from database_setup import DatabaseManager
import json
from datetime import datetime

def run_health_check():
    """Run comprehensive health check on all databases"""
    db = DatabaseManager()
    
    print("=" * 50)
    print("Database Health Check")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 50)
    
    health = db.health_check()
    
    for db_name, status in health.items():
        print(f"\n{db_name.upper()}:")
        if status.get('status') == 'healthy':
            print(f"  ✓ Status: Healthy")
            if 'paper_count' in status:
                print(f"  Papers: {status['paper_count']}")
            if 'collections' in status:
                print(f"  Collections: {status['collections']}")
            if 'node_count' in status:
                print(f"  Nodes: {status['node_count']}")
        else:
            print(f"  ✗ Status: Error")
            print(f"  Error: {status.get('error', 'Unknown error')}")
    
    db.close_connections()
    
    # Return True if all healthy
    all_healthy = all(s.get('status') == 'healthy' for s in health.values())
    print("\n" + "=" * 50)
    print(f"Overall Status: {'✓ ALL HEALTHY' if all_healthy else '✗ ISSUES DETECTED'}")
    print("=" * 50)
    
    return all_healthy

if __name__ == "__main__":
    import sys
    
    healthy = run_health_check()
    sys.exit(0 if healthy else 1)
