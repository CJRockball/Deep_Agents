# docker_manager.py - Updated version
import subprocess
import time
import os
from core.database_setup import DatabaseManager

def wait_for_neo4j(max_attempts=12, delay=10):
    """
    Wait for Neo4j to be fully ready with retry logic
    
    Args:
        max_attempts: Maximum number of connection attempts
        delay: Seconds to wait between attempts
    """
    from neo4j import GraphDatabase
    
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "arxiv_password")
    
    print(f"Waiting for Neo4j to be ready...")
    
    for attempt in range(1, max_attempts + 1):
        try:
            driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
            with driver.session() as session:
                # Simple test query
                session.run("RETURN 1")
            driver.close()
            print(f"✓ Neo4j ready after {attempt * delay} seconds")
            return True
            
        except Exception as e:
            print(f"  Attempt {attempt}/{max_attempts}: Neo4j not ready yet...")
            if attempt < max_attempts:
                time.sleep(delay)
            else:
                print(f"✗ Neo4j failed to start after {max_attempts * delay} seconds")
                print(f"  Last error: {e}")
                return False
    
    return False

def start_databases():
    """Start all database containers"""
    print("Starting database containers...")
    result = subprocess.run(["docker-compose", "up", "-d"], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error starting containers: {result.stderr}")
        return False
    
    # Wait specifically for Neo4j to be ready
    return wait_for_neo4j()

def stop_databases():
    """Stop all database containers"""
    print("Stopping database containers...")
    subprocess.run(["docker-compose", "down"])

def setup_databases():
    """Initialize database schemas"""
    print("Setting up database schemas...")
    
    # Load environment variables
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "arxiv_password")
    
    db_manager = DatabaseManager(
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password
    )
    
    db_manager.initialize_all_databases()
    
    # Health check
    health = db_manager.health_check()
    print("\nDatabase Health Check:")
    for db, status in health.items():
        print(f"  {db}: {status}")
    
    db_manager.close_connections()
    return all(status.get('status') == 'healthy' for status in health.values())

# Rest of the file remains the same...

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python docker_manager.py [start|stop|setup|restart]")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "start":
        start_databases()
    elif command == "stop":
        stop_databases()
    elif command == "setup":
        if start_databases():
            setup_databases()
    elif command == "restart":
        stop_databases()
        if start_databases():
            setup_databases()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
