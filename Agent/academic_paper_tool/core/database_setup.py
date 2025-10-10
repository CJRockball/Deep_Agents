# database_setup.py
import sqlite3
import chromadb
from neo4j import GraphDatabase
from pathlib import Path
import os
from typing import Optional
import logging
from datetime import datetime

class DatabaseManager:
    """Unified database manager for SQLite, ChromaDB, and Neo4j"""
    
    def __init__(self, 
                 sqlite_path: str = "data/papers.db",
                 chroma_path: str = "data/chroma_db",
                 neo4j_uri: str = "bolt://localhost:7687",
                 neo4j_user: str = "neo4j",
                 neo4j_password: str = "password"):
        
        self.sqlite_path = sqlite_path
        self.chroma_path = chroma_path
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        
        # Ensure data directory exists
        Path("data").mkdir(exist_ok=True)
        
        # Initialize connections
        self.sqlite_conn = None
        self.chroma_client = None
        self.neo4j_driver = None
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def initialize_all_databases(self):
        """Initialize all three databases"""
        self.logger.info("Initializing all databases...")
        
        self.initialize_sqlite()
        self.initialize_chromadb()
        self.initialize_neo4j()
        
        self.logger.info("All databases initialized successfully")
    
    def initialize_sqlite(self):
        """Initialize SQLite database with paper metadata schema"""
        self.logger.info("Initializing SQLite database...")
        
        self.sqlite_conn = sqlite3.connect(self.sqlite_path)
        cursor = self.sqlite_conn.cursor()
        
        # Papers table - core metadata
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS papers (
            arxiv_id VARCHAR(50) PRIMARY KEY,
            title TEXT NOT NULL,
            abstract TEXT,
            published_date DATETIME,
            updated_date DATETIME,
            pdf_url TEXT,
            html_url TEXT,
            abstract_url TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            processing_status VARCHAR(20) DEFAULT 'pending'
        )
        ''')
        
        # Authors table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS authors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            normalized_name TEXT,
            UNIQUE(name)
        )
        ''')
        
        # Paper-Author relationships
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS paper_authors (
            paper_id VARCHAR(50),
            author_id INTEGER,
            position INTEGER,
            PRIMARY KEY (paper_id, author_id),
            FOREIGN KEY (paper_id) REFERENCES papers(arxiv_id),
            FOREIGN KEY (author_id) REFERENCES authors(id)
        )
        ''')
        
        # Categories table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_code VARCHAR(20) UNIQUE,
            category_name TEXT
        )
        ''')
        
        # Paper-Category relationships
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS paper_categories (
            paper_id VARCHAR(50),
            category_id INTEGER,
            PRIMARY KEY (paper_id, category_id),
            FOREIGN KEY (paper_id) REFERENCES papers(arxiv_id),
            FOREIGN KEY (category_id) REFERENCES categories(id)
        )
        ''')
        
        # Document structure table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS document_sections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id VARCHAR(50),
            section_type VARCHAR(50),
            section_title TEXT,
            section_order INTEGER,
            content_start INTEGER,
            content_end INTEGER,
            parent_section_id INTEGER,
            FOREIGN KEY (paper_id) REFERENCES papers(arxiv_id),
            FOREIGN KEY (parent_section_id) REFERENCES document_sections(id)
        )
        ''')
        
        # Mathematical content table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS mathematical_content (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id VARCHAR(50),
            section_id INTEGER,
            equation_number VARCHAR(20),
            latex_content TEXT,
            context_before TEXT,
            context_after TEXT,
            position INTEGER,
            FOREIGN KEY (paper_id) REFERENCES papers(arxiv_id),
            FOREIGN KEY (section_id) REFERENCES document_sections(id)
        )
        ''')
        
        # Citations table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS citations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id VARCHAR(50),
            citation_key TEXT,
            citation_text TEXT,
            authors TEXT,
            title TEXT,
            venue TEXT,
            year INTEGER,
            doi TEXT,
            cited_arxiv_id VARCHAR(50),
            FOREIGN KEY (paper_id) REFERENCES papers(arxiv_id)
        )
        ''')
        
        # Processing status table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS processing_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id VARCHAR(50),
            processing_stage VARCHAR(50),
            status VARCHAR(20),
            error_message TEXT,
            processing_time REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (paper_id) REFERENCES papers(arxiv_id)
        )
        ''')
        
        # Create indexes for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_papers_status ON papers(processing_status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_papers_published ON papers(published_date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_citations_paper ON citations(paper_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sections_paper ON document_sections(paper_id)')
        
        self.sqlite_conn.commit()
        self.logger.info("SQLite database initialized")
    
    def initialize_chromadb(self):
        """Initialize ChromaDB for semantic search"""
        self.logger.info("Initializing ChromaDB...")
        
        # Create persistent ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path=self.chroma_path)
        
        # Create collections with different embedding strategies
        try:
            # Full papers collection
            self.papers_collection = self.chroma_client.create_collection(
                name="papers_full",
                metadata={"description": "Full paper abstracts and content"}
            )
        except Exception:
            self.papers_collection = self.chroma_client.get_collection("papers_full")
        
        try:
            # Sections collection for granular search
            self.sections_collection = self.chroma_client.create_collection(
                name="paper_sections",
                metadata={"description": "Individual paper sections"}
            )
        except Exception:
            self.sections_collection = self.chroma_client.get_collection("paper_sections")
        
        try:
            # Mathematical content collection
            self.math_collection = self.chroma_client.create_collection(
                name="mathematical_content",
                metadata={"description": "Mathematical formulas and equations"}
            )
        except Exception:
            self.math_collection = self.chroma_client.get_collection("mathematical_content")
        
        self.logger.info("ChromaDB initialized")
    
    def initialize_neo4j(self):
        """Initialize Neo4j graph database with constraints and indexes"""
        self.logger.info("Initializing Neo4j database...")
        
        self.neo4j_driver = GraphDatabase.driver(
            self.neo4j_uri,
            auth=(self.neo4j_user, self.neo4j_password)
        )
        
        with self.neo4j_driver.session() as session:
            # Create constraints
            constraints = [
                "CREATE CONSTRAINT paper_id IF NOT EXISTS FOR (p:Paper) REQUIRE p.arxiv_id IS UNIQUE",
                "CREATE CONSTRAINT author_name IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE",
                "CREATE CONSTRAINT category_code IF NOT EXISTS FOR (c:Category) REQUIRE c.code IS UNIQUE",
                "CREATE CONSTRAINT concept_name IF NOT EXISTS FOR (co:Concept) REQUIRE co.name IS UNIQUE"
            ]
            
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    if "already exists" not in str(e):
                        self.logger.warning(f"Constraint creation warning: {e}")
            
            # Create indexes for performance
            indexes = [
                "CREATE INDEX paper_date IF NOT EXISTS FOR (p:Paper) ON (p.published_date)",
                "CREATE INDEX paper_title IF NOT EXISTS FOR (p:Paper) ON (p.title)",
                "CREATE INDEX author_normalized IF NOT EXISTS FOR (a:Author) ON (a.normalized_name)",
                "CREATE INDEX citation_year IF NOT EXISTS FOR (c:Citation) ON (c.year)"
            ]
            
            for index in indexes:
                try:
                    session.run(index)
                except Exception as e:
                    if "already exists" not in str(e):
                        self.logger.warning(f"Index creation warning: {e}")
            
            self.logger.info("Neo4j database initialized")
    
    def get_sqlite_connection(self):
        """Get SQLite connection"""
        if not self.sqlite_conn:
            self.initialize_sqlite()
        return self.sqlite_conn
    
    def get_chroma_client(self):
        """Get ChromaDB client"""
        if not self.chroma_client:
            self.initialize_chromadb()
        return self.chroma_client
    
    def get_neo4j_driver(self):
        """Get Neo4j driver"""
        if not self.neo4j_driver:
            self.initialize_neo4j()
        return self.neo4j_driver
    
    def health_check(self):
        """Check health of all databases"""
        results = {}
        
        # SQLite health check
        try:
            cursor = self.get_sqlite_connection().cursor()
            cursor.execute("SELECT COUNT(*) FROM papers")
            results['sqlite'] = {'status': 'healthy', 'paper_count': cursor.fetchone()[0]}
        except Exception as e:
            results['sqlite'] = {'status': 'error', 'error': str(e)}
        
        # ChromaDB health check
        try:
            client = self.get_chroma_client()
            collections = client.list_collections()
            results['chromadb'] = {'status': 'healthy', 'collections': len(collections)}
        except Exception as e:
            results['chromadb'] = {'status': 'error', 'error': str(e)}
        
        # Neo4j health check
        try:
            driver = self.get_neo4j_driver()
            with driver.session() as session:
                result = session.run("MATCH (n) RETURN COUNT(n) as count")
                count = result.single()['count']
                results['neo4j'] = {'status': 'healthy', 'node_count': count}
        except Exception as e:
            results['neo4j'] = {'status': 'error', 'error': str(e)}
        
        return results
    
    def close_connections(self):
        """Close all database connections"""
        if self.sqlite_conn:
            self.sqlite_conn.close()
        if self.neo4j_driver:
            self.neo4j_driver.close()
        self.logger.info("All database connections closed")

# Usage example
if __name__ == "__main__":
    db_manager = DatabaseManager()
    db_manager.initialize_all_databases()
    
    # Health check
    health = db_manager.health_check()
    print("Database Health Check:")
    for db, status in health.items():
        print(f"  {db}: {status}")
    
    db_manager.close_connections()
