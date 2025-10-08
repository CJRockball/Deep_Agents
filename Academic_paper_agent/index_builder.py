
# index_builder.py - Step 10: Index Building and Optimization

import sqlite3
import logging
from database_setup import DatabaseManager

class IndexBuilder:
    """
    Builds and optimizes indexes across databases for fast retrieval.
    This is Step 10 in the pipeline.
    """
    def __init__(self, db_manager=None):
        self.db = db_manager or DatabaseManager()
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def build_sqlite_indexes(self):
        conn = self.db.get_sqlite_connection()
        c = conn.cursor()

        # Indexes for fast lookups
        indexes = [
            ('idx_sections_paper', 'document_sections', 'paper_id'),
            ('idx_sections_type', 'document_sections', 'section_type'),
            ('idx_chunks_paper', 'content_chunks', 'paper_id'),
            ('idx_chunks_section', 'content_chunks', 'canonical_section'),
            ('idx_entities_paper', 'entities', 'paper_id'),
            ('idx_entities_type', 'entities', 'entity_type'),
            ('idx_citations_paper', 'citations', 'paper_id'),
            ('idx_intext_paper', 'in_text_citations', 'paper_id'),
            ('idx_validation_paper', 'quality_validation', 'paper_id')
        ]

        for name, table, col in indexes:
            try:
                c.execute(f"CREATE INDEX IF NOT EXISTS {name} ON {table}({col})")
                self.logger.info(f"Created index {name} on {table}({col})")
            except Exception as e:
                self.logger.warning(f"Failed to create index {name}: {e}")

        conn.commit()

    def build_fts(self):
        conn = self.db.get_sqlite_connection()
        c = conn.cursor()

        # Full-text search on chunk_text
        try:
            c.execute("DROP TABLE IF EXISTS fts_content_chunks")
            c.execute(
                "CREATE VIRTUAL TABLE fts_content_chunks USING fts5(" 
                "chunk_text, paper_id UNINDEXED, content='content_chunks', content_rowid='rowid'")
            
            c.execute(
                "INSERT INTO fts_content_chunks(rowid, chunk_text, paper_id) " 
                "SELECT rowid, chunk_text, paper_id FROM content_chunks"
            )
            self.logger.info("Created FTS table fts_content_chunks and populated data")
        except Exception as e:
            self.logger.warning(f"Failed to create FTS table: {e}")

        conn.commit()

    def optimize_sqlite(self):
        conn = self.db.get_sqlite_connection()
        c = conn.cursor()

        try:
            c.execute("PRAGMA optimize")
            self.logger.info("Ran PRAGMA optimize for SQLite")
        except Exception as e:
            self.logger.warning(f"Failed to optimize SQLite: {e}")

        conn.commit()

# Simple test
if __name__ == '__main__':
    builder = IndexBuilder(DatabaseManager())
    print("Building SQLite indexes...")
    builder.build_sqlite_indexes()
    print("Building full-text search index...")
    builder.build_fts()
    print("Optimizing SQLite database...")
    builder.optimize_sqlite()
    print("Index building and optimization complete!")


# test_index_builder.py
from database_setup import DatabaseManager

def main():
    dbm = DatabaseManager()
    builder = IndexBuilder(dbm)

    print("Creating indexes...")
    builder.build_sqlite_indexes()

    print("Creating FTS table...")
    builder.build_fts()

    print("Optimizing database...")
    builder.optimize_sqlite()

    # Verify indexes exist
    conn = dbm.get_sqlite_connection()
    cursor = conn.cursor()
    cursor.execute("PRAGMA index_list('content_chunks')")
    indexes = cursor.fetchall()
    print("Indexes on content_chunks:", [idx[1] for idx in indexes])

    # Test FTS search
    test_term = "deep learning"
    cursor.execute(
        "SELECT paper_id, snippet(fts_content_chunks, 0, '[', ']', '...', 10) "
        "FROM fts_content_chunks WHERE fts_content_chunks MATCH ? LIMIT 5",
        (test_term,)
    )
    results = cursor.fetchall()
    print(f"FTS search results for '{test_term}':", results)

if __name__ == "__main__":
    main()
