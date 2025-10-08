#%%
# metadata_enricher.py - Step 8: Metadata Enrichment and Storage (FIXED)

import sqlite3
from datetime import datetime
import logging
from database_setup import DatabaseManager

class MetadataEnricher:
    def __init__(self, db_manager=None):
        self.db = db_manager or DatabaseManager()
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def enrich_paper(self, paper_id: str):
        conn = self.db.get_sqlite_connection()
        c = conn.cursor()

        stats = {}
        queries = {
            'section_count': "SELECT COUNT(*) FROM document_sections WHERE paper_id = ?",
            'equation_count': "SELECT COUNT(*) FROM mathematical_content_processed WHERE paper_id = ?",
            'reference_count': "SELECT COUNT(*) FROM citations WHERE paper_id = ?",
            'intext_citation_count': "SELECT COUNT(*) FROM in_text_citations WHERE paper_id = ?",
            'chunk_count': "SELECT COUNT(*) FROM content_chunks WHERE paper_id = ?",
            'entity_count': "SELECT COUNT(*) FROM entities WHERE paper_id = ?",
            'relationship_count': "SELECT COUNT(*) FROM entity_relationships WHERE paper_id = ?",
            'last_processed': "SELECT MAX(timestamp) FROM processing_log WHERE paper_id = ?"
        }

        for key, q in queries.items():
            try:
                c.execute(q, (paper_id,))
                row = c.fetchone()
                stats[key] = row[0] if row and row[0] is not None else 0
            except Exception as e:
                self.logger.warning(f"Error querying {key}: {e}")
                stats[key] = None

        # Create metadata table
        c.execute(
            "CREATE TABLE IF NOT EXISTS paper_metadata ("
            "paper_id TEXT PRIMARY KEY, "
            "section_count INTEGER, equation_count INTEGER, reference_count INTEGER, "
            "intext_citation_count INTEGER, chunk_count INTEGER, entity_count INTEGER, "
            "relationship_count INTEGER, last_processed TEXT)"
        )

        # Insert stats
        c.execute(
            "INSERT OR REPLACE INTO paper_metadata ("
            "paper_id, section_count, equation_count, reference_count, "
            "intext_citation_count, chunk_count, entity_count, relationship_count, last_processed) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (
                paper_id,
                stats['section_count'],
                stats['equation_count'],
                stats['reference_count'],
                stats['intext_citation_count'],
                stats['chunk_count'],
                stats['entity_count'],
                stats['relationship_count'],
                stats['last_processed'] or datetime.now().isoformat()
            )
        )
        conn.commit()
        self.logger.info(f"Enriched metadata for {paper_id}: {stats}")
        return stats

# Simple test - FIXED to look in the right tables
if __name__ == '__main__':
    db = DatabaseManager()
    enricher = MetadataEnricher(db)
    conn = db.get_sqlite_connection()
    c = conn.cursor()

    # Try to find a paper from content_downloads table (where actual data is)
    c.execute("SELECT DISTINCT arxiv_id FROM content_downloads WHERE success = 1 LIMIT 1")
    row = c.fetchone()

    if not row:
        # Try document_sections table
        c.execute("SELECT DISTINCT paper_id FROM document_sections LIMIT 1")
        row = c.fetchone()

    if not row:
        # Try content_chunks table
        c.execute("SELECT DISTINCT paper_id FROM content_chunks LIMIT 1")
        row = c.fetchone()

    if not row:
        print("No processed papers found in database")
        print("Run the previous pipeline steps first:")
        print("  python simple_test.py")
        print("  python test_math_processor.py")
        print("  python test_citation_parser.py")
        print("  python test_content_chunker.py")
        print("  python test_entity_extractor.py")
    else:
        paper_id = row[0]
        print(f"Enriching metadata for paper: {paper_id}")
        stats = enricher.enrich_paper(paper_id)
        print("\nMetadata Statistics:")
        print("="*50)
        for key, value in stats.items():
            print(f"  {key}: {value}")
        print("="*50)
        print("\nMetadata saved to paper_metadata table!")

# %%
