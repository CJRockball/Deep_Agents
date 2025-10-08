#%%

# content_chunker.py - Step 6: Content Chunking with Citation-Aware Boundaries (FIXED)

import re
import hashlib
from typing import List, Dict, Optional, Tuple, Any, Set
from dataclasses import dataclass
from datetime import datetime
import logging
import sqlite3
from bs4 import BeautifulSoup

from database_setup import DatabaseManager

@dataclass
class ContentChunk:
    """Represents a semantically coherent content chunk"""
    id: str
    paper_id: str
    chunk_text: str
    chunk_type: str  # 'section', 'paragraph', 'figure', 'equation', 'mixed'
    section_id: Optional[str] = None
    section_title: Optional[str] = None
    canonical_section: Optional[str] = None
    chunk_order: int = 0
    char_count: int = 0
    word_count: int = 0
    sentence_count: int = 0
    has_citations: bool = False
    has_math: bool = False
    has_figures: bool = False
    citations_in_chunk: List[str] = None
    math_elements_in_chunk: List[str] = None
    figure_refs_in_chunk: List[str] = None
    context_before: Optional[str] = None
    context_after: Optional[str] = None
    metadata: Dict[str, Any] = None

@dataclass
class ChunkingResult:
    """Result of content chunking process"""
    paper_id: str
    total_chunks: int
    section_chunks: int
    paragraph_chunks: int
    mixed_chunks: int
    chunks_with_citations: int
    chunks_with_math: int
    avg_chunk_size: float
    processing_errors: List[str]
    quality_score: float

class ContentChunker:
    """
    Creates citation-aware content chunks for RAG embedding.
    This is Step 6 in the academic paper processing pipeline.
    """

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db_manager = db_manager or DatabaseManager()
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Chunking configuration
        self.config = {
            'min_chunk_size': 100,      # Minimum characters per chunk
            'max_chunk_size': 1500,     # Maximum characters per chunk
            'target_chunk_size': 800,   # Target characters per chunk
            'sentence_overlap': 1,      # Sentences to overlap between chunks
            'preserve_equations': True, # Keep equations as atomic units
            'preserve_citations': True, # Keep citation context together
            'section_aware': True,      # Respect section boundaries
        }

    def chunk_paper_content(self, paper_id: str) -> ChunkingResult:
        """Main entry point for content chunking"""
        self.logger.info(f"Chunking content for paper {paper_id}")

        try:
            # Create chunks table first
            self._create_chunks_table()

            # Load structured data from previous pipeline steps
            html_content = self._load_html_content(paper_id)
            sections = self._load_document_sections(paper_id)
            math_elements = self._load_math_elements(paper_id)
            citations = self._load_citations(paper_id)

            if not html_content:
                return self._create_empty_result(paper_id, "No HTML content found")

            # Create section-aware chunks
            chunks = []

            # Process each section separately
            for section in sections:
                section_chunks = self._chunk_section(
                    paper_id, section, html_content, math_elements, citations
                )
                chunks.extend(section_chunks)

            # Calculate statistics
            result = self._calculate_chunking_stats(paper_id, chunks)

            # Save chunks to database
            if chunks:
                self._save_chunks_to_database(chunks)

            # Update processing status
            self._update_processing_status(paper_id, 'content_chunking', len(chunks) > 0)

            self.logger.info(f"Content chunking completed for {paper_id}: "
                           f"{result.total_chunks} chunks created, "
                           f"average size: {result.avg_chunk_size:.0f} chars")

            return result

        except Exception as e:
            self.logger.error(f"Error in content chunking for {paper_id}: {e}")
            return self._create_empty_result(paper_id, f"Processing failed: {str(e)}")

    def _create_chunks_table(self):
        """Create content chunks table"""
        try:
            conn = self.db_manager.get_sqlite_connection()
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS content_chunks (
                    id TEXT PRIMARY KEY,
                    paper_id VARCHAR(50),
                    chunk_text TEXT NOT NULL,
                    chunk_type VARCHAR(20),
                    section_id TEXT,
                    section_title TEXT,
                    canonical_section VARCHAR(50),
                    chunk_order INTEGER,
                    char_count INTEGER,
                    word_count INTEGER,
                    sentence_count INTEGER,
                    has_citations BOOLEAN,
                    has_math BOOLEAN,
                    has_figures BOOLEAN,
                    citations_in_chunk TEXT,
                    math_elements_in_chunk TEXT,
                    figure_refs_in_chunk TEXT,
                    context_before TEXT,
                    context_after TEXT,
                    metadata_json TEXT,
                    created_at TEXT,
                    FOREIGN KEY (paper_id) REFERENCES papers(arxiv_id)
                )
            """)
            conn.commit()
        except Exception as e:
            self.logger.error(f"Error creating chunks table: {e}")

    def _load_html_content(self, paper_id: str) -> Optional[str]:
        """Load HTML content for the paper"""

        conn = self.db_manager.get_sqlite_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT file_path FROM content_downloads 
            WHERE arxiv_id = ? AND content_type = 'html' AND success = 1
        """, (paper_id,))

        row = cursor.fetchone()
        if not row:
            return None

        try:
            with open(row[0], 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            self.logger.error(f"Error loading HTML file: {e}")
            return None

    def _load_document_sections(self, paper_id: str) -> List[Dict]:
        """Load document sections from structure parser"""

        conn = self.db_manager.get_sqlite_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT section_title, section_type, section_order, content_start, content_end
            FROM document_sections 
            WHERE paper_id = ? 
            ORDER BY section_order
        """, (paper_id,))

        sections = []
        for row in cursor.fetchall():
            sections.append({
                'title': row[0],
                'type': row[1],
                'order': row[2],
                'start': row[3] or 0,
                'end': row[4] or 0
            })

        return sections

    def _load_math_elements(self, paper_id: str) -> List[Dict]:
        """Load mathematical elements - FIXED to use correct table and columns"""

        conn = self.db_manager.get_sqlite_connection()
        cursor = conn.cursor()

        # First try mathematical_content_processed table
        try:
            cursor.execute("""
                SELECT id, cleaned_latex, original_element_id, math_type, complexity_score
                FROM mathematical_content_processed 
                WHERE paper_id = ?
                ORDER BY id
            """, (paper_id,))

            math_elements = []
            for row in cursor.fetchall():
                math_elements.append({
                    'id': row[0],
                    'latex': row[1],
                    'position': row[2] or 0,  # Use original_element_id as position
                    'type': row[3],
                    'complexity': row[4] or 0
                })

            return math_elements

        except Exception as e:
            # Fallback to original mathematical_content table
            self.logger.warning(f"Using fallback math table: {e}")
            try:
                cursor.execute("""
                    SELECT id, latex_content, 0 as position, 'equation' as type, 0 as complexity
                    FROM mathematical_content 
                    WHERE paper_id = ?
                    ORDER BY id
                """, (paper_id,))

                math_elements = []
                for row in cursor.fetchall():
                    math_elements.append({
                        'id': row[0],
                        'latex': row[1],
                        'position': row[2],
                        'type': row[3],
                        'complexity': row[4]
                    })

                return math_elements

            except Exception as e2:
                self.logger.error(f"Error loading math elements: {e2}")
                return []

    def _load_citations(self, paper_id: str) -> List[Dict]:
        """Load citation information"""

        conn = self.db_manager.get_sqlite_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT id, citation_text, citation_type, position_in_text, paragraph_context
                FROM in_text_citations 
                WHERE paper_id = ?
                ORDER BY position_in_text
            """, (paper_id,))

            citations = []
            for row in cursor.fetchall():
                citations.append({
                    'id': row[0],
                    'text': row[1],
                    'type': row[2],
                    'position': row[3] or 0,
                    'context': row[4]
                })

            return citations

        except Exception as e:
            self.logger.warning(f"Error loading citations: {e}")
            return []

    def _chunk_section(self, paper_id: str, section: Dict, html_content: str, 
                      math_elements: List[Dict], citations: List[Dict]) -> List[ContentChunk]:
        """Create chunks for a specific section"""

        chunks = []

        # Extract section content from HTML
        section_content = self._extract_section_content_from_html(
            html_content, section['title']
        )

        if not section_content or len(section_content.strip()) < self.config['min_chunk_size']:
            return chunks

        # Split into sentences for intelligent chunking
        sentences = self._split_into_sentences(section_content)

        if not sentences:
            return chunks

        # Group sentences into chunks respecting boundaries
        sentence_groups = self._group_sentences_into_chunks(sentences)

        # Create chunks with metadata
        for i, sentence_group in enumerate(sentence_groups):
            chunk_text = ' '.join(sentence_group)

            # Skip if too small
            if len(chunk_text) < self.config['min_chunk_size']:
                continue

            # Find citations and math in this chunk
            chunk_citations = self._find_citations_in_text(chunk_text, citations)
            chunk_math = self._find_math_in_text(chunk_text, math_elements)

            # Generate chunk ID
            chunk_id = self._generate_chunk_id(paper_id, section['type'], section['order'], i)

            # Create context (sentences before and after)
            context_before = self._get_context_before(sentences, sentence_group)
            context_after = self._get_context_after(sentences, sentence_group)

            chunk = ContentChunk(
                id=chunk_id,
                paper_id=paper_id,
                chunk_text=chunk_text,
                chunk_type='section',
                section_id=f"{paper_id}_{section['type']}_{section['order']}",
                section_title=section['title'],
                canonical_section=section['type'],
                chunk_order=len(chunks),
                char_count=len(chunk_text),
                word_count=len(chunk_text.split()),
                sentence_count=len(sentence_group),
                has_citations=len(chunk_citations) > 0,
                has_math=len(chunk_math) > 0,
                has_figures=self._has_figure_references(chunk_text),
                citations_in_chunk=[c['id'] for c in chunk_citations],
                math_elements_in_chunk=[str(m['id']) for m in chunk_math],
                figure_refs_in_chunk=self._extract_figure_refs(chunk_text),
                context_before=context_before,
                context_after=context_after,
                metadata={
                    'section_canonical': section['type'],
                    'section_order': section['order'],
                    'math_complexity_avg': sum(m['complexity'] or 0 for m in chunk_math) / len(chunk_math) if chunk_math else 0,
                    'citation_types': list(set(c['type'] for c in chunk_citations))
                }
            )

            chunks.append(chunk)

        return chunks

    def _extract_section_content_from_html(self, html_content: str, section_title: str) -> str:
        """Extract text content for a specific section from HTML"""

        soup = BeautifulSoup(html_content, 'html.parser')

        # Find section heading
        section_heading = None
        for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            if section_title and section_title.lower() in heading.get_text().lower():
                section_heading = heading
                break

        if not section_heading:
            # If no specific section found, get some content from body
            body = soup.find('body')
            if body:
                return body.get_text()[:2000]  # Get first 2000 chars as fallback
            return ""

        # Collect content until next heading of same or higher level
        content_elements = []
        current_level = int(section_heading.name[1])  # Extract level from h1, h2, etc.

        for sibling in section_heading.find_next_siblings():
            # Stop at next section of same or higher level
            if (sibling.name and sibling.name.startswith('h') and 
                len(sibling.name) == 2 and sibling.name[1].isdigit()):
                sibling_level = int(sibling.name[1])
                if sibling_level <= current_level:
                    break

            # Collect text content
            if hasattr(sibling, 'get_text'):
                text = sibling.get_text().strip()
                if text:
                    content_elements.append(text)

        return ' '.join(content_elements)

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using multiple heuristics"""

        # Basic sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            # Skip very short sentences (likely abbreviations or noise)
            if len(sentence) > 10 and not sentence.endswith(('et al.', 'Fig.', 'Tab.', 'Eq.')):
                cleaned_sentences.append(sentence)

        return cleaned_sentences

    def _group_sentences_into_chunks(self, sentences: List[str]) -> List[List[str]]:
        """Group sentences into appropriately sized chunks"""

        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            # Check if adding this sentence would exceed max size
            if (current_size + sentence_length > self.config['max_chunk_size'] and 
                current_chunk and current_size > self.config['min_chunk_size']):

                # Add overlap sentence if configured
                if (chunks and self.config['sentence_overlap'] > 0 and 
                    len(current_chunk) > self.config['sentence_overlap']):
                    overlap_sentences = current_chunk[-self.config['sentence_overlap']:]
                    chunks.append(current_chunk)
                    current_chunk = overlap_sentences[:]
                    current_size = sum(len(s) for s in overlap_sentences)
                else:
                    chunks.append(current_chunk)
                    current_chunk = []
                    current_size = 0

            current_chunk.append(sentence)
            current_size += sentence_length

        # Add final chunk if it has content
        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _find_citations_in_text(self, text: str, citations: List[Dict]) -> List[Dict]:
        """Find citations that appear in the given text"""

        found_citations = []
        for citation in citations:
            if citation['text'] in text:
                found_citations.append(citation)

        return found_citations

    def _find_math_in_text(self, text: str, math_elements: List[Dict]) -> List[Dict]:
        """Find mathematical elements that appear in the given text"""

        found_math = []
        for math_elem in math_elements:
            # Check if LaTeX content appears in text (simplified check)
            if math_elem['latex'] and len(math_elem['latex']) > 5:
                # Look for similar mathematical content
                if any(symbol in text for symbol in ['=', '∑', '∫', '≤', '≥', '±']):
                    found_math.append(math_elem)

        return found_math

    def _has_figure_references(self, text: str) -> bool:
        """Check if text contains figure references"""

        figure_patterns = [
            r'Figure\s+\d+',
            r'Fig\.\s+\d+',
            r'figure\s+\d+',
            r'fig\.\s+\d+'
        ]

        for pattern in figure_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        return False

    def _extract_figure_refs(self, text: str) -> List[str]:
        """Extract figure reference strings from text"""

        refs = []
        patterns = [
            r'Figure\s+(\d+)',
            r'Fig\.\s+(\d+)',
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                refs.append(f"Figure {match.group(1)}")

        return refs

    def _get_context_before(self, all_sentences: List[str], current_group: List[str]) -> str:
        """Get contextual sentences before current chunk"""

        if not current_group or not all_sentences:
            return ""

        # Find where current group starts
        first_sentence = current_group[0]
        try:
            start_idx = all_sentences.index(first_sentence)
            # Get 2 sentences before
            context_start = max(0, start_idx - 2)
            context_sentences = all_sentences[context_start:start_idx]
            return ' '.join(context_sentences)[-200:]  # Limit to 200 chars
        except ValueError:
            return ""

    def _get_context_after(self, all_sentences: List[str], current_group: List[str]) -> str:
        """Get contextual sentences after current chunk"""

        if not current_group or not all_sentences:
            return ""

        # Find where current group ends
        last_sentence = current_group[-1]
        try:
            end_idx = all_sentences.index(last_sentence)
            # Get 2 sentences after
            context_sentences = all_sentences[end_idx + 1:end_idx + 3]
            return ' '.join(context_sentences)[:200]  # Limit to 200 chars
        except ValueError:
            return ""

    def _generate_chunk_id(self, paper_id: str, section_type: str, section_order: int, chunk_idx: int) -> str:
        """Generate stable chunk ID"""

        base = f"{paper_id}_{section_type}_{section_order}_{chunk_idx}"
        hash_suffix = hashlib.md5(base.encode()).hexdigest()[:8]
        return f"{base}_{hash_suffix}"

    def _calculate_chunking_stats(self, paper_id: str, chunks: List[ContentChunk]) -> ChunkingResult:
        """Calculate chunking statistics"""

        if not chunks:
            return ChunkingResult(
                paper_id=paper_id,
                total_chunks=0,
                section_chunks=0,
                paragraph_chunks=0,
                mixed_chunks=0,
                chunks_with_citations=0,
                chunks_with_math=0,
                avg_chunk_size=0.0,
                processing_errors=[],
                quality_score=0.0
            )

        section_chunks = len([c for c in chunks if c.chunk_type == 'section'])
        para_chunks = len([c for c in chunks if c.chunk_type == 'paragraph'])
        mixed_chunks = len([c for c in chunks if c.chunk_type == 'mixed'])
        with_citations = len([c for c in chunks if c.has_citations])
        with_math = len([c for c in chunks if c.has_math])

        avg_size = sum(c.char_count for c in chunks) / len(chunks)

        # Quality score based on chunk size distribution and metadata richness
        size_quality = min(1.0, avg_size / self.config['target_chunk_size'])
        metadata_quality = (with_citations + with_math) / len(chunks)
        quality_score = (size_quality * 6.0) + (metadata_quality * 4.0)

        return ChunkingResult(
            paper_id=paper_id,
            total_chunks=len(chunks),
            section_chunks=section_chunks,
            paragraph_chunks=para_chunks,
            mixed_chunks=mixed_chunks,
            chunks_with_citations=with_citations,
            chunks_with_math=with_math,
            avg_chunk_size=avg_size,
            processing_errors=[],
            quality_score=min(quality_score, 10.0)
        )

    def _save_chunks_to_database(self, chunks: List[ContentChunk]):
        """Save chunks to database for embedding generation"""

        try:
            conn = self.db_manager.get_sqlite_connection()
            cursor = conn.cursor()

            # Save each chunk
            for chunk in chunks:
                cursor.execute("""
                    INSERT OR REPLACE INTO content_chunks 
                    (id, paper_id, chunk_text, chunk_type, section_id, section_title,
                     canonical_section, chunk_order, char_count, word_count, sentence_count,
                     has_citations, has_math, has_figures, citations_in_chunk,
                     math_elements_in_chunk, figure_refs_in_chunk, context_before,
                     context_after, metadata_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    chunk.id,
                    chunk.paper_id,
                    chunk.chunk_text,
                    chunk.chunk_type,
                    chunk.section_id,
                    chunk.section_title,
                    chunk.canonical_section,
                    chunk.chunk_order,
                    chunk.char_count,
                    chunk.word_count,
                    chunk.sentence_count,
                    chunk.has_citations,
                    chunk.has_math,
                    chunk.has_figures,
                    ','.join(chunk.citations_in_chunk or []),
                    ','.join(chunk.math_elements_in_chunk or []),
                    ','.join(chunk.figure_refs_in_chunk or []),
                    chunk.context_before,
                    chunk.context_after,
                    str(chunk.metadata) if chunk.metadata else None,
                    datetime.now().isoformat()
                ))

            conn.commit()
            self.logger.info(f"Saved {len(chunks)} content chunks to database")

        except Exception as e:
            self.logger.error(f"Error saving chunks: {e}")

    def _update_processing_status(self, paper_id: str, stage: str, success: bool):
        """Update processing status"""

        try:
            conn = self.db_manager.get_sqlite_connection()
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO processing_log (paper_id, processing_stage, status, timestamp)
                VALUES (?, ?, ?, ?)
            """, (
                paper_id,
                stage,
                'success' if success else 'failed',
                datetime.now().isoformat()
            ))

            conn.commit()
        except Exception as e:
            self.logger.error(f"Error updating processing status: {e}")

    def _create_empty_result(self, paper_id: str, error_msg: str) -> ChunkingResult:
        """Create empty result with error"""

        return ChunkingResult(
            paper_id=paper_id,
            total_chunks=0,
            section_chunks=0,
            paragraph_chunks=0,
            mixed_chunks=0,
            chunks_with_citations=0,
            chunks_with_math=0,
            avg_chunk_size=0.0,
            processing_errors=[error_msg],
            quality_score=0.0
        )

    def get_chunking_stats(self, paper_id: str) -> Dict[str, Any]:
        """Get chunking statistics for a paper"""

        try:
            conn = self.db_manager.get_sqlite_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT 
                    COUNT(*) as total_chunks,
                    AVG(char_count) as avg_char_count,
                    AVG(word_count) as avg_word_count,
                    SUM(CASE WHEN has_citations THEN 1 ELSE 0 END) as chunks_with_citations,
                    SUM(CASE WHEN has_math THEN 1 ELSE 0 END) as chunks_with_math,
                    COUNT(DISTINCT canonical_section) as sections_covered
                FROM content_chunks 
                WHERE paper_id = ?
            """, (paper_id,))

            result = cursor.fetchone()
            if result:
                return {
                    'total_chunks': result[0],
                    'avg_char_count': round(result[1] or 0, 1),
                    'avg_word_count': round(result[2] or 0, 1),
                    'chunks_with_citations': result[3],
                    'chunks_with_math': result[4],
                    'sections_covered': result[5]
                }

            return {}

        except Exception as e:
            self.logger.error(f"Error getting chunking stats: {e}")
            return {}

# Standalone function for pipeline integration
def process_paper_chunks(paper_id: str) -> ChunkingResult:
    """
    Create content chunks for a paper.
    This is called after citation parsing (Step 5).
    """
    chunker = ContentChunker()
    return chunker.chunk_paper_content(paper_id)



# test_content_chunker.py - Simple test for content chunking
from database_setup import DatabaseManager

def main():
    """Test content chunking on a paper from the database"""

    # Initialize
    db_manager = DatabaseManager()
    chunker = ContentChunker(db_manager)

    # Find a paper that has been processed through all previous steps
    conn = db_manager.get_sqlite_connection()
    cursor = conn.cursor()

    # Get a paper that has document structure, math, and citations processed
    cursor.execute("""
        SELECT DISTINCT ds.paper_id 
        FROM document_sections ds
        JOIN content_downloads cd ON ds.paper_id = cd.arxiv_id
        LEFT JOIN mathematical_content_processed mcp ON ds.paper_id = mcp.paper_id
        LEFT JOIN citations c ON ds.paper_id = c.paper_id
        WHERE cd.content_type = 'html' AND cd.success = 1
        LIMIT 1
    """)

    row = cursor.fetchone()
    if not row:
        print("No papers with complete processing found.")
        print("Run previous pipeline steps first!")
        return

    paper_id = row[0]
    print(f"Chunking content for paper: {paper_id}")

    # Check what data is available
    cursor.execute("SELECT COUNT(*) FROM document_sections WHERE paper_id = ?", (paper_id,))
    sections_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM mathematical_content_processed WHERE paper_id = ?", (paper_id,))
    math_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM in_text_citations WHERE paper_id = ?", (paper_id,))
    citations_count = cursor.fetchone()[0]

    print(f"Available data:")
    print(f"  Document sections: {sections_count}")
    print(f"  Mathematical elements: {math_count}")
    print(f"  In-text citations: {citations_count}")

    # Process content chunks
    print("\nCreating content chunks...")
    result = chunker.chunk_paper_content(paper_id)

    # Display results
    print("\n" + "="*50)
    print("CONTENT CHUNKING RESULTS")
    print("="*50)
    print(f"Paper ID: {result.paper_id}")
    print(f"Total chunks created: {result.total_chunks}")
    print(f"Section chunks: {result.section_chunks}")
    print(f"Paragraph chunks: {result.paragraph_chunks}")
    print(f"Mixed chunks: {result.mixed_chunks}")
    print(f"Chunks with citations: {result.chunks_with_citations}")
    print(f"Chunks with math: {result.chunks_with_math}")
    print(f"Average chunk size: {result.avg_chunk_size:.0f} characters")
    print(f"Quality score: {result.quality_score:.2f}/10.0")

    if result.processing_errors:
        print(f"\nProcessing errors:")
        for error in result.processing_errors:
            print(f"  - {error}")

    # Show detailed statistics
    stats = chunker.get_chunking_stats(paper_id)
    if stats:
        print(f"\nDetailed Statistics:")
        print(f"  Average characters per chunk: {stats.get('avg_char_count', 0)}")
        print(f"  Average words per chunk: {stats.get('avg_word_count', 0)}")
        print(f"  Sections covered: {stats.get('sections_covered', 0)}")
        print(f"  Citation coverage: {stats.get('chunks_with_citations', 0)}/{stats.get('total_chunks', 0)} chunks")
        print(f"  Math coverage: {stats.get('chunks_with_math', 0)}/{stats.get('total_chunks', 0)} chunks")

    # Show examples of created chunks
    cursor.execute("""
        SELECT id, section_title, canonical_section, char_count, 
               has_citations, has_math, has_figures, chunk_text
        FROM content_chunks 
        WHERE paper_id = ? 
        ORDER BY chunk_order
        LIMIT 5
    """, (paper_id,))

    chunks = cursor.fetchall()
    if chunks:
        print(f"\nExample Chunks:")
        for i, (chunk_id, section_title, canonical, char_count, 
                has_cites, has_math, has_figs, chunk_text) in enumerate(chunks):

            print(f"\n  Chunk {i+1}: {chunk_id}")
            print(f"    Section: {section_title} ({canonical})")
            print(f"    Size: {char_count} characters")

            features = []
            if has_cites:
                features.append("citations")
            if has_math:
                features.append("math")
            if has_figs:
                features.append("figures")

            if features:
                print(f"    Contains: {', '.join(features)}")

            # Show first 150 characters of chunk
            preview = chunk_text[:150].replace('\n', ' ').strip()
            print(f"    Text: {preview}{'...' if len(chunk_text) > 150 else ''}")

    # Show chunk distribution by section
    cursor.execute("""
        SELECT canonical_section, COUNT(*) as chunk_count, AVG(char_count) as avg_size
        FROM content_chunks 
        WHERE paper_id = ? 
        GROUP BY canonical_section
        ORDER BY chunk_count DESC
    """, (paper_id,))

    section_dist = cursor.fetchall()
    if section_dist:
        print(f"\nChunk Distribution by Section:")
        for section, count, avg_size in section_dist:
            print(f"  {section}: {count} chunks (avg: {avg_size:.0f} chars)")

    # Show chunks with rich metadata
    cursor.execute("""
        SELECT section_title, citations_in_chunk, math_elements_in_chunk, figure_refs_in_chunk
        FROM content_chunks 
        WHERE paper_id = ? AND (has_citations OR has_math OR has_figures)
        LIMIT 3
    """, (paper_id,))

    rich_chunks = cursor.fetchall()
    if rich_chunks:
        print(f"\nChunks with Rich Metadata:")
        for section, citations, math_elems, figure_refs in rich_chunks:
            print(f"  Section: {section}")
            if citations:
                print(f"    Citations: {citations}")
            if math_elems:
                print(f"    Math elements: {math_elems}")
            if figure_refs:
                print(f"    Figure refs: {figure_refs}")
            print()

    print("\n" + "="*50)
    print("✅ Content chunking complete!")
    print("Chunks are ready for embedding generation and RAG.")
    print("Check 'content_chunks' table for all chunk data.")
    print("="*50)

if __name__ == "__main__":
    main()

# %%
