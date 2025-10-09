#%%
# citation_reference_parser.py - Step 5: Citation and Reference Parsing

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
class Citation:
    """Represents a parsed citation entry"""
    id: str
    paper_id: str
    citation_key: Optional[str] = None  # [1], [Smith2020], etc.
    raw_text: str = ""
    authors: List[str] = None
    title: Optional[str] = None
    venue: Optional[str] = None  # Journal/Conference
    year: Optional[int] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    url: Optional[str] = None
    pages: Optional[str] = None
    volume: Optional[str] = None
    position_in_references: int = 0

@dataclass
class InTextCitation:
    """Represents an in-text citation mention"""
    id: str
    paper_id: str
    citation_text: str  # "[1]", "(Smith, 2020)", etc.
    citation_type: str  # "numeric", "author_year", "named"
    section_id: Optional[str] = None
    paragraph_context: str = ""
    position_in_text: int = 0
    linked_citation_id: Optional[str] = None

@dataclass
class CitationParsingResult:
    """Result of citation parsing process"""
    paper_id: str
    total_references: int
    parsed_references: int
    in_text_citations: int
    linked_citations: int
    citation_types: Set[str]
    processing_errors: List[str]
    quality_score: float

class CitationReferenceParser:
    """
    Parses citations and references from academic papers.
    This is Step 5 in the academic paper processing pipeline.
    """

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db_manager = db_manager or DatabaseManager()
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Citation patterns for different formats
        self.citation_patterns = {
            # Numeric citations: [1], [12], [1-3], [1,5,7]
            'numeric': r'\[\s*(\d+(?:[,-]\s*\d+)*)\s*\]',

            # Author-year: (Smith, 2020), (Smith et al., 2020)
            'author_year': r'\(([A-Z][a-z]+(?:\s+et\s+al\.)?),?\s+(\d{4})\)',

            # Named citations: Smith (2020), Smith et al. (2020)  
            'named': r'([A-Z][a-z]+(?:\s+et\s+al\.)?)\s*\((\d{4})\)',

            # Multiple variations
            'author_year_multi': r'\(([^)]+(?:;\s*[^)]+)*)\)',
        }

        # Reference entry patterns
        self.reference_patterns = {
            'authors': r'^([^.]+(?:\.[^.]*){0,2})\.',  # First part before first period
            'title': r'"([^"]+)"|\'([^\']+)\'|\b([A-Z][^.]*\.)\b',  # Quoted or capitalized
            'year': r'\b(19\d{2}|20\d{2})\b',  # 4-digit year
            'doi': r'doi:?\s*([\w.-]+/[\w.-]+)',
            'arxiv': r'arXiv:?\s*(\d{4}\.\d{4,5})',
            'url': r'https?://[^\s]+',
            'venue_patterns': [
                r'\b(Journal|Proc\.|Conference|Workshop|Symposium)\b[^.]*',
                r'\b[A-Z]{2,}\s+\d{4}\b',  # Conference acronyms
            ]
        }

    def parse_citations_and_references(self, paper_id: str) -> CitationParsingResult:
        """
        Main entry point for parsing citations and references.

        Args:
            paper_id: ArXiv paper ID to process

        Returns:
            CitationParsingResult with parsing statistics
        """
        self.logger.info(f"Parsing citations and references for paper {paper_id}")

        try:
            # Get HTML content and document structure
            html_content = self._load_html_content(paper_id)
            references_section = self._get_references_section(paper_id)

            if not html_content:
                return self._create_empty_result(paper_id, "No HTML content found")

            # Parse references section
            citations = self._parse_references_section(paper_id, html_content, references_section)

            # Parse in-text citations
            in_text_citations = self._parse_in_text_citations(paper_id, html_content)

            # Link in-text citations to reference entries
            linked_count = self._link_citations(citations, in_text_citations)

            # Calculate quality score
            quality_score = self._calculate_citation_quality_score(citations, in_text_citations, linked_count)

            # Save to database
            self._save_citations_to_database(citations, in_text_citations)

            # Update processing status
            self._update_processing_status(paper_id, 'citation_parsing', len(citations) > 0)

            result = CitationParsingResult(
                paper_id=paper_id,
                total_references=len(citations),
                parsed_references=len([c for c in citations if c.authors]),
                in_text_citations=len(in_text_citations),
                linked_citations=linked_count,
                citation_types=set(c.citation_type for c in in_text_citations),
                processing_errors=[],
                quality_score=quality_score
            )

            self.logger.info(f"Citation parsing completed for {paper_id}: "
                           f"{result.parsed_references}/{result.total_references} references parsed, "
                           f"{result.in_text_citations} in-text citations found")

            return result

        except Exception as e:
            self.logger.error(f"Error in citation parsing for {paper_id}: {e}")
            return self._create_empty_result(paper_id, f"Processing failed: {str(e)}")

    def _load_html_content(self, paper_id: str) -> Optional[str]:
        """Load HTML content for the paper"""

        conn = self.db_manager.get_sqlite_connection()
        cursor = conn.cursor()

        # Get file path from content_downloads
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

    def _get_references_section(self, paper_id: str) -> Optional[Dict]:
        """Get references section info from document structure"""

        conn = self.db_manager.get_sqlite_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT section_title, content_start, content_end 
            FROM document_sections 
            WHERE paper_id = ? AND section_type = 'references'
        """, (paper_id,))

        row = cursor.fetchone()
        if row:
            return {
                'title': row[0],
                'start': row[1], 
                'end': row[2]
            }
        return None

    def _parse_references_section(self, paper_id: str, html_content: str, references_section: Optional[Dict]) -> List[Citation]:
        """Parse bibliography entries from references section"""

        soup = BeautifulSoup(html_content, 'html.parser')
        citations = []

        # Find references section in HTML
        refs_section = None
        for heading in soup.find_all(['h1', 'h2', 'h3', 'h4']):
            if 'reference' in heading.get_text().lower():
                refs_section = heading.find_next_sibling()
                break

        if not refs_section:
            # Look for bibliography div or section
            refs_section = soup.find('div', class_=re.compile(r'.*bib.*|.*reference.*', re.I))

        if not refs_section:
            self.logger.warning(f"No references section found for {paper_id}")
            return citations

        # Extract individual reference entries
        ref_entries = []

        # Try different patterns for reference entries
        # Pattern 1: <p> tags with numbers/brackets
        numbered_refs = refs_section.find_all('p', string=re.compile(r'^\s*\[?\d+\]?'))
        if numbered_refs:
            ref_entries = numbered_refs
        else:
            # Pattern 2: <div> or <li> elements  
            ref_entries = refs_section.find_all(['div', 'li', 'p'])

        # Parse each reference entry
        for i, entry in enumerate(ref_entries):
            if not entry or len(entry.get_text().strip()) < 20:
                continue

            citation = self._parse_single_reference(paper_id, entry.get_text(), i)
            citations.append(citation)

        self.logger.info(f"Parsed {len(citations)} reference entries for {paper_id}")
        return citations

    def _parse_single_reference(self, paper_id: str, ref_text: str, position: int) -> Citation:
        """Parse a single reference entry"""

        citation_id = f"{paper_id}_ref_{position}"

        # Extract citation key (number in brackets)
        key_match = re.search(r'^\s*\[?(\d+)\]?', ref_text)
        citation_key = key_match.group(1) if key_match else str(position + 1)

        # Extract authors (usually at the beginning)
        authors = []
        author_match = re.search(r'^[^.]*([A-Z][a-z]+(?:,\s*[A-Z]\.)?(?:\s+and\s+[A-Z][a-z]+(?:,\s*[A-Z]\.)?)*)', ref_text)
        if author_match:
            author_text = author_match.group(1)
            # Split by "and" or commas
            authors = [name.strip() for name in re.split(r'\s+and\s+|,\s*(?=[A-Z])', author_text)]

        # Extract title (usually in quotes or after authors)
        title = None
        title_patterns = [
            r'"([^"]+)"',  # Quoted title
            r'"([^"]+)"',  # Single quoted title
            r'\b([A-Z][^.]*[.!?])\b'  # Capitalized sentence ending with punctuation
        ]

        for pattern in title_patterns:
            title_match = re.search(pattern, ref_text)
            if title_match:
                title = title_match.group(1).strip()
                break

        # Extract year
        year = None
        year_match = re.search(r'\b(19\d{2}|20\d{2})\b', ref_text)
        if year_match:
            year = int(year_match.group(1))

        # Extract DOI
        doi = None
        doi_match = re.search(r'doi:?\s*([\w.-]+/[\w.-]+)', ref_text, re.I)
        if doi_match:
            doi = doi_match.group(1)

        # Extract ArXiv ID
        arxiv_id = None
        arxiv_match = re.search(r'arXiv:?\s*(\d{4}\.\d{4,5})', ref_text, re.I)
        if arxiv_match:
            arxiv_id = arxiv_match.group(1)

        # Extract venue (journal/conference)
        venue = None
        venue_patterns = [
            r'\b(Journal\s+of\s+[^,]+)',
            r'\b(Proceedings\s+of\s+[^,]+)',
            r'\b([A-Z]{2,}\s+\d{4})\b',  # Conference acronym + year
        ]

        for pattern in venue_patterns:
            venue_match = re.search(pattern, ref_text)
            if venue_match:
                venue = venue_match.group(1).strip()
                break

        return Citation(
            id=citation_id,
            paper_id=paper_id,
            citation_key=citation_key,
            raw_text=ref_text.strip(),
            authors=authors,
            title=title,
            venue=venue,
            year=year,
            doi=doi,
            arxiv_id=arxiv_id,
            position_in_references=position
        )

    def _parse_in_text_citations(self, paper_id: str, html_content: str) -> List[InTextCitation]:
        """Parse in-text citation mentions throughout the document"""

        soup = BeautifulSoup(html_content, 'html.parser')
        in_text_citations = []

        # Get all text content with structure
        text_elements = soup.find_all(['p', 'div', 'section'])

        citation_id_counter = 0

        for element in text_elements:
            text = element.get_text()
            if len(text.strip()) < 10:
                continue

            # Find numeric citations [1], [2,3], [1-5]
            for match in re.finditer(self.citation_patterns['numeric'], text):
                citation_id_counter += 1
                citation = InTextCitation(
                    id=f"{paper_id}_intext_{citation_id_counter}",
                    paper_id=paper_id,
                    citation_text=match.group(0),
                    citation_type="numeric",
                    paragraph_context=text[:match.start()] + "**" + match.group(0) + "**" + text[match.end():][:100],
                    position_in_text=match.start()
                )
                in_text_citations.append(citation)

            # Find author-year citations (Smith, 2020)
            for match in re.finditer(self.citation_patterns['author_year'], text):
                citation_id_counter += 1
                citation = InTextCitation(
                    id=f"{paper_id}_intext_{citation_id_counter}",
                    paper_id=paper_id,
                    citation_text=match.group(0),
                    citation_type="author_year",
                    paragraph_context=text[:match.start()] + "**" + match.group(0) + "**" + text[match.end():][:100],
                    position_in_text=match.start()
                )
                in_text_citations.append(citation)

            # Find named citations Smith (2020)
            for match in re.finditer(self.citation_patterns['named'], text):
                citation_id_counter += 1
                citation = InTextCitation(
                    id=f"{paper_id}_intext_{citation_id_counter}",
                    paper_id=paper_id,
                    citation_text=match.group(0),
                    citation_type="named",
                    paragraph_context=text[:match.start()] + "**" + match.group(0) + "**" + text[match.end():][:100],
                    position_in_text=match.start()
                )
                in_text_citations.append(citation)

        self.logger.info(f"Found {len(in_text_citations)} in-text citations for {paper_id}")
        return in_text_citations

    def _link_citations(self, citations: List[Citation], in_text_citations: List[InTextCitation]) -> int:
        """Link in-text citations to reference entries"""

        linked_count = 0

        for in_text in in_text_citations:
            if in_text.citation_type == "numeric":
                # Extract numbers from [1], [2,3], etc.
                numbers = re.findall(r'\d+', in_text.citation_text)
                for num_str in numbers:
                    num = int(num_str)
                    # Find matching citation by key
                    for citation in citations:
                        if citation.citation_key == num_str:
                            in_text.linked_citation_id = citation.id
                            linked_count += 1
                            break

            elif in_text.citation_type in ["author_year", "named"]:
                # Extract author and year
                author_match = re.search(r'([A-Z][a-z]+)', in_text.citation_text)
                year_match = re.search(r'(\d{4})', in_text.citation_text)

                if author_match and year_match:
                    author = author_match.group(1)
                    year = int(year_match.group(1))

                    # Find matching citation by author and year
                    for citation in citations:
                        if (citation.authors and citation.year == year and 
                            any(author in auth for auth in citation.authors)):
                            in_text.linked_citation_id = citation.id
                            linked_count += 1
                            break

        return linked_count

    def _calculate_citation_quality_score(self, citations: List[Citation], 
                                        in_text_citations: List[InTextCitation], 
                                        linked_count: int) -> float:
        """Calculate quality score for citation parsing"""

        if not citations and not in_text_citations:
            return 0.0

        score = 0.0

        # Base score for finding citations
        if citations:
            score += 3.0
        if in_text_citations:
            score += 2.0

        # Quality of reference parsing
        parsed_refs = len([c for c in citations if c.authors and c.title])
        if citations:
            parse_quality = parsed_refs / len(citations)
            score += parse_quality * 3.0

        # Quality of linking
        if in_text_citations:
            link_quality = linked_count / len(in_text_citations)
            score += link_quality * 2.0

        return min(score, 10.0)

    def _save_citations_to_database(self, citations: List[Citation], in_text_citations: List[InTextCitation]):
        """Save citations and in-text citations to database"""

        try:
            conn = self.db_manager.get_sqlite_connection()
            cursor = conn.cursor()

            # Save reference citations to existing citations table
            for citation in citations:
                cursor.execute("""
                    INSERT OR REPLACE INTO citations 
                    (paper_id, citation_key, citation_text, authors, title, venue, year, doi, cited_arxiv_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    citation.paper_id,
                    citation.citation_key,
                    citation.raw_text,
                    ','.join(citation.authors) if citation.authors else None,
                    citation.title,
                    citation.venue,
                    citation.year,
                    citation.doi,
                    citation.arxiv_id
                ))

            # Create table for in-text citations
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS in_text_citations (
                    id TEXT PRIMARY KEY,
                    paper_id VARCHAR(50),
                    citation_text TEXT,
                    citation_type VARCHAR(20),
                    section_id TEXT,
                    paragraph_context TEXT,
                    position_in_text INTEGER,
                    linked_citation_id TEXT,
                    FOREIGN KEY (paper_id) REFERENCES papers(arxiv_id)
                )
            """)

            # Save in-text citations
            for in_text in in_text_citations:
                cursor.execute("""
                    INSERT OR REPLACE INTO in_text_citations 
                    (id, paper_id, citation_text, citation_type, section_id, 
                     paragraph_context, position_in_text, linked_citation_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    in_text.id,
                    in_text.paper_id,
                    in_text.citation_text,
                    in_text.citation_type,
                    in_text.section_id,
                    in_text.paragraph_context,
                    in_text.position_in_text,
                    in_text.linked_citation_id
                ))

            conn.commit()
            self.logger.info(f"Saved {len(citations)} citations and {len(in_text_citations)} in-text citations")

        except Exception as e:
            self.logger.error(f"Error saving citations: {e}")

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

    def _create_empty_result(self, paper_id: str, error_msg: str) -> CitationParsingResult:
        """Create empty result with error"""

        return CitationParsingResult(
            paper_id=paper_id,
            total_references=0,
            parsed_references=0,
            in_text_citations=0,
            linked_citations=0,
            citation_types=set(),
            processing_errors=[error_msg],
            quality_score=0.0
        )

    def get_citation_stats(self, paper_id: str) -> Dict[str, Any]:
        """Get citation statistics for a paper"""

        try:
            conn = self.db_manager.get_sqlite_connection()
            cursor = conn.cursor()

            # Get citation counts
            cursor.execute("SELECT COUNT(*) FROM citations WHERE paper_id = ?", (paper_id,))
            total_refs = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM in_text_citations WHERE paper_id = ?", (paper_id,))
            total_in_text = cursor.fetchone()[0]

            cursor.execute("""
                SELECT COUNT(*) FROM in_text_citations 
                WHERE paper_id = ? AND linked_citation_id IS NOT NULL
            """, (paper_id,))
            linked_count = cursor.fetchone()[0]

            return {
                'total_references': total_refs,
                'total_in_text_citations': total_in_text,
                'linked_citations': linked_count,
                'linking_rate': (linked_count / total_in_text * 100) if total_in_text > 0 else 0
            }

        except Exception as e:
            self.logger.error(f"Error getting citation stats: {e}")
            return {}

# Standalone function for pipeline integration
def process_paper_citations(paper_id: str) -> CitationParsingResult:
    """
    Process citations and references for a paper.
    This is called after mathematical content processing (Step 4).
    """
    parser = CitationReferenceParser()
    return parser.parse_citations_and_references(paper_id)



# test_citation_parser.py - Simple test for citation and reference parsing

from citation_reference_parser import CitationReferenceParser, process_paper_citations
from database_setup import DatabaseManager

def main():
    """Test citation parsing on a paper from the database"""

    # Initialize
    db_manager = DatabaseManager()
    parser = CitationReferenceParser(db_manager)

    # Find a paper that has been processed through document structure parsing
    conn = db_manager.get_sqlite_connection()
    cursor = conn.cursor()

    # Get a paper that has document structure parsed and HTML content
    cursor.execute("""
        SELECT DISTINCT ds.paper_id 
        FROM document_sections ds
        JOIN content_downloads cd ON ds.paper_id = cd.arxiv_id
        WHERE cd.content_type = 'html' AND cd.success = 1
        LIMIT 1
    """)

    row = cursor.fetchone()
    if not row:
        print("No papers with document structure found.")
        print("Run document structure parser first!")
        return

    paper_id = row[0]
    print(f"Processing citations for paper: {paper_id}")

    # Check if references section exists
    cursor.execute("""
        SELECT section_title FROM document_sections 
        WHERE paper_id = ? AND section_type = 'references'
    """, (paper_id,))

    refs_section = cursor.fetchone()
    if refs_section:
        print(f"Found references section: {refs_section[0]}")
    else:
        print("No references section found in document structure")

    # Process citations and references
    print("\nParsing citations and references...")
    result = parser.parse_citations_and_references(paper_id)

    # Display results
    print("\n" + "="*50)
    print("CITATION PARSING RESULTS")
    print("="*50)
    print(f"Paper ID: {result.paper_id}")
    print(f"References found: {result.total_references}")
    print(f"Successfully parsed: {result.parsed_references}")
    print(f"In-text citations: {result.in_text_citations}")
    print(f"Linked citations: {result.linked_citations}")
    print(f"Citation types found: {', '.join(result.citation_types)}")
    print(f"Quality score: {result.quality_score:.2f}/10.0")

    if result.processing_errors:
        print(f"\nProcessing errors:")
        for error in result.processing_errors:
            print(f"  - {error}")

    # Show detailed statistics
    stats = parser.get_citation_stats(paper_id)
    if stats:
        print(f"\nDetailed Statistics:")
        print(f"  Total references: {stats.get('total_references', 0)}")
        print(f"  In-text citations: {stats.get('total_in_text_citations', 0)}")
        print(f"  Successfully linked: {stats.get('linked_citations', 0)}")
        print(f"  Linking success rate: {stats.get('linking_rate', 0):.1f}%")

    # Show examples of parsed references
    cursor.execute("""
        SELECT citation_key, authors, title, venue, year 
        FROM citations 
        WHERE paper_id = ? 
        ORDER BY CAST(citation_key AS INTEGER)
        LIMIT 5
    """, (paper_id,))

    refs = cursor.fetchall()
    if refs:
        print(f"\nExample References:")
        for key, authors, title, venue, year in refs:
            print(f"  [{key}] {authors or 'Unknown'}")
            if title:
                title_str = title[:60]
                if len(title) > 60:
                    title_str += "..."
                print(f"      {title_str}")

            if venue and year:
                print(f"      {venue}, {year}")
            elif year:
                print(f"      {year}")
            print()

    # Show examples of in-text citations
    cursor.execute("""
        SELECT citation_text, citation_type, paragraph_context, linked_citation_id
        FROM in_text_citations 
        WHERE paper_id = ? 
        LIMIT 5
    """, (paper_id,))

    in_text = cursor.fetchall()
    if in_text:
        print(f"Example In-Text Citations:")
        for cite_text, cite_type, context, linked_id in in_text:
            print(f"  {cite_text} ({cite_type})")
            print(f"    Context: {context[:100]}{'...' if len(context) > 100 else ''}")
            print(f"    Linked: {'Yes' if linked_id else 'No'}")
            print()

    # Show citation network info for potential graph database
    if result.linked_citations > 0:
        print(f"Citation Network Information:")
        print(f"  This paper cites {result.total_references} other works")
        print(f"  {result.linked_citations} citations are properly linked")
        print(f"  Ready for graph database storage and network analysis")

    print("\n" + "="*50)
    print("✅ Citation parsing complete!")
    print("Check 'citations' and 'in_text_citations' tables for detailed results.")
    print("="*50)

if __name__ == "__main__":
    main()


# %%
