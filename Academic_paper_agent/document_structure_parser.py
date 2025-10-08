#%%
# document_structure_parser.py
import re
import hashlib
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import logging
from bs4 import BeautifulSoup

from database_setup import DatabaseManager
from arxiv_search import ArxivPaper

@dataclass
class DocumentElement:
    id: str
    element_type: str
    title: Optional[str] = None
    content: Optional[str] = None
    level: int = 0
    ordinal_index: int = 0
    parent_id: Optional[str] = None
    start_offset: int = 0
    end_offset: int = 0
    attributes: Optional[Dict[str, Any]] = None

@dataclass
class DocumentStructure:
    paper_id: str
    elements: List[DocumentElement]
    sections: List[DocumentElement]
    figures: List[DocumentElement]
    tables: List[DocumentElement]
    equations: List[DocumentElement]
    paragraphs: List[DocumentElement]
    references_section: Optional[DocumentElement] = None
    quality_metrics: Optional[Dict[str, Any]] = None

class DocumentStructureParser:
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db_manager = db_manager or DatabaseManager()
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.canonical_sections = {
            'abstract': ['abstract', 'summary'],
            'introduction': ['introduction', 'intro', 'background', 'motivation'],
            'related_work': ['related work', 'literature review', 'prior work', 'background'],
            'methodology': ['method', 'methods', 'methodology', 'approach', 'techniques'],
            'experiments': ['experiments', 'experimental setup', 'evaluation', 'results'],
            'results': ['results', 'findings', 'analysis', 'empirical results'],
            'discussion': ['discussion', 'analysis', 'interpretation'],
            'conclusion': ['conclusion', 'conclusions', 'future work', 'summary'],
            'acknowledgments': ['acknowledgments', 'acknowledgements', 'thanks'],
            'references': ['references', 'bibliography', 'citations']
        }

    def parse_document_structure(self, paper: ArxivPaper, html_content: str) -> DocumentStructure:
        self.logger.info(f"Parsing structure for {paper.id}")
        soup = BeautifulSoup(html_content, 'html.parser')
        doc = DocumentStructure(paper_id=paper.id, elements=[], sections=[], figures=[], tables=[], equations=[], paragraphs=[])
        self._extract_sections_from_headings(soup, doc)
        self._extract_figures_and_tables(soup, doc)
        self._extract_math(soup, doc)
        self._extract_paragraphs(soup, doc)
        self._identify_references(doc)
        doc.quality_metrics = self._quality(doc)
        self._persist(doc)
        return doc

    def _extract_sections_from_headings(self, soup: BeautifulSoup, doc: DocumentStructure):
        headings = soup.find_all(['h1','h2','h3','h4','h5','h6'])
        for idx, h in enumerate(headings):
            title = (h.get_text() or '').strip()
            level = int(h.name[1])
            sid = self._sid(doc.paper_id, 'section', idx, title)
            elem = DocumentElement(
                id=sid,
                element_type='section',
                title=title,
                level=level,
                ordinal_index=idx,
                attributes={'canonical_type': self._canonical(title), 'html_tag': h.name}
            )
            doc.sections.append(elem)
            doc.elements.append(elem)

    def _extract_figures_and_tables(self, soup: BeautifulSoup, doc: DocumentStructure):
        for idx, fig in enumerate(soup.select('figure, div[class*="figure"]')):
            cap = fig.select_one('figcaption, .caption')
            caption = (cap.get_text().strip() if cap else '')
            label = None
            m = re.search(r'(Figure|Fig\.?)[\s\xa0]*(\d+)', caption, re.IGNORECASE)
            if m:
                label = m.group(0)
            label = label or f"Figure {idx+1}"
            fid = self._sid(doc.paper_id, 'figure', idx, label)
            elem = DocumentElement(id=fid, element_type='figure', title=label, content=caption, ordinal_index=idx,
                                   attributes={'caption': caption})
            doc.figures.append(elem)
            doc.elements.append(elem)
        for idx, tab in enumerate(soup.select('table, figure[class*="table"]')):
            cap = tab.select_one('caption, figcaption, .caption')
            caption = (cap.get_text().strip() if cap else '')
            m = re.search(r'(Table|Tab\.?)[\s\xa0]*(\d+)', caption, re.IGNORECASE)
            label = m.group(0) if m else f"Table {idx+1}"
            tid = self._sid(doc.paper_id, 'table', idx, label)
            elem = DocumentElement(id=tid, element_type='table', title=label, content=caption, ordinal_index=idx,
                                   attributes={'caption': caption})
            doc.tables.append(elem)
            doc.elements.append(elem)

    def _extract_math(self, soup: BeautifulSoup, doc: DocumentStructure):
        math_nodes = soup.select('math, span[class*="math"], div[class*="equation"]')
        for idx, node in enumerate(math_nodes):
            latex = node.get('alttext') or node.get('title') or node.get_text()
            eqn = None
            num = node.find_next(string=re.compile(r'^\(\d+\)$'))
            if num:
                eqn = num.strip()
            eid = self._sid(doc.paper_id, 'equation', idx, eqn or str(idx))
            elem = DocumentElement(id=eid, element_type='equation', content=latex, ordinal_index=idx,
                                   attributes={'equation_number': eqn, 'latex_source': latex})
            doc.equations.append(elem)
            doc.elements.append(elem)

    def _extract_paragraphs(self, soup: BeautifulSoup, doc: DocumentStructure):
        paras = soup.select('p')
        for idx, p in enumerate(paras):
            text = (p.get_text() or '').strip()
            if len(text) < 10:
                continue
            pid = self._sid(doc.paper_id, 'paragraph', idx)
            elem = DocumentElement(id=pid, element_type='paragraph', content=text, ordinal_index=idx,
                                   attributes={'word_count': len(text.split())})
            doc.paragraphs.append(elem)
            doc.elements.append(elem)

    def _identify_references(self, doc: DocumentStructure):
        for s in doc.sections:
            if s.attributes.get('canonical_type') == 'references':
                doc.references_section = s
                return

    def _quality(self, doc: DocumentStructure) -> Dict[str, Any]:
        found = set()
        for s in doc.sections:
            ct = s.attributes.get('canonical_type')
            if ct and ct != 'other':
                found.add(ct)
        expected = {'abstract','introduction','methodology','results','conclusion'}
        return {
            'total_sections': len(doc.sections),
            'total_figures': len(doc.figures),
            'total_tables': len(doc.tables),
            'total_equations': len(doc.equations),
            'total_paragraphs': len(doc.paragraphs),
            'has_references': doc.references_section is not None,
            'canonical_sections_found': list(found),
            'parsing_completeness': len(found & expected) / len(expected) if expected else 1.0
        }

    def _persist(self, doc: DocumentStructure):
        try:
            conn = self.db_manager.get_sqlite_connection()
            cur = conn.cursor()
            for idx, s in enumerate(doc.sections):
                cur.execute(
                    'INSERT OR REPLACE INTO document_sections (paper_id, section_type, section_title, section_order, content_start, content_end, parent_section_id) VALUES (?, ?, ?, ?, ?, ?, ?)',
                    (doc.paper_id, s.attributes.get('canonical_type','other'), s.title, s.ordinal_index, s.start_offset, s.end_offset, None)
                )
            for eq in doc.equations:
                cur.execute(
                    'INSERT OR REPLACE INTO mathematical_content (paper_id, section_id, equation_number, latex_content, position) VALUES (?, ?, ?, ?, ?)',
                    (doc.paper_id, None, eq.attributes.get('equation_number'), eq.content, eq.ordinal_index)
                )
            cur.execute('INSERT INTO processing_log (paper_id, processing_stage, status, timestamp) VALUES (?, ?, ?, ?)',
                        (doc.paper_id, 'structure_parsing', 'success', datetime.now().isoformat()))
            conn.commit()
        except Exception as e:
            self.logger.error(f"Persist error: {e}")

    def _canonical(self, title: str) -> str:
        tl = (title or '').lower()
        for canon, variants in self.canonical_sections.items():
            for v in variants:
                if v in tl:
                    return canon
        return 'other'

    def _sid(self, paper_id: str, typ: str, idx: int, title: Optional[str] = None) -> str:
        base = f"{paper_id}_{typ}_{idx}"
        if title:
            t = re.sub(r'[^a-z0-9]', '', (title or '').lower())[:20]
            base += f"_{t}"
        return base + '_' + hashlib.md5(base.encode()).hexdigest()[:8]



# simple_test.py - Get a downloaded paper and parse it
from datetime import datetime
from pathlib import Path
from document_structure_parser import DocumentStructureParser
from database_setup import DatabaseManager
from arxiv_search import ArxivPaper

def main():
    """Get a downloaded paper and run it through the parser"""

    # Initialize
    db_manager = DatabaseManager()
    parser = DocumentStructureParser(db_manager)

    # Get first successfully downloaded HTML paper from content_downloads table
    conn = db_manager.get_sqlite_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT arxiv_id, file_path, content_size 
        FROM content_downloads 
        WHERE content_type = 'html' AND success = 1 
        LIMIT 3
    """)
    row = cursor.fetchone()

    if not row:
        print("No successfully downloaded HTML papers found in content_downloads table")
        return

    arxiv_id, file_path, content_size = row
    print(f"Found downloaded paper: {arxiv_id}")
    print(f"File path: {file_path}")
    print(f"Content size: {content_size} bytes")

    # Create minimal ArxivPaper object (we don't have full metadata)
    paper = ArxivPaper(
        id=arxiv_id,
        title=f"Paper {arxiv_id}",  # Placeholder
        authors=["Unknown"],  # Placeholder
        abstract="",  # Placeholder
        published=datetime.now(),
        updated=datetime.now(),
        categories=[],
        pdf_url="",
        abstract_url="",
        html_url=""
    )

    # Load HTML content from the file path stored in database
    html_file = Path(file_path)
    if not html_file.exists():
        print(f"HTML file not found: {html_file}")
        return

    with open(html_file, 'r', encoding='utf-8') as f:
        html_content = f.read()

    print(f"Loaded HTML content: {len(html_content)} chars")

    # Parse structure
    print("Parsing document structure...")
    doc_structure = parser.parse_document_structure(paper, html_content)

    # Show results
    print(f"✓ Parsed {len(doc_structure.sections)} sections")
    print(f"✓ Found {len(doc_structure.figures)} figures")
    print(f"✓ Found {len(doc_structure.tables)} tables") 
    print(f"✓ Found {len(doc_structure.equations)} equations")

    if doc_structure.sections:
        print("\nSections found:")
        for i, section in enumerate(doc_structure.sections[:5]):  # Show first 5
            canonical = section.attributes.get('canonical_type', 'other')
            print(f"  {i+1}. {section.title} (canonical: {canonical})")

    print("\nDone! Check document_sections and mathematical_content tables.")

if __name__ == "__main__":
    main()

# %%
