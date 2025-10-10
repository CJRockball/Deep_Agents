# wrappers/query_tools.py
# Function 3: Query Tools for Processed Academic Paper Data
# Compatible with LangGraph agents and LLM integration

import logging
import json
import re
from typing import List, Dict, Any, Optional, Union, Set
from datetime import datetime
from dataclasses import dataclass

# Import core components
from ..core.database_setup import DatabaseManager

# Module-level logger
logger = logging.getLogger(__name__)

@dataclass
class QueryResult:
    """Result of a query operation"""
    query_type: str
    success: bool
    results: List[Dict[str, Any]]
    total_matches: int
    execution_time: float
    metadata: Dict[str, Any]
    errors: List[str] = None

@dataclass
class PaperSummary:
    """Comprehensive paper summary for LLM consumption"""
    paper_id: str
    title: str
    authors: List[str]
    
    # Content overview
    sections: List[Dict[str, str]]
    total_chunks: int
    key_entities: List[Dict[str, Any]]
    mathematical_content: Dict[str, Any]
    citation_info: Dict[str, Any]
    
    # Quality metrics
    processing_quality: Dict[str, float]
    llm_ready: bool
    
    # Rich content for analysis
    abstract_text: Optional[str]
    key_findings: List[str]
    methodologies: List[str]
    datasets_used: List[str]
    
    # Metadata
    processed_at: datetime
    content_stats: Dict[str, int]

class AcademicPaperQueryEngine:
    """
    Query engine for processed academic paper data.
    Designed for LangGraph agent integration with semantic search,
    structured queries, and LLM-optimized responses.
    """
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """Initialize query engine"""
        self.db_manager = db_manager or DatabaseManager()
        
    def query_processed_data(self, 
                           query: str,
                           query_type: str = "semantic",
                           paper_ids: Optional[List[str]] = None,
                           max_results: int = 10,
                           include_context: bool = True) -> QueryResult:
        """
        Query processed academic paper data with multiple search strategies.
        
        Args:
            query: Search query string
            query_type: Type of query ('semantic', 'keyword', 'entity', 'citation', 'math')
            paper_ids: Optional list of paper IDs to restrict search to
            max_results: Maximum number of results to return
            include_context: Whether to include contextual information
            
        Returns:
            QueryResult with search results optimized for LLM consumption
            
        Example:
            result = query_processed_data(
                "What are the main approaches to quantum error correction?",
                query_type="semantic",
                max_results=5
            )
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Executing {query_type} query: '{query[:50]}...' on {len(paper_ids or [])} papers")
            
            if query_type == "semantic":
                results = self._semantic_search(query, paper_ids, max_results, include_context)
            elif query_type == "keyword":
                results = self._keyword_search(query, paper_ids, max_results, include_context)
            elif query_type == "entity":
                results = self._entity_search(query, paper_ids, max_results, include_context)
            elif query_type == "citation":
                results = self._citation_search(query, paper_ids, max_results, include_context)
            elif query_type == "math":
                results = self._mathematical_search(query, paper_ids, max_results, include_context)
            else:
                return QueryResult(
                    query_type=query_type,
                    success=False,
                    results=[],
                    total_matches=0,
                    execution_time=0.0,
                    metadata={},
                    errors=[f"Unknown query type: {query_type}"]
                )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return QueryResult(
                query_type=query_type,
                success=True,
                results=results,
                total_matches=len(results),
                execution_time=execution_time,
                metadata={
                    "query": query,
                    "paper_ids_searched": paper_ids,
                    "include_context": include_context
                },
                errors=[]
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Query execution failed: {str(e)}"
            logger.error(error_msg)
            
            return QueryResult(
                query_type=query_type,
                success=False,
                results=[],
                total_matches=0,
                execution_time=execution_time,
                metadata={"query": query},
                errors=[error_msg]
            )
    
    def _semantic_search(self, query: str, paper_ids: Optional[List[str]], 
                        max_results: int, include_context: bool) -> List[Dict[str, Any]]:
        """Perform semantic search using full-text search on content chunks"""
        conn = self.db_manager.get_sqlite_connection()
        cursor = conn.cursor()
        
        # Prepare query for FTS5
        fts_query = self._prepare_fts_query(query)
        
        # Build SQL query with optional paper ID filtering
        if paper_ids:
            placeholders = ','.join(['?' for _ in paper_ids])
            sql = f"""
                SELECT 
                    cc.paper_id,
                    cc.chunk_text,
                    cc.section_title,
                    cc.canonical_section,
                    cc.has_citations,
                    cc.has_math,
                    cc.citations_in_chunk,
                    cc.context_before,
                    cc.context_after,
                    bm25(fts_content_chunks) as relevance_score
                FROM fts_content_chunks 
                JOIN content_chunks cc ON fts_content_chunks.rowid = cc.rowid
                WHERE fts_content_chunks MATCH ? 
                AND cc.paper_id IN ({placeholders})
                ORDER BY relevance_score
                LIMIT ?
            """
            params = [fts_query] + paper_ids + [max_results]
        else:
            sql = """
                SELECT 
                    cc.paper_id,
                    cc.chunk_text,
                    cc.section_title,
                    cc.canonical_section,
                    cc.has_citations,
                    cc.has_math,
                    cc.citations_in_chunk,
                    cc.context_before,
                    cc.context_after,
                    bm25(fts_content_chunks) as relevance_score
                FROM fts_content_chunks 
                JOIN content_chunks cc ON fts_content_chunks.rowid = cc.rowid
                WHERE fts_content_chunks MATCH ?
                ORDER BY relevance_score
                LIMIT ?
            """
            params = [fts_query, max_results]
        
        cursor.execute(sql, params)
        results = []
        
        for row in cursor.fetchall():
            (paper_id, chunk_text, section_title, canonical_section, 
             has_citations, has_math, citations, context_before, 
             context_after, relevance_score) = row
            
            result = {
                "paper_id": paper_id,
                "chunk_text": chunk_text[:500] + "..." if len(chunk_text) > 500 else chunk_text,
                "section_title": section_title,
                "canonical_section": canonical_section,
                "has_citations": bool(has_citations),
                "has_math": bool(has_math),
                "relevance_score": round(float(relevance_score), 3),
                "match_type": "semantic"
            }
            
            if include_context:
                result["context_before"] = context_before
                result["context_after"] = context_after
                if citations:
                    result["citations"] = citations.split(',')
            
            results.append(result)
        
        return results
    
    def _keyword_search(self, query: str, paper_ids: Optional[List[str]], 
                       max_results: int, include_context: bool) -> List[Dict[str, Any]]:
        """Perform keyword search on content chunks"""
        conn = self.db_manager.get_sqlite_connection()
        cursor = conn.cursor()
        
        # Extract keywords from query
        keywords = [word.strip().lower() for word in query.split() if len(word.strip()) > 2]
        
        # Build LIKE conditions for keywords
        like_conditions = []
        params = []
        
        for keyword in keywords[:5]:  # Limit to 5 keywords for performance
            like_conditions.append("LOWER(cc.chunk_text) LIKE ?")
            params.append(f"%{keyword}%")
        
        if not like_conditions:
            return []
        
        where_clause = " AND ".join(like_conditions)
        
        if paper_ids:
            placeholders = ','.join(['?' for _ in paper_ids])
            where_clause += f" AND cc.paper_id IN ({placeholders})"
            params.extend(paper_ids)
        
        sql = f"""
            SELECT 
                cc.paper_id,
                cc.chunk_text,
                cc.section_title,
                cc.canonical_section,
                cc.has_citations,
                cc.has_math,
                cc.citations_in_chunk,
                cc.context_before,
                cc.context_after,
                cc.word_count
            FROM content_chunks cc
            WHERE {where_clause}
            ORDER BY cc.word_count DESC
            LIMIT ?
        """
        params.append(max_results)
        
        cursor.execute(sql, params)
        results = []
        
        for row in cursor.fetchall():
            (paper_id, chunk_text, section_title, canonical_section,
             has_citations, has_math, citations, context_before,
             context_after, word_count) = row
            
            # Calculate keyword match count
            text_lower = chunk_text.lower()
            keyword_matches = sum(1 for keyword in keywords if keyword in text_lower)
            
            result = {
                "paper_id": paper_id,
                "chunk_text": chunk_text[:500] + "..." if len(chunk_text) > 500 else chunk_text,
                "section_title": section_title,
                "canonical_section": canonical_section,
                "has_citations": bool(has_citations),
                "has_math": bool(has_math),
                "keyword_matches": keyword_matches,
                "match_type": "keyword"
            }
            
            if include_context:
                result["context_before"] = context_before
                result["context_after"] = context_after
                if citations:
                    result["citations"] = citations.split(',')
            
            results.append(result)
        
        # Sort by keyword matches
        results.sort(key=lambda x: x["keyword_matches"], reverse=True)
        return results
    
    def _entity_search(self, query: str, paper_ids: Optional[List[str]], 
                      max_results: int, include_context: bool) -> List[Dict[str, Any]]:
        """Search for papers containing specific entities"""
        conn = self.db_manager.get_sqlite_connection()
        cursor = conn.cursor()
        
        # Search for entities matching the query
        entity_search_terms = query.lower().split()
        
        # Build entity matching conditions
        entity_conditions = []
        params = []
        
        for term in entity_search_terms:
            entity_conditions.append("LOWER(e.entity_text) LIKE ? OR LOWER(e.normalized_form) LIKE ?")
            params.extend([f"%{term}%", f"%{term}%"])
        
        if not entity_conditions:
            return []
        
        entity_where = " OR ".join(entity_conditions)
        
        if paper_ids:
            placeholders = ','.join(['?' for _ in paper_ids])
            entity_where += f" AND e.paper_id IN ({placeholders})"
            params.extend(paper_ids)
        
        sql = f"""
            SELECT DISTINCT
                e.paper_id,
                e.entity_text,
                e.entity_type,
                e.confidence_score,
                e.mention_count,
                cc.chunk_text,
                cc.section_title,
                cc.canonical_section
            FROM entities e
            JOIN content_chunks cc ON e.chunk_id = cc.id
            WHERE {entity_where}
            ORDER BY e.confidence_score DESC, e.mention_count DESC
            LIMIT ?
        """
        params.append(max_results)
        
        cursor.execute(sql, params)
        results = []
        
        for row in cursor.fetchall():
            (paper_id, entity_text, entity_type, confidence_score,
             mention_count, chunk_text, section_title, canonical_section) = row
            
            result = {
                "paper_id": paper_id,
                "entity_text": entity_text,
                "entity_type": entity_type,
                "confidence_score": round(float(confidence_score), 3),
                "mention_count": mention_count,
                "chunk_text": chunk_text[:300] + "..." if len(chunk_text) > 300 else chunk_text,
                "section_title": section_title,
                "canonical_section": canonical_section,
                "match_type": "entity"
            }
            
            results.append(result)
        
        return results
    
    def _citation_search(self, query: str, paper_ids: Optional[List[str]], 
                        max_results: int, include_context: bool) -> List[Dict[str, Any]]:
        """Search citations and references"""
        conn = self.db_manager.get_sqlite_connection()
        cursor = conn.cursor()
        
        query_lower = query.lower()
        
        # Search in citations table
        params = [f"%{query_lower}%", f"%{query_lower}%", f"%{query_lower}%"]
        
        if paper_ids:
            placeholders = ','.join(['?' for _ in paper_ids])
            where_clause = f"AND c.paper_id IN ({placeholders})"
            params.extend(paper_ids)
        else:
            where_clause = ""
        
        sql = f"""
            SELECT 
                c.paper_id,
                c.title,
                c.authors,
                c.year,
                c.venue,
                c.citation_type,
                itc.citation_text,
                itc.paragraph_context
            FROM citations c
            LEFT JOIN in_text_citations itc ON c.paper_id = itc.paper_id
            WHERE (LOWER(c.title) LIKE ? OR LOWER(c.authors) LIKE ? OR LOWER(c.venue) LIKE ?)
            {where_clause}
            ORDER BY c.year DESC
            LIMIT ?
        """
        params.append(max_results)
        
        cursor.execute(sql, params)
        results = []
        
        for row in cursor.fetchall():
            (paper_id, title, authors, year, venue, citation_type,
             in_text_citation, paragraph_context) = row
            
            result = {
                "paper_id": paper_id,
                "citation_title": title,
                "citation_authors": authors,
                "citation_year": year,
                "citation_venue": venue,
                "citation_type": citation_type,
                "match_type": "citation"
            }
            
            if include_context and in_text_citation:
                result["in_text_citation"] = in_text_citation
                result["paragraph_context"] = paragraph_context[:200] + "..." if paragraph_context and len(paragraph_context) > 200 else paragraph_context
            
            results.append(result)
        
        return results
    
    def _mathematical_search(self, query: str, paper_ids: Optional[List[str]], 
                           max_results: int, include_context: bool) -> List[Dict[str, Any]]:
        """Search mathematical content"""
        conn = self.db_manager.get_sqlite_connection()
        cursor = conn.cursor()
        
        query_lower = query.lower()
        
        # Search in mathematical content
        params = [f"%{query_lower}%", f"%{query_lower}%"]
        
        if paper_ids:
            placeholders = ','.join(['?' for _ in paper_ids])
            where_clause = f"AND mcp.paper_id IN ({placeholders})"
            params.extend(paper_ids)
        else:
            where_clause = ""
        
        sql = f"""
            SELECT 
                mcp.paper_id,
                mcp.cleaned_latex,
                mcp.math_type,
                mcp.complexity_score,
                mcp.variables,
                mcp.functions,
                mcp.operators
            FROM mathematical_content_processed mcp
            WHERE (LOWER(mcp.cleaned_latex) LIKE ? OR LOWER(mcp.variables) LIKE ?)
            {where_clause}
            ORDER BY mcp.complexity_score DESC
            LIMIT ?
        """
        params.append(max_results)
        
        cursor.execute(sql, params)
        results = []
        
        for row in cursor.fetchall():
            (paper_id, latex, math_type, complexity_score,
             variables, functions, operators) = row
            
            result = {
                "paper_id": paper_id,
                "latex_content": latex,
                "math_type": math_type,
                "complexity_score": round(float(complexity_score), 2),
                "variables": variables.split(',') if variables else [],
                "functions": functions.split(',') if functions else [],
                "operators": operators.split(',') if operators else [],
                "match_type": "mathematical"
            }
            
            results.append(result)
        
        return results
    
    def get_paper_summary(self, paper_id: str, include_content_samples: bool = True) -> Optional[PaperSummary]:
        """
        Generate comprehensive paper summary for LLM consumption.
        
        Args:
            paper_id: ArXiv paper ID
            include_content_samples: Whether to include sample content for analysis
            
        Returns:
            PaperSummary with comprehensive paper information
            
        Example:
            summary = get_paper_summary("2103.01234", include_content_samples=True)
            if summary:
                print(f"Paper: {summary.title}")
                print(f"Key entities: {[e['text'] for e in summary.key_entities[:5]]}")
        """
        try:
            conn = self.db_manager.get_sqlite_connection()
            cursor = conn.cursor()
            
            # Get basic paper metadata
            cursor.execute("""
                SELECT section_count, equation_count, reference_count, 
                       chunk_count, entity_count, relationship_count
                FROM paper_metadata 
                WHERE paper_id = ?
            """, (paper_id,))
            
            metadata_row = cursor.fetchone()
            if not metadata_row:
                logger.warning(f"No metadata found for paper {paper_id}")
                return None
            
            (section_count, equation_count, reference_count,
             chunk_count, entity_count, relationship_count) = metadata_row
            
            # Get quality validation info
            cursor.execute("""
                SELECT overall_score, completeness_score, quality_score,
                       integrity_score, is_llm_ready
                FROM quality_validation 
                WHERE paper_id = ?
            """, (paper_id,))
            
            quality_row = cursor.fetchone()
            if quality_row:
                (overall_score, completeness_score, quality_score,
                 integrity_score, is_llm_ready) = quality_row
                processing_quality = {
                    "overall_score": float(overall_score),
                    "completeness_score": float(completeness_score),
                    "quality_score": float(quality_score),
                    "integrity_score": float(integrity_score)
                }
                llm_ready = bool(is_llm_ready)
            else:
                processing_quality = {}
                llm_ready = False
            
            # Get document sections
            cursor.execute("""
                SELECT section_title, section_type, section_order
                FROM document_sections 
                WHERE paper_id = ?
                ORDER BY section_order
            """, (paper_id,))
            
            sections = []
            for row in cursor.fetchall():
                sections.append({
                    "title": row[0],
                    "type": row[1],
                    "order": row[2]
                })
            
            # Get top entities by confidence
            cursor.execute("""
                SELECT entity_text, entity_type, confidence_score, mention_count
                FROM entities 
                WHERE paper_id = ?
                ORDER BY confidence_score DESC, mention_count DESC
                LIMIT 20
            """, (paper_id,))
            
            key_entities = []
            for row in cursor.fetchall():
                key_entities.append({
                    "text": row[0],
                    "type": row[1],
                    "confidence": round(float(row[2]), 3),
                    "mentions": row[3]
                })
            
            # Get mathematical content overview
            cursor.execute("""
                SELECT COUNT(*) as total_math,
                       AVG(complexity_score) as avg_complexity,
                       COUNT(DISTINCT math_type) as type_variety,
                       GROUP_CONCAT(DISTINCT variables) as all_variables
                FROM mathematical_content_processed 
                WHERE paper_id = ?
            """, (paper_id,))
            
            math_row = cursor.fetchone()
            if math_row and math_row[0]:
                mathematical_content = {
                    "total_equations": math_row[0],
                    "avg_complexity": round(float(math_row[1] or 0), 2),
                    "type_variety": math_row[2],
                    "unique_variables": len(set(math_row[3].split(',')) if math_row[3] else [])
                }
            else:
                mathematical_content = {"total_equations": 0, "avg_complexity": 0, "type_variety": 0}
            
            # Get citation info
            cursor.execute("""
                SELECT COUNT(*) as total_citations,
                       COUNT(DISTINCT citation_type) as citation_types,
                       AVG(CAST(year as INTEGER)) as avg_citation_year
                FROM citations 
                WHERE paper_id = ? AND year IS NOT NULL
            """, (paper_id,))
            
            citation_row = cursor.fetchone()
            if citation_row and citation_row[0]:
                citation_info = {
                    "total_citations": citation_row[0],
                    "citation_types": citation_row[1],
                    "avg_year": int(citation_row[2]) if citation_row[2] else None
                }
            else:
                citation_info = {"total_citations": 0}
            
            # Extract key findings, methodologies, and datasets if requested
            key_findings = []
            methodologies = []
            datasets_used = []
            abstract_text = None
            
            if include_content_samples:
                # Get sample content from different sections
                cursor.execute("""
                    SELECT cc.chunk_text, cc.canonical_section
                    FROM content_chunks cc
                    WHERE cc.paper_id = ?
                    AND cc.canonical_section IN ('abstract', 'conclusion', 'results', 'methodology')
                    ORDER BY cc.canonical_section, cc.chunk_order
                    LIMIT 10
                """, (paper_id,))
                
                for chunk_text, section_type in cursor.fetchall():
                    if section_type == 'abstract' and not abstract_text:
                        abstract_text = chunk_text[:500] + "..." if len(chunk_text) > 500 else chunk_text
                    elif section_type in ['conclusion', 'results']:
                        if len(key_findings) < 3:
                            key_findings.append(chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text)
                    elif section_type == 'methodology':
                        if len(methodologies) < 3:
                            methodologies.append(chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text)
                
                # Extract datasets from entities
                cursor.execute("""
                    SELECT DISTINCT entity_text
                    FROM entities
                    WHERE paper_id = ? AND entity_type = 'dataset'
                    ORDER BY confidence_score DESC
                    LIMIT 5
                """, (paper_id,))
                
                datasets_used = [row[0] for row in cursor.fetchall()]
            
            # Create summary
            summary = PaperSummary(
                paper_id=paper_id,
                title=f"Paper {paper_id}",  # Would need to get from search results
                authors=["Unknown"],  # Would need to get from search results
                sections=sections,
                total_chunks=chunk_count or 0,
                key_entities=key_entities,
                mathematical_content=mathematical_content,
                citation_info=citation_info,
                processing_quality=processing_quality,
                llm_ready=llm_ready,
                abstract_text=abstract_text,
                key_findings=key_findings,
                methodologies=methodologies,
                datasets_used=datasets_used,
                processed_at=datetime.now(),
                content_stats={
                    "sections": section_count or 0,
                    "equations": equation_count or 0,
                    "references": reference_count or 0,
                    "chunks": chunk_count or 0,
                    "entities": entity_count or 0,
                    "relationships": relationship_count or 0
                }
            )
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating paper summary for {paper_id}: {e}")
            return None
    
    def _prepare_fts_query(self, query: str) -> str:
        """Prepare query string for FTS5 full-text search"""
        # Remove special characters and normalize
        cleaned = re.sub(r'[^\w\s]', ' ', query.lower())
        words = [word.strip() for word in cleaned.split() if len(word.strip()) > 2]
        
        # Join with AND for phrase matching
        return ' AND '.join(f'"{word}"' for word in words[:10])  # Limit to 10 words
    
    def get_available_papers(self) -> List[Dict[str, Any]]:
        """Get list of available processed papers"""
        try:
            conn = self.db_manager.get_sqlite_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT pm.paper_id, pm.section_count, pm.chunk_count, 
                       pm.entity_count, qv.is_llm_ready, qv.overall_score
                FROM paper_metadata pm
                LEFT JOIN quality_validation qv ON pm.paper_id = qv.paper_id
                ORDER BY qv.overall_score DESC
            """)
            
            papers = []
            for row in cursor.fetchall():
                papers.append({
                    "paper_id": row[0],
                    "section_count": row[1],
                    "chunk_count": row[2],
                    "entity_count": row[3],
                    "llm_ready": bool(row[4]) if row[4] is not None else False,
                    "quality_score": float(row[5]) if row[5] is not None else 0.0
                })
            
            return papers
            
        except Exception as e:
            logger.error(f"Error getting available papers: {e}")
            return []


# Main wrapper functions for agent integration
def query_processed_data(query: str,
                        query_type: str = "semantic",
                        paper_ids: Optional[List[str]] = None,
                        max_results: int = 10,
                        include_context: bool = True) -> QueryResult:
    """
    Query processed academic paper data.
    
    This is the main query function for LangGraph agents to search through
    processed academic papers using various search strategies.
    
    Args:
        query: Natural language query or search terms
        query_type: Type of search ('semantic', 'keyword', 'entity', 'citation', 'math')
        paper_ids: Optional list to restrict search to specific papers
        max_results: Maximum results to return (default: 10)
        include_context: Whether to include contextual information
        
    Returns:
        QueryResult with search results optimized for LLM agents
        
    Example:
        # Semantic search across all papers
        result = query_processed_data(
            "What are the main quantum error correction techniques?",
            query_type="semantic"
        )
        
        # Entity search in specific papers
        result = query_processed_data(
            "neural networks",
            query_type="entity", 
            paper_ids=["2103.01234", "2104.05678"]
        )
    """
    engine = AcademicPaperQueryEngine()
    return engine.query_processed_data(query, query_type, paper_ids, max_results, include_context)


def get_paper_summary(paper_id: str, include_content_samples: bool = True) -> Optional[PaperSummary]:
    """
    Get comprehensive summary of a processed paper.
    
    This function provides LLM agents with a complete overview of a paper's
    content, structure, entities, and processing quality metrics.
    
    Args:
        paper_id: ArXiv paper ID
        include_content_samples: Whether to include sample content for analysis
        
    Returns:
        PaperSummary with comprehensive paper information, or None if not found
        
    Example:
        summary = get_paper_summary("2103.01234")
        if summary and summary.llm_ready:
            print(f"Paper has {len(summary.key_entities)} key entities")
            print(f"Quality score: {summary.processing_quality.get('overall_score', 0)}")
    """
    engine = AcademicPaperQueryEngine()
    return engine.get_paper_summary(paper_id, include_content_samples)


def get_available_papers() -> List[Dict[str, Any]]:
    """
    Get list of all available processed papers.
    
    Returns:
        List of paper metadata dictionaries with processing statistics
        
    Example:
        papers = get_available_papers()
        llm_ready_papers = [p for p in papers if p['llm_ready']]
        print(f"Found {len(llm_ready_papers)} LLM-ready papers")
    """
    engine = AcademicPaperQueryEngine()
    return engine.get_available_papers()


def format_query_results_for_llm(result: QueryResult) -> str:
    """
    Format query results as readable text for LLM consumption.
    
    Args:
        result: QueryResult from query_processed_data()
        
    Returns:
        Formatted string suitable for LLM processing
    """
    if not result.success:
        return f"Query failed: {'; '.join(result.errors or ['Unknown error'])}"
    
    if not result.results:
        return f"No results found for {result.query_type} query."
    
    lines = [
        f"Query Results ({result.query_type}):",
        f"Found {result.total_matches} matches in {result.execution_time:.2f}s",
        "=" * 50
    ]
    
    for i, item in enumerate(result.results[:5], 1):  # Show top 5 results
        lines.append(f"\nResult {i}:")
        lines.append(f"  Paper: {item.get('paper_id', 'Unknown')}")
        
        if item.get('chunk_text'):
            lines.append(f"  Content: {item['chunk_text'][:200]}...")
        if item.get('section_title'):
            lines.append(f"  Section: {item['section_title']} ({item.get('canonical_section', 'Unknown')})")
        if item.get('relevance_score'):
            lines.append(f"  Relevance: {item['relevance_score']}")
        if item.get('entity_text'):
            lines.append(f"  Entity: {item['entity_text']} ({item.get('entity_type', 'Unknown')})")
    
    if len(result.results) > 5:
        lines.append(f"\n... and {len(result.results) - 5} more results")
    
    return "\n".join(lines)


# Export main functions for agent tools
__all__ = [
    "query_processed_data", 
    "get_paper_summary", 
    "get_available_papers",
    "format_query_results_for_llm",
    "QueryResult", 
    "PaperSummary"
]