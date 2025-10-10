# wrappers/download_and_process.py
# Function 2: Download and Full Pipeline Processing for Academic Paper Search Tool
# Compatible with LangGraph agents and LLM integration

import logging
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

# Import core pipeline components
from ..core.arxiv_search import ArxivPaper
from ..core.content_downloader import ContentDownloader, DownloadResult
from ..core.document_structure_parser import DocumentStructureParser
from ..core.mathematical_content_processor import MathematicalContentProcessor
from ..core.citation_reference_parser import CitationReferenceParser
from ..core.content_chunker import ContentChunker
from ..core.entity_extractor import EntityExtractor
from ..core.metadata_enricher import MetadataEnricher
from ..core.quality_validator import QualityValidator
from ..core.index_builder import IndexBuilder
from ..core.database_setup import DatabaseManager


# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class PipelineStageResult:
    """Result of a single pipeline stage"""
    stage_name: str
    success: bool
    execution_time: float
    data_created: int
    errors: List[str]
    warnings: List[str]
    quality_score: float = 0.0

@dataclass
class CompletePipelineResult:
    """Complete pipeline processing result - optimized for LLM agents"""
    success: bool
    total_papers_processed: int
    processing_time: float
    
    # Summary statistics for LLM consumption
    summary: Dict[str, Any]
    
    # Detailed results by paper
    paper_results: Dict[str, Dict[str, Any]]
    
    # Global issues and recommendations
    issues: List[str]
    warnings: List[str]
    recommendations: List[str]
    
    # Database optimization info
    database_stats: Dict[str, Any]
    
    # LLM-ready status
    llm_ready_papers: List[str]
    total_llm_ready: int


class DownloadAndProcessPipeline:
    """
    Complete download and processing pipeline for academic papers.
    Designed for LangGraph agent integration with comprehensive error handling,
    logging, and LLM-optimized output formatting.
    """
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """Initialize pipeline with all required components"""
        self.db_manager = db_manager or DatabaseManager()
        
        # Initialize pipeline components in processing order
        self.content_downloader = ContentDownloader(db_manager=self.db_manager)
        self.document_parser = DocumentStructureParser(db_manager=self.db_manager)
        self.math_processor = MathematicalContentProcessor(db_manager=self.db_manager)
        self.citation_parser = CitationReferenceParser(db_manager=self.db_manager)
        self.content_chunker = ContentChunker(db_manager=self.db_manager)
        self.entity_extractor = EntityExtractor(db_manager=self.db_manager)
        self.metadata_enricher = MetadataEnricher(db_manager=self.db_manager)
        self.quality_validator = QualityValidator(db_manager=self.db_manager)
        self.index_builder = IndexBuilder(db_manager=self.db_manager)
        
        # Track pipeline stages for monitoring
        self.pipeline_stages = [
            ("content_download", self._process_content_download),
            ("document_structure_parsing", self._process_document_structure),
            ("mathematical_content_processing", self._process_mathematical_content),
            ("citation_reference_parsing", self._process_citation_references),
            ("content_chunking", self._process_content_chunking),
            ("entity_extraction", self._process_entity_extraction),
            ("metadata_enrichment", self._process_metadata_enrichment),
            ("quality_validation", self._process_quality_validation),
            ("index_building", self._process_index_building)
        ]

 
    def _process_content_download(self, paper: ArxivPaper) -> Dict[str, Any]:
        """Process content download stage"""
        try:
            result = self.content_downloader.download_html_content(paper)
            
            return {
                "success": result.success,
                "data_created": 1 if result.success else 0,
                "quality_score": 10.0 if result.success else 0.0,
                "errors": [result.error_message] if result.error_message else [],
                "warnings": [],
                "metadata": {
                    "file_path": result.file_path,
                    "content_size": result.content_size,
                    "download_time": result.download_time
                }
            }
        except Exception as e:
            return {
                "success": False,
                "data_created": 0,
                "quality_score": 0.0,
                "errors": [f"Download failed: {str(e)}"],
                "warnings": []
            }
    
    def _process_document_structure(self, paper_id: str) -> Dict[str, Any]:
        """Process document structure parsing stage"""
        try:
            # Load paper info and HTML content
            paper_info = self._get_paper_info(paper_id)
            html_content = self._load_html_content(paper_id)
            
            if not html_content:
                return {
                    "success": False,
                    "data_created": 0,
                    "quality_score": 0.0,
                    "errors": ["No HTML content found for document parsing"],
                    "warnings": []
                }
            
            doc_structure = self.document_parser.parse_document_structure(paper_info, html_content)
            
            return {
                "success": True,
                "data_created": len(doc_structure.elements),
                "quality_score": doc_structure.quality_metrics.get("parsing_completeness", 0) * 10,
                "errors": [],
                "warnings": [],
                "metadata": {
                    "sections_found": len(doc_structure.sections),
                    "figures_found": len(doc_structure.figures),
                    "tables_found": len(doc_structure.tables),
                    "equations_found": len(doc_structure.equations)
                }
            }
        except Exception as e:
            return {
                "success": False,
                "data_created": 0,
                "quality_score": 0.0,
                "errors": [f"Document structure parsing failed: {str(e)}"],
                "warnings": []
            }
    
    def _process_mathematical_content(self, paper_id: str) -> Dict[str, Any]:
        """Process mathematical content processing stage"""
        try:
            result = self.math_processor.process_mathematical_content(paper_id)
            
            return {
                "success": result.processed_equations > 0 or result.total_equations == 0,
                "data_created": result.processed_equations,
                "quality_score": result.quality_score,
                "errors": result.processing_errors,
                "warnings": [],
                "metadata": {
                    "total_equations": result.total_equations,
                    "inline_math": result.inline_math_count,
                    "display_math": result.display_math_count,
                    "variables_found": len(result.variables_found),
                    "functions_found": len(result.functions_found)
                }
            }
        except Exception as e:
            return {
                "success": False,
                "data_created": 0,
                "quality_score": 0.0,
                "errors": [f"Mathematical content processing failed: {str(e)}"],
                "warnings": []
            }
    
    def _process_citation_references(self, paper_id: str) -> Dict[str, Any]:
        """Process citation and reference parsing stage"""
        try:
            result = self.citation_parser.parse_citations_and_references(paper_id)
            
            return {
                "success": result.parsed_references > 0 or result.total_references == 0,
                "data_created": result.parsed_references + result.in_text_citations,
                "quality_score": result.quality_score,
                "errors": result.processing_errors,
                "warnings": [],
                "metadata": {
                    "total_references": result.total_references,
                    "parsed_references": result.parsed_references,
                    "in_text_citations": result.in_text_citations,
                    "linked_citations": result.linked_citations,
                    "citation_types": list(result.citation_types)
                }
            }
        except Exception as e:
            return {
                "success": False,
                "data_created": 0,
                "quality_score": 0.0,
                "errors": [f"Citation parsing failed: {str(e)}"],
                "warnings": []
            }
    
    def _process_content_chunking(self, paper_id: str) -> Dict[str, Any]:
        """Process content chunking stage"""
        try:
            result = self.content_chunker.chunk_paper_content(paper_id)
            
            return {
                "success": result.total_chunks > 0,
                "data_created": result.total_chunks,
                "quality_score": result.quality_score,
                "errors": result.processing_errors,
                "warnings": [],
                "metadata": {
                    "total_chunks": result.total_chunks,
                    "section_chunks": result.section_chunks,
                    "chunks_with_citations": result.chunks_with_citations,
                    "chunks_with_math": result.chunks_with_math,
                    "avg_chunk_size": result.avg_chunk_size
                }
            }
        except Exception as e:
            return {
                "success": False,
                "data_created": 0,
                "quality_score": 0.0,
                "errors": [f"Content chunking failed: {str(e)}"],
                "warnings": []
            }
    
    def _process_entity_extraction(self, paper_id: str) -> Dict[str, Any]:
        """Process entity extraction stage"""
        try:
            result = self.entity_extractor.extract_entities_and_relationships(paper_id)
            
            return {
                "success": result.total_entities > 0,
                "data_created": result.total_entities + result.total_relationships,
                "quality_score": result.quality_score,
                "errors": result.processing_errors,
                "warnings": [],
                "metadata": {
                    "total_entities": result.total_entities,
                    "total_relationships": result.total_relationships,
                    "entities_by_type": result.entities_by_type,
                    "relationships_by_type": result.relationships_by_type
                }
            }
        except Exception as e:
            return {
                "success": False,
                "data_created": 0,
                "quality_score": 0.0,
                "errors": [f"Entity extraction failed: {str(e)}"],
                "warnings": []
            }
    
    def _process_metadata_enrichment(self, paper_id: str) -> Dict[str, Any]:
        """Process metadata enrichment stage"""
        try:
            stats = self.metadata_enricher.enrich_paper(paper_id)
            
            # Calculate success based on data completeness
            total_stats = sum(v for v in stats.values() if isinstance(v, int))
            success = total_stats > 0
            
            return {
                "success": success,
                "data_created": 1,  # One metadata record created
                "quality_score": 8.0 if success else 0.0,  # Metadata enrichment is typically successful
                "errors": [],
                "warnings": [],
                "metadata": stats
            }
        except Exception as e:
            return {
                "success": False,
                "data_created": 0,
                "quality_score": 0.0,
                "errors": [f"Metadata enrichment failed: {str(e)}"],
                "warnings": []
            }
    
    def _process_quality_validation(self, paper_id: str) -> Dict[str, Any]:
        """Process quality validation stage"""
        try:
            result = self.quality_validator.validate_paper(paper_id)
            
            return {
                "success": True,  # Validation always produces results
                "data_created": 1,  # One validation record created
                "quality_score": result.overall_score / 10,  # Convert to 0-10 scale
                "errors": result.validation_issues,
                "warnings": result.validation_warnings,
                "metadata": {
                    "overall_score": result.overall_score,
                    "completeness_score": result.completeness_score,
                    "quality_score": result.quality_score,
                    "integrity_score": result.integrity_score,
                    "is_llm_ready": result.is_llm_ready,
                    "recommendations": result.recommendations
                }
            }
        except Exception as e:
            return {
                "success": False,
                "data_created": 0,
                "quality_score": 0.0,
                "errors": [f"Quality validation failed: {str(e)}"],
                "warnings": []
            }
    
    def _process_index_building(self, paper_id: str) -> Dict[str, Any]:
        """Process index building stage (runs once for all papers)"""
        try:
            # Build SQLite indexes
            self.index_builder.build_sqlite_indexes()
            
            # Build full-text search
            self.index_builder.build_fts()
            
            # Optimize database
            self.index_builder.optimize_sqlite()
            
            return {
                "success": True,
                "data_created": 1,  # Index creation
                "quality_score": 9.0,  # Index building typically succeeds
                "errors": [],
                "warnings": [],
                "metadata": {"indexes_built": True, "fts_enabled": True}
            }
        except Exception as e:
            return {
                "success": False,
                "data_created": 0,
                "quality_score": 0.0,
                "errors": [f"Index building failed: {str(e)}"],
                "warnings": []
            }
    
    def _get_paper_info(self, paper_id: str) -> ArxivPaper:
        """Get paper info from database or create minimal version"""
        # Create minimal ArxivPaper for processing
        return ArxivPaper(
            id=paper_id,
            title=f"Paper {paper_id}",
            authors=["Unknown"],
            abstract="",
            published=datetime.now(),
            updated=datetime.now(),
            categories=[],
            pdf_url="",
            abstract_url="",
            html_url=""
        )
    
    def _load_html_content(self, paper_id: str) -> Optional[str]:
        """Load HTML content from database"""
        try:
            conn = self.db_manager.get_sqlite_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT file_path FROM content_downloads 
                WHERE arxiv_id = ? AND content_type = 'html' AND success = 1
            """, (paper_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            with open(row[0], 'r', encoding='utf-8') as f:
                return f.read()
                
        except Exception as e:
            logger.error(f"Error loading HTML content for {paper_id}: {e}")
            return None
    
    def _get_llm_ready_papers(self, paper_ids: List[str]) -> List[str]:
        """Get list of LLM-ready papers from quality validation"""
        try:
            conn = self.db_manager.get_sqlite_connection()
            cursor = conn.cursor()
            
            placeholders = ','.join(['?' for _ in paper_ids])
            cursor.execute(f"""
                SELECT paper_id FROM quality_validation 
                WHERE paper_id IN ({placeholders}) AND is_llm_ready = 1
            """, paper_ids)
            
            return [row[0] for row in cursor.fetchall()]
            
        except Exception as e:
            logger.error(f"Error getting LLM-ready papers: {e}")
            return []
    
    def _generate_summary_stats(self, successful_papers: List[str], 
                               failed_papers: List[str], 
                               paper_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Generate summary statistics optimized for LLM consumption"""
        
        # Calculate averages for successful papers
        total_success = len(successful_papers)
        avg_processing_time = 0
        avg_stages_success = 0
        stage_success_rates = {}
        
        if total_success > 0:
            total_time = sum(paper_results[pid]["processing_time"] for pid in successful_papers)
            avg_processing_time = total_time / total_success
            
            # Calculate stage success rates
            all_stages = set()
            for pid in successful_papers:
                all_stages.update(paper_results[pid]["stages"].keys())
            
            for stage in all_stages:
                stage_successes = sum(
                    1 for pid in successful_papers 
                    if paper_results[pid]["stages"].get(stage, {}).get("success", False)
                )
                stage_success_rates[stage] = (stage_successes / total_success) * 100
        
        return {
            "total_papers": len(successful_papers) + len(failed_papers),
            "successful_papers": total_success,
            "failed_papers": len(failed_papers),
            "success_rate": (total_success / (total_success + len(failed_papers))) * 100 if (total_success + len(failed_papers)) > 0 else 0,
            "avg_processing_time": round(avg_processing_time, 2),
            "stage_success_rates": stage_success_rates,
            "processing_date": datetime.now().isoformat()
        }
    
    def _optimize_database(self) -> Dict[str, Any]:
        """Optimize database for better performance"""
        try:
            conn = self.db_manager.get_sqlite_connection()
            cursor = conn.cursor()
            
            # Get database stats before optimization
            cursor.execute("PRAGMA page_count")
            page_count_before = cursor.fetchone()[0]
            
            cursor.execute("PRAGMA freelist_count")
            freelist_before = cursor.fetchone()[0]
            
            # Run VACUUM to defragment and optimize
            cursor.execute("VACUUM")
            
            # Run ANALYZE to update statistics
            cursor.execute("ANALYZE")
            
            # Get stats after optimization
            cursor.execute("PRAGMA page_count")
            page_count_after = cursor.fetchone()[0]
            
            cursor.execute("PRAGMA freelist_count")
            freelist_after = cursor.fetchone()[0]
            
            return {
                "optimization_completed": True,
                "pages_before": page_count_before,
                "pages_after": page_count_after,
                "pages_freed": page_count_before - page_count_after,
                "freelist_before": freelist_before,
                "freelist_after": freelist_after,
                "space_saved_percent": ((page_count_before - page_count_after) / page_count_before * 100) if page_count_before > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Database optimization failed: {e}")
            return {"optimization_failed": True, "error": str(e)}
    
    def _generate_recommendations(self, successful_papers: List[str], 
                                 failed_papers: List[str], 
                                 paper_results: Dict[str, Dict],
                                 global_issues: List[str]) -> List[str]:
        """Generate recommendations based on processing results"""
        recommendations = []
        
        # Calculate failure rate
        total_papers = len(successful_papers) + len(failed_papers)
        failure_rate = len(failed_papers) / total_papers if total_papers > 0 else 0
        
        if failure_rate > 0.5:
            recommendations.append("High failure rate detected - review paper selection criteria")
            
        if failure_rate > 0.2:
            recommendations.append("Consider implementing retry logic for failed papers")
        
        # Analyze common failure patterns
        common_errors = {}
        for issue in global_issues:
            if "failed" in issue.lower():
                stage = issue.split("failed")[0].strip()
                common_errors[stage] = common_errors.get(stage, 0) + 1
        
        for stage, count in common_errors.items():
            if count > 1:
                recommendations.append(f"Multiple failures in {stage} - review stage implementation")
        
        # Check processing efficiency
        if successful_papers:
            avg_time = sum(paper_results[pid]["processing_time"] for pid in successful_papers) / len(successful_papers)
            if avg_time > 300:  # 5 minutes per paper
                recommendations.append("Processing time is high - consider optimization")
        
        return recommendations
    

def download_and_process_papers(papers: List[ArxivPaper], 
                              enable_database_optimization: bool = True,
                              skip_failed_papers: bool = True) -> CompletePipelineResult:
    """
    Download and process academic papers through the complete pipeline.
    
    This is the main wrapper function for LangGraph agents. It orchestrates the
    complete academic paper processing pipeline from download to indexing.
    
    Args:
        papers: List of ArxivPaper objects from search_and_select_papers()
        enable_database_optimization: Whether to run database optimizations
        skip_failed_papers: Whether to continue processing if some papers fail
        
    Returns:
        CompletePipelineResult: Comprehensive results optimized for LLM consumption
        
    Example:
        papers = search_and_select_papers("quantum machine learning", max_papers=3)
        result = download_and_process_papers(papers)
        
        if result.success:
            print(f"Processed {result.total_llm_ready} LLM-ready papers")
            # Use result.paper_results for detailed analysis
        else:
            print(f"Pipeline issues: {result.issues}")
    """
    
    pipeline = DownloadAndProcessPipeline()
    start_time = datetime.now()
    
    logger.info(f"Starting complete pipeline for {len(papers)} papers")
    
    # Initialize result tracking
    paper_results = {}
    global_issues = []
    global_warnings = []
    global_recommendations = []
    successful_papers = []
    failed_papers = []
    
    # Process each paper through the complete pipeline
    for i, paper in enumerate(papers, 1):
        paper_id = paper.id
        logger.info(f"Processing paper {i}/{len(papers)}: {paper_id}")
        
        paper_start_time = datetime.now()
        paper_stage_results = {}
        paper_success = True
        paper_errors = []
        
        # Run through all pipeline stages
        for stage_name, stage_func in pipeline.pipeline_stages:
            try:
                stage_start = datetime.now()
                stage_result = stage_func(paper if stage_name == "content_download" else paper_id)
                stage_time = (datetime.now() - stage_start).total_seconds()
                
                paper_stage_results[stage_name] = {
                    "success": stage_result.get("success", False),
                    "execution_time": stage_time,
                    "data_created": stage_result.get("data_created", 0),
                    "quality_score": stage_result.get("quality_score", 0.0),
                    "errors": stage_result.get("errors", []),
                    "warnings": stage_result.get("warnings", [])
                }
                
                # Track global issues
                if stage_result.get("errors"):
                    global_issues.extend(stage_result["errors"])
                    paper_errors.extend(stage_result["errors"])
                    
                if stage_result.get("warnings"):
                    global_warnings.extend(stage_result["warnings"])
                    
                # Check if stage failed critically
                if not stage_result.get("success", False):
                    paper_success = False
                    error_msg = f"Stage {stage_name} failed for paper {paper_id}"
                    logger.error(error_msg)
                    paper_errors.append(error_msg)
                    
                    if not skip_failed_papers:
                        break  # Stop processing this paper
                        
            except Exception as e:
                error_msg = f"Stage {stage_name} crashed for paper {paper_id}: {str(e)}"
                logger.error(error_msg)
                paper_errors.append(error_msg)
                global_issues.append(error_msg)
                paper_success = False
                
                if not skip_failed_papers:
                    break
        
        # Calculate paper processing time
        paper_time = (datetime.now() - paper_start_time).total_seconds()
        
        # Store paper results
        paper_results[paper_id] = {
            "success": paper_success,
            "processing_time": paper_time,
            "stages": paper_stage_results,
            "errors": paper_errors,
            "title": paper.title[:100] + "..." if len(paper.title) > 100 else paper.title,
            "authors": paper.authors[:3],  # First 3 authors for brevity
            "categories": paper.categories
        }
        
        # Track successful vs failed papers
        if paper_success:
            successful_papers.append(paper_id)
        else:
            failed_papers.append(paper_id)
            logger.warning(f"Paper {paper_id} failed with {len(paper_errors)} errors")
    
    # Calculate total processing time
    total_time = (datetime.now() - start_time).total_seconds()
    
    # Get LLM-ready papers from quality validation
    llm_ready_papers = pipeline._get_llm_ready_papers(successful_papers)
    
    # Generate summary statistics optimized for LLM consumption
    summary = pipeline._generate_summary_stats(successful_papers, failed_papers, paper_results)
    
    # Run database optimization if requested
    database_stats = {}
    if enable_database_optimization and successful_papers:
        try:
            database_stats = pipeline._optimize_database()
        except Exception as e:
            global_warnings.append(f"Database optimization failed: {str(e)}")
            database_stats = {"optimization_failed": True}
    
    # Generate recommendations based on results
    recommendations = pipeline._generate_recommendations(
        successful_papers, failed_papers, paper_results, global_issues
    )
    global_recommendations.extend(recommendations)
    
    # Create comprehensive result
    result = CompletePipelineResult(
        success=len(successful_papers) > 0,
        total_papers_processed=len(papers),
        processing_time=total_time,
        summary=summary,
        paper_results=paper_results,
        issues=list(set(global_issues)),  # Remove duplicates
        warnings=list(set(global_warnings)),  # Remove duplicates  
        recommendations=list(set(global_recommendations)),  # Remove duplicates
        database_stats=database_stats,
        llm_ready_papers=llm_ready_papers,
        total_llm_ready=len(llm_ready_papers)
    )
    
    # Log final results
    logger.info(f"Pipeline completed: {len(successful_papers)}/{len(papers)} papers successful, "
                        f"{len(llm_ready_papers)} LLM-ready, {total_time:.1f}s total")
    
    return result


# Convenience function for LangGraph integration
def get_pipeline_summary(result: CompletePipelineResult) -> str:
    """
    Generate a human-readable summary of pipeline results for LLM consumption.
    
    Args:
        result: CompletePipelineResult from download_and_process_papers()
        
    Returns:
        str: Formatted summary suitable for LLM processing
    """
    
    summary_lines = [
        f"Pipeline Processing Summary:",
        f"{'='*50}",
        f"Papers Processed: {result.total_papers_processed}",
        f"Success Rate: {result.summary.get('success_rate', 0):.1f}%",
        f"LLM-Ready Papers: {result.total_llm_ready}/{result.total_papers_processed}",
        f"Total Processing Time: {result.processing_time:.1f} seconds",
        f""
    ]
    
    if result.llm_ready_papers:
        summary_lines.extend([
            f"✅ Ready for Analysis:",
            f"Papers: {', '.join(result.llm_ready_papers[:5])}{'...' if len(result.llm_ready_papers) > 5 else ''}",
            f""
        ])
    
    if result.issues:
        summary_lines.extend([
            f"🚨 Issues Encountered:",
            *[f"  • {issue}" for issue in result.issues[:3]],
            f"  ... and {len(result.issues)-3} more" if len(result.issues) > 3 else "",
            f""
        ])
    
    if result.recommendations:
        summary_lines.extend([
            f"💡 Recommendations:",
            *[f"  • {rec}" for rec in result.recommendations[:3]],
            f""
        ])
    
    return "\n".join(summary_lines)


# Export main function for agent tools
__all__ = ["download_and_process_papers", "get_pipeline_summary", "CompletePipelineResult"]