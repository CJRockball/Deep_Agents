#%%
# pipeline_runner.py - Simplified Pipeline Runner (for already-downloaded papers)

import logging
from datetime import datetime
from typing import List, Dict, Any

# Import pipeline components  
from database_setup import DatabaseManager
from document_structure_parser import DocumentStructureParser
from mathematical_content_processor import MathematicalContentProcessor
from citation_reference_parser import CitationReferenceParser
from content_chunker import ContentChunker
from entity_extractor import EntityExtractor
from metadata_enricher import MetadataEnricher
from quality_validator import QualityValidator
from index_builder import IndexBuilder

class PipelineRunner:
    """
    Runs the complete academic paper processing pipeline.
    Works with already-downloaded papers.
    """

    def __init__(self):
        self.db_manager = DatabaseManager()
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Initialize processing components (skip downloader since papers already downloaded)
        self.structure_parser = DocumentStructureParser(self.db_manager)
        self.math_processor = MathematicalContentProcessor(self.db_manager)
        self.citation_parser = CitationReferenceParser(self.db_manager)
        self.chunker = ContentChunker(self.db_manager)
        self.entity_extractor = EntityExtractor(self.db_manager)
        self.metadata_enricher = MetadataEnricher(self.db_manager)
        self.quality_validator = QualityValidator(self.db_manager)
        self.index_builder = IndexBuilder(self.db_manager)

    def process_paper_complete(self, paper_id: str) -> Dict[str, Any]:
        """
        Run complete pipeline on a single already-downloaded paper.

        Args:
            paper_id: ArXiv paper ID (e.g., "2407.17293v1")

        Returns:
            Dictionary with results from each pipeline stage
        """
        self.logger.info(f"Starting pipeline processing for {paper_id}")

        results = {
            'paper_id': paper_id,
            'start_time': datetime.now().isoformat(),
            'stages': {},
            'overall_success': False,
            'final_quality_score': 0.0,
            'successful_stages': 0,
            'total_stages': 0
        }

        # Stage 1: Document Structure Parsing
        try:
            self.logger.info(f"Stage 1: Parsing document structure")
            structure_result = self.structure_parser.parse_paper_structure(paper_id)
            results['stages']['structure'] = {
                'success': structure_result.total_sections > 0,
                'sections_found': structure_result.total_sections,
                'math_elements': structure_result.math_elements_found,
                'quality_score': structure_result.quality_score
            }
        except Exception as e:
            self.logger.error(f"Stage 1 failed: {e}")
            results['stages']['structure'] = {'success': False, 'error': str(e)}

        # Stage 2: Mathematical Content Processing
        try:
            self.logger.info(f"Stage 2: Processing mathematical content")
            math_result = self.math_processor.process_paper_mathematics(paper_id)
            results['stages']['mathematics'] = {
                'success': True,
                'equations_processed': math_result.processed_equations,
                'variables_found': len(math_result.variables_found or []),
                'quality_score': math_result.quality_score
            }
        except Exception as e:
            self.logger.error(f"Stage 2 failed: {e}")
            results['stages']['mathematics'] = {'success': False, 'error': str(e)}

        # Stage 3: Citation and Reference Parsing
        try:
            self.logger.info(f"Stage 3: Parsing citations")
            citation_result = self.citation_parser.parse_citations_and_references(paper_id)
            results['stages']['citations'] = {
                'success': citation_result.total_references > 0,
                'references_found': citation_result.total_references,
                'in_text_citations': citation_result.in_text_citations,
                'quality_score': citation_result.quality_score
            }
        except Exception as e:
            self.logger.error(f"Stage 3 failed: {e}")
            results['stages']['citations'] = {'success': False, 'error': str(e)}

        # Stage 4: Content Chunking
        try:
            self.logger.info(f"Stage 4: Creating content chunks")
            chunk_result = self.chunker.chunk_paper_content(paper_id)
            results['stages']['chunking'] = {
                'success': chunk_result.total_chunks > 0,
                'chunks_created': chunk_result.total_chunks,
                'avg_chunk_size': chunk_result.avg_chunk_size,
                'quality_score': chunk_result.quality_score
            }
        except Exception as e:
            self.logger.error(f"Stage 4 failed: {e}")
            results['stages']['chunking'] = {'success': False, 'error': str(e)}

        # Stage 5: Entity Extraction
        try:
            self.logger.info(f"Stage 5: Extracting entities")
            entity_result = self.entity_extractor.extract_entities_and_relationships(paper_id)
            results['stages']['entities'] = {
                'success': entity_result.total_entities > 0,
                'entities_extracted': entity_result.total_entities,
                'relationships_found': entity_result.total_relationships,
                'quality_score': entity_result.quality_score
            }
        except Exception as e:
            self.logger.error(f"Stage 5 failed: {e}")
            results['stages']['entities'] = {'success': False, 'error': str(e)}

        # Stage 6: Metadata Enrichment
        try:
            self.logger.info(f"Stage 6: Enriching metadata")
            metadata_stats = self.metadata_enricher.enrich_paper(paper_id)
            results['stages']['metadata'] = {
                'success': True,
                'statistics': metadata_stats
            }
        except Exception as e:
            self.logger.error(f"Stage 6 failed: {e}")
            results['stages']['metadata'] = {'success': False, 'error': str(e)}

        # Stage 7: Quality Validation
        try:
            self.logger.info(f"Stage 7: Validating quality")
            quality_result = self.quality_validator.validate_paper(paper_id)
            results['stages']['quality'] = {
                'success': True,
                'overall_score': quality_result.overall_score,
                'is_llm_ready': quality_result.is_llm_ready,
                'issues': quality_result.validation_issues,
                'warnings': quality_result.validation_warnings,
                'recommendations': quality_result.recommendations
            }
            results['final_quality_score'] = quality_result.overall_score
        except Exception as e:
            self.logger.error(f"Stage 7 failed: {e}")
            results['stages']['quality'] = {'success': False, 'error': str(e)}

        # Stage 8: Index Building
        try:
            self.logger.info(f"Stage 8: Building indexes")
            self.index_builder.build_sqlite_indexes()
            self.index_builder.build_fts()
            self.index_builder.optimize_sqlite()
            results['stages']['indexing'] = {'success': True}
        except Exception as e:
            self.logger.error(f"Stage 8 failed: {e}")
            results['stages']['indexing'] = {'success': False, 'error': str(e)}

        # Calculate overall success
        successful_stages = sum(1 for stage in results['stages'].values() if stage.get('success', False))
        total_stages = len(results['stages'])
        results['successful_stages'] = successful_stages
        results['total_stages'] = total_stages
        results['overall_success'] = successful_stages >= (total_stages * 0.7)
        results['end_time'] = datetime.now().isoformat()

        self.logger.info(f"Pipeline completed: {successful_stages}/{total_stages} stages successful")

        return results

def run_complete_pipeline(paper_ids: List[str]):
    """Run complete pipeline on paper IDs."""
    runner = PipelineRunner()

    if len(paper_ids) == 1:
        result = runner.process_paper_complete(paper_ids[0])

        print("\n" + "="*60)
        print("COMPLETE PIPELINE RESULTS")
        print("="*60)
        print(f"Paper: {result['paper_id']}")
        print(f"Overall Success: {'✅ YES' if result['overall_success'] else '❌ NO'}")
        print(f"Quality Score: {result.get('final_quality_score', 0):.1f}/100")
        print(f"Successful Stages: {result['successful_stages']}/{result['total_stages']}")

        print(f"\nStage Results:")
        for stage_name, stage_result in result['stages'].items():
            status = "✅" if stage_result.get('success') else "❌"
            if 'error' in stage_result:
                print(f"  {status} {stage_name.title()}: Error - {stage_result['error']}")
            else:
                print(f"  {status} {stage_name.title()}")

        if result['stages'].get('quality', {}).get('recommendations'):
            print(f"\nRecommendations:")
            for rec in result['stages']['quality']['recommendations']:
                print(f"  • {rec}")

    else:
        # Process multiple papers
        for paper_id in paper_ids:
            result = runner.process_paper_complete(paper_id)
            status = "✅" if result['overall_success'] else "❌"
            print(f"{status} {paper_id}: {result.get('final_quality_score', 0):.1f}/100")

# Test function
if __name__ == "__main__":
    from database_setup import DatabaseManager

    db = DatabaseManager()
    conn = db.get_sqlite_connection()
    c = conn.cursor()

    # Find downloaded papers
    c.execute("SELECT DISTINCT arxiv_id FROM content_downloads WHERE success = 1 LIMIT 1")
    row = c.fetchone()

    if row:
        test_paper_id = row[0]
        print(f"Running pipeline on: {test_paper_id}")
        run_complete_pipeline([test_paper_id])
    else:
        print("No downloaded papers found.")
        print("Download papers first with content_downloader.py")
