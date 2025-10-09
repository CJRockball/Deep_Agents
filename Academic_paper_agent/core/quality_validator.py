#%%
# quality_validator.py - Step 9: Quality Validation and Verification

import sqlite3
from datetime import datetime
import logging
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from database_setup import DatabaseManager

@dataclass
class ValidationResult:
    """Result of quality validation process"""
    paper_id: str
    overall_score: float  # 0-100
    completeness_score: float
    quality_score: float
    integrity_score: float
    validation_issues: List[str]
    validation_warnings: List[str]
    recommendations: List[str]
    is_llm_ready: bool

class QualityValidator:
    """
    Validates the quality and completeness of processed academic paper data.
    This is Step 9 in the academic paper processing pipeline.
    """

    def __init__(self, db_manager=None):
        self.db = db_manager or DatabaseManager()
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Quality thresholds
        self.thresholds = {
            'min_sections': 5,          # Minimum sections for a complete paper
            'min_chunks': 10,           # Minimum chunks for good coverage
            'min_chunk_size': 100,      # Minimum characters per chunk
            'max_chunk_size': 2000,     # Maximum characters per chunk
            'min_entities': 5,          # Minimum entities for rich content
            'min_entity_confidence': 0.3,  # Minimum confidence for entities
            'min_citations': 3,         # Minimum citations for academic rigor
            'expected_sections': [      # Sections expected in academic papers
                'abstract', 'introduction', 'methodology', 'results', 
                'conclusion', 'references'
            ]
        }

    def validate_paper(self, paper_id: str) -> ValidationResult:
        """
        Main validation function for a processed paper.

        Args:
            paper_id: ArXiv paper ID to validate

        Returns:
            ValidationResult with comprehensive quality assessment
        """
        self.logger.info(f"Validating paper quality for {paper_id}")

        try:
            # Run validation checks
            completeness_result = self._check_completeness(paper_id)
            quality_result = self._check_quality(paper_id)
            integrity_result = self._check_integrity(paper_id)

            # Combine results
            overall_score = (
                completeness_result['score'] * 0.4 +  # 40% weight
                quality_result['score'] * 0.4 +       # 40% weight  
                integrity_result['score'] * 0.2       # 20% weight
            )

            # Collect all issues and recommendations
            all_issues = (
                completeness_result['issues'] + 
                quality_result['issues'] + 
                integrity_result['issues']
            )
            all_warnings = (
                completeness_result['warnings'] + 
                quality_result['warnings'] + 
                integrity_result['warnings']
            )
            all_recommendations = (
                completeness_result['recommendations'] + 
                quality_result['recommendations'] + 
                integrity_result['recommendations']
            )

            # Determine if ready for LLM
            is_llm_ready = (
                overall_score >= 70 and
                len([issue for issue in all_issues if 'critical' in issue.lower()]) == 0 and
                completeness_result['score'] >= 60
            )

            result = ValidationResult(
                paper_id=paper_id,
                overall_score=overall_score,
                completeness_score=completeness_result['score'],
                quality_score=quality_result['score'],
                integrity_score=integrity_result['score'],
                validation_issues=all_issues,
                validation_warnings=all_warnings,
                recommendations=all_recommendations,
                is_llm_ready=is_llm_ready
            )

            # Save validation results
            self._save_validation_results(result)

            self.logger.info(f"Validation completed for {paper_id}: "
                           f"Score {overall_score:.1f}/100, LLM Ready: {is_llm_ready}")

            return result

        except Exception as e:
            self.logger.error(f"Error validating {paper_id}: {e}")
            return ValidationResult(
                paper_id=paper_id,
                overall_score=0,
                completeness_score=0,
                quality_score=0,
                integrity_score=0,
                validation_issues=[f"Validation failed: {str(e)}"],
                validation_warnings=[],
                recommendations=["Re-run pipeline processing"],
                is_llm_ready=False
            )

    def _check_completeness(self, paper_id: str) -> Dict[str, Any]:
        """Check data completeness across all pipeline stages"""

        conn = self.db.get_sqlite_connection()
        c = conn.cursor()

        issues = []
        warnings = []
        recommendations = []
        score_components = []

        # Check document structure
        try:
            c.execute("SELECT COUNT(*), GROUP_CONCAT(section_type) FROM document_sections WHERE paper_id = ?", (paper_id,))
            section_count, section_types = c.fetchone()

            if section_count < self.thresholds['min_sections']:
                issues.append(f"Insufficient sections: {section_count} < {self.thresholds['min_sections']}")
                score_components.append(0)
            else:
                score_components.append(100)

            # Check for expected sections
            if section_types:
                found_sections = set(section_types.split(','))
                expected_sections = set(self.thresholds['expected_sections'])
                missing_sections = expected_sections - found_sections

                if missing_sections:
                    warnings.append(f"Missing expected sections: {', '.join(missing_sections)}")
                    recommendations.append("Check document structure parsing accuracy")
                    score_components.append(70)
                else:
                    score_components.append(100)
            else:
                issues.append("No document sections found")
                score_components.append(0)

        except Exception as e:
            issues.append(f"Error checking document structure: {e}")
            score_components.append(0)

        # Check content chunks
        try:
            c.execute("SELECT COUNT(*) FROM content_chunks WHERE paper_id = ?", (paper_id,))
            chunk_count = c.fetchone()[0]

            if chunk_count < self.thresholds['min_chunks']:
                issues.append(f"Insufficient chunks: {chunk_count} < {self.thresholds['min_chunks']}")
                score_components.append(0)
            else:
                score_components.append(100)

        except Exception as e:
            issues.append(f"Error checking content chunks: {e}")
            score_components.append(0)

        # Check mathematical content
        try:
            c.execute("SELECT COUNT(*) FROM mathematical_content_processed WHERE paper_id = ?", (paper_id,))
            math_count = c.fetchone()[0]

            if math_count == 0:
                warnings.append("No mathematical content found")
                score_components.append(80)  # Not critical for all papers
            else:
                score_components.append(100)

        except Exception as e:
            warnings.append(f"Error checking mathematical content: {e}")
            score_components.append(80)

        # Check citations
        try:
            c.execute("SELECT COUNT(*) FROM citations WHERE paper_id = ?", (paper_id,))
            citation_count = c.fetchone()[0]

            if citation_count < self.thresholds['min_citations']:
                issues.append(f"Insufficient citations: {citation_count} < {self.thresholds['min_citations']}")
                score_components.append(0)
            else:
                score_components.append(100)

        except Exception as e:
            issues.append(f"Error checking citations: {e}")
            score_components.append(0)

        # Check entities
        try:
            c.execute("SELECT COUNT(*) FROM entities WHERE paper_id = ?", (paper_id,))
            entity_count = c.fetchone()[0]

            if entity_count < self.thresholds['min_entities']:
                warnings.append(f"Few entities extracted: {entity_count} < {self.thresholds['min_entities']}")
                recommendations.append("Review entity extraction patterns")
                score_components.append(70)
            else:
                score_components.append(100)

        except Exception as e:
            warnings.append(f"Error checking entities: {e}")
            score_components.append(70)

        # Calculate completeness score
        completeness_score = sum(score_components) / len(score_components) if score_components else 0

        return {
            'score': completeness_score,
            'issues': issues,
            'warnings': warnings,
            'recommendations': recommendations
        }

    def _check_quality(self, paper_id: str) -> Dict[str, Any]:
        """Check data quality and consistency"""

        conn = self.db.get_sqlite_connection()
        c = conn.cursor()

        issues = []
        warnings = []
        recommendations = []
        score_components = []

        # Check chunk quality
        try:
            c.execute("""
                SELECT AVG(char_count), MIN(char_count), MAX(char_count), COUNT(*)
                FROM content_chunks WHERE paper_id = ?
            """, (paper_id,))
            avg_size, min_size, max_size, total_chunks = c.fetchone()

            if min_size and min_size < self.thresholds['min_chunk_size']:
                warnings.append(f"Some chunks too small: min {min_size} chars")
                score_components.append(80)
            elif max_size and max_size > self.thresholds['max_chunk_size']:
                warnings.append(f"Some chunks too large: max {max_size} chars")
                recommendations.append("Adjust chunk size parameters")
                score_components.append(85)
            else:
                score_components.append(100)

        except Exception as e:
            warnings.append(f"Error checking chunk quality: {e}")
            score_components.append(80)

        # Check entity confidence scores
        try:
            c.execute("""
                SELECT AVG(confidence_score), MIN(confidence_score), COUNT(*)
                FROM entities WHERE paper_id = ?
            """, (paper_id,))
            avg_confidence, min_confidence, entity_count = c.fetchone()

            if min_confidence and min_confidence < self.thresholds['min_entity_confidence']:
                warnings.append(f"Low confidence entities found: min {min_confidence:.2f}")
                recommendations.append("Review entity extraction confidence thresholds")
                score_components.append(75)
            else:
                score_components.append(100)

        except Exception as e:
            warnings.append(f"Error checking entity quality: {e}")
            score_components.append(80)

        # Check citation completeness
        try:
            c.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN authors IS NOT NULL THEN 1 ELSE 0 END) as with_authors,
                    SUM(CASE WHEN title IS NOT NULL THEN 1 ELSE 0 END) as with_titles
                FROM citations WHERE paper_id = ?
            """, (paper_id,))
            total_cites, with_authors, with_titles = c.fetchone()

            if total_cites > 0:
                author_completeness = with_authors / total_cites
                title_completeness = with_titles / total_cites

                if author_completeness < 0.7:
                    warnings.append(f"Many citations missing authors: {author_completeness*100:.0f}% complete")
                    score_components.append(70)
                elif title_completeness < 0.5:
                    warnings.append(f"Many citations missing titles: {title_completeness*100:.0f}% complete")
                    score_components.append(80)
                else:
                    score_components.append(100)
            else:
                score_components.append(50)  # No citations to evaluate

        except Exception as e:
            warnings.append(f"Error checking citation quality: {e}")
            score_components.append(70)

        # Calculate quality score
        quality_score = sum(score_components) / len(score_components) if score_components else 0

        return {
            'score': quality_score,
            'issues': issues,
            'warnings': warnings,
            'recommendations': recommendations
        }

    def _check_integrity(self, paper_id: str) -> Dict[str, Any]:
        """Check data integrity and relationships"""

        conn = self.db.get_sqlite_connection()
        c = conn.cursor()

        issues = []
        warnings = []
        recommendations = []
        score_components = []

        # Check for orphaned chunks (chunks without entities)
        try:
            c.execute("""
                SELECT COUNT(*) FROM content_chunks cc
                LEFT JOIN entities e ON cc.id = e.chunk_id
                WHERE cc.paper_id = ? AND e.chunk_id IS NULL
            """, (paper_id,))
            orphaned_chunks = c.fetchone()[0]

            if orphaned_chunks > 0:
                warnings.append(f"{orphaned_chunks} chunks have no entities")
                recommendations.append("Review entity extraction coverage")
                score_components.append(85)
            else:
                score_components.append(100)

        except Exception as e:
            warnings.append(f"Error checking chunk-entity relationships: {e}")
            score_components.append(90)

        # Check processing log completeness
        try:
            c.execute("""
                SELECT DISTINCT processing_stage, status
                FROM processing_log WHERE paper_id = ?
            """, (paper_id,))
            processing_stages = c.fetchall()

            expected_stages = [
                'document_structure', 'math_processing', 'citation_parsing',
                'content_chunking', 'entity_extraction'
            ]

            completed_stages = {stage for stage, status in processing_stages if status == 'success'}
            missing_stages = set(expected_stages) - completed_stages

            if missing_stages:
                issues.append(f"Incomplete pipeline stages: {', '.join(missing_stages)}")
                recommendations.append("Re-run missing pipeline stages")
                score_components.append(60)
            else:
                score_components.append(100)

        except Exception as e:
            warnings.append(f"Error checking processing integrity: {e}")
            score_components.append(80)

        # Calculate integrity score
        integrity_score = sum(score_components) / len(score_components) if score_components else 0

        return {
            'score': integrity_score,
            'issues': issues,
            'warnings': warnings,
            'recommendations': recommendations
        }

    def _save_validation_results(self, result: ValidationResult):
        """Save validation results to database"""

        try:
            conn = self.db.get_sqlite_connection()
            c = conn.cursor()

            # Create validation results table
            c.execute("""
                CREATE TABLE IF NOT EXISTS quality_validation (
                    paper_id TEXT PRIMARY KEY,
                    overall_score REAL,
                    completeness_score REAL,
                    quality_score REAL,
                    integrity_score REAL,
                    validation_issues TEXT,
                    validation_warnings TEXT,
                    recommendations TEXT,
                    is_llm_ready BOOLEAN,
                    validated_at TEXT,
                    FOREIGN KEY (paper_id) REFERENCES papers(arxiv_id)
                )
            """)

            # Insert validation results
            c.execute("""
                INSERT OR REPLACE INTO quality_validation
                (paper_id, overall_score, completeness_score, quality_score, integrity_score,
                 validation_issues, validation_warnings, recommendations, is_llm_ready, validated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.paper_id,
                result.overall_score,
                result.completeness_score,
                result.quality_score,
                result.integrity_score,
                '|'.join(result.validation_issues),
                '|'.join(result.validation_warnings),
                '|'.join(result.recommendations),
                result.is_llm_ready,
                datetime.now().isoformat()
            ))

            conn.commit()
            self.logger.info(f"Saved validation results for {result.paper_id}")

        except Exception as e:
            self.logger.error(f"Error saving validation results: {e}")

# Standalone function for pipeline integration
def validate_paper_quality(paper_id: str) -> ValidationResult:
    """
    Validate quality of processed paper.
    This is called after all processing steps are complete.
    """
    validator = QualityValidator()
    return validator.validate_paper(paper_id)



# test_quality_validator.py - Simple test for quality validation

from database_setup import DatabaseManager

def main():
    """Test quality validation on processed papers"""

    # Initialize
    db_manager = DatabaseManager()
    validator = QualityValidator(db_manager)

    # Find processed papers
    conn = db_manager.get_sqlite_connection()
    c = conn.cursor()

    # Find papers that have been processed through multiple stages
    processed_papers = set()

    # Check various processing tables
    for table in ['document_sections', 'content_chunks', 'entities']:
        try:
            c.execute(f"SELECT DISTINCT paper_id FROM {table}")
            for row in c.fetchall():
                processed_papers.add(row[0])
        except Exception as e:
            print(f"Warning: Could not check {table}: {e}")

    if not processed_papers:
        print("No processed papers found for validation!")
        print("Run the pipeline steps first:")
        print("  python simple_test.py")
        print("  python test_content_chunker.py")
        print("  python test_entity_extractor.py")
        return

    print(f"Found {len(processed_papers)} processed papers to validate")
    print("Papers:", ', '.join(processed_papers))

    # Validate each paper
    for paper_id in processed_papers:
        print(f"\n{'='*60}")
        print(f"VALIDATING: {paper_id}")
        print('='*60)

        result = validator.validate_paper(paper_id)

        # Display overall results
        print(f"Overall Quality Score: {result.overall_score:.1f}/100")
        print(f"LLM Ready: {'✅ YES' if result.is_llm_ready else '❌ NO'}")

        # Display component scores
        print(f"\nComponent Scores:")
        print(f"  Completeness: {result.completeness_score:.1f}/100")
        print(f"  Quality:      {result.quality_score:.1f}/100")
        print(f"  Integrity:    {result.integrity_score:.1f}/100")

        # Display issues
        if result.validation_issues:
            print(f"\n🚨 Critical Issues ({len(result.validation_issues)}):")
            for issue in result.validation_issues:
                print(f"  • {issue}")

        # Display warnings
        if result.validation_warnings:
            print(f"\n⚠️  Warnings ({len(result.validation_warnings)}):")
            for warning in result.validation_warnings:
                print(f"  • {warning}")

        # Display recommendations
        if result.recommendations:
            print(f"\n💡 Recommendations ({len(result.recommendations)}):")
            for rec in result.recommendations:
                print(f"  • {rec}")

        # Quality assessment
        print(f"\n📊 Quality Assessment:")
        if result.overall_score >= 90:
            print("  🏆 EXCELLENT - Premium quality data")
        elif result.overall_score >= 80:
            print("  🥇 VERY GOOD - High quality data")
        elif result.overall_score >= 70:
            print("  🥈 GOOD - Suitable for most applications")
        elif result.overall_score >= 60:
            print("  🥉 FAIR - Usable with some limitations")
        else:
            print("  ❌ POOR - Requires significant improvements")

    # Show summary statistics
    if len(processed_papers) > 1:
        print(f"\n{'='*60}")
        print("VALIDATION SUMMARY")
        print('='*60)

        # Get summary stats from database
        c.execute("""
            SELECT 
                COUNT(*) as total_papers,
                AVG(overall_score) as avg_score,
                SUM(CASE WHEN is_llm_ready THEN 1 ELSE 0 END) as llm_ready_count,
                MIN(overall_score) as min_score,
                MAX(overall_score) as max_score
            FROM quality_validation
        """)

        stats = c.fetchone()
        if stats and stats[0] > 0:
            total, avg_score, llm_ready, min_score, max_score = stats
            print(f"Total papers validated: {total}")
            print(f"Average quality score: {avg_score:.1f}/100")
            print(f"LLM-ready papers: {llm_ready}/{total} ({llm_ready/total*100:.0f}%)")
            print(f"Score range: {min_score:.1f} - {max_score:.1f}")

    print(f"\n{'='*60}")
    print("✅ Quality validation complete!")
    print("Results saved to 'quality_validation' table.")
    print("Papers marked as LLM-ready can be used for RAG.")
    print('='*60)

if __name__ == "__main__":
    main()

# %%
