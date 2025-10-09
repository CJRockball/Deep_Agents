#%%
# mathematical_content_processor.py - Step 4: Mathematical Content Processing

import re
import hashlib
from typing import List, Dict, Optional, Tuple, Any, Set
from dataclasses import dataclass
from datetime import datetime
import logging
import sqlite3

from .database_setup import DatabaseManager

@dataclass
class MathElement:
    """Enhanced mathematical element with parsed content"""
    id: str
    paper_id: str
    original_latex: str
    cleaned_latex: str
    equation_number: Optional[str] = None
    context_before: Optional[str] = None
    context_after: Optional[str] = None
    position: int = 0
    section_id: Optional[str] = None
    math_type: str = 'unknown'  # 'equation', 'inline', 'formula'
    complexity_score: float = 0.0
    variables: List[str] = None
    functions: List[str] = None
    operators: List[str] = None
    constants: List[str] = None
    processed_at: Optional[datetime] = None

@dataclass
class MathProcessingResult:
    """Result of mathematical content processing"""
    paper_id: str
    total_equations: int
    processed_equations: int
    inline_math_count: int
    display_math_count: int
    variables_found: Set[str]
    functions_found: Set[str]
    processing_errors: List[str]
    quality_score: float

class MathematicalContentProcessor:
    """
    Processes atomic mathematical elements identified by document structure parser.
    This is Step 4 in the academic paper processing pipeline.
    """

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db_manager = db_manager or DatabaseManager()
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # LaTeX patterns for different mathematical components
        self.latex_patterns = {
            'variables': r'\b[a-zA-Z]\b|\\[a-zA-Z]+\{[^}]*\}',
            'functions': r'\\(?:sin|cos|tan|log|ln|exp|sqrt|frac|sum|int|lim|max|min|arg|det)\b',
            'operators': r'\\(?:cdot|times|div|pm|mp|leq|geq|neq|approx|equiv|subset|supset|in|notin)\b',
            'constants': r'\\(?:pi|infty|alpha|beta|gamma|delta|epsilon|lambda|mu|sigma|theta|phi)\b',
            'delimiters': r'\\(?:left|right)\s*[()\[\]\{\}|]',
            'environments': r'\\begin\{([^}]*)\}.*?\\end\{\1\}',
            'equation_ref': r'\\(?:eq|eqref)\{([^}]*)\}',
            'labels': r'\\label\{([^}]*)\}'
        }

        # Common mathematical function names
        self.known_functions = {
            'sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh',
            'log', 'ln', 'exp', 'sqrt', 'abs', 'det',
            'sum', 'prod', 'int', 'lim', 'max', 'min',
            'arg', 'sup', 'inf', 'gcd', 'lcm'
        }

        # Variable patterns (single letters, Greek letters)
        self.variable_pattern = re.compile(r'\b[a-zA-Z]\b|\\[a-zA-Z]+')

    def process_mathematical_content(self, paper_id: str) -> MathProcessingResult:
        """
        Main entry point for processing mathematical content.

        Args:
            paper_id: ArXiv paper ID to process

        Returns:
            MathProcessingResult with processing statistics
        """
        self.logger.info(f"Processing mathematical content for paper {paper_id}")

        try:
            # Get mathematical content from database (saved by document structure parser)
            math_elements = self._load_math_elements_from_db(paper_id)

            if not math_elements:
                self.logger.warning(f"No mathematical content found for paper {paper_id}")
                return MathProcessingResult(
                    paper_id=paper_id,
                    total_equations=0,
                    processed_equations=0,
                    inline_math_count=0,
                    display_math_count=0,
                    variables_found=set(),
                    functions_found=set(),
                    processing_errors=["No mathematical content found"],
                    quality_score=0.0
                )

            self.logger.info(f"Found {len(math_elements)} mathematical elements to process")

            # Process each math element
            processed_elements = []
            processing_errors = []
            all_variables = set()
            all_functions = set()

            for element in math_elements:
                try:
                    processed_element = self._process_single_math_element(element)
                    processed_elements.append(processed_element)

                    if processed_element.variables:
                        all_variables.update(processed_element.variables)
                    if processed_element.functions:
                        all_functions.update(processed_element.functions)

                except Exception as e:
                    error_msg = f"Error processing math element {element.id}: {str(e)}"
                    processing_errors.append(error_msg)
                    self.logger.error(error_msg)

            # Count different types
            inline_count = sum(1 for e in processed_elements if e.math_type == 'inline')
            display_count = sum(1 for e in processed_elements if e.math_type == 'equation')

            # Calculate quality score
            quality_score = self._calculate_math_quality_score(processed_elements)

            # Save processed elements back to database
            self._save_processed_math_elements(processed_elements)

            # Update processing status
            self._update_processing_status(paper_id, 'math_processing', len(processing_errors) == 0)

            result = MathProcessingResult(
                paper_id=paper_id,
                total_equations=len(math_elements),
                processed_equations=len(processed_elements),
                inline_math_count=inline_count,
                display_math_count=display_count,
                variables_found=all_variables,
                functions_found=all_functions,
                processing_errors=processing_errors,
                quality_score=quality_score
            )

            self.logger.info(f"Mathematical processing completed for {paper_id}: "
                           f"{result.processed_equations}/{result.total_equations} processed, "
                           f"quality score: {result.quality_score:.2f}")

            return result

        except Exception as e:
            self.logger.error(f"Error in mathematical content processing for {paper_id}: {e}")
            return MathProcessingResult(
                paper_id=paper_id,
                total_equations=0,
                processed_equations=0,
                inline_math_count=0,
                display_math_count=0,
                variables_found=set(),
                functions_found=set(),
                processing_errors=[f"Processing failed: {str(e)}"],
                quality_score=0.0
            )

    def _load_math_elements_from_db(self, paper_id: str) -> List[MathElement]:
        """Load mathematical elements from database (saved by document structure parser)"""

        conn = self.db_manager.get_sqlite_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, equation_number, latex_content, context_before, 
                   context_after, position, section_id
            FROM mathematical_content 
            WHERE paper_id = ? 
            ORDER BY position
        """, (paper_id,))

        elements = []
        for row in cursor.fetchall():
            element_id, eq_num, latex, ctx_before, ctx_after, pos, section_id = row

            element = MathElement(
                id=str(element_id),  # Convert to string
                paper_id=paper_id,
                original_latex=latex or "",
                cleaned_latex="",  # Will be filled during processing
                equation_number=eq_num,
                context_before=ctx_before,
                context_after=ctx_after,
                position=pos or 0,
                section_id=section_id,
                variables=[],
                functions=[],
                operators=[],
                constants=[]
            )
            elements.append(element)

        return elements

    def _process_single_math_element(self, element: MathElement) -> MathElement:
        """Process a single mathematical element"""

        # Clean the LaTeX content
        element.cleaned_latex = self._clean_latex(element.original_latex)

        # Determine math type
        element.math_type = self._determine_math_type(element)

        # Extract mathematical components
        element.variables = self._extract_variables(element.cleaned_latex)
        element.functions = self._extract_functions(element.cleaned_latex)
        element.operators = self._extract_operators(element.cleaned_latex)
        element.constants = self._extract_constants(element.cleaned_latex)

        # Calculate complexity score
        element.complexity_score = self._calculate_complexity_score(element)

        # Record processing time
        element.processed_at = datetime.now()

        return element

    def _clean_latex(self, latex_content: str) -> str:
        """Clean and normalize LaTeX content"""

        if not latex_content:
            return ""

        # Remove common LaTeX wrapper patterns
        cleaned = latex_content.strip()

        # Remove display math delimiters but preserve content
        cleaned = re.sub(r'^\\\[(.*)\\\]$', r'\1', cleaned)
        cleaned = re.sub(r'^\\\((.*)\\\)$', r'\1', cleaned)
        cleaned = re.sub(r'^\$\$(.*)\$\$$', r'\1', cleaned)
        cleaned = re.sub(r'^\$(.*)\$$', r'\1', cleaned)

        # Normalize spacing
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = cleaned.strip()

        return cleaned

    def _determine_math_type(self, element: MathElement) -> str:
        """Determine if math is inline, display equation, or formula"""

        latex = element.original_latex or ""

        # Check for display math indicators
        if any(pattern in latex for pattern in ['\\[', '$$', 'equation}']):
            return 'equation'

        # Check for inline math indicators  
        if any(pattern in latex for pattern in ['\\(', '$']):
            return 'inline'

        # Check if it has an equation number
        if element.equation_number:
            return 'equation'

        # Default based on content complexity
        if len(latex) > 50 or '\\frac' in latex or '\\sum' in latex:
            return 'equation'

        return 'inline'

    def _extract_variables(self, latex: str) -> List[str]:
        """Extract variable names from LaTeX"""

        variables = set()

        # Find single letter variables (most common)
        single_letters = re.findall(r'\b[a-zA-Z]\b', latex)
        variables.update(single_letters)

        # Find Greek letters and LaTeX symbols
        greek_letters = re.findall(r'\\(alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|pi|rho|sigma|tau|upsilon|phi|chi|psi|omega)', latex)
        variables.update(greek_letters)

        # Remove known function names
        variables = variables - self.known_functions

        return sorted(list(variables))

    def _extract_functions(self, latex: str) -> List[str]:
        """Extract function names from LaTeX"""

        functions = set()

        # Find LaTeX function commands
        latex_functions = re.findall(r'\\(sin|cos|tan|sinh|cosh|tanh|log|ln|exp|sqrt|det|sum|prod|int|lim|max|min|arg|sup|inf)', latex)
        functions.update(latex_functions)

        # Find custom function patterns (letters followed by parentheses)
        custom_functions = re.findall(r'([a-zA-Z]+)\s*\(', latex)
        functions.update(custom_functions)

        return sorted(list(functions))

    def _extract_operators(self, latex: str) -> List[str]:
        """Extract mathematical operators from LaTeX"""

        operators = set()

        # Find LaTeX operator commands
        latex_operators = re.findall(r'\\(cdot|times|div|pm|mp|leq|geq|neq|approx|equiv|subset|supset|subseteq|supseteq|in|notin|cap|cup)', latex)
        operators.update(latex_operators)

        # Find basic operators
        basic_operators = re.findall(r'[+\-*/=<>]', latex)
        operators.update(basic_operators)

        return sorted(list(operators))

    def _extract_constants(self, latex: str) -> List[str]:
        """Extract mathematical constants from LaTeX"""

        constants = set()

        # Find LaTeX constant commands
        latex_constants = re.findall(r'\\(pi|infty|partial)', latex)
        constants.update(latex_constants)

        # Find numeric constants
        numeric_constants = re.findall(r'\b\d+(?:\.\d+)?\b', latex)
        constants.update(numeric_constants)

        return sorted(list(constants))

    def _calculate_complexity_score(self, element: MathElement) -> float:
        """Calculate complexity score for mathematical element"""

        score = 0.0
        latex = element.cleaned_latex

        # Base complexity from length
        score += len(latex) * 0.01

        # Complexity from components
        score += len(element.variables or []) * 0.5
        score += len(element.functions or []) * 1.0
        score += len(element.operators or []) * 0.3

        # Complexity from special constructs
        if '\\frac' in latex: score += 2.0
        if '\\sum' in latex or '\\int' in latex: score += 3.0
        if '\\sqrt' in latex: score += 1.0
        if '\\matrix' in latex or '\\begin{' in latex: score += 4.0

        # Normalize to 0-10 scale
        return min(score, 10.0)

    def _calculate_math_quality_score(self, elements: List[MathElement]) -> float:
        """Calculate overall quality score for mathematical content processing"""

        if not elements:
            return 0.0

        # Base score from successful processing
        processed_ratio = len(elements) / max(len(elements), 1)
        base_score = processed_ratio * 5.0

        # Bonus for variety of mathematical content
        has_functions = any(e.functions for e in elements)
        has_variables = any(e.variables for e in elements)
        has_operators = any(e.operators for e in elements)

        variety_bonus = sum([has_functions, has_variables, has_operators]) * 1.0

        # Penalty for very simple content (all single characters)
        avg_complexity = sum(e.complexity_score for e in elements) / len(elements)
        complexity_bonus = min(avg_complexity * 0.3, 3.0)

        total_score = base_score + variety_bonus + complexity_bonus
        return min(total_score, 10.0)

    def _save_processed_math_elements(self, elements: List[MathElement]):
        """Save processed mathematical elements back to database with enhanced metadata"""

        try:
            conn = self.db_manager.get_sqlite_connection()
            cursor = conn.cursor()

            # Create enhanced mathematical content table if needed
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS mathematical_content_processed (
                    id INTEGER PRIMARY KEY,
                    paper_id VARCHAR(50),
                    original_element_id INTEGER,
                    cleaned_latex TEXT,
                    math_type VARCHAR(20),
                    complexity_score REAL,
                    variables TEXT,
                    functions TEXT,
                    operators TEXT,
                    constants TEXT,
                    processed_at TEXT,
                    FOREIGN KEY (paper_id) REFERENCES papers(arxiv_id)
                )
            """)

            # Save each processed element
            for element in elements:
                cursor.execute("""
                    INSERT OR REPLACE INTO mathematical_content_processed 
                    (paper_id, original_element_id, cleaned_latex, math_type, 
                     complexity_score, variables, functions, operators, constants, processed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    element.paper_id,
                    int(element.id) if element.id.isdigit() else None,
                    element.cleaned_latex,
                    element.math_type,
                    element.complexity_score,
                    ','.join(element.variables or []),
                    ','.join(element.functions or []),
                    ','.join(element.operators or []),
                    ','.join(element.constants or []),
                    element.processed_at.isoformat() if element.processed_at else None
                ))

            conn.commit()
            self.logger.info(f"Saved {len(elements)} processed mathematical elements")

        except Exception as e:
            self.logger.error(f"Error saving processed math elements: {e}")

    def _update_processing_status(self, paper_id: str, stage: str, success: bool):
        """Update processing status in database"""

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

    def get_math_processing_stats(self, paper_id: str) -> Dict[str, Any]:
        """Get mathematical processing statistics for a paper"""

        try:
            conn = self.db_manager.get_sqlite_connection()
            cursor = conn.cursor()

            # Get basic stats
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    AVG(complexity_score) as avg_complexity,
                    COUNT(DISTINCT math_type) as type_variety
                FROM mathematical_content_processed 
                WHERE paper_id = ?
            """, (paper_id,))

            basic_stats = cursor.fetchone()

            # Get component counts
            cursor.execute("""
                SELECT 
                    SUM(LENGTH(variables) - LENGTH(REPLACE(variables, ',', '')) + 1) as total_variables,
                    SUM(LENGTH(functions) - LENGTH(REPLACE(functions, ',', '')) + 1) as total_functions
                FROM mathematical_content_processed 
                WHERE paper_id = ? AND variables != ''
            """, (paper_id,))

            component_stats = cursor.fetchone()

            return {
                'total_processed': basic_stats[0] if basic_stats else 0,
                'average_complexity': round(basic_stats[1] or 0, 2),
                'type_variety': basic_stats[2] if basic_stats else 0,
                'total_variables': component_stats[0] if component_stats and component_stats[0] else 0,
                'total_functions': component_stats[1] if component_stats and component_stats[1] else 0
            }

        except Exception as e:
            self.logger.error(f"Error getting math processing stats: {e}")
            return {}

# Usage example and integration with existing pipeline
def process_paper_mathematics(paper_id: str) -> MathProcessingResult:
    """
    Standalone function to process mathematical content for a paper.
    This is called after document structure parsing (Step 3) is complete.
    """
    processor = MathematicalContentProcessor()
    return processor.process_mathematical_content(paper_id)



# test_math_processor.py - Simple test for mathematical content processing

from .database_setup import DatabaseManager

def main():
    """Test mathematical content processing on a paper from the database"""

    # Initialize
    db_manager = DatabaseManager()
    processor = MathematicalContentProcessor(db_manager)

    # Get a paper that has mathematical content from document structure parsing
    conn = db_manager.get_sqlite_connection()
    cursor = conn.cursor()

    # Find a paper with mathematical content
    cursor.execute("""
        SELECT DISTINCT paper_id 
        FROM mathematical_content 
        WHERE latex_content IS NOT NULL 
        LIMIT 1
    """)

    row = cursor.fetchone()
    if not row:
        print("No papers with mathematical content found.")
        print("Run document structure parser first!")
        return

    paper_id = row[0]
    print(f"Processing mathematical content for paper: {paper_id}")

    # Check what math elements exist
    cursor.execute("""
        SELECT COUNT(*), 
               COUNT(CASE WHEN equation_number IS NOT NULL THEN 1 END) as numbered_equations
        FROM mathematical_content 
        WHERE paper_id = ?
    """, (paper_id,))

    count_info = cursor.fetchone()
    total_math, numbered = count_info
    print(f"Found {total_math} mathematical elements ({numbered} numbered equations)")

    # Process the mathematical content
    result = processor.process_mathematical_content(paper_id)

    # Display results
    print("\n" + "="*50)
    print("MATHEMATICAL PROCESSING RESULTS")
    print("="*50)
    print(f"Paper ID: {result.paper_id}")
    print(f"Total equations found: {result.total_equations}")
    print(f"Successfully processed: {result.processed_equations}")
    print(f"Inline math elements: {result.inline_math_count}")
    print(f"Display equations: {result.display_math_count}")
    print(f"Quality score: {result.quality_score:.2f}/10.0")

    if result.variables_found:
        print(f"\nVariables found: {', '.join(sorted(result.variables_found))}")

    if result.functions_found:
        print(f"Functions found: {', '.join(sorted(result.functions_found))}")

    if result.processing_errors:
        print(f"\nProcessing errors:")
        for error in result.processing_errors:
            print(f"  - {error}")

    # Show processing stats
    stats = processor.get_math_processing_stats(paper_id)
    if stats:
        print(f"\nDetailed Statistics:")
        print(f"  Average complexity: {stats.get('average_complexity', 0)}")
        print(f"  Total variables: {stats.get('total_variables', 0)}")
        print(f"  Total functions: {stats.get('total_functions', 0)}")

    # Show some examples of processed math
    cursor.execute("""
        SELECT cleaned_latex, math_type, complexity_score, variables, functions
        FROM mathematical_content_processed 
        WHERE paper_id = ? 
        LIMIT 3
    """, (paper_id,))

    examples = cursor.fetchall()
    if examples:
        print(f"\nExample processed equations:")
        for i, (latex, math_type, complexity, vars_str, funcs_str) in enumerate(examples):
            print(f"  {i+1}. {latex[:60]}{'...' if len(latex) > 60 else ''}")
            print(f"     Type: {math_type}, Complexity: {complexity:.1f}")
            if vars_str:
                print(f"     Variables: {vars_str}")
            if funcs_str:
                print(f"     Functions: {funcs_str}")

    print("\n" + "="*50)
    print("✅ Mathematical content processing complete!")
    print("Check 'mathematical_content_processed' table for detailed results.")
    print("="*50)

if __name__ == "__main__":
    main()

# %%
