#%%
from typing import List, Optional, Callable, Dict
from dataclasses import dataclass
from enum import Enum
from academic_paper_tool.core.arxiv_search import ArxivPaper, arxiv_search
# Using the existing ArxivPaper dataclass from your code
# (already defined in the-html-link-links-to-an-overview-page-not-the-p.md)

class SelectionStrategy(Enum):
    """Available paper selection strategies"""
    HTML_ONLY = "html_only"
    HTML_PREFERRED = "html_preferred"
    PDF_ONLY = "pdf_only"
    ANY_FORMAT = "any_format"
    METADATA_ONLY = "metadata_only"


@dataclass
class SelectionCriteria:
    """Configuration for paper selection"""
    max_papers: int = 3
    strategy: SelectionStrategy = SelectionStrategy.HTML_PREFERRED
    require_abstract: bool = True
    min_authors: int = 1
    recency_weight: float = 0.0  # 0.0 = no recency bias, 1.0 = strong recency bias
    custom_filter: Optional[Callable[[ArxivPaper], bool]] = None
    custom_scorer: Optional[Callable[[ArxivPaper], float]] = None


def calculate_paper_score(paper: ArxivPaper, criteria: SelectionCriteria) -> float:
    """
    Calculate quality/relevance score for a paper
    
    Args:
        paper: ArxivPaper object to score
        criteria: Selection criteria with scoring parameters
    
    Returns:
        Float score (higher is better)
    """
    score = 0.0
    
    # Format availability scoring
    if criteria.strategy == SelectionStrategy.HTML_ONLY:
        if not paper.html_url:
            return -1.0  # Disqualify
        score += 10.0
    elif criteria.strategy == SelectionStrategy.HTML_PREFERRED:
        if paper.html_url:
            score += 10.0
        elif paper.pdf_url:
            score += 5.0
    elif criteria.strategy == SelectionStrategy.PDF_ONLY:
        if not paper.pdf_url:
            return -1.0  # Disqualify
        score += 10.0
    elif criteria.strategy == SelectionStrategy.METADATA_ONLY:
        score += 5.0
    else:  # ANY_FORMAT
        if paper.html_url or paper.pdf_url:
            score += 10.0
    
    # Abstract completeness
    if criteria.require_abstract and paper.abstract:
        score += 2.0
        # Bonus for detailed abstracts
        if len(paper.abstract) > 500:
            score += 1.0
    
    # Author count quality indicator
    if len(paper.authors) >= criteria.min_authors:
        score += 1.0
        # Bonus for collaborative work (but not too many authors)
        if 2 <= len(paper.authors) <= 8:
            score += 0.5
    
    # Recency scoring
    if criteria.recency_weight > 0:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        days_old = (now - paper.published).days
        # Exponential decay: more recent = higher score
        recency_score = 5.0 * (0.99 ** days_old)
        score += recency_score * criteria.recency_weight
    
    # Category diversity bonus (papers with multiple categories)
    if len(paper.categories) > 1:
        score += 0.5
    
    # Custom scoring function
    if criteria.custom_scorer:
        score += criteria.custom_scorer(paper)
    
    return score


def select_papers_for_analysis(
    papers: List[ArxivPaper],
    criteria: Optional[SelectionCriteria] = None
) -> List[ArxivPaper]:
    """
    Select top papers for analysis based on configurable criteria
    
    Args:
        papers: List of ArxivPaper objects from arxiv_search()
        criteria: Selection criteria (uses defaults if None)
    
    Returns:
        List of selected ArxivPaper objects (up to max_papers)
    
    Example:
        # Default: top 3 HTML-preferred papers
        selected = select_papers_for_analysis(search_results)
        
        # HTML only
        criteria = SelectionCriteria(max_papers=5, strategy=SelectionStrategy.HTML_ONLY)
        selected = select_papers_for_analysis(search_results, criteria)
        
        # Custom filter for specific category
        criteria = SelectionCriteria(
            max_papers=3,
            custom_filter=lambda p: 'cs.AI' in p.categories
        )
        selected = select_papers_for_analysis(search_results, criteria)
    """
    if criteria is None:
        criteria = SelectionCriteria()
    
    # Apply custom filter first if provided
    filtered_papers = papers
    if criteria.custom_filter:
        filtered_papers = [p for p in papers if criteria.custom_filter(p)]
    
    # Apply basic quality filters
    quality_filtered = []
    for paper in filtered_papers:
        # Check abstract requirement
        if criteria.require_abstract and not paper.abstract:
            continue
        
        # Check minimum authors
        if len(paper.authors) < criteria.min_authors:
            continue
        
        quality_filtered.append(paper)
    
    # Score all papers
    scored_papers = []
    for paper in quality_filtered:
        score = calculate_paper_score(paper, criteria)
        if score >= 0:  # Only include non-disqualified papers
            scored_papers.append((score, paper))
    
    # Sort by score (highest first) and return top N
    scored_papers.sort(key=lambda x: x[0], reverse=True)
    selected = [paper for score, paper in scored_papers[:criteria.max_papers]]
    
    return selected


def get_selection_summary(papers: List[ArxivPaper]) -> Dict[str, any]:
    """
    Generate summary statistics for selected papers
    
    Args:
        papers: List of selected ArxivPaper objects
    
    Returns:
        Dictionary with selection statistics
    """
    if not papers:
        return {"count": 0}
    
    html_count = sum(1 for p in papers if p.html_url)
    pdf_count = sum(1 for p in papers if p.pdf_url)
    
    return {
        "count": len(papers),
        "html_available": html_count,
        "pdf_available": pdf_count,
        "html_percentage": (html_count / len(papers)) * 100,
        "avg_authors": sum(len(p.authors) for p in papers) / len(papers),
        "categories": list(set(cat for p in papers for cat in p.categories))
    }


# Example usage demonstrating different selection strategies
if __name__ == "__main__":
    # Assume we have search results from arxiv_search()
    search_results = arxiv_search("machine learning", max_results=20, fetch_html=True)
    
    # Example 1: Default - top 3 HTML-preferred papers
    selected = select_papers_for_analysis(search_results)
    print(f"Selected {len(selected)} papers (HTML preferred)")
    
    # Example 2: HTML only, 5 papers
    html_criteria = SelectionCriteria(
        max_papers=5,
        strategy=SelectionStrategy.HTML_ONLY
    )
    html_selected = select_papers_for_analysis(search_results, html_criteria)
    print(f"Selected {len(html_selected)} HTML-only papers")
    
    # Example 3: Recent papers with custom scoring
    def favor_recent(paper: ArxivPaper) -> float:
        """Custom scorer that heavily favors recent papers"""
        from datetime import datetime, timezone
        days_old = (datetime.now(timezone.utc) - paper.published).days
        if days_old < 30:
            return 5.0
        elif days_old < 90:
            return 2.0
        return 0.0
    
    recent_criteria = SelectionCriteria(
        max_papers=3,
        strategy=SelectionStrategy.HTML_PREFERRED,
        custom_scorer=favor_recent
    )
    recent_selected = select_papers_for_analysis(search_results, recent_criteria)
    
    # Example 4: Specific category filter
    ai_criteria = SelectionCriteria(
        max_papers=3,
        custom_filter=lambda p: any('cs.AI' in cat or 'cs.LG' in cat for cat in p.categories)
    )
    ai_selected = select_papers_for_analysis(search_results, ai_criteria)
    
    # Get summary statistics
    summary = get_selection_summary(selected)
    print(f"\nSelection Summary: {summary}")
